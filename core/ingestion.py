from typing import List, Optional
from pathlib import Path

from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Settings,
    load_index_from_storage,
    SimpleDirectoryReader,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.extractors import TitleExtractor
from llama_index.core.ingestion import IngestionPipeline
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.storage.docstore.redis import RedisDocumentStore
from llama_index.storage.index_store.redis import RedisIndexStore
import chromadb
from llama_index.embeddings.dashscope import DashScopeEmbedding
from llama_index.llms.openai_like import OpenAILike
from config.settings import Settings as AppSettings
from utils.logger import setup_logger
from core.pdfProcessor import MultimodalPDFProcessor

logger = setup_logger(__name__)


class DocumentIngestionPipeline:
    """文档摄取管道"""

    def __init__(self):
        self._setup_models()
        self._create_pipeline()
        self.pdf_processor = MultimodalPDFProcessor()
        self.index: Optional[VectorStoreIndex] = None
        self.storage_context: Optional[StorageContext] = None
        self.chroma_vector_store: Optional[ChromaVectorStore] = None

        # 初始化存储组件
        self._initialize_storage_components()

    def _setup_models(self):
        """设置LLM和嵌入模型（走 OpenAI 兼容端点，兼容 qwen3.x 等非官方 OpenAI 模型名）"""
        Settings.llm = OpenAILike(
            model=AppSettings.OPENAI_MODEL,
            api_key=AppSettings.OPENAI_API_KEY,
            api_base=AppSettings.API_BASE_URL,
            temperature=AppSettings.OPENAI_TEMPERATURE,
            is_chat_model=True,
        )
        Settings.embed_model = DashScopeEmbedding(
            model_name=AppSettings.EMBEDDING_MODEL,
            api_key=AppSettings.OPENAI_API_KEY,
        )

    def _create_pipeline(self):
        """创建摄取管道"""
        self.pipeline = IngestionPipeline(
            transformations=[
                SentenceSplitter(
                    chunk_size=AppSettings.CHUNK_SIZE,
                    chunk_overlap=AppSettings.CHUNK_OVERLAP
                ),
                TitleExtractor(nodes=AppSettings.TITLE_EXTRACTOR_NODES),  # 前五个标题作为所有节点的标题
                Settings.embed_model,
            ]
        )

    def _initialize_storage_components(self):
        """初始化存储组件"""
        # 初始化索引存储
        self.redis_index_store = RedisIndexStore.from_host_and_port(
            host="127.0.0.1", port=6379, namespace="redis_index"
        )
        # 初始化文档存储
        self.redis_document_store = RedisDocumentStore.from_host_and_port(
            host="127.0.0.1", port=6379, namespace="redis_docs"
        )
        # 初始化向量存储
        self._create_chroma_db()

    def _create_chroma_db(self):
        """创建Chroma向量存储"""
        chroma_client = chromadb.PersistentClient(AppSettings.CHROMA_PERSIST_DIR)
        chroma_collection = chroma_client.get_or_create_collection("quickstart")
        self.chroma_vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    def _create_storage_context(self):
        """创建存储上下文"""
        self.storage_context = StorageContext.from_defaults(
            index_store=self.redis_index_store,
            docstore=self.redis_document_store,
            vector_store=self.chroma_vector_store
        )

    def ingest_documents(self, file_paths: List[str]) -> str:
        """摄取文档并创建索引"""
        try:
            nodes = []
            documents = []
            for file_path in file_paths:
                if not Path(file_path).exists():
                    logger.warning(f"文件不存在: {file_path}")
                    continue

                if Path(file_path).suffix == ".pdf":
                    text_nodes = self.pdf_processor.create_multimodal_nodes(file_path)
                    nodes.extend(text_nodes)
                    # 按节点类型统计，方便调试
                    type_counts: dict = {}
                    for n in text_nodes:
                        t = n.metadata.get("type", "text")
                        type_counts[t] = type_counts.get(t, 0) + 1
                    logger.info(f"PDF节点统计: {type_counts}")
                else:
                    reader = SimpleDirectoryReader(input_files=[file_path])
                    docs = reader.load_data()
                    documents.extend(docs)
                logger.info(f"已读取文档: {file_path}")

            if not documents and not nodes:
                return "没有找到有效的文档"

            if documents:
                # 运行摄取管道
                logger.info("开始处理文档...")
                nodes += self.pipeline.run(documents=documents)
                logger.info(f"文档处理完成，生成了 {len(nodes)} 个节点")

            # 将文档存入Redis文档存储器中
            self.redis_document_store.add_documents(nodes)
            # 创建存储上下文
            self._create_storage_context()

            # 创建向量索引或获取索引
            logger.info("创建向量索引并存入向量数据库...如果已存在索引，将文档添加到已有索引中")
            if not self.index:
                logger.info("创建文档对应索引对象")
                self.index = VectorStoreIndex(nodes, storage_context=self.storage_context)
            else:
                logger.info("获取当前索引，进行添加节点")
                self.index.insert_nodes(nodes)

            # 调试信息：创建索引后
            print(f" 索引后 docstore 文档数: {len(self.storage_context.docstore.docs)}")
            print(f" 索引 ID: {self.index.index_id}")

            result = f"成功摄取了 {len(file_paths)} 个文档，生成了 {len(nodes)} 个节点"
            logger.info(result)
            return result

        except Exception as e:
            error_msg = f"文档摄取失败: {str(e)}"
            logger.error(error_msg)
            return error_msg

    def get_documents(self):
        """加载已存在的索引和文档"""
        logger.info("读取已有向量数据库中的文档和索引...")
        self._create_storage_context()
        # 加载已经存储的索引、文档、向量
        self.index = load_index_from_storage(self.storage_context)
        print("index:", self.index)
