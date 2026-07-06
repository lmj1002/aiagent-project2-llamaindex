from llama_index.core import VectorStoreIndex
from llama_index.core.workflow import Context, Workflow, step, StopEvent, StartEvent
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.postprocessor import LLMRerank
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever
from config.settings import Settings as AppSettings
from core.events import RAGEvents
from utils.logger import setup_logger

logger = setup_logger(__name__)


class RAGWorkflow(Workflow):
    """RAG工作流"""

    def __init__(self, index: VectorStoreIndex, **kwargs):
        super().__init__(**kwargs, timeout=None)
        self.index = index
        self._setup_components()

    def _setup_components(self):
        """设置工作流组件"""
        # 配置向量检索器
        self.retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=AppSettings.SIMILARITY_TOP_K
        )
        # 配置重排序器：使用 LLMRerank，调用已配置的 DashScope LLM，无需本地模型
        self.postprocessor = LLMRerank(
            top_n=AppSettings.RERANK_TOP_K,
            choice_batch_size=AppSettings.RERANK_CHOICE_BATCH_SIZE,
        )
        self.BM25_retriever = BM25Retriever.from_defaults(
            docstore=self.index.docstore, similarity_top_k=AppSettings.SIMILARITY_TOP_K
        )
        # 创建混合检索器
        self.queryFusionRetriever = QueryFusionRetriever(
            [
                self.retriever,
                self.BM25_retriever,
            ],
            num_queries=1,
            use_async=True,
        )
        # 配置响应合成器（不使用查询引擎，避免重复检索）
        self.response_synthesizer = get_response_synthesizer()

    @step
    async def retrieve_step(
            self,
            ctx: Context,
            ev: StartEvent
    ) -> RAGEvents.RetrievalEvent:
        """检索步骤"""
        logger.info(f"开始检索查询: {ev.query}")

        """执行检索(单个chroma检索)"""
        # nodes = self.retriever.retrieve(ev.query)

        """使用混合检索"""
        retrieve_nodes = await self.queryFusionRetriever.aretrieve(ev.query)

        logger.info(f"检索获得 {len(retrieve_nodes)} 个节点")

        return RAGEvents.RetrievalEvent(query=ev.query, nodes=retrieve_nodes)

    @step
    async def rerank_step(
            self,
            ctx: Context,
            ev: RAGEvents.RetrievalEvent
    ) -> RAGEvents.RerankEvent:
        """重排序步骤"""
        logger.info("开始重排序")

        # 应用重排序
        rerank_nodes = self.postprocessor.postprocess_nodes(ev.nodes, query_str=ev.query)

        logger.info(f"重排序后保留 {len(rerank_nodes)} 个节点")

        return RAGEvents.RerankEvent(query=ev.query, nodes=rerank_nodes)

    @step
    async def generate_step(
            self,
            ctx: Context,
            ev: RAGEvents.RerankEvent
    ) -> RAGEvents.ResponseEvent:
        """生成回答步骤"""
        logger.info("开始生成回答")

        # 直接使用检索到的节点生成回答，不再重复检索
        response = await self.response_synthesizer.asynthesize(
            query=ev.query,
            nodes=ev.nodes
        )

        logger.info(f"回答生成完成=>{response}")

        return RAGEvents.ResponseEvent(
            query=ev.query,
            nodes=ev.nodes,
            response=str(response)
        )

    @step
    async def finalize_step(
            self,
            ctx: Context,
            ev: RAGEvents.ResponseEvent
    ) -> StopEvent:
        """最终化步骤"""
        result = {
            "query": ev.query,
            "response": ev.response,
            "source_nodes": len(ev.nodes),
            "sources": [
                {
                    "content": node.node.text,
                    "score": node.score,
                    "metadata": node.node.metadata
                }
                for node in ev.nodes
            ]
        }

        return StopEvent(result=result)
