import json
from typing import List, Optional, Tuple, Dict
from core.ingestion import DocumentIngestionPipeline
from core.workflow import RAGWorkflow
from utils.logger import setup_logger
from llama_index.core import Settings
import base64
import os

logger = setup_logger(__name__)


class RAGApplication:
    """RAG应用主类"""

    def __init__(self):
        self.ingestion_pipeline = DocumentIngestionPipeline()
        self.workflow: Optional[RAGWorkflow] = None
        self.chat_history: List[List[Dict]] = []

    def upload_and_process_files(self, files) -> str:
        """上传并处理文件"""
        if not files:
            return "请上传至少一个文件"

        try:
            file_paths = []
            for file in files:
                if hasattr(file, 'name'):
                    file_paths.append(file.name)
                else:
                    file_paths.append(str(file))
            """摄取文档并创建索引"""
            result = self.ingestion_pipeline.ingest_documents(file_paths)

            """创建工作流"""
            if self.ingestion_pipeline.index:
                self.workflow = RAGWorkflow(self.ingestion_pipeline.index)

            return result

        except Exception as e:
            error_msg = f"文件处理失败: {str(e)}"
            logger.error(error_msg)
            return error_msg

    def image_to_base64(self, path):
        """将图片解析成b64格式"""
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    async def query_documents(
            self,
            query: str,
            docs_selects: List,
            history: List[Dict]
    ) -> Tuple[List[Dict], str]:
        # 如果没有选择文档就是普通对话
        if docs_selects:
            # 当工作流未被初始化的时候，才去创建工作流
            if not self.workflow:
                # 读取已有的检索器和文档
                self.ingestion_pipeline.get_documents()
                # 创建工作流
                if self.ingestion_pipeline.index:
                    self.workflow = RAGWorkflow(self.ingestion_pipeline.index)
                    logger.info("RAG工作流已创建")

            """查询文档"""
            try:
                # 运行RAG工作流
                result = await self.workflow.run(query=query, timeout=60.0)

                # 格式化回答
                response = result["response"]
                sources_info = f"\n\n📚 **相关来源** ({result['source_nodes']} 个节点):\n"

                # 将检索到的文档的内容和元数据显示在页面中
                for i, source in enumerate(result["sources"], 1):
                    score = source['score']
                    score_str = f"{score:.3f}" if score is not None else "N/A"
                    sources_info += f"{i}. 相似度: {score_str}\n"
                    sources_info += f"   内容预览: {source['content']}\n\n"
                    # 如果包含image_path属性并且不为空，那么就将图片解析到页面中
                    if 'image_paths' in source['metadata'] and source['metadata']['image_paths']:
                        # 把路径中的系统分隔符（比如 '\\'）统一替换为 '/'，从而得到跨平台一致的路径格式。
                        web_safe_image_paths = source['metadata']['image_paths'].replace(os.sep, '/')
                        image_paths = json.loads(web_safe_image_paths)
                        # 将图片转为 base64
                        for image_path in image_paths:
                            b64_img = self.image_to_base64(image_path)
                            img_html = f'<p>以下是相关图片：</p><img src="data:image/jpeg;base64,{b64_img}" width="300"/>'
                            sources_info += img_html

                full_response = response + sources_info

                # 更新历史记录
                history.append({"role": "user", "content": query})
                history.append({"role": "assistant", "content": full_response})
                self.chat_history.append(history)
                return history, ""

            except Exception as e:
                error_msg = f"查询失败: {str(e)}"
                logger.error(error_msg)
                history.append({"role": "user", "content": query})
                history.append({"role": "assistant", "content": error_msg})
                self.chat_history.append(history)
                return history, ""
        else:
            """普通对话"""
            llm_res = Settings.llm.complete(query)
            history.append({"role": "user", "content": query})
            history.append({"role": "assistant", "content": llm_res.text})
            self.chat_history.append(history)
            return history, ""

    def clear_chat(self) -> List:
        """清空聊天记录"""
        self.chat_history = []
        return []
