import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


class Settings:
    """应用配置类"""

    # OpenAI配置
    OPENAI_API_KEY: Optional[str] = os.getenv("DASHSCOPE_API_KEY")
    API_BASE_URL = os.getenv("DASHSCOPE_BASE_URL")
    OPENAI_MODEL: str = "qwen-plus-2025-07-14"
    OPENAI_TEMPERATURE: float = 0.1
    # DashScope Embedding 模型（API调用，无需本地文件）
    EMBEDDING_MODEL: str = "text-embedding-v3"

    # VLM配置（用于图片语义描述，增强图片可检索性）
    VLM_MODEL: str = "qwen-vl-plus-latest"
    ENABLE_VLM_DESCRIPTION: bool = True   # 关闭可跳过VLM调用，节省API费用

    # 文档处理配置
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    TITLE_EXTRACTOR_NODES: int = 5

    # 检索配置
    SIMILARITY_TOP_K: int = 5
    RERANK_TOP_K: int = 3          # LLMRerank 保留节点数
    RERANK_CHOICE_BATCH_SIZE: int = 10   # LLMRerank 单批处理节点数
    SIMILARITY_CUTOFF: float = 0.5

    # 存储配置
    CHROMA_PERSIST_DIR: str = "./chroma_db"
    DEFAULT_PERSIST_DIR: str = "./storage"

    # 服务器配置
    SERVER_HOST: str = "127.0.0.1"
    SERVER_PORT: int = 7860

    # 支持的文件类型
    SUPPORTED_FILE_TYPES: list = [".txt", ".pdf", ".docx", ".md"]

    # 项目根目录
    PROJECT_ROOT: Path = Path(__file__).parent.parent

    @classmethod
    def validate_api_key(cls) -> bool:
        """验证API密钥是否存在"""
        return cls.OPENAI_API_KEY is not None and len(cls.OPENAI_API_KEY.strip()) > 0
