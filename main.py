import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent))

from config.settings import Settings
from ui.interface import create_gradio_interface
from utils.logger import setup_logger

logger = setup_logger(__name__)


def main():
    """应用入口点"""

    # 检查API密钥
    if not Settings.validate_api_key():
        logger.error("⚠️  警告: 请设置 OPENAI_API_KEY 环境变量")
        print("export OPENAI_API_KEY='your-api-key-here'")
        return

    # 创建必要的目录
    Path("storage").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    # 创建并启动应用
    logger.info("启动RAG应用...")
    interface = create_gradio_interface()

    interface.launch(
        share=True,
        server_name=Settings.SERVER_HOST,
        server_port=Settings.SERVER_PORT,
        show_error=True,
        theme=gr.themes.Soft()
    )


if __name__ == "__main__":
    main()
