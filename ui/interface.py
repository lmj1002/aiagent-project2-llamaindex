import gradio as gr
from core.application import RAGApplication
from core.documentManager import DocumentManager
from config.settings import Settings


def create_gradio_interface() -> gr.Blocks:
    """创建Gradio界面 - 带可收起侧边栏"""
    app = RAGApplication()  # 主函数
    doc_manger = DocumentManager()  # 文档管理器

    with gr.Blocks(title="图灵AI") as interface:
        # 顶部标题区域
        gr.Markdown("""
        <div style="text-align: center;">
            <h1>🚀 智能文档问答对话助手</h1>
            <p>基于LlamaIndex框架构建的智能文档问答系统</p>
        </div>
        """)

        # 侧边栏切换按钮
        with gr.Row():
            sidebar_toggle = gr.Button(
                "🔧 显示/隐藏控制面板",
                variant="secondary",
                size="sm"
            )

        # 主要内容区域
        with gr.Row():
            # 左侧控制面板 - 可动态显示/隐藏
            sidebar_column = gr.Column(scale=1, visible=True)

            with sidebar_column:
                gr.Markdown("## 📄 文档管理")

                # 文档上传区域
                with gr.Group():
                    gr.Markdown("### 上传文档")
                    file_upload = gr.File(
                        label="选择文件",
                        file_count="multiple",
                        file_types=Settings.SUPPORTED_FILE_TYPES,
                        height=120
                    )

                    process_btn = gr.Button(
                        "🔄 处理文档",
                        variant="primary",
                        size="lg"
                    )

                # 状态显示区域
                with gr.Group():
                    gr.Markdown("### 系统状态")
                    process_status = gr.Textbox(
                        label="处理状态",
                        lines=3,
                        interactive=False,
                        placeholder="等待上传文档...",
                    )

                with gr.Group():
                    gr.Markdown("### 文档选择")
                    docs_selects = gr.Dropdown(
                        doc_manger.get_document_names_only(),
                        multiselect=True,
                        label="选择文档",
                    )

                # 快速操作
                with gr.Group():
                    gr.Markdown("### 快速操作")
                    with gr.Row():
                        clear_all_btn = gr.Button(
                            "🗑️ 重置系统",
                            variant="stop",
                            size="sm"
                        )
                        refresh_btn = gr.Button(
                            "🔄 刷新状态",
                            size="sm"
                        )

                # 帮助信息（折叠）
                with gr.Accordion("❓ 使用帮助", open=False):
                    gr.Markdown("""
                    **支持格式**: txt, pdf, docx, md

                    **使用步骤**:
                    1. 📤 上传文档文件
                    2. 🔄 点击"处理文档"
                    3. ⏳ 等待处理完成
                    4. 💬 开始智能问答

                    **提示**: 
                    - 可保存索引避免重复处理
                    - 支持多文件批量上传
                    - 大文件和PDF文件处理需要更多时间
                    """)

            # 右侧对话区域
            chat_column = gr.Column(scale=3)

            with chat_column:
                gr.Markdown("## 💬 智能问答")

                # 对话状态指示器
                chat_status = gr.Textbox(
                    label="对话状态",
                    value="📝 请输入您的问题进行对话",
                    interactive=False,
                    container=False
                )

                # 聊天机器人界面
                chatbot = gr.Chatbot(
                    label="RAG 对话历史",
                    height=450,
                    show_label=False,
                    placeholder="您好，我是图灵AI智能助手",
                    avatar_images=(
                        "./avatars/avatar.png",
                        "./avatars/assistant.png"
                    ),
                )

                # 输入区域
                with gr.Group():
                    with gr.Row():
                        query_input = gr.Textbox(
                            label="请输入问题",
                            placeholder="请输入您想要查询的问题...",
                            scale=5,
                            lines=2,
                            max_lines=4
                        )
                        with gr.Column(scale=1, min_width=100):
                            submit_btn = gr.Button(
                                "发送 📤",
                                variant="primary",
                                size="lg"
                            )
                            clear_btn = gr.Button(
                                "清空 🗑️",
                                variant="secondary"
                            )

        # 侧边栏显示/隐藏状态
        sidebar_visible = gr.State(True)

        def toggle_sidebar(visible):
            """切换侧边栏显示状态"""
            new_visible = not visible

            # 根据侧边栏状态调整聊天区域的比例
            if new_visible:
                # 显示侧边栏：左1右2
                return (
                    gr.update(visible=True, scale=1),  # 设置侧边栏所占区域
                    gr.update(scale=2),  # 聊天栏所占的区域
                    new_visible,  # 侧边栏状态
                    "🔧 隐藏控制面板"  # 按钮文字
                )
            else:
                # 隐藏侧边栏：聊天区域占满
                return (
                    gr.update(visible=False, scale=0),  # 设置侧边栏所占区域
                    gr.update(scale=1),  # 聊天栏所占的区域
                    new_visible,  # 侧边栏状态
                    "🔧 显示控制面板"  # 按钮文字
                )

        def update_docs_dropdown():
            """更新文档下拉框选项"""
            doc_names = doc_manger.get_document_names_only()
            return gr.update(choices=doc_names, value=doc_names if doc_names else [])

        def update_chat_status(status_text):
            """更新对话状态"""
            if "成功" in status_text or "完成" in status_text:
                return "✅ 文档已处理完成，可以开始文档检索"
            elif "失败" in status_text or "错误" in status_text:
                return "❌ 文档处理失败，请检查文件格式"
            elif "处理中" in status_text:
                return "⏳ 正在处理文档，请稍候..."
            else:
                return "📝 请先上传并处理文档"

        def clear_all_data():
            """清空所有数据"""
            return (
                "",  # 系统状态
                "📝 请先上传并处理文档",  # 聊天状态
                [],  # 聊天窗口
                gr.update(choices=[], value=[])  # 清空下拉框
            )

        """事件绑定"""

        # 侧边栏切换
        sidebar_toggle.click(
            fn=toggle_sidebar,
            inputs=[sidebar_visible],
            outputs=[sidebar_column, chat_column, sidebar_visible, sidebar_toggle]
        )

        # 文档处理
        process_btn.click(
            fn=app.upload_and_process_files,
            inputs=[file_upload],
            outputs=[process_status]
        ).then(
            fn=update_chat_status,
            inputs=[process_status],
            outputs=[chat_status]
        ).then(
            fn=update_docs_dropdown,
            outputs=[docs_selects]
        )

        # 对话功能
        submit_btn.click(
            fn=app.query_documents,
            inputs=[query_input, docs_selects, chatbot],
            outputs=[chatbot, query_input]
        )

        query_input.submit(
            fn=app.query_documents,
            inputs=[query_input, docs_selects, chatbot],
            outputs=[chatbot, query_input]
        )

        # 清空对话
        clear_btn.click(
            fn=app.clear_chat,
            outputs=[chatbot]
        )

        # 快速操作
        clear_all_btn.click(
            fn=clear_all_data,
            outputs=[process_status, chat_status, chatbot, docs_selects]
        )

        refresh_btn.click(
            fn=lambda: "状态已刷新 ✅",
            outputs=[process_status]
        )

    return interface
