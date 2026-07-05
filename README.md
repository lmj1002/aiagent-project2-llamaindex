# 智能文档问答助手

> 基于 LlamaIndex 框架构建的多模态 RAG（检索增强生成）文档问答系统，支持 PDF 图文表格解析、混合检索、重排序和 Gradio 可视化交互界面。

---

## 目录

- [技术选型](#技术选型)
- [系统架构](#系统架构)
- [整体链路](#整体链路)
- [快速开始](#快速开始)
- [配置说明](#配置说明)
- [项目结构](#项目结构)

---

## 系统架构

```
┌─────────────────────────────────────────────────────────┐
│                      Gradio 前端界面                      │
│          文档上传 │ 文档选择 │ 问答对话 │ 系统状态          │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│                   RAGApplication                         │
│          upload_and_process_files()                      │
│          query_documents()  ← 双模式：RAG / 普通对话      │
└──────────┬──────────────────────────┬───────────────────┘
           │                          │
           ▼                          ▼
┌──────────────────────┐   ┌─────────────────────────────┐
│ DocumentIngestion    │   │      RAGWorkflow             │
│ Pipeline             │   │  (LlamaIndex Workflow)       │
│                      │   │                             │
│ · SentenceSplitter   │   │  Step1: retrieve_step       │
│ · TitleExtractor     │   │  Step2: rerank_step         │
│ · HuggingFace Embed  │   │  Step3: generate_step       │
│ · MultimodalPDF      │   │  Step4: finalize_step       │
│   Processor          │   │                             │
└──────────┬───────────┘   └──────────┬──────────────────┘
           │                          │
           ▼                          ▼
┌──────────────────────────────────────────────────────────┐
│                       存储层                              │
│   ChromaDB（向量）  │  Redis docstore  │  Redis indexstore │
└──────────────────────────────────────────────────────────┘
```

### 核心模块说明

| 模块 | 文件 | 职责 |
|------|------|------|
| 入口 | `main.py` | 校验 API Key、创建目录、启动 Gradio |
| 前端 | `ui/interface.py` | Gradio 界面布局与事件绑定 |
| 应用层 | `core/application.py` | 协调摄取与查询，维护对话历史 |
| 摄取管道 | `core/ingestion.py` | 文档加载、分块、嵌入、存储 |
| PDF 处理 | `core/pdfProcessor.py` | 多模态 PDF 解析（图/文/表） |
| RAG 工作流 | `core/workflow.py` | 检索→重排→生成→返回 四步流水线 |
| 工作流事件 | `core/events.py` | 定义步骤间传递的类型化事件 |
| 文档管理 | `core/documentManager.py` | 从 ChromaDB 查询已索引文档列表 |
| 配置 | `config/settings.py` | 统一管理所有可配置参数 |
| 日志 | `utils/logger.py` | 双输出（控制台 + 文件）日志工厂 |

---

## 整体链路

### 1. 文档摄取链路（Ingestion Pipeline）

```
用户上传文件
    │
    ├─ 非 PDF（txt / docx / md）
    │       └─ SimpleDirectoryReader 读取
    │             └─ IngestionPipeline
    │                   ├─ SentenceSplitter（chunk=512, overlap=50）
    │                   ├─ TitleExtractor（前5节点提取标题）
    │                   └─ HuggingFaceEmbedding（bge-large-zh-v1.5）
    │
    └─ PDF
            └─ MultimodalPDFProcessor
                  ├─ partition_pdf（hi_res + OCR）
                  │     提取：Image / Text / Table
                  │
                  ├─ 图片-文本空间关联
                  │     计算 bbox 欧几里得距离
                  │     最近文本节点打上 image_paths 标注
                  │
                  ├─ 句子感知分块（在。！？边界处切割）
                  │     → 文本 TextNode（type=text）
                  │
                  ├─ 表格节点
                  │     → TextNode（type=table，[表格]\n内容）
                  │
                  └─ VLM 图片语义描述
                        qwen-vl-plus-latest 生成描述
                        → TextNode（type=image_description）
                              含 image_paths 元数据，命中后前端渲染图片
    │
    ▼
所有 TextNode
    ├─ 存入 Redis docstore（namespace: redis_docs）
    └─ 存入 ChromaDB 向量库（collection: quickstart）
          VectorStoreIndex 自动触发 Embedding
```

### 2. 查询链路（Query Pipeline）

```
用户输入 Query
    │
    ├─ 未选择文档（普通对话模式）
    │       └─ Settings.llm.complete(query) → 直接返回
    │
    └─ 选择了文档（RAG 模式）
            │
            ▼
      RAGWorkflow（LlamaIndex Workflow）
            │
            ├─ Step 1: retrieve_step
            │       QueryFusionRetriever（混合检索）
            │         ├─ VectorIndexRetriever（Chroma 向量检索，top_k=5）
            │         └─ BM25Retriever（Redis docstore 关键词检索，top_k=5）
            │       └─ 返回融合去重后的节点列表
            │
            ├─ Step 2: rerank_step
            │       SentenceTransformerRerank（bge-reranker-large）
            │       保留 top_n=3 最相关节点
            │
            ├─ Step 3: generate_step
            │       ResponseSynthesizer
            │       将节点内容 + Query 发给 Qwen Plus 生成回答
            │
            └─ Step 4: finalize_step
                    组装结果：{response, source_nodes, sources}
                    │
                    ▼
              前端渲染
                ├─ 回答文本
                ├─ 相关来源（相似度分数 + 内容预览）
                └─ 关联图片（base64 内联渲染）
```

---

## 技术选型

| 层级 | 技术 / 模型 | 说明 |
|------|------------|------|
| **LLM** | DashScope `qwen-plus-2025-07-14` | 对话生成与答案合成，通过阿里云 DashScope API 调用 |
| **VLM** | DashScope `qwen-vl-plus-latest` | 对 PDF 中提取的图片生成语义描述，使图片内容可被检索 |
| **Embedding** | `BAAI/bge-large-zh-v1.5`（本地） | 中文向量化，将文本块转为高维向量 |
| **Reranker** | `BAAI/bge-reranker-large`（本地） | 对召回节点进行精排，过滤低相关结果 |
| **向量数据库** | ChromaDB | 持久化存储文档向量，支持相似度检索 |
| **文档/索引存储** | Redis | 存储原始文档节点（docstore）和索引元数据（index store） |
| **PDF 解析** | `unstructured`（hi_res + OCR） | 高精度 PDF 解析，支持图片、文本、表格三类元素提取 |
| **RAG 框架** | LlamaIndex | 统一管理摄取管道、工作流、检索器和响应合成器 |
| **前端界面** | Gradio | 可视化文档上传、管理和对话交互界面 |

---

## 快速开始

### 环境要求

- Python 3.10+
- Redis 7.x（本地运行，端口 6379）
- 本地模型文件（路径在 `config/settings.py` 中配置）：
  - `BAAI/bge-large-zh-v1.5`（Embedding）
  - `BAAI/bge-reranker-large`（Reranker）

### 安装依赖

```bash
pip install -r requirements.txt
```

> PDF 高精度解析依赖 `unstructured[hi_res]`，首次运行会自动下载 `detectron2` 等模型文件，耗时较长。

### 配置环境变量

在项目根目录新建 `.env` 文件：

```env
DASHSCOPE_API_KEY=your_dashscope_api_key_here
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

> DashScope API Key 申请地址：https://dashscope.console.aliyun.com/

### 启动 Redis

```bash
# Windows（WSL 或 Docker）
docker run -d -p 6379:6379 redis:7

# macOS / Linux
redis-server
```

### 启动应用

```bash
python main.py
```

启动后访问 `http://127.0.0.1:7860`，或使用控制台输出的 Gradio 公网分享链接。

### 使用步骤

1. 在左侧面板**上传文档**（支持 `.pdf` `.txt` `.docx` `.md`）
2. 点击「🔄 处理文档」等待摄取完成
3. 在「文档选择」下拉框中**勾选要检索的文档**
4. 在右侧输入框提问，点击「发送」
   - 勾选了文档 → **RAG 模式**，回答附带来源引用和关联图片
   - 未勾选文档 → **普通对话模式**，直接调用 LLM 回答

---

## 配置说明

所有配置集中在 `config/settings.py`：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `OPENAI_API_KEY` | `DASHSCOPE_API_KEY` 环境变量 | DashScope API Key |
| `API_BASE_URL` | `DASHSCOPE_BASE_URL` 环境变量 | DashScope 接入地址 |
| `OPENAI_MODEL` | `qwen-plus-2025-07-14` | LLM 模型名 |
| `VLM_MODEL` | `qwen-vl-plus-latest` | 图片语义描述模型 |
| `ENABLE_VLM_DESCRIPTION` | `True` | 关闭可跳过 VLM 调用，节省 API 费用 |
| `EMBEDDING_MODEL_PATH` | 本地绝对路径 | bge-large-zh-v1.5 模型目录 |
| `RERANK_MODEL_PATH` | 本地绝对路径 | bge-reranker-large 模型目录 |
| `CHUNK_SIZE` | `512` | 分块字符数上限 |
| `CHUNK_OVERLAP` | `50` | 相邻块重叠字符数 |
| `SIMILARITY_TOP_K` | `5` | 混合检索召回节点数 |
| `RERANK_TOP_K` | `3` | 重排后保留节点数 |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | ChromaDB 持久化目录 |
| `SERVER_HOST` | `127.0.0.1` | Gradio 监听地址 |
| `SERVER_PORT` | `7860` | Gradio 端口 |

---

## 项目结构

```
.
├── main.py                  # 应用入口
├── requirements.txt         # 依赖清单
├── .env                     # 环境变量（不提交 Git）
├── config/
│   └── settings.py          # 统一配置类
├── core/
│   ├── application.py       # RAGApplication：协调摄取与查询
│   ├── ingestion.py         # 文档摄取管道（分块/嵌入/存储）
│   ├── pdfProcessor.py      # 多模态 PDF 处理（图/文/表 + VLM）
│   ├── workflow.py          # RAGWorkflow：检索→重排→生成→返回
│   ├── events.py            # 工作流步骤间的类型化事件定义
│   └── documentManager.py   # 已索引文档列表查询
├── ui/
│   └── interface.py         # Gradio 界面与事件绑定
├── utils/
│   └── logger.py            # 日志工厂（控制台 + 文件双输出）
├── avatars/                 # 对话头像图片
├── file/images/             # PDF 解析时提取的图片（运行时生成）
├── chroma_db/               # ChromaDB 向量库（运行时生成）
├── logs/                    # 日志文件（运行时生成）
├── CLAUDE.md                # Claude Code 工作指引
└── PDF处理方案分析与优化.md   # PDF 处理方案分析文档
```

---

