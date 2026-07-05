# PDF 处理方案分析与优化建议

> 基于当前 LlamaIndex 实战项目的 PDF 处理链路梳理，对比主流方案，列出可优化点。

---

## 一、当前项目 PDF 处理链路

```
PDF文件输入
    │
    ▼
MultimodalPDFProcessor.create_multimodal_nodes()
    │
    ├─ partition_pdf()  ← unstructured, strategy='hi_res' + OCR
    │       提取三类元素：
    │         · Image
    │         · Text / Title / NarrativeText
    │         · Table（提取了但未使用）
    │
    ├─ get_context_around_image()
    │       空间关联：计算图片 bbox 与文本 bbox 的欧几里得距离
    │       将最近文本块打上 image_paths 标注（JSON 字符串存入 metadata）
    │
    ├─ 手动分块（按 CHUNK_SIZE=512 字符累加）
    │       重叠通过 text_chunk_overlap 实现
    │       image_paths 以 JSON 字符串存入 metadata
    │
    └─ 构造 TextNode 列表返回
            │
            ▼
    DocumentIngestionPipeline
            ├─ 存入 Redis docstore
            └─ 存入 Chroma 向量库
```

### 关键特征说明

| 特征 | 现状 |
|------|------|
| 表格处理 | 提取了但代码注释掉，**未进入索引** |
| 图片理解 | 仅存路径到 metadata，无 VLM 语义描述，**不可被检索** |
| 分块策略 | 手写字符累加，与非PDF路径（SentenceSplitter）**逻辑分裂** |
| 图片渲染 | 前端 base64 `<img>` 展示，不参与检索 |

---

## 二、主流方案横向对比

### 2.1 解析层（Parse）

| 方案 | 代表工具 | 优势 | 局限 |
|------|---------|------|------|
| 规则 + OCR | **unstructured**（本项目）、pdfminer、pypdf | 本地运行、可控 | hi_res 模式极慢；复杂布局识别差 |
| 专用 PDF 解析服务 | **LlamaParse**、Adobe PDF Extract API | 表格/公式识别准，结构化输出 Markdown | 收费、需联网 |
| 多模态 VLM 解析 | **GPT-4o / Qwen-VL** 对每页截图做理解 | 最强语义理解，图表可被检索 | 费用高、速度慢 |
| 布局检测模型 | **Docling**（IBM开源）、PP-StructureV2 | 识别多栏/跨页表格，开源免费 | 部署成本较高 |

### 2.2 分块层（Chunk）

| 策略 | 说明 | 适用场景 |
|------|------|---------|
| 固定字符窗口（本项目） | 按字符数累加 | 快，但语义割裂风险高 |
| Semantic Chunking | 按句向量相似度边界切割 | 语义完整，推荐 |
| 按文档结构切 | 按标题层级（H1/H2）切分 | 适合有明确目录的文档 |
| 父子块（Parent-Child） | 小块检索、大块送入 LLM | 精度 + 上下文兼顾，LlamaIndex 原生支持 |

### 2.3 检索层（Retrieve）

| 策略 | 本项目 | 主流最佳实践 |
|------|--------|-------------|
| 向量检索 | ✅ Chroma + bge | — |
| BM25 稀疏检索 | ✅ | — |
| 混合融合 | ✅ QueryFusionRetriever | — |
| 重排序 | ✅ bge-reranker | — |
| 图片/表格可检索 | ❌ | 多模态向量（CLIP / VLM embedding） |
| 查询改写/扩展 | ❌ | HyDE、多查询扩展 |

---

## 三、可优化点清单

### 🔴 高优先级（影响核心质量）

#### 1. 表格内容完全丢失

- **现状**：`tables` 列表被提取但代码注释掉，从未进入索引
- **建议**：将表格转为 Markdown 格式（unstructured 已支持 `infer_table_structure=True`），作为独立节点存入，并打上 `type=table` 的 metadata 标签
- **涉及文件**：`core/pdfProcessor.py`，`create_multimodal_nodes()` 方法末尾注释部分

#### 2. 图片没有语义，无法被检索

- **现状**：图片路径挂在文本 metadata 里，只能靠关联文本间接召回
- **建议**：对每张图片调用 VLM（如 Qwen-VL）生成描述文字，将描述文本作为独立 TextNode 进索引，`image_path` 作为 metadata
- **涉及文件**：`core/pdfProcessor.py`，`_process_image_element()` 方法

#### 3. PDF 分块走手写逻辑，不走 IngestionPipeline

- **现状**：非 PDF 走 `SentenceSplitter`，PDF 走手写字符累加，逻辑分裂且语义边界差
- **建议**：`MultimodalPDFProcessor` 返回 `Document` 对象（含文本 + metadata），统一走 `IngestionPipeline` 的 `SentenceSplitter`，保证分块策略一致
- **涉及文件**：`core/ingestion.py` 的 `ingest_documents()`，`core/pdfProcessor.py` 的返回值类型

---

### 🟡 中优先级（影响检索效果）

#### 4. 缺少查询改写 / HyDE

- **现状**：用户原始 query 直接检索
- **建议**：在 `retrieve_step` 前加一步 `query_rewrite_step`，用 LLM 生成假设性答案（HyDE）或多个改写变体，再做 QueryFusion，召回率提升明显
- **涉及文件**：`core/workflow.py`，`core/events.py`（新增 `QueryRewriteEvent`）

#### 5. 相似度阈值已定义但未使用

- **现状**：`SIMILARITY_CUTOFF=0.5` 已在 `config/settings.py` 中定义，但 `RAGWorkflow` 里没有使用
- **建议**：在 `rerank_step` 后加 `SimilarityPostprocessor(cutoff=AppSettings.SIMILARITY_CUTOFF)`，避免低质量节点污染回答
- **涉及文件**：`core/workflow.py`，`_setup_components()` 方法

#### 6. 父子块（Parent-Child Retrieval）未启用

- **现状**：检索块和送入 LLM 的块大小相同（512）
- **建议**：用小块（128～256）做向量检索，命中后取其父块（512～1024）送 LLM，兼顾精准召回和上下文完整性
- **涉及文件**：`core/ingestion.py`，`_create_pipeline()` 方法

---

### 🟢 低优先级（工程健壮性）

#### 7. hi_res 策略速度极慢，缺少解析缓存

- **现状**：每次上传都重新解析，大 PDF 体验极差
- **建议**：以文件 MD5 为 key，将解析结果缓存到本地（pickle/json），命中缓存则跳过 `partition_pdf`
- **涉及文件**：`core/pdfProcessor.py`，`extract_images_and_text()` 方法入口

#### 8. 多页 PDF 缺少跨页上下文保护

- **现状**：分块时不感知页边界，可能将跨页的同一段话切断
- **建议**：在元素遍历时记录页码，页边界作为强制切块边界之一
- **涉及文件**：`core/pdfProcessor.py`，`get_context_around_image()` 分块逻辑

#### 9. DocumentManager 与 DocumentIngestionPipeline 各自独立连接 Chroma

- **现状**：两个类分别创建 `chromadb.PersistentClient`，存在重复连接
- **建议**：将 Chroma client 封装为单例，通过 `config/settings.py` 或依赖注入共享
- **涉及文件**：`core/documentManager.py`，`core/ingestion.py`

---

## 四、优化优先顺序总结

```
表格索引
    → 图片 VLM 语义描述
        → 统一分块管道（PDF 走 IngestionPipeline）
            → HyDE 查询改写
                → 相似度阈值过滤（补全已定义的配置）
                    → 父子块检索
                        → 解析结果缓存（MD5）
                            → 页边界保护
                                → Chroma 单例连接
```

---

## 五、参考资料

- [LlamaIndex 官方文档 - Ingestion Pipeline](https://docs.llamaindex.ai/en/stable/module_guides/loading/ingestion_pipeline/)
- [unstructured 文档 - partition_pdf](https://docs.unstructured.io/open-source/core-functionality/partitioning)
- [Docling - IBM 开源 PDF 解析](https://github.com/DS4SD/docling)
- [HyDE 论文 - Precise Zero-Shot Dense Retrieval](https://arxiv.org/abs/2212.10496)
- [LlamaIndex Parent-Child Retriever](https://docs.llamaindex.ai/en/stable/examples/retrievers/auto_merging_retriever/)
