from typing import List, Dict, Tuple, Optional
from pathlib import Path
import hashlib
import json
import os
import math
import base64
import re
from openai import OpenAI
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import Image, Text, Title, NarrativeText, Table
from llama_index.core.schema import TextNode
from config.settings import Settings as AppSettings
from utils.logger import setup_logger

logger = setup_logger(__name__)


class MultimodalPDFProcessor:
    """多模态PDF处理器 - 支持提取图片、文本和表格"""

    def __init__(self, image_output_dir: str = "file/images"):
        self.image_output_dir = image_output_dir
        os.makedirs(image_output_dir, exist_ok=True)

    def extract_images_and_text(self, pdf_path: str) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """从PDF中提取图片和对应的文档内容"""
        logger.info(f"开始解析PDF文件: {pdf_path}")

        # 解析PDF
        raw_pdf_elements = partition_pdf(
            filename=pdf_path,  # 指定要解析的 PDF 文件路径
            extract_images_in_pdf=True,  # 是否提取 PDF 中的图片（将其作为 ImageBlock 返回）
            infer_table_structure=True,  # 是否尝试识别并解析表格结构（会输出结构化的 Table 类型元素）
            strategy=AppSettings.PDF_STRATEGY,  # hi_res（高精度）或 fast（无需下载模型）
            extract_image_block_output_dir=self.image_output_dir  # 图片输出目录，提取出的图片会保存到这个路径
        )

        images = []
        texts = []
        tables = []

        logger.info(f"总共解析到 {len(raw_pdf_elements)} 个元素")

        # ── unstructured 返回空时，用 pypdf 兜底提取文本 ──────────────────
        # 常见场景：PPT 转 PDF、矢量图形文字，fast 策略下 pdfminer 读不到内容
        if not raw_pdf_elements:
            logger.warning(
                f"unstructured 未提取到内容（strategy={AppSettings.PDF_STRATEGY}），"
                "切换到 pypdf 兜底提取文本"
            )
            texts = self._fallback_extract_with_pypdf(pdf_path)
            logger.info(f"pypdf 兜底完成，提取文本块: {len(texts)}")
            return [], texts, []

        # 遍历所有元素
        for i, element in enumerate(raw_pdf_elements):
            # 创建每个元素的内容
            element_info = {
                # 元素类型
                'type': type(element).__name__,
                # 元素页码
                'page': getattr(element.metadata, 'page_number', None) if hasattr(element, 'metadata') else None,
                # 获取坐标-整个元素的坐标点是从左上逆时针开始计算的
                'bbox': getattr(element.metadata, 'coordinates', None) if hasattr(element, 'metadata') else None,
                # 获取坐标系
                'content': str(element),
                'element_id': f"{Path(pdf_path).stem}_{i}"
            }

            # 处理图片元素
            if isinstance(element, Image):
                # 处理图片元素
                image_info = self._process_image_element(element, element_info, pdf_path)
                if image_info:
                    images.append(image_info)

            # 处理文本元素
            elif isinstance(element, (Text, Title, NarrativeText)):
                texts.append(element_info)

            # 处理表格元素
            elif isinstance(element, Table):
                tables.append(element_info)

        logger.info(f"提取完成 - 图片: {len(images)}, 文本: {len(texts)}, 表格: {len(tables)}")
        return images, texts, tables

    def _fallback_extract_with_pypdf(self, pdf_path: str) -> List[Dict]:
        """pypdf 兜底：按页提取文本，返回与 unstructured 相同格式的 texts 列表"""
        try:
            from pypdf import PdfReader
            reader = PdfReader(pdf_path)
            texts = []
            for page_num, page in enumerate(reader.pages, start=1):
                content = page.extract_text() or ""
                content = content.strip()
                if content:
                    texts.append({
                        'type': 'NarrativeText',
                        'page': page_num,
                        'bbox': None,
                        'content': content,
                        'element_id': f"{Path(pdf_path).stem}_page{page_num}",
                    })
            logger.info(f"pypdf 提取完成: {len(reader.pages)} 页，有文字页: {len(texts)}")
            return texts
        except Exception as e:
            logger.error(f"pypdf 兜底也失败: {e}")
            return []

    def _process_image_element(self, element: Image, element_info: Dict, pdf_path: str) -> Optional[Dict]:
        """处理图片元素"""
        # 拷贝元素的内容
        image_info = element_info.copy()

        # 获取图片的base64数据
        if hasattr(element.metadata, 'image_base64') and element.metadata.image_base64:
            # 获取图片的bs4的内容
            image_info['image_data'] = element.metadata.image_base64
            # 图片类型
            image_info['image_format'] = getattr(element.metadata, 'image_mime_type', 'image/png')

        # 获取图片路径
        elif hasattr(element.metadata, 'image_path') and element.metadata.image_path:
            image_info['image_path'] = element.metadata.image_path

        # 检查其他可能的图片数据属性
        elif hasattr(element.metadata, 'image_data') and element.metadata.image_data:
            image_info['image_data'] = element.metadata.image_data
        # 如果 image_info 字典中包含 'image_data' 或 'image_path' 这两个键中任意一个，就返回 image_info 本身；否则返回 None。
        return image_info if any(key in image_info for key in ['image_data', 'image_path']) else None

    def _describe_image_with_vlm(self, image_path: str) -> str:
        """调用 VLM 生成图片语义描述，使图片内容可被检索"""
        if not AppSettings.ENABLE_VLM_DESCRIPTION:
            return ""
        try:
            client = OpenAI(
                api_key=AppSettings.OPENAI_API_KEY,
                base_url=AppSettings.API_BASE_URL
            )
            # 读取图片并转为 base64
            with open(image_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")

            # 根据后缀推断 MIME 类型
            suffix = Path(image_path).suffix.lower()
            mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                        ".png": "image/png", ".gif": "image/gif", ".webp": "image/webp"}
            mime_type = mime_map.get(suffix, "image/jpeg")

            response = client.chat.completions.create(
                model=AppSettings.VLM_MODEL,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime_type};base64,{image_data}"}
                        },
                        {
                            "type": "text",
                            "text": (
                                "请详细描述这张图片的内容，"
                                "包括图表数据、文字信息、图像含义等，"
                                "描述需准确、完整，用于文档检索。"
                            )
                        }
                    ]
                }],
                max_tokens=512
            )
            description = response.choices[0].message.content
            logger.info(f"VLM描述生成成功: {Path(image_path).name}")
            return description
        except Exception as e:
            logger.warning(f"VLM描述生成失败 ({Path(image_path).name}): {e}")
            return ""

    def euclidean_distance(self, p1: float, p2: float) -> float:
        """计算欧几里得距离"""
        return math.sqrt((p1 - p2) ** 2)

    def get_context_around_image(self, images: List[Dict], texts: List[Dict]) -> List[Dict]:
        """获取图片周围的文本内容作为上下文"""

        # 第一部分：建立图片-文本关联
        for image in images:
            image_page = image.get('page')  # 获取图片的页面
            image_bbox = image.get('bbox')  # 获取图片的坐标
            image_path = image.get('image_path')  # 获取图片的路径

            # 数据有效性检查
            if not all([image_page, image_bbox, image_path]):
                continue

            # 找到同一页最近的文本
            closest_text = None  # 就等于文本对象
            min_dist = float("inf")  # 创建一个无穷大的浮点数

            # 计算图片和哪个文档最接近
            for text in texts:
                text_page = text.get('page')  # 获取文档的页码
                text_bbox = text.get('bbox')  # 获取文档的坐标
                text_content = text.get('content')  # 获取文档的内容

                # 检查数据有效性和是否在相同页面， 当一个页面中只有图片没有文字的时候，这个图片会失效
                if not all([text_page, text_bbox, text_content]) or text_page != image_page:
                    continue

                try:
                    # 图片和文字的四个角的坐标- 从左上坐标开始逆时针直到右上角坐标
                    # 图片的上下坐标  (  [左上[X，Y]], [左下[X，Y]], [右下[X，Y]], [右上[X，Y]]        )
                    image_top = int(image_bbox.points[0][1])  # 图片的左上Y坐标
                    image_bottom = int(image_bbox.points[1][1])  # 图片的左下Y坐标

                    # 文本的上下坐标
                    text_top = int(text_bbox.points[0][1])  # 文本的左上Y坐标
                    text_bottom = int(text_bbox.points[1][1])  # 文本的左下Y坐标

                    # 计算距离
                    if image_top > text_bottom:  # 文字在图片上面
                        dist = self.euclidean_distance(image_top, text_bottom)
                    elif text_top > image_bottom:  # 文字在图片下面
                        dist = self.euclidean_distance(text_top, image_bottom)
                    else:  # 文字与图片有重叠
                        dist = 0  # 重叠时距离最小
                    # 但计算的距离小于上次计算的内容，那么就代表当前文档和图片更接近
                    if dist < min_dist:
                        min_dist = dist
                        closest_text = text

                except (IndexError, ValueError, AttributeError) as e:
                    logger.warning(f"处理坐标时出错: {e}")
                    continue

            # 创建图片-文本关联
            if closest_text:
                # 支持多图片关联
                if 'image_paths' not in closest_text:
                    closest_text['image_paths'] = []
                closest_text['image_paths'].append(image_path)

        # 第二部分：句子感知分块处理（优化点3：在中文句子边界切割，避免语义割裂）
        # 匹配中文/英文句子结束符后的切割点
        SENTENCE_END = re.compile(r'(?<=[。！？!?\n])')

        results: List[Dict] = []
        buffer_content = ""
        buffer_image_paths: List[str] = []

        def _flush_chunk(content: str, image_paths: List[str]) -> Dict:
            """将缓冲区内容打包成一个结果块"""
            chunk: Dict = {"content": content}
            if image_paths:
                chunk["image_paths"] = json.dumps(image_paths)
            return chunk

        for text in texts:
            content = text.get("content", "")
            if not content:
                continue

            # 合并当前文本元素的图片路径（去重）
            cur_paths: List[str] = text.get("image_paths", []) if "image_paths" in text else []
            existing = set(buffer_image_paths)
            buffer_image_paths += [p for p in cur_paths if p not in existing]
            buffer_content += content

            # 缓冲区达到分块大小时，寻找最近的句子边界进行切割
            while len(buffer_content) >= AppSettings.CHUNK_SIZE:
                # 在 CHUNK_SIZE 往后最多 80 字的窗口内寻找句子边界
                window = buffer_content[:AppSettings.CHUNK_SIZE + 80]
                parts = SENTENCE_END.split(window)

                cut_pos = 0
                for part in parts:
                    next_pos = cut_pos + len(part)
                    if next_pos > AppSettings.CHUNK_SIZE:
                        break
                    cut_pos = next_pos

                # 单个句子本身就超过 CHUNK_SIZE，则强制在 CHUNK_SIZE 处截断
                if cut_pos == 0:
                    cut_pos = AppSettings.CHUNK_SIZE

                results.append(_flush_chunk(buffer_content[:cut_pos], buffer_image_paths.copy()))

                # 保留重叠部分，图片路径不随重叠段重复携带
                overlap_start = max(0, cut_pos - AppSettings.CHUNK_OVERLAP)
                buffer_content = buffer_content[overlap_start:]
                buffer_image_paths = []

        # 处理缓冲区中的最后一块
        if buffer_content.strip():
            results.append(_flush_chunk(buffer_content, buffer_image_paths))

        return results

    def create_multimodal_nodes(self, pdf_path: str) -> List[TextNode]:
        """创建多模态文档节点：文本节点 + 表格节点 + 图片语义描述节点"""
        images, texts, tables = self.extract_images_and_text(pdf_path)
        texts = self.get_context_around_image(images, texts)

        file_name = Path(pdf_path).name
        doc_id = hashlib.md5(file_name.encode("utf-8")).hexdigest()
        text_nodes: List[TextNode] = []

        # ── 1. 文本节点（含关联图片路径）────────────────────────────
        for text in texts:
            node = TextNode(
                text=text["content"],
                metadata={
                    "type": "text",
                    "image_paths": text.get("image_paths"),
                    "file_name": file_name,
                    "doc_id": doc_id,
                }
            )
            text_nodes.append(node)

        # ── 2. 表格节点（优化点1：原代码注释掉未使用，现在启用）──────
        table_count = 0
        for table in tables:
            content = table.get("content", "").strip()
            if not content:
                continue
            node = TextNode(
                text=f"[表格]\n{content}",
                metadata={
                    "type": "table",
                    "page": str(table.get("page", "")),
                    "file_name": file_name,
                    "doc_id": doc_id,
                }
            )
            text_nodes.append(node)
            table_count += 1
        if table_count:
            logger.info(f"表格节点已添加: 共 {table_count} 个")

        # ── 3. 图片语义描述节点（优化点2：VLM生成描述使图片可检索）──
        image_desc_count = 0
        if AppSettings.ENABLE_VLM_DESCRIPTION:
            for image in images:
                image_path = image.get("image_path")
                if not image_path or not Path(image_path).exists():
                    continue
                description = self._describe_image_with_vlm(image_path)
                if not description:
                    continue
                node = TextNode(
                    text=f"[图片描述]\n{description}",
                    metadata={
                        "type": "image_description",
                        "image_paths": json.dumps([image_path]),
                        "page": str(image.get("page", "")),
                        "file_name": file_name,
                        "doc_id": doc_id,
                    }
                )
                text_nodes.append(node)
                image_desc_count += 1

        logger.info(
            f"节点构建完成 ← {file_name} | "
            f"文本: {len(texts)}  表格: {table_count}  图片描述: {image_desc_count}"
        )
        return text_nodes
