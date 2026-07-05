# from unstructured.partition.pdf import partition_pdf
# from unstructured.documents.elements import Image, Text, Title, NarrativeText, Table
# from llama_index.core.schema import Document
# import os
# from PIL import Image as PILImage
# import base64
# import math
#
#
# def extract_images_and_text(pdf_path):
#     """
#     从PDF中提取图片和对应的文档内容
#     """
#     # 解析PDF
#     raw_pdf_elements = partition_pdf(
#         filename=pdf_path,
#         extract_images_in_pdf=True,  # 提取PDF中的图片
#         infer_table_structure=True,  # 启用表格结构识别
#         max_characters=4000,  # 每个文本块最大字符数
#         new_after_n_chars=3800,  # 达到3800个字符后分新块
#         combine_text_under_n_chars=2000,  # 合并小于2000个字符的碎片文本
#         strategy='hi_res',
#         extract_image_block_output_dir="image"
#     )
#
#     images = []
#     texts = []
#     tables = []
#
#     print(f"总共解析到 {len(raw_pdf_elements)} 个元素")
#
#     # 遍历所有元素
#     for i, element in enumerate(raw_pdf_elements):
#         # 获取元素的基本信息
#         element_info = {
#             'type': type(element).__name__,
#             'page': getattr(element.metadata, 'page_number', None) if hasattr(element, 'metadata') else None,
#             'bbox': getattr(element.metadata, 'coordinates', None) if hasattr(element, 'metadata') else None,
#             'content': str(element)
#         }
#
#         print(f"元素 {i}: {element_info['type']}, 页码: {element_info['page']}")
#
#         # 处理图片元素
#         if isinstance(element, Image):
#             image_info = element_info.copy()
#
#             # 打印图片元素的所有属性，用于调试
#             if hasattr(element, 'metadata'):
#                 # print(f"图片元素属性: {dir(element.metadata)}")
#                 for attr in dir(element.metadata):
#                     if not attr.startswith('_'):
#                         value = getattr(element.metadata, attr, None)
#                         # print(f"  {attr}: {type(value)} - {str(value)[:100] if value else 'None'}")
#
#             # 获取图片的base64数据（如果有）
#             if hasattr(element.metadata, 'image_base64') and element.metadata.image_base64:
#                 image_info['image_data'] = element.metadata.image_base64
#                 image_info['image_format'] = getattr(element.metadata, 'image_mime_type', 'image/png')
#
#             # 获取图片路径（如果保存到了文件）
#             if hasattr(element.metadata, 'image_path') and element.metadata.image_path:
#                 image_info['image_path'] = element.metadata.image_path
#
#             # 检查其他可能的图片数据属性
#             if hasattr(element.metadata, 'image_data') and element.metadata.image_data:
#                 image_info['image_data'] = element.metadata.image_data
#
#             images.append(image_info)
#
#         # 处理文本元素
#         elif isinstance(element, (Text, Title, NarrativeText)):
#             texts.append(element_info)
#
#         # 处理表格元素
#         elif isinstance(element, Table):
#             tables.append(element_info)
#
#     return images, texts, tables
#
#
# def euclidean(p1, p2):
#     return math.sqrt((p1 - p2) ** 2)
#
#
# def get_context_around_image(images, texts):
#     """
#     获取图片周围的文本内容作为上下文
#     """
#
#     for image in images:
#         image_page = image['page']
#         image_bbox = image['bbox']
#
#         # 找到同一页或相邻页的文本
#         nearest_text = None
#         min_dist = float("inf")
#         for text in texts:
#             text_page = text['page']
#             text_bbox = text['bbox']
#             # 检查是否在相同页面
#             if text_page and image_page:
#                 if text_page == image_page:
#                     # 图片的左上角坐标和左下角坐标的y轴
#                     image_left_top = int(image_bbox.points[0][1])
#                     image_left_down = int(image_bbox.points[1][1])
#                     # 文档的左上角坐标的y轴
#                     text_left_top = int(text_bbox.points[0][1])
#                     text_left_down = int(text_bbox.points[1][1])
#
#                     y_num = 0
#                     # 判断文字是否在图片上面=> 图片的左上和文档的左下进行比较
#                     if image_left_top > text_left_down:
#                         dist = euclidean(image_left_top, text_left_down)
#                         if dist < min_dist:
#                             min_dist = dist
#                             nearest_text = text
#                     else:
#                         # 文字在图片下面
#                         dist = euclidean(text_left_top, image_left_down)
#                         if dist < min_dist:
#                             min_dist = dist
#                             nearest_text = text
#         if nearest_text:
#             nearest_text["image_path"] = image["image_path"]
#
#     # 最终返回内容
#     results = []
#     # 每个块内容
#     text_chunk = {}
#     # 每个块重复数据
#     text_chunk_overlap = ""
#     # 每个块的长度
#     text_length = 0
#     for text in texts:
#         # 如果文本长度为0，创建一个新的文本块，将文本和图片信息(如果有)填充
#         if text_length == 0:
#             text_chunk["content"] = text_chunk_overlap + text["content"]
#             text_chunk["image_path"] = text['image_path'] if 'image_path' in text else None
#             text_length += len(text["content"])
#         else:
#             # 如果如果文本长度不为0，将文本和图片信息(如果有)填充
#             text_chunk["content"] += text["content"]
#             # 如果包含图片路径就存入最终的文档块中
#             if 'image_path' in text:
#                 text_chunk["image_path"] = text['image_path']
#             text_length += len(text["content"])
#             # 如果文本块的长度大于或等于200将文本块存入最终的返回值中，将文本块内容清空，文本块长度设置为0
#             if text_length >= 200:
#                 results.append(text_chunk)
#                 text_chunk_overlap = text_chunk["content"][-50:]
#                 text_chunk = {}
#                 text_length = 0
#
#     print(results)
#
#
# # 使用示例
# def main():
#     pdf_path = "斗破苍穹.pdf"  # 替换为你的PDF文件路径
#
#     # 提取图片和文本
#     images, texts, tables = extract_images_and_text(pdf_path)
#     # 获取图片和对应的上下文
#     get_context_around_image(images, texts)
#
#
# if __name__ == "__main__":
#     main()


import gradio as gr


sorted()
def respond(message, history):
    img_html = '<p>以下是相关图片：</p><img src="file/images/figure-1-1.jpg" width="300"/>'
    history.append((message, img_html))
    return history, ""


with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()

    msg.submit(respond, [msg, chatbot], [chatbot, msg])

demo.launch()
