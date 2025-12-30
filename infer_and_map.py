import json
import cv2
import numpy as np
import os  # 新增：用于文件夹遍历
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

# -------------------------- 1. 配置Linux中文字体 --------------------------
# 方案1：Noto CJK字体（已安装）
# FONT_PATH = "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"
# 方案2：手动下载的SimHei.ttf（注释上方，启用下方）
FONT_PATH = "garbage/SimHei.ttf"

# 初始化字体（字号12，标签变长可适当调大，比如11）
try:
    font = ImageFont.truetype(FONT_PATH, 11)  # 字号略调小，避免标签过长
    print("✅ 中文字体加载成功")
except IOError:
    print(f"⚠️ 警告：未找到字体 {FONT_PATH}，请检查路径！")
    font = ImageFont.load_default()


# -------------------------- 2. 兼容Pillow版本的文本尺寸计算 --------------------------
def get_text_size(draw, text, font):
    """兼容Pillow<10.0.0（textsize）和Pillow≥10.0.0（textbbox）"""
    if hasattr(draw, 'textsize'):
        return draw.textsize(text, font=font)
    else:
        bbox = draw.textbbox((0, 0), text, font=font)
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        return (width, height)


# -------------------------- 3. 加载模型和映射表 --------------------------
small_class_model = YOLO('garbage/garbage_train/small_class_model12/weights/best.pt')

# 加载小类→大类映射表
with open('garbage/category_mapping.json', 'r', encoding='utf-8') as f:
    mapping_data = json.load(f)
small2big = mapping_data["小类→大类映射"]
big_categories = mapping_data["大类列表"]

# 获取小分类名称列表
with open('garbage/train_classes.txt', 'r', encoding='utf-8') as f:
    small_categories = [line.strip() for line in f.readlines()]


# -------------------------- 4. 自定义中文标注函数 --------------------------
def draw_chinese_label(img, box, label, color=(0, 255, 0), line_width=2):
    """绘制带中文的检测框（大分类_小分类 置信度格式）"""
    # 1. 绘制检测框
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, line_width)

    # 2. 绘制中文标签（PIL处理）
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    # 计算文本尺寸（兼容所有Pillow版本）
    text_size = get_text_size(draw, label, font=font)
    text_w, text_h = text_size

    # 计算文本位置，防止越界（标签变长，增加右侧越界判断）
    text_x = x1
    text_y = max(y1 - text_h - 2, 2)  # 避免顶部越界
    # 避免标签超出图片右侧
    if text_x + text_w > img.shape[1]:
        text_x = img.shape[1] - text_w - 2

    # 绘制文本背景（半透明，适配长标签）
    draw.rectangle(
        [(text_x, text_y), (text_x + text_w, text_y + text_h)],
        fill=(0, 255, 0, 128)  # 绿色背景，半透明
    )
    # 绘制中文文本
    try:
        draw.text((text_x, text_y), label, font=font, fill=(0, 0, 0))
    except:
        draw.text((text_x, text_y), label, font=font, fill=(0, 0, 0), anchor="lt")

    # PIL转回OpenCV
    img[:] = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


# -------------------------- 5. 推理+归类函数（完全未改！） --------------------------
def infer_and_map_big_category(img_path, conf_thres=0.5):
    """
    输入：图片路径、置信度阈值
    输出：标注后的图片（含大分类_小分类 置信度）、大分类结果列表
    """
    # 1. 模型推理
    results = small_class_model(img_path, conf=conf_thres)

    # 2. 解析结果并映射
    big_category_results = []
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"无法读取图片：{img_path}，请检查路径是否正确")

    for r in results:
        boxes = r.boxes
        if boxes is None:
            continue
        for box in boxes:
            # 获取小分类信息
            cls_id = int(box.cls[0])
            small_cls_name = small_categories[cls_id] if cls_id < len(small_categories) else "未知小类"
            conf = float(box.conf[0])

            # 映射到大分类
            big_cls_name = small2big.get(small_cls_name, "未知分类")

            # 保存结果
            big_category_results.append({
                "大分类": big_cls_name,
                "小分类": small_cls_name,
                "置信度": round(conf, 3),
                "检测框": box.xyxy[0].tolist()
            })

            # 核心修改：标签格式为「大分类_小分类 置信度」
            label = f"{big_cls_name}_{small_cls_name} {conf:.2f}"
            # 标注到图片
            draw_chinese_label(img, box.xyxy[0], label, color=(0, 255, 0))

    # 保存标注后的图片
    annotated_img_path = "annotated_" + img_path.split("/")[-1]
    cv2.imwrite(annotated_img_path, img)

    return annotated_img_path, big_category_results


# -------------------------- 新增：批量处理文件夹函数（仅新增这部分！） --------------------------
def process_folder_images(folder_path, conf_thres=0.5):
    """
    批量处理指定文件夹下的所有图片（调用原有infer_and_map_big_category函数）
    :param folder_path: 图片文件夹路径
    :param conf_thres: 置信度阈值
    :return: 所有图片的处理结果汇总
    """
    # 支持的图片格式
    supported_ext = [".jpg", ".jpeg", ".png", ".bmp", ".tif"]
    # 遍历文件夹下所有文件
    all_results = []
    img_files = [f for f in os.listdir(folder_path) if os.path.splitext(f)[-1].lower() in supported_ext]

    if not img_files:
        print(f"❌ 文件夹 {folder_path} 中无有效图片（支持格式：{supported_ext}）")
        return all_results

    print(f"\n📁 开始批量处理文件夹 {folder_path} 中的 {len(img_files)} 张图片...")
    for img_file in img_files:
        img_path = os.path.join(folder_path, img_file)
        try:
            # 调用原有函数处理单张图片
            annotated_img, img_results = infer_and_map_big_category(img_path, conf_thres)
            all_results.append({
                "图片名": img_file,
                "标注路径": annotated_img,
                "检测结果": img_results
            })
            print(f"✅ 处理完成：{img_file} → 保存为 {annotated_img}")
        except Exception as e:
            print(f"❌ 处理失败：{img_file} → 原因：{str(e)}")
            continue
    return all_results


# -------------------------- 6. 测试运行（可选单张/批量，原有单张逻辑保留） --------------------------
if __name__ == "__main__":

    test_img_path = "garbage_sorting/images/val/img_733_7.jpg"
    try:
        annotated_img, results = infer_and_map_big_category(test_img_path, conf_thres=0.5)
        # 打印结果（同步改为大分类_小分类格式）
        print("\n📊 大分类归类结果：")
        for idx, res in enumerate(results, 1):
            print(f"{idx}. {res['大分类']}_{res['小分类']}（置信度：{res['置信度']}）")
        print(f"\n✅ 标注后的图片已保存至：{annotated_img}")
    except Exception as e:
        print(f"\n❌ 运行出错：{e}")

    # # ========== 选项2：批量处理文件夹（新增，注释掉上面单张即可用） ==========
    # test_folder_path = "../garbage_sorting/images/val"  # 你的图片文件夹路径
    # batch_results = process_folder_images(test_folder_path, conf_thres=0.5)
    #
    # # 打印批量结果汇总
    # if batch_results:
    #     print("\n📊 批量处理汇总结果：")
    #     for idx, res in enumerate(batch_results, 1):
    #         print(f"\n{idx}. 图片：{res['图片名']}")
    #         print(f"   标注路径：{res['标注路径']}")
    #         print(f"   检测结果：")
    #         for det in res['检测结果']:
    #             print(f"     - {det['大分类']}_{det['小分类']}（置信度：{det['置信度']}）")
    # else:
    #     print("\n📊 无有效批量处理结果")