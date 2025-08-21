"""从labelme生成的SegmentationClassPNG 图像中提取指定像素值的掩码"""

import os
import os.path as osp
import glob
import sys
from PIL import Image
import numpy as np


def extract_mask_by_value(input_png_path, output_png_path, target_value):
    """
    从一个 LabelMe 生成的 SegmentationClassPNG 图像中提取指定像素值的掩码。

    Args:
        input_png_path (str): 输入的 SegmentationClassPNG 文件路径。
        output_png_path (str): 输出的二值掩码文件路径。
        target_value (int): 需要提取的像素值（对应 LabelMe 中的类别 ID）。
    """
    try:
        # 打开图像
        # LabelMe 的 SegmentationClassPNG 通常是 P 模式 (paletted)，像素值是索引
        image = Image.open(input_png_path)

        # 确保图像是 P 模式，如果不是，尝试转换为 L 模式（灰度）
        if image.mode != 'P':
            print(f"Warning: Image {input_png_path} is not in P mode ({image.mode}). Attempting to convert to L mode.")
            image = image.convert('L')
            # 如果转换后仍然不是 L 模式，或者转换失败，后续 NumPy 操作可能会有问题
            if image.mode != 'L':
                print(f"Error: Could not process image {input_png_path} in mode {image.mode}. Skipping.")
                return

        # 将图像转换为 NumPy 数组以便快速处理
        img_array = np.array(image)

        # 创建二值掩码：像素值等于 target_value 的地方设为 255，否则设为 0
        # 使用 astype(np.uint8) 确保数据类型正确
        mask_array = (img_array == target_value).astype(np.uint8) * 255

        # 将 NumPy 数组转换回 PIL 图像 (灰度模式 'L')
        new_image = Image.fromarray(mask_array, 'L')

        # 确保输出目录存在
        output_dir = osp.dirname(output_png_path)
        if not osp.exists(output_dir):
            os.makedirs(output_dir)

        # 保存处理后的图像
        new_image.save(output_png_path)
        # print(f"Extracted mask for value {target_value} from {osp.basename(input_png_path)} to {output_png_path}")

    except FileNotFoundError:
        print(f"Error: Input file not found: {input_png_path}. Skipping.")
    except Exception as e:
        print(f"An error occurred while processing {input_png_path} for value {target_value}: {e}. Skipping.")


def main():
    # --- 配置你的路径 ---
    # 上一步脚本输出的根目录，包含 150 个样本文件夹
    previous_output_root = r"C:\srp_OCT\图像数据集（含原图、掩码、文本报告）\图像数据集(文本报告)\labelme2voc结果\PACG合并白内障组mask"  # <-- 请修改为你的实际路径

    # 最终存放提取出的掩码的新根目录
    final_output_root = r"C:\srp_OCT\图像数据集（含原图、掩码、文本报告）\图像数据集(文本报告)\segment_mask\PACG合并白内障组mask"  # <-- 请修改为你的实际路径

    # LabelMe 标注时使用的 labels.txt 文件路径
    labels_file_path = r"C:\srp_OCT\图像数据集（含原图、掩码、文本报告）\图像数据集(文本报告)\labels.txt"  # <-- 请修改为你的实际路径
    # --------------------

    # 检查输入根目录是否存在
    if not osp.isdir(previous_output_root):
        print(f"Error: Previous output root directory not found: {previous_output_root}")
        sys.exit(1)

    # 检查 labels.txt 文件是否存在
    if not osp.exists(labels_file_path):
        print(f"Error: Labels file not found: {labels_file_path}")
        sys.exit(1)

    # 创建最终输出根目录（如果不存在）
    if osp.exists(final_output_root):
        print(f"Warning: Final output root directory already exists: {final_output_root}. Files might be overwritten.")
        # 如果你不想覆盖现有文件，可以在这里添加 sys.exit(1)
    else:
        os.makedirs(final_output_root)
        print(f"Created final output root directory: {final_output_root}")

    # 加载类别名称和创建 ID 到名称的映射
    # 我们需要知道哪些 ID 对应你需要提取的四个部位
    mask_id_to_name_map = {}
    try:
        with open(labels_file_path, 'r', encoding='utf-8') as f:  # 尝试使用 utf-8 编码打开 labels 文件
            for i, line in enumerate(f):
                class_id = i - 1  # LabelMe 默认 __ignore__ 是 -1, _background_ 是 0
                class_name = line.strip()
                # 排除 __ignore__ 和 _background_
                if class_name not in ['__ignore__', '_background_'] and class_id >= 1:
                    mask_id_to_name_map[class_id] = class_name

        if not mask_id_to_name_map:
            print(f"Error: No valid mask classes found in {labels_file_path} (excluding __ignore__ and _background_).")
            sys.exit(1)

        print('Mask classes to extract:', mask_id_to_name_map)

    except FileNotFoundError:
        print(f"Error: Labels file not found at {labels_file_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading labels file {labels_file_path}: {e}")
        sys.exit(1)

    # 遍历上一步脚本输出根目录下的所有子文件夹 (样本文件夹)
    items_in_previous_output_root = os.listdir(previous_output_root)

    processed_samples_count = 0
    for item_name in items_in_previous_output_root:
        sample_folder_path = osp.join(previous_output_root, item_name)

        # 只处理是目录的项 (样本文件夹)
        if osp.isdir(sample_folder_path):
            input_seg_png_folder = osp.join(sample_folder_path, 'SegmentationClassPNG')

            # 检查当前样本文件夹中是否存在 SegmentationClassPNG 文件夹
            if not osp.isdir(input_seg_png_folder):
                print(f"Warning: SegmentationClassPNG folder not found in {sample_folder_path}. Skipping this sample.")
                continue  # 跳过当前样本文件夹

            print(f"Processing sample folder: {item_name}")

            # 遍历 SegmentationClassPNG 文件夹中的所有 PNG 文件
            png_files = glob.glob(osp.join(input_seg_png_folder, '*.png'))

            if not png_files:
                print(f"Warning: No PNG files found in {input_seg_png_folder}. Skipping this sample.")
                continue  # 跳过当前样本文件夹

            # 假设每个样本只有一个主要的掩码 PNG 文件
            if len(png_files) > 1:
                print(
                    f"Warning: Found multiple PNG files in {input_seg_png_folder}. Processing only the first one: {png_files[0]}")

            input_png_file = png_files[0]  # 只处理找到的第一个 PNG 文件

            # 对于当前样本的主掩码文件，提取每个类别的掩码
            for target_value, class_name in mask_id_to_name_map.items():
                # 构建当前类别在最终输出目录下的子文件夹路径
                # 例如: C:\Your\Path\To\Final_Masks_Output\Mask_Cornea
                output_mask_type_folder_name = f'Mask_{class_name}'  # 可以自定义文件夹命名规则
                output_mask_type_folder_path = osp.join(final_output_root, output_mask_type_folder_name)

                # 构建当前样本的掩码文件在最终输出目录下的路径
                # 例如: C:\Your\Path\To\Final_Masks_Output\Mask_Cornea\sample_001.png
                output_png_filename = osp.basename(input_png_file)  # 保留原始文件名
                output_png_path = osp.join(output_mask_type_folder_path, output_png_filename)

                # 调用函数提取并保存掩码
                extract_mask_by_value(input_png_file, output_png_path, target_value)

            processed_samples_count += 1

    print(f"\nFinished processing. Attempted to process {processed_samples_count} sample directories.")
    print(f"Extracted masks are saved in {final_output_root}")


if __name__ == '__main__':
    main()
