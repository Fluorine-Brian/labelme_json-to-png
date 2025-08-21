#!/usr/bin/env python

from __future__ import print_function

import glob
import json
import os
import os.path as osp
import sys

import numpy as np
import PIL.Image

# 确保 labelme 库可以被导入
# 如果你的 labelme 库没有安装在标准位置，你可能需要调整 sys.path
try:
    import labelme
    import labelme.utils
except ImportError:
    print("Error: labelme library not found.")
    print("Please install it using: pip install labelme")
    sys.exit(1)


def process_single_json_folder(input_folder, output_folder, class_name_to_id, class_names, colormap):
    """
    处理单个包含 LabelMe JSON 文件的文件夹，并将其输出到指定的输出文件夹。

    Args:
        input_folder (str): 包含图像和 LabelMe JSON 文件的输入文件夹路径。
        output_folder (str): 存储处理结果的输出文件夹路径。
        class_name_to_id (dict): 类别名称到 ID 的映射字典。
        class_names (tuple): 类别名称元组。
        colormap (np.ndarray): 用于可视化的颜色映射。
    """
    print(f"Processing folder: {input_folder}")

    # 创建当前样本的输出子目录结构
    output_jpeg_dir = osp.join(output_folder, 'JPEGImages')
    output_segclass_dir = osp.join(output_folder, 'SegmentationClass')
    output_segclasspng_dir = osp.join(output_folder, 'SegmentationClassPNG')
    output_viz_dir = osp.join(output_folder, 'SegmentationClassVisualization')

    os.makedirs(output_jpeg_dir, exist_ok=True)  # exist_ok=True 避免重复创建报错
    os.makedirs(output_segclass_dir, exist_ok=True)
    os.makedirs(output_segclasspng_dir, exist_ok=True)
    os.makedirs(output_viz_dir, exist_ok=True)

    json_files = glob.glob(osp.join(input_folder, '*.json'))

    if not json_files:
        print(f"Warning: No JSON file found in {input_folder}. Skipping.")
        return

    # 假设每个样本文件夹只有一个 JSON 文件
    if len(json_files) > 1:
        print(f"Warning: Found multiple JSON files in {input_folder}. Processing only the first one: {json_files[0]}")

    label_file = json_files[0] # 只处理找到的第一个 JSON 文件

    try:
        with open(label_file, 'r', encoding='utf-8') as f:  # 尝试使用 utf-8 编码打开
            data = json.load(f)

        base = osp.splitext(osp.basename(label_file))[0]  # JSON 文件名（不含扩展名）

        # 构建输出文件路径
        out_img_file = osp.join(output_jpeg_dir, base + '.jpg')
        out_lbl_file = osp.join(output_segclass_dir, base + '.npy')
        out_png_file = osp.join(output_segclasspng_dir, base + '.png')
        out_viz_file = osp.join(output_viz_dir, base + '.jpg')

        # 获取原始图像路径并读取
        # imagePath 在 JSON 中通常是相对于 JSON 文件本身的路径
        img_file_relative = data.get('imagePath')
        if not img_file_relative:
             print(f"Error: 'imagePath' not found in JSON file: {label_file}. Skipping.")
             return

        img_file = osp.join(input_folder, img_file_relative) # 假设 imagePath 是相对于 input_folder 的

        if not osp.exists(img_file):
             print(f"Error: Image file not found: {img_file} (specified in {label_file}). Skipping.")
             return

        # 尝试读取图像
        try:
            img = np.asarray(PIL.Image.open(img_file))
        except Exception as e:
            print(f"Error: Could not read image file {img_file}: {e}. Skipping.")
            return

        # 保存原始图像到输出目录
        PIL.Image.fromarray(img).save(out_img_file)

        # 将 LabelMe 标注转换为掩码
        lbl = labelme.utils.shapes_to_label(
            img_shape=img.shape,
            shapes=data.get('shapes', []), # 使用 .get() 并提供默认空列表，防止 shapes 键不存在
            label_name_to_value=class_name_to_id,
        )

        # 保存掩码为 PNG 格式 (LabelMe 默认行为)
        labelme.utils.lblsave(out_png_file, lbl)

        # 保存掩码为 NumPy 格式 (可选，如果需要的话)
        np.save(out_lbl_file, lbl)

        # 创建可视化图像
        viz = labelme.utils.draw_label(
            lbl, img, class_names, colormap=colormap)
        PIL.Image.fromarray(viz).save(out_viz_file)

        print(f"Successfully processed {osp.basename(label_file)}")

    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON file: {label_file}. Skipping.")
    except Exception as e:
        print(f"An unexpected error occurred while processing {label_file}: {e}. Skipping.")


def main():
    # --- 配置你的路径 ---
    # 包含 150 个样本文件夹的根目录
    input_root_directory = r"C:\srp_OCT\图像数据集（含原图、掩码、文本报告）\图像数据集(文本报告)\PACG合并白内障组"  # <-- 请修改为你的实际路径

    # 存放所有处理结果的新根目录
    output_root_directory = r"C:\srp_OCT\图像数据集（含原图、掩码、文本报告）\图像数据集(文本报告)\labelme2voc结果/PACG合并白内障组mask"  # <-- 请修改为你的实际路径

    # LabelMe 标注时使用的 labels.txt 文件路径
    labels_file_path = r"C:\srp_OCT\图像数据集（含原图、掩码、文本报告）\图像数据集(文本报告)\labels.txt"  # <-- 请修改为你的实际路径
    # --------------------

    # 检查输入根目录是否存在
    if not osp.isdir(input_root_directory):
        print(f"Error: Input root directory not found: {input_root_directory}")
        sys.exit(1)

    # 检查 labels.txt 文件是否存在
    if not osp.exists(labels_file_path):
        print(f"Error: Labels file not found: {labels_file_path}")
        sys.exit(1)

    # 创建输出根目录（如果不存在）
    if osp.exists(output_root_directory):
        print(f"Warning: Output root directory already exists: {output_root_directory}. Files might be overwritten.")
        # sys.exit(1) # 如果你不想覆盖现有文件，可以取消注释这行
    else:
        os.makedirs(output_root_directory)
        print(f"Created output root directory: {output_root_directory}")

    # 加载类别名称和创建 ID 映射 (只需要加载一次)
    class_names = []
    class_name_to_id = {}
    try:
        with open(labels_file_path, 'r', encoding='utf-8') as f:  # 尝试使用 utf-8 编码打开 labels 文件
            for i, line in enumerate(f):
                class_id = i - 1  # starts with -1 (__ignore__)
                class_name = line.strip()
                class_name_to_id[class_name] = class_id
                if class_id == -1:
                    assert class_name == '__ignore__'
                    continue
                elif class_id == 0:
                    assert class_name == '_background_'
                class_names.append(class_name)
        class_names = tuple(class_names)
        print('Loaded class_names:', class_names)
        print('Loaded class_name_to_id:', class_name_to_id)

    except FileNotFoundError:
         print(f"Error: Labels file not found at {labels_file_path}")
         sys.exit(1)
    except Exception as e:
         print(f"Error loading labels file {labels_file_path}: {e}")
         sys.exit(1)

    # 生成颜色映射 (只需要生成一次)
    colormap = labelme.utils.label_colormap(255)

    # 遍历输入根目录下的所有子文件夹
    # os.listdir() 列出目录下的所有文件和文件夹
    items_in_input_root = os.listdir(input_root_directory)

    processed_folders_count = 0
    for item_name in items_in_input_root:
        input_folder_path = osp.join(input_root_directory, item_name)

        # 只处理是目录的项
        if osp.isdir(input_folder_path):
            # 构建当前样本的输出文件夹路径
            output_folder_path = osp.join(output_root_directory, item_name)

            # 调用处理单个文件夹的函数
            process_single_json_folder(input_folder_path, output_folder_path, class_name_to_id, class_names, colormap)
            processed_folders_count += 1

    print(f"\nFinished processing. Attempted to process {processed_folders_count} directories found in {input_root_directory}.")
    print(f"Results are saved in {output_root_directory}")


if __name__ == '__main__':
    main()
