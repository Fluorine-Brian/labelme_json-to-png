from PIL import Image
import os

# 输入和输出文件夹路径
input_folder = "C:\srp_OCT\掩码比较验证/normalX5_coco\SegmentationClassPNG"  # 替换为你的输入文件夹路径
output_folder = "C:\srp_OCT\掩码比较验证/normalX5_coco/1"  # 替换为你的输出文件夹路径
getval=1  #每次修改这个值 ，获得不同部位的掩码
# 确保输出文件夹存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍历输入文件夹中的所有文件
for filename in os.listdir(input_folder):
    if filename.endswith('.png'):
        # 打开图像文件
        image_path = os.path.join(input_folder, filename)
        image = Image.open(image_path)

        # 确保图像是灰度图或RGB图，因为我们需要访问每个像素的具体值
        # 如果图像不是灰度图或RGB图，你可能需要先转换它
        # if image.mode not in ['L', 'RGB']:
        #     image = image.convert('L')  # 转换为灰度图作为处理的基础
        image = image.convert('P')
        # 创建一个与原图大小相同的新图像，用于保存处理后的像素值
        new_image = Image.new('L', image.size)  # 使用灰度模式，因为我们要处理的是像素亮度值

        # 遍历图像的每个像素点，并根据条件设置新的像素值
        for y in range(image.size[1]):
            for x in range(image.size[0]):
                pixel_value = image.getpixel((x, y))
                if pixel_value == getval:
                    new_image.putpixel((x, y), 255)  # 将值为getval的点设置为255
                else:
                    new_image.putpixel((x, y), 0)  # 将其他所有点设置为0

        # 保存处理后的图像到输出文件夹
        output_path = os.path.join(output_folder, filename)
        new_image.save(output_path)

print("处理完成，所有处理后的图像已保存到输出文件夹。")