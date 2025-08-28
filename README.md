# labelme_json-to-png
After using labelme to sketch the mask of object in one image, this code can be used to change labelme's json file into png.
This code used AS-OCT image as an example.

Add specific part of your mask into 'labels.txt'

labelme2voc的使用：项目安装和运行可参考“labelme总结文档”

# 数据集转换

### 1 在github下载labelme=v3.16.2的项目

### 2 项目中example中有labelme2voc脚本 复制下来放入标注数据文件夹同级目录

### 3 创建labels.txt 注意要和实际标注时的标签一样

    注意labels.txt格式 前两行固定不要动 后面每行跟一个label名称（不要是中文）

```
__ignore__
_background_
hongmo
hongmo2
jingzuangti
```

### 4 进行转换

```
python .\labelme2voc.py   标注文件夹名称  输出转化后文件夹名称    --labels   标签命名文件（文件中最好不要有中文）
```

将标注的图片转换成voc格式 命令行执行：

```
python .\labelme2voc.py mylabelme data_dataset_coco  --labels  labels.txt
```

