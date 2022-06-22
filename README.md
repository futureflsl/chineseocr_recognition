# chineseocr_recongnition
crnn pytorch code for train and test

# environment

torch>=1.7.0  
torchvision>=0.8.1  

# How to Train

1、将数据集放在data文件夹里面，图片放在data/images里面，标签txt放在data/labels里面，分别为train.txt和val.txt,txt内容格式为：  
图片路径[TAB]不定长字符串  
2、执行python get_ocr_chars.py生成字符集文件  
3、开始训练python train.py  
测试：使用demo.py即可测试  

# 核心源码

源码改自：https://github.com/chineseocr/chineseocr/tree/master/train/ocr  

