# Traditional-clothing-detection-demo
# Yolov5

## 源码安装

```
git clone https://github.com/ultralytics/yolov5  # clone
cd yolov5
pip install -r requirements.txt  # install
```

在vscode或pycharm终端中输入命令，若无法连接网络直接使用进入https://github.com/ultralytics/yolov5网站下载压缩包进行安装。使用pip命令下载所需依赖。==注意：yolov5要求Python>=3.8.0，Pytorch>=1.8==

## 介绍

解压后的源码如下所示：

![image-20231224150646706](C:\Users\CSS\AppData\Roaming\Typora\typora-user-images\image-20231224150646706.png)

yolov5可以用于分类，检测和分割，此次实验是检测，要使用到的文件包括：

data（存放自建数据集的.yaml.文件）

models（存放不同网络模型的.yaml文件）

train.py（训练）

test.py（验证）

detect.py（推理）

requirment.txt（安装依赖）

## 自建数据集

使用labelimg标注自己的数据集，标注完的数据集每张图片有一个与图片同名的.txt文件即为标签。

### 标记软件使用:

1、打开Anaconda终端切换到用于实验的环境，输入pip install labelimg，等待下载完成，注意安装的位置

2、找到labelimg安装的问价夹，通常为：C:\Anaconda3\envs\Pytorch\Lib\site-packages\labelImg\data

在data文件夹下创建一个xxx.yaml文件用于读取数据，.yaml文件构成如下所示：

![image-20231224184246635](C:\Users\CSS\AppData\Roaming\Typora\typora-user-images\image-20231224184246635.png)

dataset与yolov5同级，dataset目录下有一个自建的数据集文件夹mydataset，mydataset下包含train、val和test（可有可无），train和test问价夹下又包含images用于存放图片；labels用于存放标签。YOLOv5 通过将每个图像路径中最后一个实例“/images/”替换为“/labels/”来自动定位每个图像的标签。

train和val也可以是存有所有训练（验证）图像路径的txt文件，如图所示

![image-20231224184408022](C:\Users\CSS\AppData\Roaming\Typora\typora-user-images\image-20231224184408022.png)

获取所有文件路径保存为txt的脚本已给出：get_txt.py

## 选择模型

Yolov5有简单到复杂有五种型号可供选择，如下图下所：

![image-20231224154537538](C:\Users\CSS\AppData\Roaming\Typora\typora-user-images\image-20231224154537538.png)

由于我们的数据量目前不是很大选择5s或5m即可（扩充数据后可以尝试较复杂模型），示例中我选择5s。每一个模型对应的都有预训练权重可供迁移学习使用。

## 参数配置

yolov5源码中的配置共有三处。

1、超参数

超参数配置文件位于data/hyps/目录下，其中可以选择![image-20231224161008323](C:\Users\CSS\AppData\Roaming\Typora\typora-user-images\image-20231224161008323.png)![image-20231224161046247](C:\Users\CSS\AppData\Roaming\Typora\typora-user-images\image-20231224161046247.png)![image-20231224161109430](C:\Users\CSS\AppData\Roaming\Typora\typora-user-images\image-20231224161109430.png)分别代表增强的有无和高低。

2、模型配置

模型配置位于models/下的.yaml文件，可以选择不同的模型以及更改模型

3、训练参数

训练参数位于train.py中的parse_opt（）函数下，如图所示：

![image-20231224155207358](C:\Users\CSS\AppData\Roaming\Typora\typora-user-images\image-20231224155207358.png)

调参时可以直接在函数中修改对应参数，也可以在终端中使用命令行指定参数，如：python train.py --img 640 --epochs 3 --data coco128.yaml --weights yolov5s.pt。--后跟参数名称，空格，参数值，各个参数间使用空格分隔。

主要使用参数：

--weight：模型预训练权重，可从[Releases · ultralytics/yolov5 (github.com)](https://github.com/ultralytics/yolov5/releases)下载，也可在终端中指定--weights yolov5s.pt自动下载

--cfg：模型.yaml路径（更换模型，改进模型）

--data：数据配置文件.yaml路径

--hpy：超参数配置文件.yaml路径

--epochs：训练轮数

--batch_size：批量大小

--imgsz：训练数据尺寸

--resume：恢复最近的训练，默认为False，若因以外训练中断可以使用此参数继续训练

--nosave：默认为True，即只保留最后一个epoch的权重，若想保留中间权重可以改为False

--noplots：是否保存训练中绘制的图像，默认为False

--device：指定训练设备（cpu或gpu编号）

--optimizer：优化器

--single-cls：多个类别是依然将目标当成单个类别（前景和背景）训练，类似于多个但类别检测

--project：模型参数保存位置

--name：每次运行模型参数保存文件的名称

--patience：早停轮数，损失停止降低x轮后提前停止训练

--save-period：保存周期，每x轮保存一次权重，需配合--nosave使用

## 训练

使用train.py脚本进行训练，更改脚本中的训练配置或使用命令行指定训练配置，如python train.py --img 640 --epochs 3 --data coco128.yaml --weights yolov5s.pt==（注意路径问题）==，训练时脚本会新建一个runs/train/exp文件夹，其中保存了训练权重、超参、训练配置和训练过程中的一些可视化结果。训练过程中以及训练结束后可以使用tensorboard查看训练过程，在终端中输入命令命令：tensorboard --logdir runs/train/exp14==（跑几次实验就会出现几个exp，此示例是展示exp14的训练过程）==。训练过程如图所示：

![image-20231224185649654](C:\Users\CSS\AppData\Roaming\Typora\typora-user-images\image-20231224185649654.png)

## 测试

测试模型使用验证模型使用val.py脚本，验证是需要使用前面准备好的data文件夹下的数据配置文件xxx.yaml和训练好的模型权重（保存在runs/train/exp/best.pt或last.pt，best为训练过程中最好权重，但可能过拟合，last为最后一个eopch的权重，也可能过拟合，具体要看训练曲线，若在train.py中设置了--save-period和--nosave会有更多.pt文件可供选择）。

与训练脚本一样，验证脚本中也有参数配置，如下图所示：

![image-20231224190922925](C:\Users\CSS\AppData\Roaming\Typora\typora-user-images\image-20231224190922925.png)

常用的有：

--data：验证图像保存位置，前面准备好的.yaml配置文件

--weight：训练好的权重文件的路径

--batch-size：验证批量大小

--imgsz：验证图像输入尺寸

--conf-thres：置信度阈值，超过置信度阈值的框才会保留

--iou-thres：NMS（非极大值抑制）中的IOU阈值，用于去除同一中心点的重复框，域值越大抑制效果越弱，保留的框数越多

--max-det：每张图像最多预测框个数

--device：指定验证设备，cpu或gpu编号

--single-cls：多个类别是依然将目标当成单个类别（前景和背景）训练，类似于多个但类别检测

--augment：是否启用数据增强

--project：结果保存位置

--name：保存结果的文件的名称

验证时，更改脚本中的验证配置或使用命令行指定验证配置，如python val.py --img 640  --data coco128.yaml --weights runs/train/exp23/weights/best.pt==（注意路径问题）==，验证时脚本会新建一个runs/val/exp文件夹，其中保存了验证过程中的一些可视化结果。验证过程如图所示：

![image-20231224193217437](C:\Users\CSS\AppData\Roaming\Typora\typora-user-images\image-20231224193217437.png)

## 推理

推理使用detect.py，需要使用训练并验证好的权重，输入需要预测的图片，推理结果为可视化框的图片

推理脚本的参数配置如下：

![image-20231224194655184](C:\Users\CSS\AppData\Roaming\Typora\typora-user-images\image-20231224194655184.png)

常用的有：

--weight：使用的推理权重

--source：要推理的图片、视频文件（夹）路径

--conf-thres：置信度阈值，超过置信度阈值的框才会保留

--iou-thres：NMS（非极大值抑制）中的IOU阈值，用于去除同一中心点的重复框，域值越大抑制效果越弱，保留的框数越多

--device：指定验证设备，cpu或gpu编号

--save-txt：保存每张图片推理框的类别和坐标，类似于训练时的标签

--save-csv：将所有预测结果保存到一个csv文件中

--project：结果保存位置

--name：每次运行保存结果的文件的名称

推理时，更改脚本中的推理配置或使用命令行指定推理配置，如python detect.py --weight /home/deng/css/detection/yolov5/runs/train/exp23/weights/last.pt --source /home/deng/css/detection/yolov5/data/images  --save-txt --save-csv==（注意路径问题）==，推理时脚本会新建一个runs/detect/exp文件夹，其中保存了推理的结果。推理过程如图所示：

![image-20231224195625635](C:\Users\CSS\AppData\Roaming\Typora\typora-user-images\image-20231224195625635.png)

在此配置下，推理结果包含带框的图片，每张图片的txt推理结果以及保存有所有结果的csv文件。
