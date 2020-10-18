本项目探索了efficientnet-b0,densenet121,resnet50在CIFAR10数据集上的效果，所有的模型都加载了基于ImageNet的预训练模型，并将最后的输出层的神经元个数改成了10。
实验数据均基于Tesla P100 GPU，由于模型文件较大，densenet121,resnet50训练后的模型见

链接：https://pan.baidu.com/s/1e7fET8WcXtMmEiX6k3Cdxw 提取码：2020 

efficientnet-b0：96.56%        20.48M

densenet121：96.01%            31.04M

resnet50：94.32%               97.81M
