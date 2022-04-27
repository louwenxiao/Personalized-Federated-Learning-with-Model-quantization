# FedPAQ

## 算法说明

这是第一个将模型量化用到PS架构下的联邦学习，量化函数用的是NIPS2017的QSGD（论文连接：https://proceedings.neurips.cc/paper/2017/file/6c340f25839e6acdc73414517203f5f0-Paper.pdf ）。算法为：

![image](https://user-images.githubusercontent.com/86142265/165467149-5f52e46c-4d11-47c0-9b7d-62f5b4c5da34.png)


## 本文件说明

本文件夹包含两个文件：client和worker。只需要将这两个文件替换掉FedAvg文件夹中的文件即可。（FedAvg中也提供了这个量化函数，不过为了区分，我还是将这两个分开了）

