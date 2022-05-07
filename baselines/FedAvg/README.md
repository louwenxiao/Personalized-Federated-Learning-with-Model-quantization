# FedAvg

## 说明

FedAvg算法是最基本的联邦学习算法，通过本地多次迭代，发送到server端聚合，再送到每一个worker。反复迭代多次后，实现收敛。论文地址：http://proceedings.mlr.press/v54/mcmahan17a.html

算法：

![image](https://user-images.githubusercontent.com/86142265/165031369-4f04810a-14d2-4eac-a3c7-ac84fafbbcc6.png)

## 项目说明

--只需要运行server文件即可，代码会自动创建多个worker运行
