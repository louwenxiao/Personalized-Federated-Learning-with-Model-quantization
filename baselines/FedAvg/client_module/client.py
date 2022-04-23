import os
import time
import socket
import pickle
import argparse
import asyncio
import concurrent.futures
import threading
import math
import copy
import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from pulp import *
import random
from config import ClientConfig, CommonConfig
from client_comm_utils import *
from training_utils import train, test
import datasets, models

parser = argparse.ArgumentParser(description='Distributed Client')
parser.add_argument('--idx', type=str, default="0",
                    help='index of worker')
parser.add_argument('--master_ip', type=str, default="127.0.0.1",
                    help='IP address for controller or ps')
parser.add_argument('--master_port', type=int, default=58000, metavar='N',
                    help='')
parser.add_argument('--visible_cuda', type=str, default='-1')
parser.add_argument('--use_cuda', action="store_false", default=True)

args = parser.parse_args()

if args.visible_cuda == '-1':
    os.environ['CUDA_VISIBLE_DEVICES'] = str(int(args.idx) % 4)
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.visible_cuda
device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")

def main():
    client_config = ClientConfig(
        common_config=CommonConfig()
    )
    recorder = SummaryWriter("logs/log_"+str(args.idx))
    # receive config
    master_socket = connect_get_socket(args.master_ip, args.master_port)
    config_received = get_data_socket(master_socket)
    #这里跟服务器通信然后获取配置文件，get_data_socket是堵塞的。
    for k, v in config_received.__dict__.items():
        setattr(client_config, k, v)

    # computation = client_config.custom["computation"]
    dynamics=client_config.custom["dynamics"]

    common_config = CommonConfig()
    common_config.model_type = client_config.common_config.model_type
    common_config.dataset_type = client_config.common_config.dataset_type
    common_config.batch_size = client_config.common_config.batch_size
    common_config.data_pattern=client_config.common_config.data_pattern
    common_config.lr = client_config.common_config.lr
    common_config.decay_rate = client_config.common_config.decay_rate
    common_config.min_lr=client_config.common_config.min_lr
    common_config.epoch = client_config.common_config.epoch
    common_config.weight_decay = client_config.common_config.weight_decay

    # init config
    print(common_config.__dict__)

    # create model
    local_model = models.create_model_instance(common_config.dataset_type)
    torch.nn.utils.vector_to_parameters(client_config.para, local_model.parameters())
    local_model.to(device)
    local_para = torch.nn.utils.parameters_to_vector(local_model.parameters()).detach()
    model_size = local_para.nelement() * 4 / 1024 / 1024
    print("Model Size: {} MB".format(model_size))
    accumulated_error = torch.zeros_like(local_para)
    

    # create dataset
    print(len(client_config.custom["train_data_idxes"]))
    train_dataset, test_dataset = datasets.load_datasets(common_config.dataset_type)
    train_loader = datasets.create_dataloaders(train_dataset, batch_size=common_config.batch_size, 
                                               selected_idxs=client_config.custom["train_data_idxes"])
    # test_loader = datasets.create_dataloaders(test_dataset, batch_size=128, shuffle=False)
    test_loader = datasets.create_dataloaders(test_dataset, batch_size=32, 
                                    selected_idxs=client_config.custom["test_data_idxes"], shuffle=False)
    local_iters = math.ceil(len(train_loader.loader.dataset) / train_loader.loader.batch_size)
    print("Barch Size:",local_iters)

    epoch_lr = common_config.lr
    for epoch in range(1, 1+common_config.epoch):
        time1 = time.time()         # 开始训练
        
        if epoch > 1 and epoch % 1 == 0:
            epoch_lr = max((common_config.decay_rate * epoch_lr, common_config.min_lr))
        print("epoch-{} lr: {}".format(epoch, epoch_lr))
        
        
        # 接收量化参数
        compre_ratio=get_data_socket(master_socket)
        print("Compression Ratio: ", compre_ratio,"bits")

        begin_time = time.time()                                    # 模型训练
        old_para = torch.nn.utils.parameters_to_vector(local_model.parameters()).clone().detach()
        optimizer = optim.SGD(local_model.parameters(), lr=epoch_lr, momentum=0.9, weight_decay=common_config.weight_decay)
        train_loss = train(local_model, train_loader, optimizer, local_iters=local_iters,device=device)
        new_para = torch.nn.utils.parameters_to_vector(local_model.parameters()).clone().detach()
        time2 = time.time()
        print("训练时间：",time2-time1)
        
        grad_para = new_para - old_para + accumulated_error
        model_update = model_quantization(grad_para,compre_ratio)   # 模型量化
        accumulated_error = grad_para - model_update                # 更新误差
        print("量化后的误差为：",torch.norm(accumulated_error))
        end_time = time.time()
        
        train_and_send_time = (end_time-begin_time)*10              # 计算传输训练时间和流量
        train_time,traffic = get_time_and_traffic(train_and_send_time,model_size,compre_ratio)
        time3 = time.time()
        print("模型压缩时间",time3-time2)
        
        
        # 发送模型、训练时间和流量
        send_data_socket((model_update,train_time,traffic), master_socket)
        print("train time:",train_time)
        time4 = time.time()
        print("模型发送时间：",time4-time3)
        
        
        # 获得全局模型
        # time.sleep(100)
        model = get_data_socket(master_socket)
        torch.nn.utils.vector_to_parameters(model.to(device), local_model.parameters())
        time5 = time.time()
        print("模型接收时间：",time5-time4)
        
        
        # 模型测试
        test_loss, acc = test(local_model, test_loader, device)
        time6 = time.time()
        print("模型测试时间：",time6-time5)
        send_data_socket((acc, test_loss), master_socket)
        # print("train time: ",train_time)
        
        
        recorder.add_scalar('acc_worker-' + str(args.idx), acc, epoch)
        recorder.add_scalar('test_loss_worker-' + str(args.idx), test_loss, epoch)
        recorder.add_scalar('train_loss_worker-' + str(args.idx), train_loss, epoch)
        print("epoch: {}, train loss: {}, test loss: {}, test accuracy: {}".format(epoch, train_loss, test_loss, acc))
        print("训练一轮的时间：",time.time()-time1)
        print("\n\n")
        
        
    master_socket.shutdown(2)
    master_socket.close()

def get_time_and_traffic(train_and_send_time,model_size,compre_ratio):
    train_time = np.random.normal(loc=train_and_send_time, scale=np.sqrt(3))    # 使用高斯分布模拟训练发送时间
    if compre_ratio == None:
        size = model_size
    else:
        size = model_size * compre_ratio / 32.0
    return train_time,size


# 模型量化
def model_quantization(local_para, ratio):
    if ratio == None:
        return local_para
    
    return local_para

if __name__ == '__main__':
    main()
