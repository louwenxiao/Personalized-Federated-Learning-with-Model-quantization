import os
import sys
import argparse
import socket
import pickle
import asyncio
import concurrent.futures
import json
import random
import time
import numpy as np
import threading
import torch
import copy
import math
from config import *
import torch.nn.functional as F
import datetime
import pandas as pd
import datasets, models
from training_utils import test

#init parameters
parser = argparse.ArgumentParser(description='Distributed Client')
parser.add_argument('--dataset_type', type=str, default='CIFAR100',choices=["CIFAR10", "CIFAR100","FashionMNIST"])
# CIFAR10 lr=0.01 epoch=400 decay_rate=0.993
# FMNIST lr=0.002 epoch=150 decay_rate=1
# CIFAR100 lr=0.01 epoch=400 decay_rate=0.996
parser.add_argument('--model_type', type=str, default='ResNet18')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--data_pattern', type=int, default=0)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--decay_rate', type=float, default=0.995)
parser.add_argument('--min_lr', type=float, default=0.002)
parser.add_argument('--epoch', type=int, default=600)
parser.add_argument('--weight_decay', type=float, default=0.001)
parser.add_argument('--use_cuda', action="store_false", default=True)

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
# device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


def main():
    start_time = datetime.datetime.now().strftime("%m-%d_%H-%M")
    # init config
    for arg in vars(args):
        print(arg, ":", getattr(args, arg))
    RESULT = [[0],[0],[0],[0],[0],[0]]          # 分别用来保存：epoch,带宽MB，等待时间，总时间s，精度，损失
    common_config = CommonConfig()
    common_config.master_listen_port_base += random.randint(0, 20) * 20
    common_config.model_type = args.model_type
    common_config.dataset_type = args.dataset_type
    common_config.batch_size = args.batch_size
    common_config.data_pattern=args.data_pattern
    common_config.lr = args.lr
    common_config.decay_rate = args.decay_rate
    common_config.min_lr=args.min_lr
    common_config.epoch = args.epoch
    common_config.weight_decay = args.weight_decay

    #read the worker_config.json to init the worker node
    with open("worker_config.json") as json_file:
        workers_config = json.load(json_file)

    global_model = models.create_model_instance(common_config.dataset_type)
    init_para = torch.nn.utils.parameters_to_vector(global_model.parameters())
    common_config.para_nums=init_para.nelement()
    model_size = init_para.nelement() * 4 / 1024 / 1024
    print("para num: {}".format(common_config.para_nums))
    print("Model Size: {} MB".format(model_size))

    # create workers
    workers_config['worker_config_list'] = workers_config['worker_config_list'][:20]
    worker_num = len(workers_config['worker_config_list'])
    worker_list: List[Worker] = list()
    for worker_idx, worker_config in enumerate(workers_config['worker_config_list']):
        custom = dict()
        custom["computation"] = worker_config["computation"]
        custom["dynamics"] = worker_config["dynamics"]
        worker_list.append(
            Worker(config=ClientConfig(common_config=common_config,custom=custom),
                    idx=worker_idx,
                    client_ip=worker_config['ip_address'],
                    user_name=worker_config['user_name'],
                    pass_wd=worker_config['pass_wd'],
                    remote_scripts_path=workers_config['scripts_path']['remote'],
                    master_port=common_config.master_listen_port_base+worker_idx,
                    location='local'
                    )
        )

    # Create model instance
    train_data_partition, test_data_partition = partition_data(common_config.dataset_type, common_config.data_pattern,worker_num)
    for worker_idx, worker in enumerate(worker_list):
        worker.config.para = init_para
        worker.config.custom["train_data_idxes"] = train_data_partition.use(worker_idx)
        worker.config.custom["test_data_idxes"] = test_data_partition.use(worker_idx)

    # connect socket and send init config
    communication_parallel(worker_list, action="init")

    recoder: SummaryWriter = SummaryWriter()


    local_steps = int(np.ceil(50000 / worker_num / common_config.batch_size))
    print("local steps: {}".format(local_steps))        # 本地迭代次数是-1，不再改变
    
    total_time = 0.0
    total_traffic = 0.0
    init_para = init_para.to(device)
    for epoch_num in range(1, 1+common_config.epoch):
        print("第{}轮次...".format(epoch_num))
        
        # 更新每一轮的量化参数和聚合参数
        aggregate_vertor = compute_quantization_para(worker_list)
        
        # 发送量化参数到每一个worker
        communication_parallel(worker_list, action="send_para")
        
        # 获得模型，训练时间和通信开销
        communication_parallel(worker_list, action="get_para")
        
        # 发送全局模型到每一个worker
        init_para = aggregate_model(init_para,worker_list,aggregate_vertor)
        communication_parallel(worker_list, action="send_model",data=init_para)
        
        # 接收测试结果
        communication_parallel(worker_list, "get_res")

        
        avg_wait_time,avg_acc,avg_loss,total_time,total_traffic = get_time_and_traffic(worker_list,
                                                    total_time,total_traffic)
        print("等待时间{}，通信开销{}，通信流量{}".format(avg_wait_time,total_time,total_traffic))
        print("平均精度{}，平均损失{}".format(avg_acc,avg_loss))

        
        # 记录每一轮的精度、损失和等待时间。以及通信开销和时间
        recoder.add_scalar('Accuracy/average', avg_acc, epoch_num)
        recoder.add_scalar('Test_loss/average', avg_loss, epoch_num)
        recoder.add_scalar('Wait_time/average', avg_wait_time, epoch_num)
        
        # 横坐标为 时间
        recoder.add_scalar('Accuracy/average_time', avg_acc, total_time)
        recoder.add_scalar('Test_loss/average_time', avg_loss, total_time)
        recoder.add_scalar('resource_time', total_traffic, total_time)
        
        RESULT[0].append(epoch_num)
        RESULT[1].append(total_traffic)
        RESULT[2].append(avg_wait_time)
        RESULT[3].append(total_time)
        RESULT[4].append(avg_acc)
        RESULT[5].append(avg_loss)
        print("\n")
        
        pd.DataFrame(RESULT).to_csv('/data/wxlou/PS_mywork/FedAvg/result/'+
                            '/{}_FedAvg_{}_{}_lr{}_batch{}.csv'.format(start_time,
                            args.data_pattern,args.dataset_type,args.lr,args.batch_size))    
        
    for worker in worker_list:
        worker.socket.shutdown(2)

def communication_parallel(worker_list, action, data=None):
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=len(worker_list),)
        tasks = []
        for worker in worker_list:
            if action == "init":
                tasks.append(loop.run_in_executor(executor, worker.send_init_config))
            elif action == "get_para":
                tasks.append(loop.run_in_executor(executor, get_quantizated_model,worker))
            # elif action == "get_time":
                # tasks.append(loop.run_in_executor(executor, get_time_and_traffic,worker))
            elif action == "send_model":
                tasks.append(loop.run_in_executor(executor, worker.send_data, data))
            elif action == "send_para":
                data=worker.config.compre_ratio
                tasks.append(loop.run_in_executor(executor, worker.send_data,data))
            elif action == "get_res":
                tasks.append(loop.run_in_executor(executor, get_acc_and_loss,worker))
            
        loop.run_until_complete(asyncio.wait(tasks))
        loop.close()
    except:
        sys.exit(0)

   
    

def get_time_and_traffic(worker_list,total_time,total_traffic):
    time_list = []
    traffic = []
    loss = []
    acc = []
    for worker in worker_list:
        time_list.append(worker.config.train_time)
        traffic.append(worker.config.traffic)
        loss.append(worker.loss)
        acc.append(worker.acc)
    total_time += max(time_list)
    total_traffic += sum(traffic)
    wait_time = (max(time_list)*len(worker_list)*1.0-sum(time_list))*1.0/len(worker_list)
    avg_loss = sum(loss)/len(loss)
    avg_acc = sum(acc)/len(acc)
    return wait_time,avg_acc,avg_loss,total_time,total_traffic

def get_quantizated_model(worker):
    received_para,train_time,traffic = get_data_socket(worker.socket)
    worker.model_update = received_para.to(device)
    # train_time,traffic = get_data_socket(worker.socket)
    worker.config.train_time = train_time
    worker.config.traffic = traffic

def partition_data(dataset_type, data_pattern, worker_num=30):      # 使用6个用户训练，需要添加一个IID
    train_dataset, test_dataset = datasets.load_datasets(dataset_type)
    if dataset_type=="CIFAR10" or dataset_type=="FashionMNIST":
        partition_sizes = np.ones((10, worker_num)) * (1 / worker_num)  
    else:
        partition_sizes = np.ones((100, worker_num)) * (1 / worker_num)  
    train_data_partition = datasets.LabelwisePartitioner(train_dataset, partition_sizes=partition_sizes)
    test_data_partition = datasets.LabelwisePartitioner(test_dataset, partition_sizes=partition_sizes)            # 这种方式与训练集数据保持一致
    
    return train_data_partition, test_data_partition


# 计算每一轮的压缩比特和聚合参数向量
def compute_quantization_para(worker_list):
    for worker in worker_list:
        worker.config.compre_ratio = None
    aggregate_vertor = [1.0/len(worker_list)]*len(worker_list)
    return aggregate_vertor

def aggregate_model(local_para, worker_list,aggregate_vertor):
    with torch.no_grad():
        para_delta = torch.zeros_like(local_para)
        average_weight=1.0/len(worker_list)
        for worker in worker_list:
            mid_model = worker.model_update
            # para_delta += worker.config.average_weight * mid_model
            para_delta += average_weight * mid_model
        para_delta += local_para

    return para_delta


def get_acc_and_loss(worker):
    acc,loss = get_data_socket(worker.socket)
    worker.acc = acc
    worker.loss = loss


if __name__ == "__main__":
    main()
