import os
from typing import List
import paramiko
from scp import SCPClient
from torch.utils.tensorboard import SummaryWriter
from comm_utils import *


class ClientAction:
    LOCAL_TRAINING = "local_training"


class Worker:
    def __init__(self,
                 config,
                 idx,
                 client_ip,
                 user_name,
                 pass_wd,
                 remote_scripts_path,
                 master_port,
                 location
                 ):
        #这个config就是后面的client_config
        self.config=config
        self.idx=idx
        self.client_ip=client_ip
        self.user_name=user_name
        self.pass_wd=pass_wd
        self.remote_scripts_path=remote_scripts_path
        self.master_port=master_port
        self.location=location
        self.socket = None
        self.train_info = None
        self.model_update = None
        self.acc = 0.0
        self.loss = 0.0

        if self.location=="local":
            self.client_ip="127.0.0.1"
            self.__start_local_worker_process()
        else:
            self.__start_remote_worker_process()

    def __start_remote_worker_process(self):
        s = paramiko.SSHClient()
        s.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        s.connect(self.config.client_ip, username=self.user_name, password=self.pass_wd)
        stdin, stdout, stderr = s.exec_command('cd ' + self.remote_scripts_path + ';ls')
        print(stdout.read().decode('utf-8'))
        if self.idx==0: #only edge01
            s1='cd ' + self.remote_scripts_path + '/client_module' + ';nohup python3 client.py'+ ' --master_port ' + str(
                self.config.master_port) +' --master_ip ' + str(self.config.client_ip)+ ' --idx ' + str(self.idx) + '&'
            print(s1)
            s.exec_command(s1)
        else:
            s1='cd ' + self.remote_scripts_path + '/client_module' + ';nohup python3 client.py'+ ' --master_port ' + str(
                self.config.master_port) +' --master_ip ' + str(self.config.client_ip)+ ' --idx ' + str(self.idx) + '&'
            print(s1)
            s.exec_command(s1)

        print("start process at ", self.user_name, ": ", self.config.client_ip)

    def __start_local_worker_process(self):
        python_path = '/opt/anaconda3/envs/pytorch/bin/python'
        python_path = '/data/yxu/software/Anaconda/envs/torch1.6/bin/python'
        os.system('cd ' + os.getcwd() + '/client_module' + ';nohup  ' + python_path + ' -u client.py --master_ip ' 
                     + self.client_ip + ' --master_port ' + str(self.master_port)  + ' --idx ' + str(self.idx) 
                     + ' > client_' + str(self.idx) + '_log.txt 2>&1 &')

        print("start process at ", self.user_name, ": ", self.client_ip)

    def send_data(self, data):
        send_data_socket(data, self.socket)

    def send_init_config(self):
        self.socket = connect_send_socket(self.client_ip, self.master_port)
        send_data_socket(self.config, self.socket)

    def get_config(self):
        self.train_info=get_data_socket(self.socket)


class CommonConfig:
    def __init__(self):
        self.master_listen_port_base=57600

        self.model_type = None
        self.dataset_type = None
        self.batch_size = None
        self.data_pattern = None
        self.lr = None
        self.decay_rate = None
        self.min_lr = None
        self.epoch = None
        self.momentum=None
        self.weight_decay=None

        #这里用来存worker的


class ClientConfig:
    def __init__(self,
                common_config,
                custom: dict = dict()
                ):
        #custom 表示邻居
        self.custom = custom
        self.para = dict()
        self.common_config=common_config

        self.average_weight=0.1
        self.local_steps=50
        self.compre_ratio=1
        self.train_time=0
        self.traffic=0
        self.neighbor_paras=None
        self.neighbor_indices=None
        
