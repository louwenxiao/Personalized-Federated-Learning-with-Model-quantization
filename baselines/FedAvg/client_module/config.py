from typing import List

class ClientAction:
    LOCAL_TRAINING = "local_training"

class ServerAction:
    LOCAL_TRAINING = "local_training"

class ClientConfig:
    def __init__(self,
                common_config,
                custom: dict = dict()
                ):
        #self.action = action
        self.custom = custom
        self.common_config=common_config
        self.para = dict()

class CommonConfig:
    def __init__(self):
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