from Aplus.utils import LogManager
from Aplus.utils import CheckPoint, LMCheckPoint
import torch
from torch import nn
from torch.utils.data import DataLoader
from Aplus.utils import DataMeter
import matplotlib.pyplot as plt
import os
import torch.distributed as dist
import copy


class BaseTrainer:
    def __init__(self, model: nn.Module, data, optimizer, batch_size, loss_func):
        """
        Used for manage training process.
        Args:
            model: Your model.
            data: Dataset object. You can build data via Aplus.data.BaseDataset
            optimizer: Model's optimizer.
            batch_size: /
            loss_func: /
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.data = data
        self.epoch = 0
        self.batch_size = batch_size
        self.log_manager = LogManager(items=["epoch", "loss_train", "loss_eval"])
        self.checkpoint = None

    def save(self, folder_path, model_name=None):
        if self.checkpoint is None:
            self.checkpoint = CheckPoint(
                model=self.model, optimizer=self.optimizer, log_manager=self.log_manager
            )
        print(f"saving checkpoint ...", end="")
        os.makedirs(folder_path, exist_ok=True)
        self.checkpoint.save(
            save_folder_path=folder_path, epoch=self.epoch, model_name=model_name
        )
        print("done")

    def restore(self, checkpoint_path, load_optimizer=True):
        checkpoint_dict = CheckPoint.load(file_path=checkpoint_path)
        self.model.load_state_dict(checkpoint_dict["model"])
        if load_optimizer:
            if isinstance(self.optimizer, list):
                for i, optim in enumerate(self.optimizer):
                    optim.load_state_dict(checkpoint_dict["optimizer"][i])
            else:
                self.optimizer.load_state_dict(checkpoint_dict["optimizer"])
        self.log_manager.load_data(data=checkpoint_dict["log"])
        self.epoch = checkpoint_dict["epoch"]
        print(f"training continue from epoch {self.epoch}")
        self.log_manager.print_latest()

    def log_export(self, path):
        """
        Export training log.
        :param path: e.g. './log.xlsx'
        :return: None
        """
        log_folder = os.path.dirname(path)
        os.makedirs(log_folder, exist_ok=True)
        self.log_manager.to_excel(path)

    def get_model_device(self):
        return next(self.model.parameters()).device

    def run(self, epoch, data_shuffle=True, evaluator=None):

        data_loader = DataLoader(
            dataset=self.data,
            batch_size=self.batch_size,
            shuffle=data_shuffle,
            drop_last=False,
        )

        # 获取当前模型所在device
        device = self.get_model_device()

        # AverageMeter用于计算整个epoch的loss
        avg_meter_loss = DataMeter()

        for e in range(epoch):

            # AverageMeter需要在每个epoch开始时置0
            avg_meter_loss.reset()
            self.model.train()
            for i, data in enumerate(data_loader):
                self.optimizer.zero_grad()

                x, y = data
                x = x.to(device)
                y = y.to(device)
                y_hat = self.model(x)

                loss = self.loss_func(y_hat, y)
                loss.backward()

                self.optimizer.step()

                # 每个batch记录一次
                avg_meter_loss.update(value=loss.item(), n_sample=len(y))

            # 获取整个epoch的loss
            loss_train = avg_meter_loss.get_avg()
            self.epoch += 1

            if evaluator is not None:
                loss_eval = evaluator.run()
            else:
                loss_eval = -1

            # 记录当前epoch的训练集 & 验证集loss
            self.log_manager.update(
                {"epoch": self.epoch, "loss_train": loss_train, "loss_eval": loss_eval}
            )

            # 打印最新一个epoch的训练记录
            self.log_manager.print_latest()


class LMTrainer:
    def __init__(self, model=None, data=None, optimizer=None, ema_decay=None, ema_start_step=None):
        """
        For large model training management.
        Args:
            model: Your model.
            data: Dataset object. You can build data via Aplus.data.BaseDataset
            optimizer: Model's optimizer.
            ema_decay: Exponential moving average decay rate (default: 0.9999)
            ema_start_step: Step to start EMA updates (default: 2000)
        """
        self.model = model
        self.optimizer = optimizer
        self.data = data
        self.epoch = 0
        self.iter = 0
        
        # EMA相关参数
        self.ema_decay = ema_decay
        self.ema_start_step = ema_start_step
        self.ema_model = None
        
        # 初始化EMA模型
        if ema_decay is not None:
            self._init_ema_model()

    def _init_ema_model(self):
        """初始化EMA模型副本"""
        if self.model is None:
            return
        
        # 创建EMA模型副本（与主模型相同的结构）

        self.ema_model = copy.deepcopy(self.model)
        self.ema_model.eval()  # EMA模型始终处于评估模式
        
        # 将EMA模型移动到与主模型相同的设备
        device = self.get_model_device()
        self.ema_model.to(device)
        
        # 设置EMA模型参数为不需要梯度
        for param in self.ema_model.parameters():
            param.requires_grad = False

    def update_ema(self):
        """更新EMA模型权重"""
        if self.ema_model is None or self.iter < self.ema_start_step:
            return
        
        with torch.no_grad():
            # 获取主模型的状态字典（处理分布式训练情况）
            if hasattr(self.model, "module"):
                # 分布式训练：模型被包装为DistributedDataParallel
                model_state_dict = self.model.module.state_dict()
            else:
                # 单卡训练：直接获取模型状态字典
                model_state_dict = self.model.state_dict()
            
            ema_state_dict = self.ema_model.state_dict()
            
            # 更新EMA权重：ema = decay * ema + (1 - decay) * model
            for key in model_state_dict:
                if key in ema_state_dict:
                    ema_state_dict[key].copy_(
                        self.ema_decay * ema_state_dict[key] + 
                        (1 - self.ema_decay) * model_state_dict[key]
                    )

    def save(self, folder_path, model_name=None, model_only=False):
        checkpoint = LMCheckPoint(model=self.model, optimizer=self.optimizer, ema_model=self.ema_model)
        print(f"saving checkpoint ...", end="")
        os.makedirs(folder_path, exist_ok=True)
        checkpoint.save(
            save_folder_path=folder_path,
            epoch=self.epoch,
            iter=self.iter,
            model_name=model_name,
            model_only=model_only,
        )
        print("done")

    def restore(self, folder_path, iter, model_name, model_only=False):
        checkpoint_dict = LMCheckPoint.load(
            folder_path=folder_path,
            iter=iter,
            model_name=model_name,
            model_only=model_only,
        )
        self.model.load_state_dict(checkpoint_dict["model"])
        
        # 加载EMA模型权重（如果存在）
        if "ema_model" in checkpoint_dict and self.ema_model is not None:
            self.ema_model.load_state_dict(checkpoint_dict["ema_model"])
            print(f"EMA model loaded from checkpoint")
        
        if not model_only:
            if isinstance(self.optimizer, tuple):
                for i, optim in enumerate(self.optimizer):
                    optim.load_state_dict(checkpoint_dict["optimizer"][i])
            else:
                self.optimizer.load_state_dict(checkpoint_dict["optimizer"])
        self.epoch = checkpoint_dict["epoch"]
        self.iter = checkpoint_dict["iter"]
        print(f"training continue from iter {self.iter}")

    def get_model_device(self):
        return next(self.model.parameters()).device


class DistributedLMTrainer(LMTrainer):
    def __init__(self, model=None, data=None, optimizer=None, ema_decay=None, ema_start_step=None):
        """
        For large model training management.
        Args:
            model: Your model.
            data: Dataset object. You can build data via Aplus.data.BaseDataset
            optimizer: Model's optimizer.
        """
        super(DistributedLMTrainer, self).__init__(model=model, data=data, optimizer=optimizer, ema_decay=ema_decay, ema_start_step=ema_start_step)
        self.init_distributed()

    def save(self, folder_path, model_name=None, model_only=False):
        # 处理模型保存：支持单卡和多卡情况
        if hasattr(self.model, "module"):
            # 多卡情况：模型被包装为DistributedDataParallel
            model_to_save = self.model.module
        else:
            # 单卡情况：直接保存模型
            model_to_save = self.model
        
        # 处理EMA模型保存
        ema_model_to_save = self.ema_model
        if self.ema_model is not None and hasattr(self.model, "module"):
            # 对于分布式训练，EMA模型应该与基础模型（非DDP包装）对应
            pass  # ema_model_to_save已经是基础模型，不需要额外处理

        checkpoint = LMCheckPoint(model=model_to_save, optimizer=self.optimizer, ema_model=ema_model_to_save)
        print(f"saving checkpoint ...", end="")
        os.makedirs(folder_path, exist_ok=True)
        checkpoint.save(
            save_folder_path=folder_path,
            epoch=self.epoch,
            iter=self.iter,
            model_name=model_name,
            model_only=model_only,
        )
        print("done")

    def restore(self, folder_path, iter, model_name, model_only=False):
        print(f"restoring checkpoint ...")
        checkpoint_dict = LMCheckPoint.load(
            folder_path=folder_path,
            iter=iter,
            model_name=model_name,
            model_only=model_only,
        )

        # 处理模型状态加载：支持单卡和多卡情况
        if hasattr(self.model, "module"):
            # 多卡情况：模型被包装为DistributedDataParallel
            self.model.module.load_state_dict(checkpoint_dict["model"])
        else:
            # 单卡情况：直接加载模型状态
            self.model.load_state_dict(checkpoint_dict["model"])
        
        # 加载EMA模型权重（如果存在）
        if "ema_model" in checkpoint_dict and self.ema_model is not None:
            self.ema_model.load_state_dict(checkpoint_dict["ema_model"])
            print(f"EMA model loaded from checkpoint")

        if not model_only:
            if isinstance(self.optimizer, tuple):
                for i, optim in enumerate(self.optimizer):
                    optim.load_state_dict(checkpoint_dict["optimizer"][i])
            else:
                self.optimizer.load_state_dict(checkpoint_dict["optimizer"])
        self.epoch = checkpoint_dict["epoch"]
        self.iter = checkpoint_dict["iter"]
        print(f"training continue from iter {self.iter}")

    def get_model_device(self):
        # 处理模型设备获取：支持单卡和多卡情况
        if hasattr(self.model, "module"):
            # 多卡情况：模型被包装为DistributedDataParallel
            return next(self.model.module.parameters()).device
        else:
            # 单卡情况：直接获取模型参数
            return next(self.model.parameters()).device

    def init_distributed(self):
        self.distributed = False
        self.gpu = 0
        self.world_size = 1
        self.rank = 0

        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            self.rank = int(os.environ["RANK"])
            self.world_size = int(os.environ["WORLD_SIZE"])
            self.gpu = int(os.environ["LOCAL_RANK"])  # 本地GPU索引（整数）
            self.distributed = self.world_size > 1

        # 创建设备对象
        self.device = torch.device(
            f"cuda:{self.gpu}" if torch.cuda.is_available() else "cpu"
        )

        if self.distributed:
            # 关键：提前绑定当前进程到指定GPU
            torch.cuda.set_device(self.device)  # 使用device对象绑定

            # 移除device_id参数，现代PyTorch不需要
            dist.init_process_group(
                backend="nccl",
                init_method="env://",
                world_size=self.world_size,
                rank=self.rank,
                # 此处删除device_id参数
            )

            # 同步时显式指定设备索引（整数）
            dist.barrier(device_ids=[self.gpu])

    def is_main_process(self):
        """检查当前进程是否为主进程"""
        return self.rank == 0
