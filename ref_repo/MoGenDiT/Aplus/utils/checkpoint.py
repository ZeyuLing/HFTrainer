import os.path

import torch
class CheckPoint():
    def __init__(self, model, optimizer, log_manager):
        self.model = model
        self.optimizer = optimizer
        self.log_manager = log_manager

    def save(self, save_folder_path, epoch, model_name=None):
        save_state = {
            'epoch': epoch,
            'model': self.model.state_dict(),
            'log': self.log_manager.log,
        }
        # 多个optimizer可以以list存储
        if isinstance(self.optimizer, tuple):
            optimizer_states = []
            for i, optim in enumerate(self.optimizer):
                optimizer_states.append(optim.state_dict())
            save_state.update({'optimizer': tuple(optimizer_states)})
        else:
            save_state.update({'optimizer': self.optimizer.state_dict()})
        if model_name is None:
            model_name = type(self.model).__name__
        torch.save(save_state, os.path.join(save_folder_path, f'{model_name}_{epoch}.pth'))

    @staticmethod
    def load(file_path):
        try:
            checkpoint_dict = torch.load(file_path)
            print(f"check point [{file_path}] loaded")
        except FileNotFoundError as r:
            print(f"Error: check point [{file_path}] doesn't exist")

        return checkpoint_dict

class LMCheckPoint():
    def __init__(self, model, optimizer, ema_model=None):
        """
        Check point for large model
        :param model: main model
        :param optimizer: model's optimizer
        :param ema_model: exponential moving average model (optional)
        """
        self.model = model
        self.optimizer = optimizer
        self.ema_model = ema_model

    def save(self, save_folder_path, epoch, iter, model_name=None, model_only=False):
        model_state = {
            'epoch': epoch,
            'iter': iter,
            'model': self.model.state_dict(),
        }
        if model_name is None:
            model_name = type(self.model).__name__
        iter_id = f'{"%010d" % iter}'
        os.makedirs(os.path.join(save_folder_path, model_name), exist_ok=True)
        torch.save(model_state, os.path.join(save_folder_path, model_name, f'model_{iter_id}.pth'))

        # 保存EMA模型权重（如果存在）
        if self.ema_model is not None:
            ema_model_state = {
            'epoch': epoch,
            'iter': iter,
            'model': self.ema_model.state_dict(),
            }
            torch.save(ema_model_state, os.path.join(save_folder_path, model_name, f'ema_model_{iter_id}.pth'))
        # 多个optimizer可以以list存储
        if not model_only:
            opt_state = {}
            if isinstance(self.optimizer, list):
                optimizer_states = []
                for i, optim in enumerate(self.optimizer):
                    optimizer_states.append(optim.state_dict())
                opt_state.update({'optimizer': optimizer_states})
            else:
                opt_state.update({'optimizer': self.optimizer.state_dict()})
            torch.save(opt_state, os.path.join(save_folder_path, model_name, f'opt_{iter_id}.pth'))


    @staticmethod
    def load(folder_path, model_name, iter, model_only=False):
        iter_id = f'{"%010d" % iter}'
        model_ckpt_path = os.path.join(folder_path, model_name, f'model_{iter_id}.pth')
        opt_ckpt_path = os.path.join(folder_path, model_name, f'opt_{iter_id}.pth')
        try:
            checkpoint_dict = torch.load(model_ckpt_path)
            print(f"model check point [{model_ckpt_path}] loaded")
        except FileNotFoundError as r:
            print(f"Error: check point [{model_ckpt_path}] doesn't exist")
            raise r

        if not model_only:
            try:
                checkpoint_dict.update(torch.load(opt_ckpt_path))
                print(f"optimizer check point [{opt_ckpt_path}] loaded")
            except FileNotFoundError as r:
                print(f"Error: optimizer check point [{opt_ckpt_path}] doesn't exist")
                raise r

        return checkpoint_dict



