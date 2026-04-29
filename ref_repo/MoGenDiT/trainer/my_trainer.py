import pdb

import numpy as np
import torch
from Aplus.utils import DataMeter

from Aplus.tools.annotations import timing
from Aplus.data.process import add_gaussian_noise
from Aplus.runner import *

from EasyDiffusion.resample import create_named_schedule_sampler
from tqdm import tqdm
from .mask_scheduler import MotionMaskScheduler
from torch.utils.tensorboard import SummaryWriter
import math
from .utils import random_index
from queue import Queue
from threading import Thread
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from trainer.geometric_loss import geometric_loss_batch
import random
from animo.skeleton.smpl_body import AnimoSMPLBody
from articulate.math.angular import angle_between, r6d_to_rotation_matrix

class MoGenDitDistributedTrainer(DistributedLMTrainer):
    def __init__(
        self,
        args,
        train_platform,
        model: nn.Module,
        diffusion,
        data,
        motion_rep,
        # 移除显式设备指定，改为通过rank确定
    ):
        """
        支持多GPU训练的修改版本
        添加了分布式训练相关参数
        """
        # 调用基类构造函数，传递EMA参数
        super(MoGenDitDistributedTrainer, self).__init__(
            model=model,
            data=data,
            optimizer=None,  # 稍后创建优化器
            ema_decay=getattr(args, 'ema_decay', 0.999),
            ema_start_step=getattr(args, 'ema_start_step', 2000)
        )
        
        self.args = args
        self.train_platform = train_platform
        self.batch_size = args.batch_size
        self.log_interval = args.log_interval
        self.save_interval = args.save_interval
        self.schedule_sampler_type = args.schedule_sampler_type
        self.save_dir = args.save_dir
        self.log_dir = args.log_dir
        self.mask_scheduler = MotionMaskScheduler(motion_rep=motion_rep)

        self.body_model = AnimoSMPLBody()
        
        # 注意：self.model已经在基类中设置，这里不需要重复设置
        
        # 只在主进程创建日志写入器
        self.log_writer = (
            SummaryWriter(log_dir=os.path.join(self.log_dir, args.model_name))
            if self.is_main_process()
            else None
        )

        self.checkpoint = None
        self.loss_func = nn.MSELoss()
        self.diffusion = diffusion
        self.schedule_sampler = create_named_schedule_sampler(
            self.schedule_sampler_type, diffusion
        )
        self.motion_rep = motion_rep

        # ========== 关键修改1：仅分布式模式下使用DDP封装模型 ==========
        self.model = self.model.to(self.device)
        if self.distributed:
            # 仅多卡时用DDP封装
            self.model = DDP(
                self.model, device_ids=[self.gpu], find_unused_parameters=False
            )
            
            # 对于分布式训练，需要重新初始化EMA模型（因为主模型被DDP包装）
            if self.ema_model is not None:
                # 获取基础模型（非DDP包装）用于EMA
                base_model = self.model.module
                # 重新创建EMA模型
                self.ema_model = copy.deepcopy(base_model)
                self.ema_model.eval()
                self.ema_model.to(self.device)
                for param in self.ema_model.parameters():
                    param.requires_grad = False
                print(f"EMA model reinitialized for distributed training")

        # ========== 关键修改2：分布式模式下才创建DistributedSampler ==========
        # 在模型移动到GPU后创建优化器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        
        # 更新基类中的optimizer引用
        self.optimizer = self.optimizer

        # 仅分布式模式下使用DistributedSampler
        self.data_sampler = DistributedSampler(self.data) if self.distributed else None

        self.data_loader = DataLoader(
            dataset=self.data,
            batch_size=self.batch_size,
            shuffle=(self.data_sampler is None),  # 分布式模式下由sampler控制shuffle
            sampler=self.data_sampler,
            drop_last=False,
            num_workers=4,
            pin_memory=True,
        )
        self.data_queue = None

        # 初始化迭代和epoch计数（已经在基类中初始化，这里确保一致性）
        self.iter = self.iter
        self.epoch = self.epoch

    def data_producer(self, keyframe_modes):
        """数据生产者线程，负责加载和预处理数据"""
        device = self.device
        # 多进程下使用tqdm时只在主进程显示进度条
        pbar = tqdm(
            self.data_loader, desc="Loading data", disable=not self.is_main_process()
        )

        for data in pbar:
            batch, length = data
            length = length.to(device, non_blocking=True)
            batch = batch.to(device, non_blocking=True)
            cond = None

            t, weight = self.schedule_sampler.sample(batch.shape[0], device)
            # b x seq x dim
            keyframe_mask = self.mask_scheduler.get_formulated_mask(
                motion=batch,
                length=length,
                mode_formula=keyframe_modes,
            )
            # b x seq x dim
            bool_length_mask = self.mask_scheduler.get_length_mask_bool(
                motion=batch, length=length
            )
            # b x seq
            padding_mask = (
                bool_length_mask[..., 0] == 0
            ).float()  # 全False的帧为填充帧

            # uncond_idx = random_index(
            #     data_len=batch.shape[0], sampling_rate=self.uncond_rate
            # )
            # cond[uncond_idx] *= 0
            # keyframe_mask[uncond_idx] *= 0

            # 对50%的数据进行degradation处理
            x0 = batch.clone()
            if self.args.motion_degradation and self.args.degrade_rate > 0:
                degradation_idx = random_index(
                    data_len=batch.shape[0], sampling_rate=self.args.degrade_rate
                )
                keyframe_mask[degradation_idx] *= 0
                # 干净的第一帧
                ref_frames = random.randint(1,10)
                keyframe_mask[degradation_idx, :ref_frames] += 1

                x0[degradation_idx] = self.motion_rep.motion_degradation_batch(
                    motion=batch[degradation_idx],
                    keyframe_mask=keyframe_mask[degradation_idx],
                    length=length[degradation_idx],
                    bool_length_mask=bool_length_mask[degradation_idx],
                )

            x_t, noise = self.diffusion.q_sample(
                x0=x0, t=t, obs_mask=keyframe_mask, length_mask=bool_length_mask
            )
            x_wrapped = (
                self.model.module.wrap_inputs(
                    x=x_t, cond=cond, mask=keyframe_mask, padding_mask=padding_mask
                )
                if self.distributed
                else self.model.wrap_inputs(
                    x=x_t, cond=cond, mask=keyframe_mask, padding_mask=padding_mask
                )
            )

            self.data_queue.put(
                (x_wrapped, t, batch, noise, weight, bool_length_mask, length)
            )
        self.data_queue.put(None)  # 数据加载完毕信号

    def train(self, iters, keyframe_modes):
        """训练主函数, 支持多GPU分布式训练"""
        print(f"Training start on {self.device} (rank: {self.rank})")
        if self.rank == 0:
            print(self.args)

        torch.cuda.empty_cache()  # 强制回收未使用的GPU缓存

        total_steps_per_epoch = len(self.data_loader)
        num_epochs = math.ceil(iters / total_steps_per_epoch)

        # 设置模型为训练模式
        self.model.train()

        for epoch in range(num_epochs):
            if self.distributed:
                # 每个epoch打乱数据分布
                self.data_sampler.set_epoch(epoch)

            if self.is_main_process():
                print(f"Starting epoch {epoch}/{num_epochs}")

            self.data_queue = Queue(maxsize=12)

            # 启动生产者线程
            data_producer_thread = Thread(
                target=self.data_producer, kwargs={"keyframe_modes": keyframe_modes}
            )
            data_producer_thread.start()
            target_iter = self.iter + iters

            while self.iter < target_iter:
                train_data_patch = self.data_queue.get()
                if train_data_patch is None:  # 数据处理完毕
                    data_producer_thread.join(timeout=1)
                    break

                losses = self.forward_backward(train_data_patch)

                self.iter += 1

                # 只在主进程记录日志
                if self.is_main_process() and self.iter % self.log_interval == 0:
                    base_model = self.model.module if self.distributed else self.model
                    losses["metrics"].update(
                        {
                            "paras_norm": base_model.get_avg_parameter_norm(),
                            "batch_size": self.batch_size,
                        }
                    )
                    for main_key, sub_dict in losses.items():
                        # 遍历子字典中的具体键值对（如loss中的"total_loss"、metrics中的"paras_norm"）
                        for sub_key, value in sub_dict.items():
                            # 构造层级标签（如"loss/total_loss"、"metrics/paras_norm"）
                            tag = f"{main_key}/{sub_key}"
                            # 记录标量（确保value是标量，如tensor.item()或Python数值）
                            self.log_writer.add_scalar(tag, value, self.iter)
                    self.log_writer.flush()

                # 只在主进程保存模型
                if self.is_main_process() and self.iter % self.save_interval == 0:
                    self.save(
                        folder_path=self.save_dir, model_name=self.args.model_name
                    )

            self.epoch += 1

            # 检查是否达到迭代次数
            if self.iter >= iters:
                break

        # 训练结束，清理分布式环境
        if self.distributed:
            dist.destroy_process_group()

    def forward_backward(self, train_data_patch):
        """前向传播和反向传播，支持分布式梯度同步"""
        self.optimizer.zero_grad()

        x_wrapped, t, x_0, noise, weight, bool_length_mask, length = train_data_patch
        # pdb.set_trace()

        # 前向传播
        pred_x0 = self.model(x_wrapped, t)
        # copy值用于计算mpjpe
        pred_x0_copy = pred_x0.clone()
        x0_copy = x_0.clone()

        # if self.args.use_v_loss:
        #     pred_x0 = self.diffusion.x0_to_v_t(pred_x0, x_wrapped["x_t"], t)
        #     x_0 = self.diffusion.x0_to_v_t(x_0, x_wrapped["x_t"], t)

        # 拆分预测和真实值
        pred_pose, gt_pose = (
            pred_x0[:, :, self.motion_rep.pose_mask],
            x_0[:, :, self.motion_rep.pose_mask],
        )
        pred_joint, gt_joint = (
            pred_x0[:, :, self.motion_rep.joint_mask],
            x_0[:, :, self.motion_rep.joint_mask],
        )
        pred_trans, gt_trans = (
            pred_x0[:, :, self.motion_rep.trans_mask],
            x_0[:, :, self.motion_rep.trans_mask],
        )
        b, seq_len = x_0.shape[0], x_0.shape[1]

        pred_global_joint = pred_joint.reshape(
            b, seq_len, self.motion_rep.n_joint, 3
        ) + pred_trans.reshape(b, seq_len, 1, 3)
        gt_global_joint = gt_joint.reshape(
            b, seq_len, self.motion_rep.n_joint, 3
        ) + gt_trans.reshape(b, seq_len, 1, 3)

        pred_vel = pred_global_joint[:, 1:] - pred_global_joint[:, :-1]
        gt_vel = gt_global_joint[:, 1:] - gt_global_joint[:, :-1]

        # 计算各部分损失
        loss_pose = self.weighted_masked_loss(
            pred=pred_pose,
            gt=gt_pose,
            weight=weight,
            mask=bool_length_mask[:, :, self.motion_rep.pose_mask],
            l1_weight=self.args.l1_weight_x0,
            l2_weight=self.args.l2_weight_x0,
        )

        loss_joint = self.weighted_masked_loss(
            pred=pred_joint,
            gt=gt_joint,
            weight=weight,
            mask=bool_length_mask[:, :, self.motion_rep.joint_mask],
            l1_weight=self.args.l1_weight_x0,
            l2_weight=self.args.l2_weight_x0,
        )

        loss_vel = (
            self.weighted_masked_loss(
                pred=pred_vel.flatten(2),
                gt=gt_vel.flatten(2),
                weight=weight,
                mask=bool_length_mask[:, 1:, self.motion_rep.joint_mask],
                l1_weight=self.args.l1_weight_x0,
                l2_weight=self.args.l2_weight_x0,
            )
        )

        loss_trans = self.weighted_masked_loss(
            pred=pred_trans,
            gt=gt_trans,
            weight=weight,
            mask=bool_length_mask[:, :, self.motion_rep.trans_mask],
            l1_weight=self.args.l1_weight_x0,
            l2_weight=self.args.l2_weight_x0,
        )

        loss = loss_pose + loss_joint + loss_trans + loss_vel

        loss_consis = 0.0
        if self.args.consis_loss:
            pose_6d = self.motion_rep.get_component(pred_x0_copy, "pose")
            joint = self.motion_rep.get_component(pred_x0_copy, "joint")
            loss_consis = self.motion_rep.kinematic_loss_batch(
                pose_6d,
                joint,
                length=length,
                l1_weight=self.args.l1_weight_consis,
                l2_weight=self.args.l2_weight_consis,
            )
            # import pdb; pdb.set_trace()
            if self.iter > self.args.consis_start_step:
                loss += loss_consis

        # 反向传播
        loss.backward()

        # 梯度裁剪
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=1.0
        )

        # 参数更新
        self.optimizer.step()
        
        # 更新EMA模型权重
        self.update_ema()

        # 计算全局关节位置用于监控（不影响训练）
        pose_pred = r6d_to_rotation_matrix(
            pred_x0_copy[:, :, self.motion_rep.pose_mask][
                bool_length_mask[:, :, self.motion_rep.pose_mask]
            ]
            .reshape(-1, 6)
            .detach()
        )
        
        pose_gt = r6d_to_rotation_matrix(
            x0_copy[:, :, self.motion_rep.pose_mask][
                bool_length_mask[:, :, self.motion_rep.pose_mask]
            ]
            .reshape(-1, 6)
            .detach()
        )

        if self.args.global_pose == False:
            pose_pred = self.body_model.forward_kinematics(pose_pred)
            pose_gt = self.body_model.forward_kinematics(pose_gt)

        local_joint_pred = (
            pred_x0_copy[:, :, self.motion_rep.joint_mask][
                bool_length_mask[:, :, self.motion_rep.joint_mask]
            ]
            .reshape(-1, self.motion_rep.n_joint, 3)
            .detach()
        )
        local_joint_gt = (
            x0_copy[:, :, self.motion_rep.joint_mask][
                bool_length_mask[:, :, self.motion_rep.joint_mask]
            ]
            .reshape(-1, self.motion_rep.n_joint, 3)
            .detach()
        )
        trans_pred = (
            pred_x0_copy[:, :, self.motion_rep.trans_mask][
                bool_length_mask[:, :, self.motion_rep.trans_mask]
            ]
            .reshape(-1, 1, 3)
            .detach()
        )
        trans_gt = (
            x0_copy[:, :, self.motion_rep.trans_mask][
                bool_length_mask[:, :, self.motion_rep.trans_mask]
            ]
            .reshape(-1, 1, 3)
            .detach()
        )
        # pdb.set_trace()
        global_joint_pred = local_joint_pred + trans_pred
        global_joint_gt = local_joint_gt + trans_gt

        global_ori_diff = angle_between(pose_pred, pose_gt).reshape(-1, self.motion_rep.n_joint) * 180 / torch.pi

        wa_mpjae = global_ori_diff.mean()
        arms_wa_mpjae = global_ori_diff[:, [16, 17, 18, 19]].mean()
        legs_wa_mpjae = global_ori_diff[:, [1, 2, 4, 5]].mean()

        mpjpe = (
            torch.mean(torch.norm(global_joint_pred - global_joint_gt, dim=-1)) * 100
        )
        mpjpe_local = (
            torch.mean(torch.norm(local_joint_pred - local_joint_gt, dim=-1)) * 100
        )

        results = {
            "losses": {
                "loss_pose": loss_pose.item(),
                "loss_joint": loss_joint.item(),
                "loss_vel": loss_vel.item(),
                "loss_trans": loss_trans.item(),
            },
            "metrics": {
                "wa-mpjae": wa_mpjae.item(),
                "arms-wa-mpjae": arms_wa_mpjae.item(),
                "legs-wa-mpjae": legs_wa_mpjae.item(),
                "wa-mpjpe": mpjpe.item(),
                "mpjpe": mpjpe_local.item(),
                "grad_norm": grad_norm.item(),
            },
        }

        if self.args.consis_loss:
            results["losses"].update(
                {
                    "loss_consis": loss_consis.item(),
                }
            )

        return results

    def weighted_masked_loss(
        self, pred, gt, weight=None, mask=None, l1_weight=0.0, l2_weight=1.0
    ):
        """带权重和掩码的MSE损失计算"""
        if weight is not None:
            weight = weight.reshape(-1, 1, 1)
        else:
            weight = 1

        assert l1_weight + l2_weight > 0

        loss = torch.zeros(1).to(pred.device)

        if l2_weight > 0:
            l2_loss = weight * torch.nn.functional.mse_loss(pred, gt, reduction="none")
            if mask is not None:
                assert mask.dtype == torch.bool
                l2_loss = l2_loss[mask].mean()
            else:
                l2_loss = l2_loss.mean()
            loss += l2_weight * l2_loss

        if l1_weight > 0:
            l1_loss = weight * torch.nn.functional.l1_loss(pred, gt, reduction="none")
            if mask is not None:
                l1_loss = l1_loss[mask].mean()
            else:
                l1_loss = l1_loss.mean()
            loss += l1_weight * l1_loss

        return loss

    def total_step(self):
        return self.iter
