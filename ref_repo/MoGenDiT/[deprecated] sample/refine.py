import torch
import numpy as np
from EasyDiffusion import GaussianDiffusion, BetaSchedule, ModelMeanType
from articulate.math import axis_angle_to_rotation_matrix
from my_data.amass import NpzMotion
from trainer import load_args
from model.my_model import *
from motion_process.motion_representation import Motion291Rep, OccamMotionRep
import os
from motion_process.processors import Npz2VecProcessor
from trainer.mask_scheduler import MotionMaskScheduler
from matplotlib.animation import FuncAnimation
from matplotlib import pyplot as plt
from sample.load_gen_args import load_setting
from model.more_diff import get_MoreDiff_model


args = load_setting(json_path="./sample/gen_settings.json")
# motion_root_path = "./web_fbx/motion数据/origin_v2m/origin"
motion_root_path = args.data_source_path

out_put_path = f"./web2/motion_data/refine/{args.caption}"
use_vel = True
model_name = args.model_name
restore_iter = args.restore_iter
eraly_stop = None

custom_timesteps = [
    999,
    750,
    500,
    250,
    100,
    50,
    25,
    10,
    5,
    0,
]

denoise_steps = args.denoise_steps
os.makedirs(out_put_path, exist_ok=True)
mode = "gen-from-denoise-rot-kpts"


class MoreDiffRefiner:
    """
    运动精炼封装类，用于封装运动精炼相关功能
    """
    
    def __init__(self, motion_rep, model, diffusion):
        """
        初始化运动精炼器
        Args:
            motion_rep: 运动表示对象
            model: 模型对象
            diffusion: 扩散模型对象
        """
        self.motion_rep = motion_rep
        self.model = model
        self.diffusion = diffusion
    
    def _denoise_mode(
        self,
        motion,
        cond,
        mask,
        keep_mask,
        step=10,
        eta=1.0,
        imputation_mode="skip_last",
    ):
        """
        独立的denoise模式处理函数
        Args:
            motion: 输入运动数据
            cond: 条件输入
            mask: 原始mask
            keep_mask: 需要保持的部分mask
            eta: 噪声缩放因子，默认为1.0
            imputation_mode: 插补模式
        Returns:
            处理后的运动数据
        """
        device = motion.device
        _motion = self.motion_rep.normalization(motion.squeeze(0)).unsqueeze(0)
        
        with torch.no_grad():
            x_wrap = self.model.wrap_inputs(_motion, cond, mask, None)
            _motion = self.diffusion.denoise(
                x_wrap=x_wrap,
                model=self.model,
                num_timesteps=step,
                eta=eta,
                mask=keep_mask,
                imputation_mode=imputation_mode,
            )
        
        return _motion
    
    def _regen_mode(
        self,
        motion,
        cond,
        mask,
        keep_mask=None,
        eta=0.0,
        early_stop=None,
        custom_timesteps=custom_timesteps,
        fast_sampling=True,
        imputation_mode="skip_last",
    ):
        """
        独立的regen模式处理函数，支持可选的keep_mask参数
        Args:
            motion: 输入运动数据
            cond: 条件输入
            mask: 原始mask
            keep_mask: 可选的保持部分mask，如果为None则使用mask
            eta: 噪声缩放因子，默认为0.0
            early_stop: 早期停止参数
            custom_timesteps: 自定义时间步
            imputation_mode: 插补模式
        Returns:
            处理后的运动数据
        """
        device = motion.device
        _motion = self.motion_rep.normalization(motion.squeeze(0)).unsqueeze(0)
        
        # 如果未提供keep_mask，则使用mask
        if keep_mask is None:
            keep_mask = mask.bool()
        
        with torch.no_grad():
            x_wrap = self.model.wrap_inputs(_motion, cond, mask, None)
            _motion = self.diffusion.ddim_sample_loop(
                x_wrap=x_wrap,
                model=self.model,
                eta=eta,
                mask=keep_mask,
                early_stop=early_stop,
                custom_timesteps=custom_timesteps if fast_sampling else None,
                imputation_mode=imputation_mode,
            )
        
        return _motion

    def refine(
        self,
        motion,
        cond,
        step=10,
        eta=1.0,
        early_stop=None,
        custom_timesteps=None,
        imputation_mode="skip_last",
        mode="denoise",
        window_size=224,
        prev_padding=20,
        use_windowed=False,
    ):
        """
        统一的运动精炼接口函数，支持三种不同的精炼模式和窗口化处理
        Args:
            motion: 输入运动数据
            cond: 条件输入
            mask: 原始mask
            keep_mask: 可选的保持部分mask
            step: 去噪步数，默认为10
            eta: 噪声缩放因子，默认为1.0
            early_stop: 早期停止参数
            custom_timesteps: 自定义时间步
            imputation_mode: 插补模式
            mode: 精炼模式，可选值为 denoise / trans_regen / ada_denoise
            window_size: 窗口大小，默认为224
            prev_padding: 前向填充大小，默认为20
            use_windowed: 是否使用窗口化处理，默认为False
        Returns:
            处理后的运动数据
        """
        device = motion.device
        
        # 检查模式合法性
        valid_modes = ["denoise", "trans-regen", "ada-denoise"]
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode: {mode}. Valid modes are: {valid_modes}")
        
        # 如果不需要窗口化处理，使用原始的单次处理逻辑
        if not use_windowed:
            mask = torch.zeros_like(motion).to(device)
            if mode in ["denoise", "ada-denoise"]:
                mask[:, :1] += 1
            elif mode in ["trans-regen"]:
                mask[:, :, self.motion_rep.pose_mask] += 1
                mask[:, :, self.motion_rep.joint_mask] += 1
            mask = mask.clamp(0, 1)
            keep_mask = mask.bool()
            return self._non_windowed_refine(
                motion=motion,
                cond=cond,
                mask=_mask,
                keep_mask=_keep_mask,
                step=step,
                eta=eta,
                early_stop=early_stop,
                custom_timesteps=custom_timesteps,
                imputation_mode=imputation_mode,
                mode=mode,
            )
        
        # 窗口化处理逻辑
        current_idx = 0
        prev_frame_pad = 0
        while True:
            begin = current_idx
            end = min(begin + window_size, motion.shape[1])
            _motion = motion[:, begin:end]
            length = end - begin
            _mask = torch.zeros_like(_motion).to(device)
            _mask[:, :prev_frame_pad] += 1
            if mode in ["denoise", "ada-denoise"]:
                _mask[:, :1] += 1
            elif mode in ["trans-regen"]:
                _mask[:, :, self.motion_rep.pose_mask] += 1
                _mask[:, :, self.motion_rep.joint_mask] += 1
            _mask = _mask.clamp(0, 1)
            _keep_mask = _mask.bool()

            # 使用非窗口化版本处理当前窗口
            _motion = self._non_windowed_refine(
                motion=_motion,
                cond=cond,
                mask=_mask,
                keep_mask=_keep_mask,
                step=step,
                eta=eta,
                early_stop=early_stop,
                custom_timesteps=custom_timesteps,
                imputation_mode=imputation_mode,
                mode=mode,
            )

            # 根据【生成结果】的位移来判断截断位置
            _trans = _motion[..., self.motion_rep.trans_mask].reshape(-1, 3)

            cutoff_in_segment = None  # 片段内的截断索引（相对于片段的起始）
            if _trans.shape[0] > 0:
                # 计算片段内每帧相对于片段第一帧的3D欧氏距离
                first_frame_trans = _trans[0:1, :]  # 片段第一帧的平移数据 [1, 3]
                frame_distances = torch.norm(
                    _trans - first_frame_trans, dim=1
                )  # 片段内每帧距离 [片段帧数]

                # 找到片段内第一个距离>3m的帧索引
                distance_exceed_mask = frame_distances > 3.0
                cutoff_indices = torch.where(distance_exceed_mask)[0]
                if len(cutoff_indices) > 0:
                    # 确保索引是整数类型
                    cutoff_indices = cutoff_indices.long()
                    cutoff_in_segment = cutoff_indices[0].item()  # 片段内的第一个超3m索引

            # 更新end：基于片段内的截断索引，映射回原始motion的索引
            if cutoff_in_segment is not None:
                # 原始end更新为：begin + 片段内第一个超3m的索引（不包含该帧）
                end = begin + cutoff_in_segment
                # 双重边界保护：确保end不超过原end、不超过motion总长度
                end = max(end, begin + prev_frame_pad + 30)  # 确保至少包含30帧生成内容
                end = min(
                    end, motion.shape[1])

                # 重新截取motion到最终的end位置（截断后）
                _motion = _motion[:, :(end - begin)]
                # import pdb; pdb.set_trace()

            # 数据缝合
            _motion = self.motion_rep.pre_stitch(
                _motion[0, :],
                ref_motion=motion[0, [begin]],
                reset_height=False,
                stitch_joint_idx=0,
            )

            # print(begin, end)
            
            # 更新原始运动数据
            motion[0, begin:end] = _motion
            prev_frame_pad = prev_padding
            current_idx += (end - begin) - prev_frame_pad
            if end >= motion.shape[1]:
                break

        return motion
    
    def _non_windowed_refine(
        self,
        motion,
        cond,
        mask,
        keep_mask=None,
        step=10,
        eta=1.0,
        early_stop=None,
        custom_timesteps=None,
        imputation_mode="skip_last",
        mode="denoise",
    ):
        """
        非窗口化的运动精炼函数，支持三种不同的精炼模式
        Args:
            motion: 输入运动数据
            cond: 条件输入
            mask: 原始mask
            keep_mask: 可选的保持部分mask
            step: 去噪步数，默认为10
            eta: 噪声缩放因子，默认为1.0
            early_stop: 早期停止参数
            custom_timesteps: 自定义时间步
            imputation_mode: 插补模式
            mode: 精炼模式，可选值为 denoise / trans_regen / ada_denoise
        Returns:
            处理后的运动数据
        """
        device = motion.device
        
        # 检查模式合法性
        valid_modes = ["denoise", "trans-regen", "ada-denoise"]
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode: {mode}. Valid modes are: {valid_modes}")
        
        # 模式1: denoise 模式
        if mode == "denoise":
            return self._denoise_mode(
                motion=motion,
                cond=cond,
                mask=mask,
                keep_mask=keep_mask if keep_mask is not None else mask.bool(),
                step=step,
                eta=eta,
                imputation_mode=imputation_mode,
            )
        
        # 模式2: trans_regen 模式（对应 windowed_refine 中的 gen-from-rot-kpts）
        elif mode == "trans-regen":
            return self._regen_mode(
                motion=motion,
                cond=cond,
                mask=mask,
                keep_mask=keep_mask if keep_mask is not None else mask.bool(),
                eta=eta,
                early_stop=early_stop,
                custom_timesteps=custom_timesteps,
                imputation_mode=imputation_mode,
            )
        
        # 模式3: ada_denoise 模式（自适应去噪）
        elif mode == "ada-denoise":
            # 第一阶段：执行标准 denoise
            denoised_motion = self._denoise_mode(
                motion=motion,
                cond=cond,
                mask=mask,
                keep_mask=keep_mask if keep_mask is not None else mask.bool(),
                step=min(step*2, 100),  # 第一阶段使用更大的步数以获得更明显的变化
                eta=1,
                imputation_mode=imputation_mode,
            )
            
            # 第二阶段：计算 denoise 前后的变化值
            # 将运动数据归一化以便比较
            
            # 计算变化值（欧氏距离）
            change_values = torch.abs(
                motion - denoised_motion
            )
            
            # 基于变化值阈值生成新的 mask
            # 变化值 > 0.1 的区域标记为需要重新处理的区域（mask = 0）
            # 变化值 <= 0.1 的区域标记为保持区域（keep_mask = 1）
            change_threshold = 0.1
            high_change_mask = change_values > change_threshold
            low_change_mask = change_values <= change_threshold
            
            # 创建新的 mask 和 keep_mask
            # 需要重新处理的区域（高变化区域）：mask=0（不保持）
            # 需要保持的区域（低变化区域）：keep_mask=1
            new_mask = mask.clone()
            # 在低变化区域设置 keep_mask=1（保持这些区域）
            # 在高变化区域设置 keep_mask=0（重新处理这些区域）
            new_mask[low_change_mask] += 1
            new_mask = new_mask.clamp(0, 1)
            new_keep_mask = new_mask.bool()
            
            # 第三阶段：基于计算的 mask 和 keep_mask，对原始运动数据再次执行 denoise
            return self._denoise_mode(
                motion=motion,
                cond=cond,
                mask=mask,
                keep_mask=new_keep_mask,
                step=step,
                eta=eta,
                imputation_mode=imputation_mode,
            )


def windowed_refine(
    motion,
    cond,
    model,
    diffusion,
    motion_rep,
    window_size=224,
    prev_padding=20,
    mode="gen-from-denoise-rot-kpts",
    step=10,
    eta=1.0,
    imputation_mode="skip_last",
    early_stop=None,
    custom_timesteps=None,
):
    """
    窗口化处理函数，支持多种模式
    Args:
        motion: 输入运动数据
        cond: 条件输入
        model: 模型对象
        diffusion: 扩散模型对象
        motion_rep: 运动表示对象
        window_size: 窗口大小
        prev_padding: 前向填充大小
        mode: 处理模式，可选值为 gen-from-denoise-rot-kpts / gen-from-rot-kpts / gen-from-kpts
        step: 去噪步数，默认为10
        eta: 噪声缩放因子，默认为1.0
        imputation_mode: 插补模式
        early_stop: 早期停止参数
        custom_timesteps: 自定义时间步
    Returns:
        处理后的运动数据
    """
    device = motion.device
    
    # 创建精炼器实例
    refiner = MoreDiffRefiner(motion_rep, model, diffusion)
    
    # 创建keep_mask
    keep_mask = torch.zeros_like(motion).to(device)
    keep_mask[:, 0] += 1
    
    # 将windowed_refine的模式映射到refine支持的模式
    mode_mapping = {
        "gen-from-denoise-rot-kpts": "denoise",
        "gen-from-rot-kpts": "trans_regen",
        "gen-from-kpts": "trans_regen",
    }
    
    if mode not in mode_mapping:
        raise ValueError(f"Invalid mode for windowed_refine: {mode}")
    
    refined_mode = mode_mapping[mode]
    
    # 直接使用refine函数的窗口化处理功能
    # 创建全1的mask，表示整个序列都是有效的
    full_mask = torch.ones_like(motion).to(device)
    
    return refiner.refine(
        motion=motion,
        cond=cond,
        mask=full_mask,
        keep_mask=keep_mask,
        step=step,
        eta=eta,
        early_stop=early_stop,
        custom_timesteps=custom_timesteps,
        imputation_mode=imputation_mode,
        mode=refined_mode,
        window_size=window_size,
        prev_padding=prev_padding,
        use_windowed=True,
    )


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset_path = motion_root_path
    motion_rep = OccamMotionRep(keep_hand=False, global_pose=args.global_pose)
    mp = Npz2VecProcessor(keep_hand=False)

    # training settings
    model = get_MoreDiff_model(
        data_dim=motion_rep.data_dim, version=args.model_version
    ).to(device)

    model.restore(
        checkpoint_path=f"./save/{model_name}/model_{'%010d' % restore_iter}.pth"
    )
    model.eval()

    diffusion = GaussianDiffusion(
        num_timesteps=1000,
        beta_schedule=BetaSchedule.COSINE,
        model_mean_type=ModelMeanType.START_X,
        # noise_remap_mode="clip",
    )
    data_dict = NpzMotion.load_data(dataset_path, min_len=20, motion_rep=motion_rep)
    data = NpzMotion(
        data=data_dict, motion_rep=motion_rep, fix_len=600, use_vel=use_vel
    )
    file_names = data_dict["file_name"]
    for i, fname in enumerate(file_names):
        print(f"processing {fname} [{i+1}/{len(file_names)}]")
        motion, length, cond = data.__getitem__(index=i)
        beta = data_dict["beta"][i]
        length = torch.tensor([length])
        motion = motion[:length]

        motion = motion_rep.normalization(motion)
        # pdb.set_trace()
        motion = motion.unsqueeze(0).to(device)
        cond = cond.unsqueeze(0).to(device)

        gen_motion = windowed_refine(
            motion,
            cond,
            model,
            diffusion,
            motion_rep,
            step=10,
            mode=mode,
        )
        # mask = torch.zeros_like(motion).to(device)
        # mask[:, 0] += 1
        # # bool_length_mask = self.mask_scheduler.get_length_mask_bool(motion=motion, length=length)

        # x_wrap = model.wrap_inputs(motion, cond, mask, None)

        # # 迭代去噪
        # with torch.no_grad():
        #     for i in range(1):
        #         x_wrap = model.wrap_inputs(motion, cond, mask, None)
        #         motion = diffusion.denoise(
        #             x_wrap=x_wrap, model=model, num_timesteps=10, eta=0
        #         )
        # gen_motion = motion

        # with torch.no_grad():
        #     gen_motion = diffusion.denoise(
        #         x_wrap=x_wrap, model=model, num_timesteps=denoise_steps, eta=1
        #     )

        pose, joint, trans = motion_rep.decode(gen_motion[0])
        # trans = joint[:, 0]

        # fig, ani = create_animation(np.array(joint.cpu()), stationary=None)

        out_npz = mp.motion2npz_dict(pose=pose, trans=trans, frame_rate=30, betas=beta)
        npz_path = os.path.join(out_put_path, fname)
        print("Saving SMPL params to [{}]".format(npz_path))
        np.savez(npz_path, **out_npz)


if __name__ == "__main__":
    main()
