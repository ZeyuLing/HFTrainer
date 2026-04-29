import torch
import numpy as np
from EasyDiffusion import GaussianDiffusion, BetaSchedule, ModelMeanType
from articulate.math import axis_angle_to_rotation_matrix
from my_data.amass import NpzMotion
from trainer import load_args
from model.my_model import *
from motion_process.motion_representation import Motion291Rep, HM263XRep, OccamMotionRep
import os
from motion_process.processors import Npz2VecProcessor
from trainer.mask_scheduler import MotionMaskScheduler
from matplotlib.animation import FuncAnimation
from matplotlib import pyplot as plt
from sample.load_gen_args import load_setting
from pathlib import Path
from model.more_diff import get_MoreDiff_model

from pathlib import Path
from .refine import windowed_refine


def traverse_subfolders(root_dir):
    """遍历 root_dir 下的所有末端文件夹（无任何子文件夹的文件夹）"""
    root_path = Path(root_dir)
    # 检查根目录是否存在
    if not root_path.is_dir():
        print(f"错误：目录不存在 - {root_dir}")
        return

    # 存储所有末端文件夹路径
    end_subfolders = []

    # 递归匹配所有路径，过滤出文件夹（排除根目录本身）
    for path in root_path.rglob("*"):
        if path.is_dir() and path != root_path:
            # 检查当前文件夹是否有子文件夹
            # 遍历当前文件夹下的所有条目，判断是否存在子文件夹
            has_subfolders = any(p.is_dir() for p in path.iterdir())

            # 如果没有子文件夹，则是末端文件夹，加入列表
            if not has_subfolders:
                end_subfolders.append(str(path))
    if len(end_subfolders) == 0:
        end_subfolders.append(str(root_dir))

    return end_subfolders


args = load_setting(json_path="./sample/gen_settings.json")
# motion_root_path = "./web_fbx/motion_data/origin_v2m/origin"
motion_root_path = args.data_source_path

use_vel = True
model_name = args.model_name
restore_iter = args.restore_iter
early_stop = None
if args.fast_sampling:
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
else:
    custom_timesteps = None

denoise_steps = [10]
save_GT = False

from animo.simulator import FlatGroundSimulator
from animo.skeleton.smpl_body import AnimoSMPLBody

phys_refine = False

refine_modes = ["denoise", "gen-from-rot-kpts", "gen-from-kpts"]
refine_modes = ["gen-from-denoise-rot-kpts"]
refine_modes = ["denoise", "gen-from-rot-kpts", "gen-from-denoise-rot-kpts"]


def main():
    # 配置路径（请根据实际情况修改这两个路径）
    out_root = "web2/data"
    # source_root = "/apdcephfs_cq10/share_1467498/datasets/motion_gen_arena/coverage_test/single_actions"
    # dataset_name = "motion_gen_arena"

    # source_root = "/apdcephfs_cq10/share_1467498/datasets/motion_gen_arena/evaluation_20251125/demo_20251217"
    # dataset_name = "1222动作修复测试"

    source_root = "/apdcephfs_cq10/share_1467498/home/chengxuzuo/projects/amass_process/degrade_motion_test"
    dataset_name = "0120动作修复测试"

    # source_root = "/apdcephfs_cq10/share_1467498/datasets/motion_data/heping/retarget_to_smpl/part1-2561-npz/"
    # dataset_name = "HePing位移重建测试"

    # out_root = "web2/data"
    # source_root = "hunyuan_motion_data/bad_motions"
    # dataset_name = "Academic动作修复测试"
    # 源目录（包含子文件夹的根目录）

    # 加载模型
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    motion_rep = OccamMotionRep(keep_hand=False, global_pose=args.global_pose)
    mp = Npz2VecProcessor(keep_hand=False)

    # model = MoGenDiT_V2(
    #     d_motion=291,
    #     d_model=512,
    #     d_cond=2 * 24 * 3,
    #     n_head=8,
    #     n_stack=8,
    #     dropout=0,
    #     window_size=args.mask_window,
    # ).to(device)

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

    # 检查源目录是否存在
    if not Path(source_root).exists():
        print(f"错误：源目录不存在 - {source_root}")
        return

    # 递归查找所有.npz文件
    folders = traverse_subfolders(source_root)

    out_put_path_gt = os.path.join(out_root, "Input", dataset_name)
    os.makedirs(out_put_path_gt, exist_ok=True)

    for sub_folder in folders:
        sub_folder_name = sub_folder.split("/")[-1]
        print(f"Processing folder: {sub_folder}")

        data_dict = NpzMotion.load_data(sub_folder, min_len=1, motion_rep=motion_rep)
        data = NpzMotion(
            data=data_dict, motion_rep=motion_rep, fix_len=600, use_vel=use_vel
        )
        file_names = data_dict["file_name"]
        for refine_mode in refine_modes:
            if phys_refine:
                out_put_path = os.path.join(
                    out_root,
                    f"{refine_mode}_phys",
                    dataset_name,
                )
            else:
                out_put_path = os.path.join(out_root, f"{refine_mode}", dataset_name)
            os.makedirs(out_put_path, exist_ok=True)
            for i, fname in enumerate(file_names):
                print(f"processing {fname} [{i+1}/{len(file_names)}]")
                fname = f"{sub_folder_name}_{fname}"
                npz_path = os.path.join(out_put_path, fname)
                # if Path(npz_path).exists():
                #     print(f"File already exists, skipping: {npz_path}")
                #     continue

                motion, length, cond = data.__getitem__(index=i)
                beta = data_dict["beta"][i]
                length = torch.tensor([length])
                motion = motion[:length]

                motion = motion_rep.normalization(motion.to(device))
                # pdb.set_trace()
                origin_motion = motion.unsqueeze(0).to(device)
                cond = cond.unsqueeze(0).to(device)
                gen_motion = windowed_refine(
                    origin_motion,
                    cond,
                    model,
                    diffusion,
                    motion_rep,
                    step=10,
                    mode=refine_mode,
                )

                # mp = Npz2VecProcessor(keep_hand=True)

                pose, joint, trans = motion_rep.decode(gen_motion[0])

                if phys_refine:
                    # 使用物理模拟器进行运动修正
                    body_model = AnimoSMPLBody()
                    body_model.set_joint_offset(pose[0], joint[0])
                    simulator = FlatGroundSimulator(skeleton=body_model, fps=30)
                    pose, trans, contact_flags = simulator.simulate(pose=pose, vel=vel)

                # fig, ani = create_animation(np.array(joint.cpu()), stationary=None)
                # body_model = AnimoSMPLBody()

                out_npz = mp.motion2npz_dict(
                    pose=pose, trans=trans, frame_rate=30, betas=beta
                )

                print("Saving SMPL params to [{}]".format(npz_path))
                np.savez(npz_path, **out_npz)

                if save_GT:
                    npz_path_gt = os.path.join(out_put_path_gt, fname)
                    os.makedirs(os.path.dirname(npz_path_gt), exist_ok=True)
                    gt_pose, gt_joint, gt_trans = motion_rep.decode(
                        origin_motion[0].cpu()
                    )

                    out_npz = mp.motion2npz_dict(
                        pose=gt_pose, trans=gt_trans, frame_rate=30, betas=beta
                    )

                    print("Saving SMPL params to [{}]".format(npz_path_gt))
                    np.savez(npz_path_gt, **out_npz)


if __name__ == "__main__":
    main()
