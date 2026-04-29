import torch
import numpy as np
from EasyDiffusion import GaussianDiffusion, BetaSchedule, ModelMeanType
from articulate.math import axis_angle_to_rotation_matrix
from my_data.amass import NpzMotion
from trainer import load_args
from model.my_model import *
from motion_process.motion_representation import *
import os
from motion_process.processors import Npz2VecProcessor
from trainer.mask_scheduler import MotionMaskScheduler
from sample.load_gen_args import load_setting
from model.more_diff import get_MoreDiff_model


def auto_loop_interp_args(motion_joint22, fps=30):
    ani_frames = motion_joint22.shape[0]  # [n, 22, 3]
    trans = motion_joint22[:, [0]]
    hands_move = motion_joint22[:, [20, 21]] - trans
    loop_k = min(30, ani_frames // 5)  # 取动画长度的1/5
    trans_A = motion_joint22[0, 0, [0, 2]]
    trans_B = motion_joint22[-1, 0, [0, 2]]
    distance_gap = np.linalg.norm(trans_A - trans_B)  # 计算起始和结束位置的距离
    n_interpolation_return = int(
        fps * distance_gap / 0.65
    )  # 允许平均0.8m/s的速度回到起始点

    avg_hands_move_speed = np.linalg.norm(
        hands_move[1:] - hands_move[:-1], axis=-1
    ).mean()
    hands_return_distance = np.linalg.norm(
        hands_move[-1] - hands_move[0], axis=-1
    ).mean()
    n_interpolation_motion = int(hands_return_distance / avg_hands_move_speed) * 1.5
    n_interpolation_motion = int(n_interpolation_motion)

    n_interpolation = n_interpolation_motion + n_interpolation_return
    # n_interpolation = n_interpolation_motion + n_interpolation_return
    n_interpolation = min(
        n_interpolation, 300 - loop_k * 2
    )  # 限制总输入长度不超过300帧
    return loop_k, max(n_interpolation, 30)


args = load_setting(json_path="./sample/gen_settings.json")

# motion_root_path = "./web_fbx/motion_data/origin_v2m/origin"
motion_root_path = args.data_source_path
out_put_path = f"./web2/motion_data/loop/{args.caption}"
use_vel = True
model_name = args.model_name
restore_iter = args.restore_iter
early_stop = None
if args.fast_sampling:
    custom_timesteps = [
        999,
        800,
        700,
        600,
        500,
        400,
        300,
        200,
        100,
        50,
        25,
        10,
        8,
        6,
        4,
        2,
        1,
        0,
    ]
else:
    custom_timesteps = None


os.makedirs(out_put_path, exist_ok=True)


def main():
    # training settings
    args = load_setting(json_path="./sample/gen_settings.json")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset_path = motion_root_path
    motion_rep = OccamMotionRep(keep_hand=False, global_pose=args.global_pose)
    mp = Npz2VecProcessor(keep_hand=False)

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
    data_dict = NpzMotion.load_data(dataset_path, motion_rep=motion_rep, min_len=20)
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

        pose, joint, trans = motion_rep.decode(motion)
        loop_k, n_interp = auto_loop_interp_args(joint + trans.unsqueeze(1))
        motion = motion_rep.normalization(
            motion, ref_idx=-loop_k
        )  # 以motion B起始帧作为normalization参考

        motion_A = motion[:loop_k]
        motion_B = motion[-loop_k:]
        motion_template = torch.zeros(n_interp + 2 * loop_k, motion_rep.data_dim).to(
            device
        )
        motion_template[:loop_k] = motion_B
        motion_template[-loop_k:] = motion_A

        mask_scheduler = MotionMaskScheduler()

        motion_template = motion_template.unsqueeze(0).to(device)
        cond = cond.unsqueeze(0).to(device)
        # bool_length_mask = self.mask_scheduler.get_length_mask_bool(motion=motion, length=length)

        mask = mask_scheduler.get_temporal_mask(
            motion=motion_template, length=length, mode="fix_start_end", loop_k=loop_k
        )
        mask_bool = mask_scheduler.get_temporal_mask(
            motion=motion_template,
            length=length,
            mode="fix_start_end",
            loop_k=loop_k,
            dtype="bool",
        )

        x_wrap = model.wrap_inputs(motion_template, cond, mask, None)

        with torch.no_grad():
            gen_motion = diffusion.ddim_sample_loop(
                x_wrap=x_wrap,
                model=model,
                eta=1,
                mask=mask_bool,
                early_stop=early_stop,
                custom_timesteps=custom_timesteps,
            )

        interp_motion = gen_motion[0, loop_k:-loop_k].cpu()
        loop_motion = torch.cat([motion, interp_motion], dim=0)
        loop_motion = motion_rep.normalization(loop_motion)

        pose, joint, trans = motion_rep.decode(loop_motion)
        # from sample.gen_test import create_animation
        # create_animation(data=np.array(joint.cpu()))
        # trans = joint[:, 0]
        # pdb.set_trace()

        out_npz = mp.motion2npz_dict(pose=pose, trans=trans, frame_rate=30, betas=beta)
        npz_path = os.path.join(out_put_path, fname)
        print("Saving SMPL params to [{}]".format(npz_path))
        np.savez(npz_path, **out_npz)


if __name__ == "__main__":
    main()
