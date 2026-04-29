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

args = load_setting(json_path="./sample/gen_settings.json")
motion_root_path = args.data_source_path

out_put_path = f"./web2/motion_data/stitch/{args.caption}"
out_put_path_refine = f"./web2/motion_data/stitch_post_refine/{args.caption}"
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
denoise_steps = args.denoise_steps
os.makedirs(out_put_path, exist_ok=True)
os.makedirs(out_put_path_refine, exist_ok=True)

from .refine import windowed_refine


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset_path = motion_root_path
    args = load_setting(json_path="./sample/gen_settings.json")
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
        # 读取2段动作数据
        print(f"processing {fname} [{i+1}/{len(file_names)}]")
        # if fname.find("150_originalframes_000000_000320") == -1:
        #     continue

        motion_1, length_1, cond_1 = data.__getitem__(index=i)
        motion_2, length_2, cond_2 = data.__getitem__(index=(i + 1) % len(file_names))

        motion_1 = motion_1.to(device)
        motion_2 = motion_2.to(device)

        beta_1 = data_dict["beta"][i]
        beta_2 = data_dict["beta"][(i + 1) % len(file_names)]

        motion_1 = motion_1[:length_1]
        motion_2 = motion_2[:length_2]

        length_1 = torch.tensor([length_1])
        length_2 = torch.tensor([length_2])

        mask_scheduler = MotionMaskScheduler()

        cond = cond_1.unsqueeze(0).to(device)

        loop_k = min(30, len(motion_1), len(motion_2))
        n_interp = 20
        template_length = loop_k * 2 + n_interp

        # 朝向位移骨架对齐
        motion_2 = motion_rep.pre_stitch(
            motion=motion_2,
            ref_motion=motion_1[[-1]],
            reset_height=True,
            sync_skeleton=True,
        )

        # 构造生成模板
        # pdb.set_trace()
        motion_template = torch.zeros(template_length, motion_1.shape[1]).to(device)
        motion_template[:loop_k] += motion_1[-loop_k:]
        motion_template[-loop_k:] += motion_2[:loop_k]
        # 输入模型之前进行归一化
        motion_template = motion_rep.normalization(motion_template.to(device))
        # pdb.set_trace()
        motion_template = motion_template.unsqueeze(0)

        mask = mask_scheduler.get_temporal_mask(
            motion=motion_template,
            length=torch.tensor(template_length),
            mode="fix_start_end",
            loop_k=loop_k,
        )
        mask_bool = mask == 1

        x_wrap = model.wrap_inputs(motion_template, cond, mask, None)
        # pdb.set_trace()
        with torch.no_grad():
            gen_motion = diffusion.ddim_sample_loop(
                x_wrap=x_wrap,
                model=model,
                eta=1,
                mask=mask_bool,
                early_stop=early_stop,
                custom_timesteps=custom_timesteps,
            )
        gen_motion = gen_motion[0]

        motion_A2B = motion_rep.pre_stitch(
            gen_motion[loop_k:-loop_k], ref_motion=motion_1[[-1]]
        )
        motion_A_stitch_B = torch.cat(
            [motion_1, motion_A2B, motion_2],
            dim=0,
        )
        pose, joint, trans = motion_rep.decode(motion_A_stitch_B)
        # from sample.gen_test import create_animation
        # create_animation(data=np.array(joint.cpu()))
        # trans = joint[:, 0]

        out_npz = mp.motion2npz_dict(
            pose=pose, trans=trans, frame_rate=30, betas=beta_1
        )
        npz_path = os.path.join(out_put_path, fname)
        print("Saving SMPL params to [{}]".format(npz_path))
        np.savez(npz_path, **out_npz)

        # 后refine处理
        motion_A_stitch_B = windowed_refine(
            motion=motion_A_stitch_B.unsqueeze(0),
            cond=cond,
            model=model,
            diffusion=diffusion,
            motion_rep=motion_rep,
        )[0]
        pose, joint, trans = motion_rep.decode(motion_A_stitch_B)

        out_npz = mp.motion2npz_dict(
            pose=pose, trans=trans, frame_rate=30, betas=beta_1
        )
        npz_path = os.path.join(out_put_path_refine, fname)
        print("Saving SMPL params to [{}]".format(npz_path))
        np.savez(npz_path, **out_npz)


if __name__ == "__main__":
    main()
