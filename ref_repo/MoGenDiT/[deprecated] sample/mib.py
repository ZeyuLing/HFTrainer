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
from sample.load_gen_args import load_setting
from model.more_diff import get_MoreDiff_model

args = load_setting(json_path="./sample/gen_settings.json")
motion_root_path = args.data_source_path

out_put_path = f"./web2/motion_data/MIB/{args.caption}"
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


def main():
    # training settings
    args = load_setting(json_path="./sample/gen_settings.json")
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
    data_dict = NpzMotion.load_data(dataset_path, motion_rep=motion_rep, min_len=20)
    data = NpzMotion(
        data=data_dict, motion_rep=motion_rep, fix_len=300, use_vel=use_vel
    )
    file_names = data_dict["file_name"]
    for i, fname in enumerate(file_names):
        print(f"processing {fname} [{i+1}/{len(file_names)}]")

        motion, length, cond = data.__getitem__(index=i)
        beta = data_dict["beta"][i]
        length = torch.tensor([length])
        motion = motion[:length]

        pose_6d, joint, _ = motion_rep.decode(motion)
        loop_k = length // 5

        motion_template = torch.zeros_like(motion).to(device)

        mask_scheduler = MotionMaskScheduler()

        motion_template = motion_template.unsqueeze(0).to(device)
        cond = cond.unsqueeze(0).to(device)

        # bool_length_mask = self.mask_scheduler.get_length_mask_bool(motion=motion, length=length)

        mask = mask_scheduler.get_temporal_mask(
            motion=motion_template,
            length=length,
            mode="fix_start_end",
            loop_k=loop_k,
        )
        mask_bool = mask == 1
        motion = motion.view_as(motion_template).to(device)
        motion_template[mask_bool] = motion[mask_bool]

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

        pose, joint, trans = motion_rep.decode(gen_motion[0])

        out_npz = mp.motion2npz_dict(pose=pose, trans=trans, frame_rate=30, betas=beta)
        npz_path = os.path.join(out_put_path, fname)
        print("Saving SMPL params to [{}]".format(npz_path))
        np.savez(npz_path, **out_npz)

        # motion = motion[0]
        # motion = motion_rep.normalization(motion)
        # pose_6d, joint, t = motion_rep.decode(motion[:length])
        # trans = joint[:, 0]
        # # pdb.set_trace()

        # out_npz = mp.motion2npz_dict(
        #     pose=pose_6d, trans=trans, frame_rate=30, betas=beta
        # )
        # npz_path = os.path.join(motion_root_path, fname)
        # np.savez(npz_path, **out_npz)


if __name__ == "__main__":
    main()
