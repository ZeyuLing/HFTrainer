import torch
from EasyDiffusion import GaussianDiffusion, BetaSchedule, ModelMeanType
from trainer.train_platforms import TensorboardPlatform
from trainer.my_trainer import *
from trainer import load_args
from model.my_model import *
from model.more_diff import get_MoreDiff_model
from trainer.data_loader import AdHocMotionData
from motion_process.motion_representation import OccamMotionRep
import os
import shutil
from utils.fixseed import fixseed

# torchrun --nproc_per_node=4 --master_port=29500 -m train.train_multi_GPUs


def main():
    # training settings
    train_args_path = "./train/train_args_0.json"
    args = load_args(json_path=train_args_path)
    
    # 在训练开始时转存args配置文件
    # 创建目标目录：args.save_dir/args.model_name
    config_save_dir = os.path.join(args.save_dir, args.model_name)
    os.makedirs(config_save_dir, exist_ok=True)
    
    # 复制args_default.json文件到目标目录
    source_json_path = train_args_path
    target_json_path = os.path.join(config_save_dir, "train_args.json")
    shutil.copy2(source_json_path, target_json_path)
    
    print(f"[INFO] Training configuration saved to: {target_json_path}")
    
    # dataset_path = 'F:\Dataset\processed_AMASS'
    # dataset_path = "/apdcephfs_cq10/share_1467498/home/chengxuzuo/projects/MoGenDIT/processed_amass"
    pose_type = "global" if args.global_pose else "local"

    dataset_paths = {
        "amass": f"/apdcephfs_cq10/share_1467498/home/chengxuzuo/projects/MoreDiff_Data/data/amass_hq2",
        "academic": f"/apdcephfs_cq10/share_1467498/home/chengxuzuo/projects/MoreDiff_Data/data/academic_hq2",
    }

    n_GPU = int(os.environ.get("WORLD_SIZE", 1))  # 默认为1（单进程）
    rank = int(os.environ.get("LOCAL_RANK", 0))

    fixseed(args.seed, offset=rank)

    args.save_interval = args.save_interval // n_GPU

    # 可用的mask patterns
    # "random_frame", "random_phrase", "random_start_end",
    # "block_pose", "block_trans_and_vel", "joint_only"

    keyframe_modes = {
        "random_frame": 0.2,
        "random_phrase": 0.2,
        "random_start_end": 0.2,
        "block_trans": 0.1,
        "joint_only": 0.1,
        "uncond": 0.2,
    }

    # # restoration
    # keyframe_modes = {
    #     "uncond": 1.0,
    # }

    # # trans inpanting
    # keyframe_modes = {
    #     "block_trans": 1.0,
    # }

    # 训练所需实例
    if rank == 0:
        train_platform = TensorboardPlatform(
            save_dir=os.path.join(args.log_dir, args.model_name)
        )
    else:
        train_platform = None

    diffusion = GaussianDiffusion(
        num_timesteps=1000,
        beta_schedule=BetaSchedule.COSINE,
        model_mean_type=ModelMeanType.START_X,
        # noise_remap_mode="sphere_norm",
    )
    data_dict = None
    print(args.train_data)
    for dataset_name in args.train_data:
        print(f"Loading dataset: {dataset_name}")
        if data_dict is None:
            data_dict = AdHocMotionData.load_data(
                dataset_paths[dataset_name.lower()],
                min_len=30,
            )
        else:
            new_data_dict = AdHocMotionData.load_data(
                dataset_paths[dataset_name.lower()],
                min_len=30,
            )
            data_dict = AdHocMotionData.merge(data_dict, new_data_dict)

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    motion_rep = OccamMotionRep(keep_hand=False, global_pose=args.global_pose)
    data = AdHocMotionData(
        data=data_dict,
        motion_rep=motion_rep,
        fix_len=224,
    )
    model = get_MoreDiff_model(
        data_dim=motion_rep.data_dim, version=args.model_version
    ).to(device)

    trainer = MoGenDitDistributedTrainer(
        args=args,
        train_platform=train_platform,
        model=model,
        diffusion=diffusion,
        data=data,
        motion_rep=motion_rep,
    )

    # trainer.restore(folder_path=args.save_dir, iter=430000, model_name=args.model_name)

    # args.model_name += "_l1_sft"

    trainer.train(iters=4000000 // n_GPU, keyframe_modes=keyframe_modes)

    if train_platform is not None:
        train_platform.close()


if __name__ == "__main__":
    main()
