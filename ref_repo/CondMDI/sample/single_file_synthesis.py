# This code is based on https://github.com/openai/guided-diffusion
"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
from utils.fixseed import fixseed
import os
import numpy as np
import torch
from utils.parser_util import cond_synt_args
from utils.model_util import create_model_and_diffusion, load_saved_model
from utils import dist_util
from model.cfg_sampler import ClassifierFreeSampleModel
from data_loaders.get_data import get_dataset_loader, DatasetConfig
from data_loaders.humanml.scripts.motion_process import recover_from_ric
from data_loaders import humanml_utils
import data_loaders.humanml.utils.paramUtil as paramUtil
from data_loaders.humanml.utils.plot_script import plot_3d_motion
import shutil
from pathlib import Path
from utils.editing_util import get_keyframes_mask, load_fixed_dataset
from data_loaders.humanml.utils.plotting import plot_conditional_samples
import json

def get_std_mean():
    std = np.load('./dataset/HumanML3D/Std_abs_3d.npy')
    mean = np.load('./dataset/HumanML3D/Mean_abs_3d.npy')
    return std, mean


def load_motion_and_mask(motion_file, mask_file, max_frames=196):
    """
    Load motion data from a file and build obs_mask based on the mask file.
    Ensures that:
        - input_motions has shape (1, 263, 1, 196)
        - obs_mask has shape (1, 263, 1, 196) and is boolean
        - input_masks has shape (1, 1, 1, 196) and is boolean
        - obs_joint_mask has shape (1, 22, 1, 196) with frames matching obs_mask.
    Any frames beyond 196 are truncated, and missing frames are zero-padded.
    """
    # Load motion data, assuming .npy format with shape (nframes, njoints)
    input_motions = np.load(motion_file)

    # Z Normalization
    std, mean = get_std_mean()  # Retrieve the std and mean from the dataset
    input_motions = (input_motions - mean) / std  # Apply Z normalization
    
    # Get nframes and njoints
    nframes, njoints = input_motions.shape
    
    # Ensure input_motions has max_frames length
    if nframes > max_frames:
        input_motions = input_motions[:max_frames]
    else:
        padding = np.zeros((max_frames - nframes, njoints))
        input_motions = np.vstack([input_motions, padding])
    
    # Reshape input_motions to (1, 263, 1, 196)
    input_motions = input_motions.T  # Transpose to (njoints, nframes)
    input_motions = np.expand_dims(input_motions, axis=(0, 2))  # Reshape to (1, 263, 1, 196)
    
    # Convert to PyTorch tensor
    input_motions = torch.tensor(input_motions, dtype=torch.float32)
    
    # Generate input_masks: shape (1, 1, 1, max_frames) and boolean type
    input_masks = torch.zeros((1, 1, 1, max_frames), dtype=torch.bool)
    input_masks[0, 0, 0, :min(nframes, max_frames)] = True  # Set first nframes positions to True
    
    # Set input_length as the actual number of frames (up to max_frames)
    input_length = torch.tensor([min(nframes, max_frames)], dtype=torch.int64)
    
    # Generate obs_mask: shape (1, 263, 1, max_frames) and boolean type
    obs_mask = torch.zeros((1, njoints, 1, max_frames), dtype=torch.bool)
    
    # Load mask file and set the corresponding frames in obs_mask to True
    with open(mask_file, 'r') as f:
        frame_indices = f.readlines()
    
    for frame_idx in frame_indices:
        frame_idx = int(frame_idx.strip())  # Remove any whitespace and convert to int
        if 0 <= frame_idx < max_frames:
            obs_mask[0, :, 0, frame_idx] = True  # Set all joints for this frame to True

    # Generate obs_joint_mask: shape (1, 22, 1, max_frames), same True frames as obs_mask
    obs_joint_mask = torch.zeros((1, 22, 1, max_frames), dtype=torch.bool)
    obs_joint_mask[0, :, 0, :] = obs_mask[0, :22, 0, :]  # Use the first 22 joints from obs_mask

    return input_motions, input_masks, input_length, obs_mask, obs_joint_mask

def detect_foot_contacts(positions, fid_l, fid_r, velfactor=0.02):
    velocities = positions[:, 1:, :, :] - positions[:, :-1, :, :]  # (batch_size, n_frames-1, n_joints, 3)

    # 计算左脚关节的速度平方和
    feet_l_v = velocities[:, :, fid_l, :] ** 2  # (batch_size, n_frames-1, len(fid_l), 3)
    feet_l_v = feet_l_v.sum(dim=-1).mean(dim=-1)  # (batch_size, n_frames-1)

    # 计算右脚关节的速度平方和
    feet_r_v = velocities[:, :, fid_r, :] ** 2
    feet_r_v = feet_r_v.sum(dim=-1).mean(dim=-1)

    # 判断接触
    feet_l_contact = (feet_l_v < velfactor)  # (batch_size, n_frames-1)
    feet_r_contact = (feet_r_v < velfactor)

    # 补齐帧数
    pad = torch.zeros((feet_l_contact.shape[0], 1), dtype=torch.bool, device=positions.device)
    feet_l_contact = torch.cat([pad, feet_l_contact], dim=1)  # (batch_size, n_frames)
    feet_r_contact = torch.cat([pad, feet_r_contact], dim=1)

    return feet_l_contact, feet_r_contact

def detect_foot_contacts_simple(positions, fid_heels, fid_toes, velfactor=0.02):
    velocities = positions[:, 1:, :, :] - positions[:, :-1, :, :]  # (batch_size, n_frames - 1, n_joints, 3)

    foot_contacts = {}
    for foot, fid in zip(['left_heel', 'right_heel', 'left_toe', 'right_toe'], fid_heels + fid_toes):
        foot_v = velocities[:, :, fid, :] ** 2  # (batch_size, n_frames - 1, 3)
        foot_v = foot_v.sum(dim=-1)  # (batch_size, n_frames - 1)
        foot_contact = (foot_v < velfactor)  # (batch_size, n_frames - 1)
        # 补齐帧数
        foot_contact = torch.cat([foot_contact.new_zeros((foot_contact.shape[0], 1), dtype=torch.bool), foot_contact], dim=1)
        foot_contacts[foot] = foot_contact

    return foot_contacts

def select_reference_foot(foot_contacts, fid_heels, fid_toes):
    left_contact = foot_contacts['left_heel'] & foot_contacts['left_toe']
    right_contact = foot_contacts['right_heel'] & foot_contacts['right_toe']

    both_feet_contact = left_contact & right_contact

    ref_foot = None
    contact_frames = None
    foot_indices = None

    if both_feet_contact.any():
        ref_foot = 'left'
        contact_frames = left_contact
        foot_indices = [fid_heels[0], fid_toes[0]]
    elif left_contact.any():
        ref_foot = 'left'
        contact_frames = left_contact
        foot_indices = [fid_heels[0], fid_toes[0]]
    elif right_contact.any():
        ref_foot = 'right'
        contact_frames = right_contact
        foot_indices = [fid_heels[1], fid_toes[1]]

    return ref_foot, contact_frames, foot_indices

def adjust_root_positions(positions, original_positions, contact_frames, foot_indices):
    """
    在接触期间调整根部关节的位置，使参考脚保持静止。
    在非接触期间，保持根部的相对位移与原始数据一致。

    参数：
    - positions: 调整后的关节位置，形状为 (n_frames, n_joints, 3)
    - original_positions: 原始的关节位置，未调整过，形状相同
    - contact_frames: 布尔张量，形状为 (n_frames)，表示参考脚接触的帧
    - foot_indices: 参考脚的关节索引列表

    返回：
    - positions: 调整后的关节位置
    """
    n_frames = positions.shape[0]

    for t in range(1, n_frames):
        if contact_frames[t]:
            # 接触帧，调整根部位置使脚部保持静止

            # 计算脚部在当前帧和上一帧的位置差
            foot_displacement = positions[t, foot_indices, :] - positions[t - 1, foot_indices, :]
            foot_displacement = foot_displacement.mean(dim=0)  # 平均多个关节的位移

            # 将位移的相反值添加到根部和其他关节的位置上
            positions[t, 0, :] -= foot_displacement  # 调整根部位置

            # 调整其他关节的位置（不包括根部和参考脚）
            other_indices = [i for i in range(positions.shape[1]) if i != 0 and i not in foot_indices]
            positions[t, other_indices, :] -= foot_displacement
            # 参考脚的位置保持不变

        else:
            # 非接触帧，保持根部的相对位移与原始数据一致

            # 计算原始数据中根部的相对位移
            root_displacement = original_positions[t, 0, :] - original_positions[t - 1, 0, :]

            # 更新调整后的根部位置
            positions[t, 0, :] = positions[t - 1, 0, :] + root_displacement

            # 计算其他关节相对于根部的位置
            relative_positions = original_positions[t, :, :] - original_positions[t, 0, :].unsqueeze(0)

            # 更新其他关节的位置，使其相对于新的根部位置保持一致
            positions[t, :, :] = positions[t, 0, :].unsqueeze(0) + relative_positions

    return positions


def fabrik_multi_constraints(positions, bones, fixed_joint_indices, target_positions, max_iterations=10, tolerance=1e-3):
    n_joints = positions.shape[0]
    bones_length = torch.norm(positions - positions[bones], dim=1)  # (n_joints)
    bones_length[bones == -1] = 0  # 根节点没有父节点

    fixed_positions = dict(zip(fixed_joint_indices, target_positions))

    for iteration in range(max_iterations):
        prev_positions = positions.clone()

        # 后向阶段
        for i in reversed(range(n_joints)):
            if i in fixed_positions:
                positions[i] = fixed_positions[i]
            else:
                child_indices = (bones == i).nonzero(as_tuple=False).squeeze()
                if child_indices.numel() > 0:
                    for child in child_indices:
                        r = torch.norm(positions[i] - positions[child])
                        l = bones_length[child]
                        if r > 1e-8:
                            positions[i] = positions[child] + (positions[i] - positions[child]) * (l / r)

        # 前向阶段
        for i in range(n_joints):
            parent = bones[i]
            if parent != -1:
                r = torch.norm(positions[i] - positions[parent])
                l = bones_length[i]
                if r > 1e-8:
                    positions[i] = positions[parent] + (positions[i] - positions[parent]) * (l / r)
            elif i in fixed_positions:
                positions[i] = fixed_positions[i]  # 根节点固定

        # 检查收敛
        errors = [torch.norm(positions[idx] - fixed_positions[idx]) for idx in fixed_joint_indices]
        mean_error = torch.mean(torch.stack(errors))
        if mean_error < tolerance:
            break

    return positions

def apply_foot_contact_constraints(sample, method='ik', bones=None, contact_info=None, fid_l=[7, 10], fid_r=[8, 11], velfactor=0.02):
    """
    应用脚部接触约束，提供两种方法：'ik'（基于逆运动学）和 'root'（调整根部位置）。

    参数：
    - sample: 关节位置序列，形状为 (batch_size, n_joints, 3, n_frames)
    - method: 字符串，'ik' 或 'root'，选择使用的方法
    - bones: 父关节索引列表，长度为 n_joints，根节点的父索引为 -1
        - 当 method='ik' 时需要提供
    - contact_info: 可选，字典，包含左脚和右脚的接触信息
        - 'left': 左脚接触信息，形状为 (batch_size, n_frames)，布尔类型
        - 'right': 右脚接触信息，形状为 (batch_size, n_frames)，布尔类型
        - 如果未提供，将自动计算
    - fid_l: 左脚关节索引列表
    - fid_r: 右脚关节索引列表
    - velfactor: 速度阈值

    返回：
    - sample: 调整后的关节位置序列，形状与输入相同
    """
    positions = sample.permute(0, 3, 1, 2).clone()
    original_positions = positions.clone()  # 保存原始的关节位置
    batch_size, n_frames, n_joints, _ = positions.shape

    if contact_info is None:
        feet_l_contact, feet_r_contact = detect_foot_contacts(positions, fid_l, fid_r, velfactor)
        contact_info = {'left': feet_l_contact, 'right': feet_r_contact}
    else:
        feet_l_contact = contact_info['left']
        feet_r_contact = contact_info['right']

    if method == 'ik':
        # IK 方法的处理逻辑（保持不变）
        ...

    elif method == 'root':
        fid_heels = [fid_l[0], fid_r[0]]  # 脚跟关节索引
        fid_toes = [fid_l[1], fid_r[1]]   # 脚趾关节索引

        for b in range(batch_size):
            positions_b = positions[b]  # (n_frames, n_joints, 3)
            original_positions_b = original_positions[b]

            # 检测脚部接触信息
            foot_contacts = detect_foot_contacts_simple(positions_b.unsqueeze(0), fid_heels, fid_toes, velfactor)
            # 选择参考脚
            ref_foot, contact_frames, foot_indices = select_reference_foot(foot_contacts, fid_heels, fid_toes)

            if ref_foot is not None:
                # 调整根部位置，传入原始的关节位置
                positions_b = adjust_root_positions(positions_b, original_positions_b, contact_frames[0], foot_indices)
                positions[b] = positions_b
            else:
                # 没有检测到接触的脚，保持原始的位移
                positions[b] = original_positions_b  # 或者无需修改

    else:
        raise ValueError("Invalid method. Choose 'ik' or 'root'.")

    sample = positions.permute(0, 2, 3, 1)  # (batch_size, n_joints, 3, n_frames)
    return sample

def extract_contact_info_from_sample(sample, n_frames):
    feet_contact = sample[:, -4:, 0, :]  # (batch_size, 4, n_frames)
    print(sample.shape)
    print(feet_contact)
    feet_contact = feet_contact > 0.5
    feet_l_contact = feet_contact[:, [0, 1], :].all(dim=1)  # 左脚，(batch_size, n_frames)
    feet_r_contact = feet_contact[:, [2, 3], :].all(dim=1)  # 右脚，(batch_size, n_frames)
    contact_info = {
        'left': feet_l_contact,
        'right': feet_r_contact
    }
    return contact_info

def main():
    args = cond_synt_args()
    fixseed(args.seed)

    sample_file = 'generated_samples.npy'
    samples_loaded = False

    assert args.dataset == 'humanml' and args.abs_3d # Only humanml dataset and the absolute root representation is supported for conditional synthesis
    assert args.keyframe_conditioned

    out_path = args.output_dir
    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')
    max_frames = 196 if args.dataset in ['kit', 'humanml'] else (200 if args.dataset == 'trajectories' else 60)
    fps = 12.5 if args.dataset == 'kit' else 20
    dist_util.setup_dist(args.device)
    if out_path == '':
        checkpoint_name = os.path.split(os.path.dirname(args.model_path))[-1]
        model_results_path = os.path.join('save/results', checkpoint_name)
        method = ''
        if args.imputate:
            method += '_' + 'imputation'
        if args.reconstruction_guidance:
            method += '_' + 'recg'

        if args.editable_features != 'pos_rot_vel':
            edit_mode = args.edit_mode + '_' + args.editable_features
        else:
            edit_mode = args.edit_mode
        out_path = os.path.join(model_results_path,
                                '{}_condsamples{}_{}_{}_T={}_CI={}_CRG={}_KGP={}_seed{}'.format(os.path.splitext(os.path.basename(args.motion_file))[0], niter, method,
                                                                                      edit_mode, args.transition_length,
                                                                                      args.stop_imputation_at, args.stop_recguidance_at,
                                                                                      args.keyframe_guidance_param, args.seed))
        if args.text_prompt != '':
            out_path += '_' + args.text_prompt.replace(' ', '_').replace('.', '')
        elif args.input_text != '':
            out_path += '_' + os.path.basename(args.input_text).replace('.txt', '').replace(' ', '_').replace('.', '')

    # this block must be called BEFORE the dataset is loaded
    use_test_set_prompts = False
    if args.text_prompt != '':
        texts = [args.text_prompt]
        args.num_samples = 1
    elif args.input_text != '':
        assert os.path.exists(args.input_text)
        with open(args.input_text, 'r') as fr:
            texts = fr.readlines()
        texts = [s.replace('\n', '') for s in texts]
        args.num_samples = len(texts)
    elif args.action_name:
        action_text = [args.action_name]
        args.num_samples = 1
    elif args.action_file != '':
        assert os.path.exists(args.action_file)
        with open(args.action_file, 'r') as fr:
            action_text = fr.readlines()
        action_text = [s.replace('\n', '') for s in action_text]
        args.num_samples = len(action_text)
    elif args.no_text:
        texts = [''] * args.num_samples
        args.guidance_param = 0.  # Force unconditioned generation # TODO: This is part of inbetween.py --> Will I need it here?
    else:
        # use text from the test set
        use_test_set_prompts = True

    print('Loading dataset...')
    assert args.num_samples <= args.batch_size, \
        f'Please either increase batch_size({args.batch_size}) or reduce num_samples({args.num_samples})'
    # So why do we need this check? In order to protect GPU from a memory overload in the following line.
    # If your GPU can handle batch size larger then default, you can specify it through --batch_size flag.
    # If it doesn't, and you still want to sample more prompts, run this script with different seeds
    # (specify through the --seed flag)
    args.batch_size = args.num_samples # Sampling a single batch from the testset, with exactly args.num_samples
    split = 'fixed_subset' if args.use_fixed_subset else 'test'
    data = load_dataset(args, max_frames, split=split)

    # data.fixed_length = n_frames
    total_num_samples = args.num_samples * args.num_repetitions

    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, data)

    ###################################
    # LOADING THE MODEL FROM CHECKPOINT
    print(f"Loading checkpoints from [{args.model_path}]...")
    load_saved_model(model, args.model_path) # , use_avg_model=args.gen_avg_model)
    if args.guidance_param != 1 and args.keyframe_guidance_param != 1:
        raise NotImplementedError('Classifier-free sampling for keyframes not implemented.')
    elif args.guidance_param != 1:
        model = ClassifierFreeSampleModel(model)  # wrapping model with the classifier-free sampler
    model.to(dist_util.dev())
    model.eval()  # disable random masking
    ###################################

    # Load motion and mask data
    input_motions, input_masks, input_length, obs_mask, obs_joint_mask = load_motion_and_mask(args.motion_file, args.mask_file)

    # Move data to the appropriate device (GPU or CPU)
    input_motions = input_motions.to(dist_util.dev())
    input_masks = input_masks.to(dist_util.dev())
    input_length = input_length.to(dist_util.dev())
    obs_mask = obs_mask.to(dist_util.dev())

    # Prepare model_kwargs with the loaded data
    model_kwargs = {
        'y': {
            'mask': input_masks,
            'lengths': input_length,
        }
    }

    # iterator = iter(data)

    # input_motions, model_kwargs = next(iterator)
    # print(input_motions.shape)
    # print(model_kwargs)

    if args.use_fixed_dataset: # TODO: this is for debugging - need a neater way to do this for the final version - num_samples should be 10
        assert args.dataset == 'humanml' and args.abs_3d
        input_motions, model_kwargs = load_fixed_dataset(args.num_samples)

    input_motions = input_motions.to(dist_util.dev()) # [nsamples, njoints=263/1, nfeats=1/3, nframes=196/200]
    input_masks = model_kwargs["y"]["mask"]  # [nsamples, 1, 1, nframes]
    # print(input_masks.shape)
    input_lengths = model_kwargs["y"]["lengths"]  # [nsamples]

    model_kwargs['obs_x0'] = input_motions
    model_kwargs['obs_mask'] = obs_mask
    # model_kwargs['obs_mask'], obs_joint_mask = get_keyframes_mask(data=input_motions, lengths=input_lengths, edit_mode=args.edit_mode,
    #                                                               feature_mode=args.editable_features, trans_length=args.transition_length,
    #                                                               get_joint_mask=True, n_keyframes=args.n_keyframes) # [nsamples, njoints, nfeats, nframes]
    # print(model_kwargs['obs_mask'])

    assert max_frames == input_motions.shape[-1]

    # Arguments
    model_kwargs['y']['text'] = texts if not use_test_set_prompts else model_kwargs['y']['text']
    model_kwargs['y']['diffusion_steps'] = args.diffusion_steps

    # Add inpainting mask according to args
    if args.zero_keyframe_loss: # if loss is 0 over keyframes durint training, then must impute keyframes during inference
        model_kwargs['y']['imputate'] = 1
        model_kwargs['y']['stop_imputation_at'] = 0
        model_kwargs['y']['replacement_distribution'] = 'conditional'
        model_kwargs['y']['inpainted_motion'] = model_kwargs['obs_x0']
        model_kwargs['y']['inpainting_mask'] = model_kwargs['obs_mask'] # used to do [nsamples, nframes] --> [nsamples, njoints, nfeats, nframes]
        model_kwargs['y']['reconstruction_guidance'] = False
    elif args.imputate: # if loss was present over keyframes during training, we may use imputation at inference time
        model_kwargs['y']['imputate'] = 1
        model_kwargs['y']['stop_imputation_at'] = args.stop_imputation_at
        model_kwargs['y']['replacement_distribution'] = 'conditional' # TODO: check if should also support marginal distribution
        model_kwargs['y']['inpainted_motion'] = model_kwargs['obs_x0']
        model_kwargs['y']['inpainting_mask'] = model_kwargs['obs_mask']
        if args.reconstruction_guidance: # if loss was present over keyframes during training, we may use guidance at inference time
            model_kwargs['y']['reconstruction_guidance'] = args.reconstruction_guidance
            model_kwargs['y']['reconstruction_weight'] = args.reconstruction_weight
            model_kwargs['y']['gradient_schedule'] = args.gradient_schedule
            model_kwargs['y']['stop_recguidance_at'] = args.stop_recguidance_at
    elif args.reconstruction_guidance: # if loss was present over keyframes during training, we may use guidance at inference time
        model_kwargs['y']['inpainted_motion'] = model_kwargs['obs_x0']
        model_kwargs['y']['inpainting_mask'] = model_kwargs['obs_mask']
        model_kwargs['y']['reconstruction_guidance'] = args.reconstruction_guidance
        model_kwargs['y']['reconstruction_weight'] = args.reconstruction_weight
        model_kwargs['y']['gradient_schedule'] = args.gradient_schedule
        model_kwargs['y']['stop_recguidance_at'] = args.stop_recguidance_at

    all_motions = []
    all_lengths = []
    all_text = []
    all_observed_motions = []
    all_observed_masks = []

    if os.path.exists(sample_file):
        # 加载样本
        sample_data = np.load(sample_file)
        sample = torch.from_numpy(sample_data).float().to(0)

        print(f"Samples have been loaded from '{sample_file}'")
        samples_loaded = True
    else:
        print(f"Sample file '{sample_file}' does not exist. Proceeding to generate samples.")

    for rep_i in range(args.num_repetitions):
        print(f'### Start sampling [repetitions #{rep_i}]')

        if not samples_loaded:
            # add CFG scale to batch
            if args.guidance_param != 1:
                # text classifier-free guidance
                model_kwargs['y']['text_scale'] = torch.ones(args.batch_size, device=dist_util.dev()) * args.guidance_param
            if args.keyframe_guidance_param != 1:
                # keyframe classifier-free guidance
                model_kwargs['y']['keyframe_scale'] = torch.ones(args.batch_size, device=dist_util.dev()) * args.keyframe_guidance_param

            sample_fn = diffusion.p_sample_loop

            sample = sample_fn(
                    model,
                    (args.batch_size, model.njoints, model.nfeats, max_frames),
                    clip_denoised=False,
                    model_kwargs=model_kwargs,
                    skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
                    init_image=None,
                    progress=True,
                    dump_steps=None,
                    noise=None,
                    const_noise=False,
                ) # [nsamples, njoints, nfeats, nframes]

            np.save(sample_file, sample.cpu().numpy())
            print(f"Samples have been saved to '{sample_file}'")

        # # Unnormalize samples and recover XYZ *positions*
        # if model.data_rep == 'hml_vec':
        #     n_joints = 22 if (sample.shape[1] in [263, 264]) else 21
        #     sample = sample.cpu().permute(0, 2, 3, 1)
        #     sample = data.dataset.t2m_dataset.inv_transform(sample).float()
        #     sample = recover_from_ric(sample, n_joints, abs_3d=args.abs_3d)
        #     sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1) # batch_size, n_joints=22, 3, n_frames

        # Unnormalize samples and recover XYZ positions
        if model.data_rep == 'hml_vec':
            n_joints = 22 if (sample.shape[1] in [263, 264]) else 21
            n_frames = sample.shape[-1]

            # 提取接触信息
            contact_info = extract_contact_info_from_sample(sample, n_frames)

            # 恢复关节位置
            sample = sample.cpu().permute(0, 2, 3, 1)
            sample = data.dataset.t2m_dataset.inv_transform(sample).float()
            sample = recover_from_ric(sample, n_joints, abs_3d=args.abs_3d)
            sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1)  # (batch_size, n_joints, 3, n_frames)

            # 定义骨骼的父关节索引列表 bones
            # 根节点的父索引为 -1，其余关节的父索引为对应的索引
            bones = [
                -1,  # 0: Pelvis
                0,   # 1: L_Hip
                0,   # 2: R_Hip
                0,   # 3: Spine1
                1,   # 4: L_Knee
                2,   # 5: R_Knee
                3,   # 6: Spine2
                4,   # 7: L_Ankle
                5,   # 8: R_Ankle
                6,   # 9: Spine3
                7,   # 10: L_Foot
                8,  # 11: R_Foot
                9,  # 12: Neck
                12,  # 13: Head
                9,  # 14: L_Collar
                9,  # 15: R_Collar
                14,  # 16: L_Shoulder
                15,  # 17: R_Shoulder
                16,  # 18: L_Elbow
                17,  # 19: R_Elbow
                18,  # 20: L_Wrist
                19,  # 21: R_Wrist
            ]

            # 应用脚部接触约束
            # 选择使用 'ik' 方法
            # sample = apply_foot_contact_constraints(sample, method='ik', bones=bones, contact_info=contact_info)

            # 或者选择使用 'root' 方法
            sample = apply_foot_contact_constraints(sample, method='root', contact_info=contact_info)

        all_motions.append(sample.cpu().numpy())
        all_lengths.append(model_kwargs['y']['lengths'].cpu().numpy())

        if args.unconstrained:
            all_text += ['unconstrained'] * args.num_samples
        else:
            text_key = 'text' if 'text' in model_kwargs['y'] else 'action_text'
            all_text += model_kwargs['y'][text_key]

        print(f"created {len(all_motions) * args.batch_size} samples")
        # Sampling is done!

    # Unnormalize observed motions and recover XYZ *positions*
    if model.data_rep == 'hml_vec':
        input_motions = input_motions.cpu().permute(0, 2, 3, 1)
        input_motions = data.dataset.t2m_dataset.inv_transform(input_motions).float()
        input_motions = recover_from_ric(data=input_motions, joints_num=n_joints, abs_3d=args.abs_3d)
        input_motions = input_motions.view(-1, *input_motions.shape[2:]).permute(0, 2, 3, 1)
        input_motions = input_motions.cpu().numpy()
        inpainting_mask = obs_joint_mask.cpu().numpy()

    all_motions = np.stack(all_motions) # [num_rep, num_samples, 22, 3, n_frames]
    all_text = np.stack(all_text) # [num_rep, num_samples]
    all_lengths = np.stack(all_lengths) # [num_rep, num_samples]
    all_observed_motions = input_motions # [num_samples, 22, 3, n_frames]
    all_observed_masks = inpainting_mask

    os.makedirs(out_path, exist_ok=True)

    # Write run arguments to json file an save in out_path
    with open(os.path.join(out_path, 'edit_args.json'), 'w') as fw:
        json.dump(vars(args), fw, indent=4, sort_keys=True)

    npy_path = os.path.join(out_path, f'results.npy')
    print(f"saving results file to [{npy_path}]")
    np.save(npy_path,
            {'motion': all_motions, 'text': all_text, 'lengths': all_lengths,
             'num_samples': args.num_samples, 'num_repetitions': args.num_repetitions,
             'observed_motion': all_observed_motions, 'observed_mask': all_observed_masks})
    with open(npy_path.replace('.npy', '.txt'), 'w') as fw:
        fw.write('\n'.join(all_text)) # TODO: Fix this for datasets other thah trajectories
    with open(npy_path.replace('.npy', '_len.txt'), 'w') as fw:
        fw.write('\n'.join([str(l) for l in all_lengths]))

    if args.dataset == 'humanml':
        plot_conditional_samples(motion=all_motions,
                                 lengths=all_lengths,
                                 texts=all_text,
                                 observed_motion=all_observed_motions,
                                 observed_mask=all_observed_masks,
                                 num_samples=args.num_samples,
                                 num_repetitions=args.num_repetitions,
                                 out_path=out_path,
                                 edit_mode=args.edit_mode, #FIXME: only works for selected edit modes
                                 stop_imputation_at=0)


def load_dataset(args, max_frames, split='test'):
    conf = DatasetConfig(
        name=args.dataset,
        batch_size=args.batch_size,
        num_frames=max_frames,
        split=split,
        hml_mode='train',  # in train mode, you get both text and motion.
        use_abs3d=args.abs_3d,
        traject_only=args.traj_only,
        use_random_projection=args.use_random_proj,
        random_projection_scale=args.random_proj_scale,
        augment_type='none',
        std_scale_shift=args.std_scale_shift,
        drop_redundant=args.drop_redundant,
    )
    data = get_dataset_loader(conf)
    return data


if __name__ == "__main__":
    main()