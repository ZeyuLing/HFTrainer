"""Real CUDA math coverage; tiny random weights, not pretrained-quality evidence."""
import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_joint_ltx_lora_backward_and_eval():
    from hftrainer.models.lora import apply_lora, get_lora_state_dict, set_lora_state_dict
    from hftrainer.models.ltx_video.network.model.transformer.model import LTXModel
    from hftrainer.models.ltx_video.network.model.transformer.modality import Modality
    from hftrainer.trainers.ltx_video.native.training_strategies.base_strategy import ModelInputs
    from hftrainer.trainers.ltx_video.native.training_strategies.flexible import FlexibleStrategy, FlexibleStrategyConfig

    torch.manual_seed(42)
    model = LTXModel(
        num_attention_heads=2, attention_head_dim=32, in_channels=8, out_channels=8,
        num_layers=1, cross_attention_dim=64,
        audio_num_attention_heads=2, audio_attention_head_dim=32,
        audio_in_channels=8, audio_out_channels=8, audio_cross_attention_dim=64,
        positional_embedding_max_pos=[20,2048,2048], audio_positional_embedding_max_pos=[20],
        cross_attention_adaln=True, use_prompt_adaln_single=False,
        ff_bias=False, audio_ff_bias=False,
    ).cuda()
    # Production loaders fill these empty tables from pretrained weights.
    # Initialize them explicitly in this random-weight math test.
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if 'scale_shift_table' in name:
                parameter.normal_(mean=0.0, std=0.02)
    apply_lora(model, dict(rank=2,alpha=2,target_modules=['to_q','to_k','to_v','to_out.0']))
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-3)

    def modality(axes):
        pos = torch.arange(4,device='cuda').float()[None,None,:,None].expand(1,axes,4,2).clone()
        pos[...,1] += 1
        return Modality(latent=torch.randn(1,4,8,device='cuda'),sigma=torch.tensor([.5],device='cuda'),
                        timesteps=torch.full((1,4),.5,device='cuda'),positions=pos,
                        context=torch.randn(1,3,64,device='cuda'))
    video,audio = modality(3),modality(1)
    strategy=FlexibleStrategy(FlexibleStrategyConfig(video={'is_generated':True,'latents_dir':'latents'},audio={'is_generated':True,'latents_dir':'audio_latents'}))
    inputs=ModelInputs(video=video,audio=audio,
                       video_targets=torch.randn_like(video.latent),audio_targets=torch.randn_like(audio.latent),
                       video_loss_mask=torch.ones(1,4,device='cuda',dtype=torch.bool),
                       audio_loss_mask=torch.ones(1,4,device='cuda',dtype=torch.bool))
    model.train()
    for _ in range(2):
        v,a=model(video,audio,None)
        loss=strategy.compute_loss(v,a,inputs).mean()
        assert torch.isfinite(loss)
        loss.backward()
        grads=[p.grad for p in trainable if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert any(torch.count_nonzero(g) for g in grads)
        optimizer.step();optimizer.zero_grad()
    saved={k:v.clone() for k,v in get_lora_state_dict(model).items()}
    assert saved
    with torch.no_grad():
        for p in trainable: p.zero_()
    set_lora_state_dict(model,saved)
    model.eval()
    with torch.no_grad():
        v,a=model(video,audio,None)
        assert torch.isfinite(strategy.compute_loss(v,a,inputs)).all()
