from types import SimpleNamespace

import torch
import torch.nn as nn


class _DummyClip(nn.Module):
    def encode_text(self, text):
        return torch.zeros(text.shape[0], 512, device=text.device)


def test_mogents_tiny_generate_motion(monkeypatch):
    from hftrainer.models.motion.mogents.network import RVQVAE, generate_motion
    import hftrainer.models.motion.mogents.network.transformer.transformer_aux as aux
    import hftrainer.models.motion.mogents.network.transformer.transformer_ts as ts

    monkeypatch.setattr(aux, "_load_openai_clip_model", lambda clip_version, device: _DummyClip())
    monkeypatch.setattr(ts, "_load_openai_clip_model", lambda clip_version, device: _DummyClip())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    opt = SimpleNamespace(
        num_tokens1d=8,
        num_tokens2d=8,
        num_quantizers=3,
        device=device,
        attnj=False,
        attnt=False,
    )
    kwargs = dict(
        cond_mode="text",
        latent_dim=8,
        ff_size=16,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
        opt=opt,
        clip_dim=512,
        clip_version="dummy",
    )
    mask_aux = aux.MaskTransformer(4, **kwargs)
    mask_ts = ts.MaskTransformer2D(4, **kwargs)
    res_aux = aux.ResidualTransformer(4, **kwargs, share_weight=True)
    res_ts = ts.ResidualTransformer2D(4, **kwargs, share_weight=True)
    vq_args = SimpleNamespace(
        dataset_name="humanml3d",
        code_dim1d=4,
        nb_code1d=8,
        code_dim2d=4,
        nb_code2d=8,
        num_quantizers=3,
        shared_codebook=False,
        quantize_dropout_prob=0.0,
        mu=0.99,
    )
    vq = RVQVAE(
        vq_args,
        263,
        down_t=2,
        stride_t=2,
        width=8,
        depth=1,
        dilation_growth_rate=1,
        activation="relu",
        norm=None,
    )

    for module in (mask_aux, mask_ts, res_aux, res_ts, vq):
        module.eval().to(device)

    with torch.no_grad():
        motion = generate_motion(
            mask_aux,
            mask_ts,
            res_aux,
            res_ts,
            vq,
            ["walk"],
            torch.tensor([2], device=device),
            time_steps=1,
            cond_scale=1.0,
            res_cond_scale=1.0,
            n_joint_groups=6,
        )

    assert tuple(motion.shape) == (1, 8, 263)
