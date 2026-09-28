# StyleGAN2-ADA source provenance

The checkpoint-compatible graph under `network/ada` is adapted from
[NVIDIA StyleGAN2-ADA](https://github.com/NVlabs/stylegan2-ada-pytorch), revision `d72cc7d041b42ec8e806021a205ed9349f87c6a4`.
The NVIDIA Source Code License in `LICENSE.nvidia` applies to this graph and
its derivatives, including its non-commercial research/evaluation restriction.
Original copyright notices are retained. HFTrainer removes persistence/pickle
execution, uses local namespaces and the PyTorch reference operators, and
adapts the public generator/discriminator calls to the shared bundle boundary.
The earlier compact graph remains readable for legacy artifacts only.
