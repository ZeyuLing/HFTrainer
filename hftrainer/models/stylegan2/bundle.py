"""StyleGAN2 bundle."""

import torch

from hftrainer.models.base_model_bundle import ModelBundle
from hftrainer.models.stylegan2.network import (
    StyleGAN2Discriminator,
    StyleGAN2Generator,
)
from hftrainer.registry import MODEL_BUNDLES


from hftrainer.models.stylegan2.network import StyleGAN2ADAGenerator, StyleGAN2ADADiscriminator


@MODEL_BUNDLES.register_module()
class StyleGAN2Bundle(ModelBundle):
    """ModelBundle for StyleGAN2-style adversarial image generation."""

    def __init__(self, generator: dict, discriminator: dict):
        super().__init__()
        self._build_modules({
            'generator': generator,
            'discriminator': discriminator,
        })
        expected = {
            'generator': (StyleGAN2Generator, StyleGAN2ADAGenerator),
            'discriminator': (StyleGAN2Discriminator, StyleGAN2ADADiscriminator),
        }
        for name, expected_type in expected.items():
            component = getattr(self, name)
            if type(component) not in expected_type:
                raise TypeError(
                    f'StyleGAN2Bundle.{name} must be the repository-owned '
                    f'{[cls.__name__ for cls in expected_type]}; got '
                    f'{type(component).__module__}.{type(component).__name__}.'
                )

    def save_pretrained(
        self,
        save_directory: str,
        *,
        safe_serialization: bool = True,
        **kwargs,
    ):
        """Save architecture and weights as one HFTrainer-owned artifact."""

        if kwargs:
            unknown = ', '.join(sorted(kwargs))
            raise TypeError(f'Unexpected StyleGAN2 export options: {unknown}')
        from hftrainer.models.stylegan2.checkpoint import save_artifact

        generator_config = self.get_module_build_cfg('generator')
        discriminator_config = self.get_module_build_cfg('discriminator')
        # The strict registry also accepts the exact repository-owned class
        # object.  Artifacts, however, must remain portable JSON and must not
        # serialize Python object identities.
        generator_config['type'] = type(self.generator).__name__
        discriminator_config['type'] = type(self.discriminator).__name__
        config = {
            'type': type(self).__name__,
            'generator': generator_config,
            'discriminator': discriminator_config,
        }
        state = {
            f'{component}.{name}': tensor
            for component in ('generator', 'discriminator')
            for name, tensor in getattr(self, component).state_dict().items()
        }
        save_artifact(save_directory, config, state, safe=safe_serialization)
        from hftrainer.artifacts import write_artifact_config

        write_artifact_config(save_directory, type(self).__name__)

        return save_directory

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        """Load a strict, self-contained StyleGAN2 artifact."""

        from hftrainer.models.stylegan2.checkpoint import load_artifact

        config, state = load_artifact(pretrained_model_name_or_path)
        config.pop('type', None)
        config = cls._merge_nested_dict(config, kwargs)
        bundle = cls(**config)
        nested = {
            component: {
                name[len(component) + 1:]: tensor
                for name, tensor in state.items()
                if name.startswith(f'{component}.')
            }
            for component in ('generator', 'discriminator')
        }
        for component, values in nested.items():
            missing, unexpected = getattr(bundle, component).load_state_dict(values, strict=True)
            if missing or unexpected:
                raise RuntimeError(
                    f'Invalid {component} artifact: missing={missing}, unexpected={unexpected}'
                )
        return bundle

    def sample(
        self,
        z: torch.Tensor,
        truncation_psi: float = 1.0,
        truncation_cutoff=None,
        return_latents: bool = False,
    ):
        return self.generator(
            z,
            truncation_psi=truncation_psi,
            truncation_cutoff=truncation_cutoff,
            return_latents=return_latents,
        )

    def discriminate(self, images: torch.Tensor) -> torch.Tensor:
        return self.discriminator(images)
