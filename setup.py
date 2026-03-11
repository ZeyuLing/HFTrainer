from setuptools import setup, find_packages

setup(
    name='hftrainer',
    version='0.1.0',
    description='A unified training framework built on HuggingFace ecosystem',
    packages=find_packages(),
    python_requires='>=3.9',
    install_requires=[
        'torch>=2.0',
        'torchvision>=0.15',
        'accelerate>=1.1,<2',
        'transformers>=4.30',
        'diffusers>=0.20',
        'peft>=0.17,<0.18',
        'mmengine>=0.7',
        'safetensors>=0.3',
        'datasets>=2.0',
    ],
    extras_require={
        'dev': [
            'pytest>=7.4',
        ],
    },
)
