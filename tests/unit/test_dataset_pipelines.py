from pathlib import Path

from PIL import Image

from hftrainer.datasets.classification.imagefolder_dataset import ImageFolderDataset


def test_imagefolder_dataset_uses_mmengine_style_pipeline(tmp_path):
    class_dir = tmp_path / 'cat'
    class_dir.mkdir(parents=True)
    image_path = class_dir / 'sample.png'
    Image.new('RGB', (12, 10), color=(255, 0, 0)).save(image_path)

    dataset = ImageFolderDataset(
        data_root=str(tmp_path),
        pipeline=[
            dict(type='LoadImage'),
            dict(type='ResizeImage', size=(8, 8)),
            dict(type='HFTrainerImageToTensor', image_key='image', output_key='pixel_values'),
            dict(type='RenameKeys', mapping={'label': 'labels'}),
        ],
    )

    sample = dataset[0]

    assert len(dataset) == 1
    assert sample['pixel_values'].shape == (3, 8, 8)
    assert sample['labels'] == 0
    assert Path(sample['img_path']) == image_path
