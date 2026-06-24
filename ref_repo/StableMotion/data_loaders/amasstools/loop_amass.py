import os
from glob import glob
from tqdm import tqdm


def loop_amams(
    base_folder,
    new_base_folder,
    ext=".npz",
    newext=".npz",
    force_redo=False,
    only_mirror=False,
    exclude=None,
):
    match_str = f"**/*{ext}"

    if only_mirror:
        match_str = f"M/**/*{ext}"

    _abs = glob(os.path.join(base_folder, match_str), recursive=True)
    _rel = [os.path.relpath(_p, base_folder) for _p in _abs]
    for motion_file in tqdm(_rel):
        if exclude and exclude in motion_file:
            continue

        motion_path = os.path.join(base_folder, motion_file)

        if motion_path.endswith("shape.npz"):
            continue

        new_motion_path = os.path.join(
            new_base_folder, motion_file.replace(ext, newext)
        )
        if not force_redo and os.path.exists(new_motion_path):
            continue

        new_folder = os.path.split(new_motion_path)[0]
        os.makedirs(new_folder, exist_ok=True)

        yield motion_path, new_motion_path
