from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from mmcv.transforms import BaseTransform

from hftrainer.registry import TRANSFORMS


def _import_librosa():
    try:
        import librosa
    except Exception as exc:
        raise RuntimeError(
            "LoadAudio requires librosa and its runtime dependencies "
            "(for example libsndfile)."
        ) from exc
    return librosa


@TRANSFORMS.register_module(force=True)
class LoadAudio(BaseTransform):
    def __init__(
        self,
        key: str = "audio",
        sr_key: Union[int, str] = "sr",
        target_sr: Optional[int] = None,
        allow_none: bool = False,
        duration_diff_threshold: float = 0.02,
    ):
        """
        :param keys: keys of audio need to be loaded
        :param sr: 1) if sr is str, it means the key of sr in the input dict,
         all loaded sr will be transformed to the sr saved in the dict.
         2) if sr in int, transform all the loaded audio to this sr.
        """
        self.key = key
        self.sr_key = sr_key
        self.target_sr = target_sr
        self.allow_none = allow_none
        self.duration_diff_threshold = duration_diff_threshold

    def transform(self, results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        assert self.target_sr is None or isinstance(
            self.target_sr, int
        ), f"target_sr must be None or int, but got {self.target_sr}"

        filename = results.get(f"{self.key}_path")
        if filename is None and self.allow_none:
            return results

        load_kwargs = dict(sr=self.target_sr, mono=True)
        offset = results.get("_motion_audio_crop_start")
        duration = results.get("_motion_audio_crop_duration")
        if offset is not None:
            load_kwargs["offset"] = float(offset)
        if duration is not None:
            load_kwargs["duration"] = float(duration)

        try:
            librosa = _import_librosa()
            # Load only the crop window when it is already planned by
            # MotionAudioRandomCrop. This avoids decoding the entire long-form
            # wav before trimming down to a 12 s segment.
            audio, sr = librosa.load(filename, **load_kwargs)
        except Exception as e:
            raise RuntimeError(
                f"LoadAudio failed for {filename} with load_kwargs={load_kwargs}"
            ) from e

        expected_motion_frames = results.get("_motion_audio_crop_num_frames")
        fps = results.get("fps")
        if expected_motion_frames is not None and fps is not None and sr:
            expected_audio_frames = int(round(int(expected_motion_frames) / float(fps) * sr))
            if audio.shape[0] > expected_audio_frames:
                audio = audio[:expected_audio_frames]
            elif audio.shape[0] < expected_audio_frames:
                audio = np.pad(audio, (0, expected_audio_frames - audio.shape[0]))

            motion_duration = int(expected_motion_frames) / float(fps)
            audio_duration = audio.shape[0] / sr
            if abs(motion_duration - audio_duration) > self.duration_diff_threshold:
                raise ValueError(
                    f"LoadAudio cropped segment mismatch for {filename}: "
                    f"motion_duration={motion_duration:.6f}, "
                    f"audio_duration={audio_duration:.6f}"
                )

        results[self.key] = audio
        results[self.sr_key] = sr
        results[f"{self.key}_num_frames"] = audio.shape[0]
        results[f"{self.key}_duration"] = audio.shape[0] / sr

        return results
