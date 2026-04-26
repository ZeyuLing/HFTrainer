from typing import Dict, List, Optional, Tuple, Union
import os
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


def _load_audio_fast(
    filename: str,
    sr: Optional[int] = None,
    mono: bool = True,
    offset: Optional[float] = None,
    duration: Optional[float] = None,
) -> Tuple[np.ndarray, int]:
    """Load audio using soundfile (fast C backend) with librosa fallback.

    soundfile is ~5-10x faster than librosa for WAV/FLAC files because it
    uses libsndfile directly without the FFmpeg overhead.  For non-WAV
    formats (mp3, m4a, etc.) we fall back to librosa.

    Returns:
        (audio, sample_rate): 1D float32 numpy array and sample rate.
    """
    ext = os.path.splitext(filename)[1].lower()

    # soundfile supports: wav, flac, ogg, aiff
    sf_extensions = {'.wav', '.flac', '.ogg', '.aiff', '.aif'}

    if ext in sf_extensions:
        try:
            import soundfile as sf

            # Compute frame range for offset/duration
            sf_kwargs = {}
            if offset is not None or duration is not None:
                info = sf.info(filename)
                native_sr = info.samplerate
                start_frame = 0
                if offset is not None:
                    start_frame = int(round(offset * native_sr))
                if duration is not None:
                    num_frames = int(round(duration * native_sr))
                    sf_kwargs['frames'] = num_frames
                sf_kwargs['start'] = start_frame

            audio, native_sr = sf.read(
                filename, dtype='float32', always_2d=False, **sf_kwargs,
            )

            # Convert to mono if needed
            if mono and audio.ndim > 1:
                audio = audio.mean(axis=1)

            # Resample if target sr differs
            if sr is not None and sr != native_sr:
                librosa = _import_librosa()
                audio = librosa.resample(audio, orig_sr=native_sr, target_sr=sr)
                native_sr = sr

            return audio.astype(np.float32), native_sr

        except Exception:
            pass  # Fall back to librosa

    # Fallback: librosa (handles mp3, m4a, and any format via FFmpeg)
    librosa = _import_librosa()
    load_kwargs = dict(sr=sr, mono=mono)
    if offset is not None:
        load_kwargs['offset'] = float(offset)
    if duration is not None:
        load_kwargs['duration'] = float(duration)
    audio, out_sr = librosa.load(filename, **load_kwargs)
    return audio, out_sr


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

        offset = results.get("_motion_audio_crop_start")
        duration = results.get("_motion_audio_crop_duration")

        try:
            audio, sr = _load_audio_fast(
                filename,
                sr=self.target_sr,
                mono=True,
                offset=float(offset) if offset is not None else None,
                duration=float(duration) if duration is not None else None,
            )
        except Exception as e:
            raise RuntimeError(
                f"LoadAudio failed for {filename} "
                f"(offset={offset}, duration={duration})"
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
