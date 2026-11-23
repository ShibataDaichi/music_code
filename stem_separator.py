"""
Meta社のAI音源分離モデル Demucs (htdemucs) を用いて、
音声ファイルを「drums（ドラム）」「bass（ベース）」「other（ギター・ピアノ・シンセ等の伴奏和音）」「vocals（ボーカル）」の
4ステムへ高精度に分離するモジュール。
Apple Silicon (MPS) アクセラレーションに自動対応し、一度ロードしたモデルはメモリ上にキャッシュします。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import soundfile as sf
import librosa

# モデルキャッシュをプロジェクト内の .cache ディレクトリに保存する
_CACHE_ROOT = Path(__file__).parent.resolve() / ".cache"
os.environ.setdefault("HF_HOME", str(_CACHE_ROOT / "huggingface"))
os.environ.setdefault("TORCH_HOME", str(_CACHE_ROOT / "torch"))

_MODEL_CACHE = None
_DEVICE_CACHE = None


def _get_demucs_model():
    """
    htdemucs モデルをロードしてメモリキャッシュする。
    """
    global _MODEL_CACHE, _DEVICE_CACHE
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE, _DEVICE_CACHE

    import torch
    from demucs.pretrained import get_model

    model = get_model("htdemucs")
    model.eval()

    # Apple Silicon GPU (MPS) が利用可能なら優先使用
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    try:
        model.to(device)
    except Exception:
        device = torch.device("cpu")
        model.to(device)

    _MODEL_CACHE = model
    _DEVICE_CACHE = device
    return _MODEL_CACHE, _DEVICE_CACHE


def separate_stems_for_analysis(
    audio_path: str,
    target_sr: int = 22050,
    backing_wav_out: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """
    音声ファイルを Demucs (htdemucs) で4ステム分離し、解析用のサンプリングレート (target_sr=22050Hz) の
    モノラル波形辞書 {'drums', 'bass', 'other', 'vocals', 'mix'} を返す。
    backing_wav_out が指定されている場合は、ボーカルとドラムを除去した伴奏トラック (bass + other) を WAV 保存する。
    """
    import torch
    from demucs.apply import apply_model

    model, device = _get_demucs_model()
    model_sr = int(model.samplerate)  # 通常 44100 Hz

    # soundfile でステレオ読み込み
    wav_data, file_sr = sf.read(audio_path, dtype="float32", always_2d=True)
    # shape: (samples, channels) -> (channels, samples)
    wav_ch = wav_data.T
    if wav_ch.shape[0] == 1:
        wav_ch = np.vstack([wav_ch, wav_ch])
    elif wav_ch.shape[0] > 2:
        wav_ch = wav_ch[:2, :]

    if file_sr != model_sr:
        left = librosa.resample(wav_ch[0], orig_sr=file_sr, target_sr=model_sr)
        right = librosa.resample(wav_ch[1], orig_sr=file_sr, target_sr=model_sr)
        wav_ch = np.vstack([left, right])

    # 正規化してテンソル化: shape (1, 2, samples)
    ref_mean = float(np.mean(wav_ch))
    ref_std = float(np.std(wav_ch)) + 1e-8
    wav_norm = (wav_ch - ref_mean) / ref_std

    tensor_in = torch.from_numpy(wav_norm.astype(np.float32)).unsqueeze(0)

    try:
        with torch.no_grad():
            sources = apply_model(
                model,
                tensor_in.to(device),
                shifts=0,
                split=True,
                overlap=0.25,
                progress=False,
            )[0]
            sources_np = sources.cpu().numpy()
    except Exception:
        # 万一 MPS 等で未対応演算エラーが出た場合は CPU にフォールバック
        model.to("cpu")
        with torch.no_grad():
            sources = apply_model(
                model,
                tensor_in.to("cpu"),
                shifts=0,
                split=True,
                overlap=0.25,
                progress=False,
            )[0]
            sources_np = sources.cpu().numpy()

    # 元のスケールに戻す: shape (num_sources, 2, samples)
    sources_np = sources_np * ref_std + ref_mean
    source_names = list(model.sources)  # ['drums', 'bass', 'other', 'vocals']

    stereo_stems: Dict[str, np.ndarray] = {}
    for idx, name in enumerate(source_names):
        stereo_stems[name] = sources_np[idx]

    # ボーカル＆ドラム除去の伴奏トラック (bass + other) および各パート単独トラック (bass / other / drums / vocals) を保存
    if backing_wav_out:
        out_path = Path(backing_wav_out)
        prefix = out_path.name.replace("_backing.wav", "")
        parent_dir = out_path.parent

        def _write_normalized_wav(file_path: Path, stereo_arr: np.ndarray, min_target_peak: float = 0.0) -> None:
            arr_t = stereo_arr.T
            peak = float(np.max(np.abs(arr_t)))
            if peak > 0.98:
                arr_t = arr_t / peak * 0.95
            elif min_target_peak > 0.0 and 0.02 < peak < min_target_peak:
                # ベース単体など音量が小さくなりやすいステムを聴き取りやすい音量に持ち上げる
                gain = min(2.2, min_target_peak / peak)
                arr_t = arr_t * gain
            sf.write(str(file_path), arr_t, model_sr, subtype="PCM_16")

        if "bass" in stereo_stems and "other" in stereo_stems:
            _write_normalized_wav(out_path, stereo_stems["bass"] + stereo_stems["other"], min_target_peak=0.85)
        if "bass" in stereo_stems:
            _write_normalized_wav(parent_dir / f"{prefix}_bass.wav", stereo_stems["bass"], min_target_peak=0.88)
        if "other" in stereo_stems:
            _write_normalized_wav(parent_dir / f"{prefix}_other.wav", stereo_stems["other"], min_target_peak=0.85)
        if "drums" in stereo_stems:
            _write_normalized_wav(parent_dir / f"{prefix}_drums.wav", stereo_stems["drums"], min_target_peak=0.85)
        if "vocals" in stereo_stems:
            _write_normalized_wav(parent_dir / f"{prefix}_vocals.wav", stereo_stems["vocals"], min_target_peak=0.85)

    # 解析用に target_sr (22050Hz) のモノラル配列へ変換
    result_mono: Dict[str, np.ndarray] = {}
    for name, st in stereo_stems.items():
        mono_44k = 0.5 * (st[0] + st[1])
        if model_sr != target_sr:
            result_mono[name] = librosa.resample(mono_44k, orig_sr=model_sr, target_sr=target_sr)
        else:
            result_mono[name] = mono_44k

    mix_44k = 0.5 * (wav_ch[0] + wav_ch[1])
    result_mono["mix"] = (
        librosa.resample(mix_44k, orig_sr=model_sr, target_sr=target_sr)
        if model_sr != target_sr
        else mix_44k
    )

    return result_mono
