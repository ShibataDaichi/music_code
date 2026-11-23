"""
YouTube等のストリーミングURLからの音源取得・WAV変換モジュール、
および動作確認・テスト用の高品位デモ音源（王道進行・丸サ進行・カノン進行）生成モジュール。
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import soundfile as sf
import yt_dlp

# imageio-ffmpeg から同梱の ffmpeg バイナリを取得し、標準名 'ffmpeg' として参照可能にする
FFMPEG_EXE = "ffmpeg"
try:
    import imageio_ffmpeg
    _raw_ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    if _raw_ffmpeg and os.path.exists(_raw_ffmpeg):
        bin_dir = Path(__file__).parent / ".bin"
        bin_dir.mkdir(exist_ok=True)
        symlink_path = bin_dir / "ffmpeg"
        if not symlink_path.exists():
            try:
                os.symlink(_raw_ffmpeg, symlink_path)
            except OSError:
                pass
        FFMPEG_EXE = str(symlink_path if symlink_path.exists() else _raw_ffmpeg)
        os.environ["PATH"] = f"{bin_dir}{os.pathsep}{os.path.dirname(_raw_ffmpeg)}{os.pathsep}{os.environ.get('PATH', '')}"
except Exception:
    pass


def convert_to_wav(input_path: str, output_wav_path: str, sr: int = 44100) -> str:
    """
    任意の音声/動画ファイルをブラウザ再生およびLibrosa解析に最適なWAVファイルに変換する。
    """
    cmd = [
        FFMPEG_EXE,
        "-y",
        "-i",
        input_path,
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sr),
        "-ac",
        "2",
        output_wav_path,
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0 or not os.path.exists(output_wav_path):
        raise RuntimeError(f"音声変換に失敗しました: {proc.stderr[-400:]}")
    return output_wav_path


def download_from_url(url: str, output_dir: Path, job_id: str) -> Dict[str, Any]:
    """
    yt-dlp を用いて YouTube 等の URL から最高音質の音声ストリームを取得し、WAV に変換して返す。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_template = str(output_dir / f"{job_id}_raw.%(ext)s")
    wav_path = str(output_dir / f"{job_id}.wav")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": raw_template,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "ffmpeg_location": FFMPEG_EXE,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        downloaded_file = ydl.prepare_filename(info)

    # 拡張子が変化している場合に備えて探す
    if not os.path.exists(downloaded_file):
        candidates = list(output_dir.glob(f"{job_id}_raw.*"))
        if not candidates:
            raise FileNotFoundError("URLからの音声ファイルのダウンロードに失敗しました。")
        downloaded_file = str(candidates[0])

    # WAV へ変換して元の一時ファイルを削除
    convert_to_wav(downloaded_file, wav_path)
    try:
        if os.path.exists(downloaded_file) and downloaded_file != wav_path:
            os.remove(downloaded_file)
    except OSError:
        pass

    return {
        "title": info.get("title") or "YouTube Audio Track",
        "uploader": info.get("uploader") or info.get("channel") or "YouTube",
        "duration": info.get("duration") or 0,
        "thumbnail": info.get("thumbnail") or "",
        "webpage_url": info.get("webpage_url") or url,
        "video_id": info.get("id") or "",
        "source_type": "youtube",
        "audio_path": wav_path,
    }


# デモ音源プリセット定義
DEMO_PRESETS: Dict[str, Dict[str, Any]] = {
    "royal_road": {
        "title": "Demo 1: J-Pop 王道進行 (FM7 - G7 - Em7 - Am7)",
        "uploader": "Built-in Synthesizer (Key: C Major / 120 BPM)",
        "bpm": 120.0,
        # (bass_midi, [chord_midi_notes], beats_count)
        "progression": [
            # Bar 1-4: FM7 -> G7 -> Em7 -> Am7
            (41, [65, 69, 72, 76], 4),   # FM7 (F2 + F4 A4 C5 E5)
            (43, [67, 71, 74, 77], 4),   # G7  (G2 + G4 B4 D5 F5)
            (40, [64, 67, 71, 74], 4),   # Em7 (E2 + E4 G4 B4 D5)
            (45, [69, 72, 76, 79], 4),   # Am7 (A2 + A4 C5 E5 G5)
            # Bar 5-8: Dm7 -> G7sus4 -> G7 -> CM7 -> C7
            (38, [62, 65, 69, 72], 4),   # Dm7 (D2 + D4 F4 A4 C5)
            (43, [67, 72, 74, 77], 2),   # G7sus4 (G2 + G4 C5 D5 F5)
            (43, [67, 71, 74, 77], 2),   # G7  (G2 + G4 B4 D5 F5)
            (36, [60, 64, 67, 71], 2),   # CM7 (C2 + C4 E4 G4 B4)
            (36, [60, 64, 67, 70], 2),   # C7  (C2 + C4 E4 G4 Bb4)
        ],
    },
    "marusa": {
        "title": "Demo 2: 丸サ進行 / Just the Two of Us (FM7 - E7 - Am7 - Gm7 - C7)",
        "uploader": "Built-in Synthesizer (Key: A Minor / 108 BPM)",
        "bpm": 108.0,
        "progression": [
            (41, [65, 69, 72, 76], 4),   # FM7
            (40, [64, 68, 71, 74], 4),   # E7
            (45, [69, 72, 76, 79], 4),   # Am7
            (43, [67, 70, 74, 77], 2),   # Gm7
            (36, [60, 64, 67, 70], 2),   # C7
            (41, [65, 69, 72, 76], 4),   # FM7
            (40, [64, 68, 71, 74], 4),   # E7
            (45, [69, 72, 76, 79], 8),   # Am7
        ],
    },
    "canon": {
        "title": "Demo 3: カノン進行・オンコード含む (C - G/B - Am7 - Em7 - FM7 - C/E - Dm7 - G7)",
        "uploader": "Built-in Synthesizer (Key: C Major / 112 BPM)",
        "bpm": 112.0,
        "progression": [
            (36, [60, 64, 67], 4),       # C
            (47, [67, 71, 74], 4),       # G/B (Bass B2)
            (45, [69, 72, 76, 79], 4),   # Am7
            (40, [64, 67, 71, 74], 4),   # Em7
            (41, [65, 69, 72, 76], 4),   # FM7
            (40, [60, 64, 67], 4),       # C/E (Bass E2)
            (38, [62, 65, 69, 72], 4),   # Dm7
            (43, [67, 71, 74, 77], 4),   # G7
        ],
    },
}


def generate_demo_song(preset_key: str, output_dir: Path, job_id: str) -> Dict[str, Any]:
    """
    指定されたプリセットのコード進行を持つリッチなエレピ＋ベース＋ドラム音源を合成しWAV出力する。
    """
    preset = DEMO_PRESETS.get(preset_key, DEMO_PRESETS["royal_road"])
    bpm = float(preset["bpm"])
    sr = 44100
    beat_sec = 60.0 / bpm

    total_beats = sum(item[2] for item in preset["progression"])
    total_sec = total_beats * beat_sec
    n_samples = int(total_sec * sr)

    audio = np.zeros(n_samples, dtype=np.float64)

    current_beat = 0
    for bass_midi, chord_midis, beats in preset["progression"]:
        seg_start = int(current_beat * beat_sec * sr)
        seg_end = min(n_samples, int((current_beat + beats) * beat_sec * sr))
        seg_len = seg_end - seg_start
        if seg_len <= 0:
            continue

        t = np.arange(seg_len) / sr

        # エレクトリックピアノ風の減衰エンベロープ＋ビートごとの軽いコンピング刻み
        env_sustain = np.exp(-t * 0.55) * 0.75 + 0.25
        attack_samples = min(seg_len, int(0.015 * sr))
        env_sustain[:attack_samples] *= np.linspace(0.0, 1.0, attack_samples)
        release_samples = min(seg_len, int(0.04 * sr))
        env_sustain[-release_samples:] *= np.linspace(1.0, 0.05, release_samples)

        seg_wave = np.zeros(seg_len, dtype=np.float64)

        # ベース音 (豊かな基本波＋第2倍音)
        bass_freq = 440.0 * (2.0 ** ((bass_midi - 69.0) / 12.0))
        bass_wave = 0.48 * np.sin(2.0 * np.pi * bass_freq * t) + 0.16 * np.sin(4.0 * np.pi * bass_freq * t)
        seg_wave += bass_wave * env_sustain

        # コード構成音
        for m_note in chord_midis:
            freq = 440.0 * (2.0 ** ((m_note - 69.0) / 12.0))
            note_wave = (
                0.22 * np.sin(2.0 * np.pi * freq * t)
                + 0.05 * np.sin(4.0 * np.pi * freq * t)
                + 0.015 * np.sin(6.0 * np.pi * freq * t)
            )
            seg_wave += note_wave * env_sustain

        audio[seg_start:seg_end] += seg_wave
        current_beat += beats

    # ビートトラッキングおよびリズム感のために各拍に心地よいキック＆ハイハットを付加
    rng = np.random.default_rng(42)
    for b in range(total_beats):
        b_start = int(b * beat_sec * sr)
        hit_len = min(int(0.12 * sr), n_samples - b_start)
        if hit_len <= 0:
            continue
        th = np.arange(hit_len) / sr
        if b % 2 == 0:
            # ソフトなキック
            f_sweep = 110.0 * np.exp(-th * 28.0) + 45.0
            phase = 2.0 * np.pi * np.cumsum(f_sweep) / sr
            kick = 0.22 * np.sin(phase) * np.exp(-th * 22.0)
            audio[b_start : b_start + hit_len] += kick
        else:
            # ソフトなスネア/リム＋ハイハット
            noise = rng.normal(0.0, 1.0, hit_len)
            snare = 0.10 * noise * np.exp(-th * 35.0)
            audio[b_start : b_start + hit_len] += snare

    # 正規化してステレオWAVとして書き出し
    peak = float(np.max(np.abs(audio)))
    if peak > 1e-6:
        audio = audio / peak * 0.88

    stereo = np.column_stack([audio, audio])
    output_dir.mkdir(parents=True, exist_ok=True)
    wav_path = str(output_dir / f"{job_id}.wav")
    sf.write(wav_path, stereo, sr, subtype="PCM_16")

    return {
        "title": preset["title"],
        "uploader": preset["uploader"],
        "duration": round(total_sec, 2),
        "thumbnail": "",
        "webpage_url": "",
        "video_id": "",
        "source_type": "demo",
        "audio_path": wav_path,
    }
