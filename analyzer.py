"""
高精度コード進行・キー・BPM解析エンジン (v5 - Meta Demucs AI 4ステム音源分離統合版)
- Demucs (htdemucs) により音源を「drums」「bass」「other（伴奏和音）」「vocals」へAI分離
  - ボーカルとドラムを100%除去した「other」ステムからコード構成音クロマ (C2〜B5) を抽出
  - 純粋な「bass」ステムから低域ルート音・オンコードベース音クロマ (C1〜B2) を抽出
  - 純粋な「drums + bass」ステムから連続周波数フーリエ・コム相関（0.1 BPM精度）でBPM・拍位置をロック
- ハーモニック・フラックス最大化による小節頭（1拍目・ダウンビート）自動アライメント
- NNLS（非負最小二乗法）倍音逆畳み込み＋拍構造認識ビタビデコーディング
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import scipy.ndimage
import scipy.optimize
import librosa

try:
    import imageio_ffmpeg
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    ffmpeg_dir = os.path.dirname(ffmpeg_exe)
    if ffmpeg_dir not in os.environ.get("PATH", ""):
        os.environ["PATH"] = f"{ffmpeg_dir}{os.pathsep}{os.environ.get('PATH', '')}"
except Exception:
    pass


PITCH_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]

MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def _build_harmonic_dictionary(g3: float = 0.21, g5: float = 0.10) -> np.ndarray:
    H = np.eye(12, dtype=np.float64)
    for p in range(12):
        H[(p + 7) % 12, p] += g3
        H[(p + 4) % 12, p] += g5
    return H


HARMONIC_DICT_H = _build_harmonic_dictionary(g3=0.20, g5=0.09)
HARMONIC_DICT_BASS = _build_harmonic_dictionary(g3=0.26, g5=0.11)


def _deconvolve_chroma_nnls(vec: np.ndarray, H: np.ndarray) -> np.ndarray:
    if np.max(vec) < 1e-8:
        return vec
    x, _ = scipy.optimize.nnls(H, vec)
    return x


@dataclass
class ChordTemplate:
    quality: str
    suffix: str
    intervals: List[int]
    weights: List[float]
    complexity_penalty: float
    simple_suffix: str


CHORD_QUALITIES: List[ChordTemplate] = [
    # トライアド
    ChordTemplate("maj", "", [0, 4, 7], [1.0, 0.90, 0.82], 0.00, ""),
    ChordTemplate("min", "m", [0, 3, 7], [1.0, 0.90, 0.82], 0.00, "m"),
    # セブンス系
    ChordTemplate("7", "7", [0, 4, 7, 10], [1.0, 0.88, 0.74, 0.88], 0.026, "7"),
    ChordTemplate("maj7", "M7", [0, 4, 7, 11], [1.0, 0.88, 0.74, 0.88], 0.030, "M7"),
    ChordTemplate("min7", "m7", [0, 3, 7, 10], [1.0, 0.88, 0.74, 0.88], 0.024, "m7"),
    ChordTemplate("m7b5", "m7(b5)", [0, 3, 6, 10], [1.0, 0.88, 0.88, 0.86], 0.058, "m7(b5)"),
    ChordTemplate("dim7", "dim7", [0, 3, 6, 9], [1.0, 0.85, 0.85, 0.85], 0.068, "dim"),
    # サスペンド・オーギュメント・テンション系
    ChordTemplate("sus4", "sus4", [0, 5, 7], [1.0, 0.95, 0.80], 0.048, "sus4"),
    ChordTemplate("7sus4", "7sus4", [0, 5, 7, 10], [1.0, 0.92, 0.72, 0.88], 0.048, "7sus4"),
    ChordTemplate("aug", "aug", [0, 4, 8], [1.0, 0.90, 0.90], 0.068, "aug"),
    ChordTemplate("add9", "add9", [0, 2, 4, 7], [1.0, 0.82, 0.88, 0.74], 0.062, ""),
]


def _build_template_matrix() -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    templates = []
    out_masks = []
    penalties = []
    meta = []

    for root_idx, root_name in enumerate(PITCH_NAMES):
        for cq in CHORD_QUALITIES:
            vec = np.zeros(12, dtype=np.float64)
            mask = np.ones(12, dtype=np.float64)
            notes = []
            for interval, w in zip(cq.intervals, cq.weights):
                pc = (root_idx + interval) % 12
                vec[pc] = w
                mask[pc] = 0.0
                notes.append(PITCH_NAMES[pc])

            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm

            templates.append(vec)
            out_masks.append(mask)
            penalties.append(cq.complexity_penalty)
            meta.append({
                "root_idx": root_idx,
                "root": root_name,
                "quality": cq.quality,
                "suffix": cq.suffix,
                "simple_suffix": cq.simple_suffix,
                "chord": f"{root_name}{cq.suffix}",
                "simple_chord": f"{root_name}{cq.simple_suffix}",
                "notes": notes,
                "intervals": cq.intervals,
            })

    return np.array(templates), np.array(out_masks), np.array(penalties), meta


TEMPLATE_MATRIX, OUT_MASKS, TEMPLATE_PENALTIES, TEMPLATE_META = _build_template_matrix()


def estimate_key(chroma_mean: np.ndarray) -> Dict[str, Any]:
    c_norm = chroma_mean - np.mean(chroma_mean)
    c_std = np.linalg.norm(c_norm)
    if c_std < 1e-8:
        return {
            "key": "C Major",
            "root": "C",
            "root_idx": 0,
            "scale": "Major",
            "confidence": 0.5,
            "relative_key": "A Minor",
            "diatonic_roots": [0, 2, 4, 5, 7, 9, 11],
        }

    maj_norm = MAJOR_PROFILE - np.mean(MAJOR_PROFILE)
    maj_norm /= np.linalg.norm(maj_norm)

    min_norm = MINOR_PROFILE - np.mean(MINOR_PROFILE)
    min_norm /= np.linalg.norm(min_norm)

    candidates = []
    for i in range(12):
        rolled_maj = np.roll(maj_norm, i)
        corr_maj = float(np.dot(c_norm / c_std, rolled_maj))
        candidates.append((corr_maj, i, "Major"))

        rolled_min = np.roll(min_norm, i)
        corr_min = float(np.dot(c_norm / c_std, rolled_min))
        candidates.append((corr_min, i, "Minor"))

    candidates.sort(key=lambda x: x[0], reverse=True)
    best_corr, best_root_idx, best_scale = candidates[0]
    best_root = PITCH_NAMES[best_root_idx]

    if best_scale == "Major":
        rel_idx = (best_root_idx + 9) % 12
        relative_key = f"{PITCH_NAMES[rel_idx]} Minor"
        scale_intervals = [0, 2, 4, 5, 7, 9, 11]
    else:
        rel_idx = (best_root_idx + 3) % 12
        relative_key = f"{PITCH_NAMES[rel_idx]} Major"
        scale_intervals = [0, 2, 3, 5, 7, 8, 10]

    diatonic_roots = [(best_root_idx + iv) % 12 for iv in scale_intervals]

    return {
        "key": f"{best_root} {best_scale}",
        "root": best_root,
        "root_idx": best_root_idx,
        "scale": best_scale,
        "confidence": round(max(0.0, min(1.0, (best_corr + 1.0) / 2.0)), 3),
        "relative_key": relative_key,
        "diatonic_roots": diatonic_roots,
    }


def _build_key_chord_priors(key_root_idx: int, key_scale: str) -> np.ndarray:
    priors = np.full(len(TEMPLATE_META), -0.038, dtype=np.float64)
    maj_root = key_root_idx if key_scale == "Major" else (key_root_idx + 3) % 12

    diatonic_map = {
        0:  ({"maj", "maj7", "sus4", "add9"}, {"7", "7sus4"}),          # I
        2:  ({"min", "min7"}, {"7", "maj"}),                            # ii / II7
        4:  ({"min", "min7"}, {"7", "maj"}),                            # iii / III7 (V7/vi)
        5:  ({"maj", "maj7", "add9"}, {"min", "min7", "7"}),            # IV / IVm / IV7
        7:  ({"maj", "7", "sus4", "7sus4", "add9"}, {"min", "min7"}),   # V / Gm7
        9:  ({"min", "min7"}, {"7", "maj"}),                            # vi / VI7
        11: ({"m7b5", "dim7"}, {"min", "maj", "7"}),                    # vii°
        10: ({"maj", "7", "maj7"}, set()),                              # bVII
        8:  ({"maj", "maj7", "7"}, set()),                              # bVI
        3:  ({"maj", "maj7", "7"}, set()),                              # bIII
    }

    for idx, meta in enumerate(TEMPLATE_META):
        rel = (meta["root_idx"] - maj_root) % 12
        q = meta["quality"]
        if rel in diatonic_map:
            pref, acc = diatonic_map[rel]
            if q in pref:
                priors[idx] = 0.035
            elif q in acc:
                priors[idx] = 0.015
            else:
                priors[idx] = -0.018

    return priors


def compute_roman_numeral(
    chord_root_idx: int,
    quality: str,
    suffix: str,
    key_root_idx: int,
    key_scale: str,
) -> str:
    if chord_root_idx < 0:
        return "N.C."
    rel_semitone = (chord_root_idx - key_root_idx) % 12
    degree_names = ["I", "bII", "II", "bIII", "III", "IV", "#IV", "V", "bVI", "VI", "bVII", "VII"]
    base = degree_names[rel_semitone]

    is_minor_chord = quality in ("min", "min7", "m7b5", "dim7", "m6")
    if is_minor_chord:
        prefix = ""
        roman_part = base
        if base.startswith(("b", "#")):
            prefix = base[0]
            roman_part = base[1:]
        roman_part = roman_part.lower()
        extra = suffix[1:] if suffix.startswith("m") else suffix
        return f"{prefix}{roman_part}{extra}"
    else:
        return f"{base}{suffix}"


def _viterbi_bar_aware(
    scores: np.ndarray,
    bar_head_penalty: float = 0.062,
    half_bar_penalty: float = 0.100,
    off_beat_penalty: float = 0.240,
) -> np.ndarray:
    n_states, n_frames = scores.shape
    if n_frames == 0:
        return np.array([], dtype=int)

    dp = np.zeros((n_states, n_frames), dtype=np.float64)
    backtrack = np.zeros((n_states, n_frames), dtype=np.int32)

    dp[:, 0] = scores[:, 0]

    for t in range(1, n_frames):
        beat_in_bar = t % 4
        if beat_in_bar == 0:
            penalty = bar_head_penalty
        elif beat_in_bar == 2:
            penalty = half_bar_penalty
        else:
            penalty = off_beat_penalty

        prev = dp[:, t - 1]
        best_prev_idx = int(np.argmax(prev))
        best_prev_val = prev[best_prev_idx]

        if n_states > 1:
            temp = prev.copy()
            temp[best_prev_idx] = -np.inf
            second_prev_idx = int(np.argmax(temp))
            second_prev_val = temp[second_prev_idx]
        else:
            second_prev_idx = best_prev_idx
            second_prev_val = best_prev_val

        for s in range(n_states):
            stay_val = prev[s]
            if s != best_prev_idx:
                switch_val = best_prev_val - penalty
                switch_idx = best_prev_idx
            else:
                switch_val = second_prev_val - penalty
                switch_idx = second_prev_idx

            if stay_val >= switch_val:
                dp[s, t] = stay_val + scores[s, t]
                backtrack[s, t] = s
            else:
                dp[s, t] = switch_val + scores[s, t]
                backtrack[s, t] = switch_idx

    path = np.zeros(n_frames, dtype=np.int32)
    path[-1] = int(np.argmax(dp[:, -1]))
    for t in range(n_frames - 1, 0, -1):
        path[t - 1] = backtrack[path[t], t]

    return path


def _estimate_locked_beats(y_rhythm: np.ndarray, sr: int, duration: float) -> Tuple[float, np.ndarray]:
    """
    ドラム＋ベースの分離リズム信号から高分解能オンセット包絡線 (hop=128 = 5.8ms) と
    連続周波数フーリエ・コム相関を用いて正確なBPMと拍グリッドを算出する。
    """
    hop = 128
    oenv = librosa.onset.onset_strength(y=y_rhythm, sr=sr, hop_length=hop)
    oenv_norm = oenv / (np.std(oenv) + 1e-8)
    times = librosa.frames_to_time(np.arange(len(oenv)), sr=sr, hop_length=hop)

    tempo_raw, beat_frames_raw = librosa.beat.beat_track(
        onset_envelope=oenv, sr=sr, hop_length=hop, start_bpm=130.0, tightness=180
    )
    bt_raw = librosa.frames_to_time(beat_frames_raw, sr=sr, hop_length=hop)

    if len(bt_raw) >= 4:
        med_bpm = float(60.0 / np.median(np.diff(bt_raw)))
    else:
        med_bpm = float(np.ndim(tempo_raw) and tempo_raw[0] or tempo_raw) or 120.0

    if med_bpm < 78.0:
        med_bpm *= 2.0

    cands = np.arange(max(70.0, med_bpm - 8.0), min(225.0, med_bpm + 8.0), 0.1)
    comb_scores = np.zeros(len(cands), dtype=np.float64)
    for idx, b in enumerate(cands):
        f = b / 60.0
        p = np.exp(-2j * np.pi * f * times)
        comb_scores[idx] = (
            np.abs(np.dot(oenv_norm, p))
            + 0.45 * np.abs(np.dot(oenv_norm, p ** 2))
            + 0.25 * np.abs(np.dot(oenv_norm, p ** 0.5))
        )

    best_idx = int(np.argmax(comb_scores))
    exact_bpm = float(cands[best_idx])
    peak_ratio = float(comb_scores[best_idx] / (np.mean(comb_scores) + 1e-8))

    nearest_int = round(exact_bpm)
    if abs(exact_bpm - nearest_int) <= 0.12:
        exact_bpm = float(nearest_int)

    period = 60.0 / exact_bpm

    if peak_ratio >= 1.50 or len(bt_raw) < 8:
        freq = exact_bpm / 60.0
        c_val = np.dot(oenv_norm, np.exp(-2j * np.pi * freq * times))
        phase_sec = float((-np.angle(c_val) / (2.0 * np.pi * freq)) % period)
        grid_times = np.arange(phase_sec, duration, period)

        win_frames = max(1, int(0.015 * sr / hop))
        refined_times = []
        for gt in grid_times:
            center_f = int(round(gt * sr / hop))
            f0 = max(0, center_f - win_frames)
            f1 = min(len(oenv) - 1, center_f + win_frames)
            local_peak_f = f0 + int(np.argmax(oenv[f0 : f1 + 1]))
            refined_t = 0.72 * gt + 0.28 * float(local_peak_f * hop / sr)
            refined_times.append(refined_t)
        beat_times = np.array(refined_times, dtype=np.float64)
    else:
        repaired = [float(bt_raw[0])]
        for t_val in bt_raw[1:]:
            dt = float(t_val) - repaired[-1]
            if dt > 1.65 * period:
                n_missing = int(round(dt / period))
                step = dt / n_missing
                for m in range(1, n_missing):
                    repaired.append(repaired[-1] + step)
                repaired.append(float(t_val))
            elif dt < 0.55 * period:
                continue
            else:
                repaired.append(float(t_val))
        beat_times = np.array(repaired, dtype=np.float64)

        while len(beat_times) > 0 and beat_times[0] > period * 0.85:
            beat_times = np.insert(beat_times, 0, max(0.0, beat_times[0] - period))
            if beat_times[0] == 0.0:
                break

    return exact_bpm, beat_times


def extract_waveform_peaks(y: np.ndarray, num_points: int = 900) -> List[float]:
    if len(y) == 0:
        return [0.0] * num_points
    chunk_size = max(1, len(y) // num_points)
    trimmed_len = chunk_size * num_points
    if trimmed_len > len(y):
        y_padded = np.pad(y, (0, trimmed_len - len(y)))
    else:
        y_padded = y[:trimmed_len]

    reshaped = np.abs(y_padded.reshape(num_points, chunk_size))
    peaks = np.mean(reshaped, axis=1) * 0.6 + np.max(reshaped, axis=1) * 0.4
    max_val = float(np.max(peaks)) if np.max(peaks) > 1e-6 else 1.0
    normalized = np.clip(peaks / max_val, 0.02, 1.0)
    return [round(float(v), 4) for v in normalized]


def analyze_audio_file(
    audio_path: str,
    use_demucs: bool = True,
    backing_wav_out: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Demucs AI 4ステム音源分離（drums / bass / other / vocals）＋連続フーリエ・コム同期＋NNLS倍音逆畳み込みを用いて
    BPM・ビート・小節頭（1拍目）・キー・コード進行を最高精度で解析する。
    """
    sr = 22050
    demucs_applied = False

    if use_demucs:
        try:
            from stem_separator import separate_stems_for_analysis
            stems = separate_stems_for_analysis(
                audio_path=audio_path,
                target_sr=sr,
                backing_wav_out=backing_wav_out,
            )
            y_mix = stems["mix"]
            # 100% ボーカル・ドラムを除去した伴奏ステム (other) と ベースステム (bass)
            y_other = stems["other"]
            y_bass_stem = stems["bass"]
            y_drums = stems["drums"]

            # 軽いHPSSで伴奏内のギターカッティングノイズ等をさらに滑らかにする
            y_harm_chords, _ = librosa.effects.hpss(y_other, kernel_size=21, margin=(1.3, 1.5))
            y_harm_bass, _ = librosa.effects.hpss(y_bass_stem, kernel_size=21, margin=(1.3, 1.5))
            # ビートトラッキングには純粋なドラム＋ベース信号を使用
            y_rhythm = 0.72 * y_drums + 0.28 * y_bass_stem
            demucs_applied = True
        except Exception as exc:
            print(f"[!] Demucs fallback to M/S: {exc}")
            demucs_applied = False

    if not demucs_applied:
        y_raw, sr = librosa.load(audio_path, sr=sr, mono=False)
        if y_raw.ndim == 2 and y_raw.shape[0] >= 2:
            y_mix = 0.5 * (y_raw[0] + y_raw[1])
            y_side = 0.5 * (y_raw[0] - y_raw[1])
            side_rms = float(np.sqrt(np.mean(y_side ** 2)))
            mid_rms = float(np.sqrt(np.mean(y_mix ** 2)))
            y_for_chords = (0.52 * y_mix + 0.85 * y_side) if side_rms > 0.05 * mid_rms else y_mix
        else:
            y_mix = y_raw if y_raw.ndim == 1 else y_raw[0]
            y_for_chords = y_mix

        y_harm_chords, _ = librosa.effects.hpss(y_for_chords, kernel_size=31, margin=(1.8, 2.0))
        y_harm_bass, _ = librosa.effects.hpss(y_mix, kernel_size=31, margin=(1.6, 2.0))
        y_rhythm = y_mix

    duration = float(librosa.get_duration(y=y_mix, sr=sr))
    if duration < 1.0:
        raise ValueError("音声ファイルが短すぎます（1秒以上の音源を指定してください）。")

    # 1. A4チューニングずれの推定（ボーカルピッチ揺れの影響を受けない伴奏＋ベースから推定）
    tuning = float(librosa.estimate_tuning(y=y_harm_chords + 0.5 * y_harm_bass, sr=sr))

    # 2. 連続周波数フーリエ・コム相関＋フェーズロックによるBPM・ビート抽出
    bpm, beat_times = _estimate_locked_beats(y_rhythm, sr, duration)

    # 3. CQTクロマグラム計算
    hop_length = 512
    beat_frames = librosa.time_to_frames(beat_times, sr=sr, hop_length=hop_length)

    # (a) コード本体クロマ: C2〜B5 (4オクターブ)
    chroma_harm = librosa.feature.chroma_cqt(
        y=y_harm_chords,
        sr=sr,
        hop_length=hop_length,
        fmin=float(librosa.note_to_hz("C2")),
        n_octaves=4,
        bins_per_octave=36,
        tuning=tuning,
    )
    chroma_harm = scipy.ndimage.median_filter(chroma_harm, size=(1, 7))

    # (b) 低音（ベースライン・ルート音）クロマ: C1〜B2 (2オクターブ)
    chroma_bass = librosa.feature.chroma_cqt(
        y=y_harm_bass,
        sr=sr,
        hop_length=hop_length,
        fmin=float(librosa.note_to_hz("C1")),
        n_octaves=2,
        bins_per_octave=36,
        tuning=tuning,
    )
    chroma_bass = scipy.ndimage.median_filter(chroma_bass, size=(1, 7))

    # (c) 伴奏＋ベースの調波エネルギー（無音・トークのみの区間を N.C. にするため y_harm_chords + y_harm_bass を使用！）
    # ※ Demucs有効時はトーク（vocals）が除去されているため、冒頭トーク区間がさらに確実に N.C. になります！
    rms_harm = librosa.feature.rms(y=y_harm_chords + y_harm_bass, hop_length=hop_length)[0]

    # 4. ビート同期 ＆ NNLS倍音逆畳み込み
    max_frame = min(chroma_harm.shape[1], chroma_bass.shape[1], len(rms_harm)) - 1
    valid_mask = beat_frames < max_frame
    beat_frames = beat_frames[valid_mask]
    beat_times = beat_times[valid_mask]

    if len(beat_frames) == 0:
        beat_frames = np.array([0])
        beat_times = np.array([0.0])

    boundary_frames = np.append(beat_frames, max_frame)
    boundary_times = np.append(beat_times, duration)

    n_beats = len(beat_frames)
    beat_chroma = np.zeros((12, n_beats), dtype=np.float64)
    beat_bass = np.zeros((12, n_beats), dtype=np.float64)
    beat_rms = np.zeros(n_beats, dtype=np.float64)

    for i in range(n_beats):
        f_start = int(boundary_frames[i])
        f_end = max(f_start + 1, int(boundary_frames[i + 1]))
        seg_harm = chroma_harm[:, f_start:f_end]
        seg_bass = chroma_bass[:, f_start:f_end]

        beat_rms[i] = float(np.median(rms_harm[f_start:f_end]))

        c_raw = np.median(seg_harm, axis=1)
        b_raw = np.median(seg_bass, axis=1)

        c_deconv = _deconvolve_chroma_nnls(c_raw, HARMONIC_DICT_H)
        b_deconv = _deconvolve_chroma_nnls(b_raw, HARMONIC_DICT_BASS)

        combined = c_deconv + 0.32 * b_deconv
        norm_c = np.linalg.norm(combined)
        beat_chroma[:, i] = combined / norm_c if norm_c > 1e-8 else combined

        norm_b = np.linalg.norm(b_deconv)
        beat_bass[:, i] = b_deconv / norm_b if norm_b > 1e-8 else b_deconv

    # 5. 有音ビートのみを使って曲全体のキー（調）を推定
    rms_ref = float(np.percentile(beat_rms, 80)) if len(beat_rms) > 0 else 1.0
    active_beat_mask = beat_rms > (rms_ref * 0.12)
    if np.any(active_beat_mask):
        chroma_global = (
            np.mean(beat_chroma[:, active_beat_mask], axis=1)
            + 0.38 * np.mean(beat_bass[:, active_beat_mask], axis=1)
        )
    else:
        chroma_global = np.mean(chroma_harm, axis=1)
    key_info = estimate_key(chroma_global)

    # 6. グローバル・ハーモニック・フラックス最大化による小節頭（1拍目）位相の自動特定
    if n_beats >= 8:
        c_diff = np.linalg.norm(beat_chroma[:, 1:] - beat_chroma[:, :-1], axis=0)
        b_diff = np.linalg.norm(beat_bass[:, 1:] - beat_bass[:, :-1], axis=0)
        flux = np.insert(c_diff + 0.55 * b_diff, 0, 0.0)
        flux = flux * active_beat_mask.astype(np.float64)

        phase_scores = np.zeros(4, dtype=np.float64)
        for p in range(4):
            m1 = float(np.mean(flux[p::4])) if len(flux[p::4]) > 0 else 0.0
            p3 = (p + 2) % 4
            m3 = float(np.mean(flux[p3::4])) if len(flux[p3::4]) > 0 else 0.0
            phase_scores[p] = m1 + 0.25 * m3
        bar_offset = int(np.argmax(phase_scores))
    else:
        bar_offset = 0

    # 7. 全テンプレートとの類似度スコア計算（Demucs分離時はベースステムの信頼度が非常に高いため bass_support を強化）
    in_chord_sim = np.dot(TEMPLATE_MATRIX, beat_chroma)
    out_chord_energy = np.sqrt(np.dot(OUT_MASKS, beat_chroma ** 2))
    raw_scores = in_chord_sim - 0.25 * out_chord_energy

    key_priors = _build_key_chord_priors(key_info["root_idx"], key_info["scale"])
    adjusted_scores = np.zeros_like(raw_scores)
    bass_weight = 0.18 if demucs_applied else 0.14

    for idx, meta in enumerate(TEMPLATE_META):
        score_row = raw_scores[idx, :] - TEMPLATE_PENALTIES[idx] + key_priors[idx]
        root_idx = meta["root_idx"]
        bass_support = beat_bass[root_idx, :] * bass_weight
        adjusted_scores[idx, :] = score_row + bass_support

    if bar_offset > 0 and n_beats > bar_offset:
        pad_cols = (4 - bar_offset) % 4
        padded_scores = np.pad(adjusted_scores, ((0, 0), (pad_cols, 0)), mode="edge")
        padded_seq = _viterbi_bar_aware(padded_scores)
        best_state_seq = padded_seq[pad_cols:]
    else:
        best_state_seq = _viterbi_bar_aware(adjusted_scores)

    # 8. ビート単位のコード情報生成
    raw_beat_chords: List[Dict[str, Any]] = []
    for i in range(n_beats):
        t_start = float(boundary_times[i])
        t_end = float(boundary_times[i + 1])

        shifted_beat = i + ((4 - bar_offset) % 4)
        bar_num = (shifted_beat // 4) + 1
        beat_in_bar = (shifted_beat % 4) + 1

        if not active_beat_mask[i]:
            raw_beat_chords.append({
                "beat_index": i,
                "bar": bar_num,
                "beat_in_bar": beat_in_bar,
                "start": round(t_start, 3),
                "end": round(t_end, 3),
                "chord": "N.C.",
                "simple_chord": "N.C.",
                "root": "N.C.",
                "root_idx": -1,
                "quality": "nc",
                "suffix": "",
                "simple_suffix": "",
                "bass": "N.C.",
                "notes": [],
                "intervals": [],
                "roman": "N.C.",
                "confidence": 1.0,
            })
            continue

        state_idx = int(best_state_seq[i])
        meta = TEMPLATE_META[state_idx]
        conf = float(np.clip(in_chord_sim[state_idx, i], 0.0, 1.0))
        root_pc = meta["root_idx"]

        roman = compute_roman_numeral(
            chord_root_idx=root_pc,
            quality=meta["quality"],
            suffix=meta["suffix"],
            key_root_idx=key_info["root_idx"],
            key_scale=key_info["scale"],
        )

        raw_beat_chords.append({
            "beat_index": i,
            "bar": bar_num,
            "beat_in_bar": beat_in_bar,
            "start": round(t_start, 3),
            "end": round(t_end, 3),
            "chord": meta["chord"],
            "simple_chord": meta["simple_chord"],
            "root": meta["root"],
            "root_idx": root_pc,
            "quality": meta["quality"],
            "suffix": meta["suffix"],
            "simple_suffix": meta["simple_suffix"],
            "bass": meta["root"],
            "notes": meta["notes"],
            "intervals": meta["intervals"],
            "roman": roman,
            "confidence": round(conf, 3),
        })

    # 9. 同一小節内の連続同一コードをマージ
    segments: List[Dict[str, Any]] = []
    for bc in raw_beat_chords:
        if (
            segments
            and segments[-1]["chord"] == bc["chord"]
            and segments[-1]["bar"] == bc["bar"]
        ):
            prev = segments[-1]
            prev["end"] = bc["end"]
            prev["beat_end"] = bc["beat_index"] + 1
            prev["beats_count"] = prev.get("beats_count", 1) + 1
            prev["duration"] = round(prev["end"] - prev["start"], 3)
            prev["confidence"] = round((prev["confidence"] + bc["confidence"]) / 2.0, 3)
        else:
            segments.append({
                "index": len(segments),
                "start": bc["start"],
                "end": bc["end"],
                "duration": round(bc["end"] - bc["start"], 3),
                "beat_start": bc["beat_index"],
                "beat_end": bc["beat_index"] + 1,
                "beats_count": 1,
                "bar": bc["bar"],
                "beat_in_bar": bc["beat_in_bar"],
                "chord": bc["chord"],
                "simple_chord": bc["simple_chord"],
                "root": bc["root"],
                "root_idx": bc["root_idx"],
                "quality": bc["quality"],
                "suffix": bc["suffix"],
                "simple_suffix": bc["simple_suffix"],
                "bass": bc["bass"],
                "notes": bc["notes"],
                "intervals": bc["intervals"],
                "roman": bc["roman"],
                "confidence": bc["confidence"],
            })

    # セグメント単位（2拍以上持続する場合のみ）でのオンコード判定
    for seg in segments:
        if seg["root_idx"] < 0 or seg["beats_count"] < 2:
            continue
        b_start = seg["beat_start"]
        b_end = seg["beat_end"]
        mean_bass_vec = np.mean(beat_bass[:, b_start:b_end], axis=1)
        bass_pc = int(np.argmax(mean_bass_vec))
        root_pc = int(seg["root_idx"])
        bass_name = PITCH_NAMES[bass_pc]

        if (
            bass_pc != root_pc
            and mean_bass_vec[bass_pc] > 0.60
            and (mean_bass_vec[bass_pc] - mean_bass_vec[root_pc]) > 0.22
            and bass_name in seg["notes"]
        ):
            seg["bass"] = bass_name
            seg["chord"] = f"{seg['root']}{seg['suffix']}/{bass_name}"
            seg["simple_chord"] = f"{seg['root']}{seg['simple_suffix']}/{bass_name}"

    waveform = extract_waveform_peaks(y_mix, num_points=900)

    return {
        "duration": round(duration, 3),
        "bpm": round(bpm, 1),
        "tuning_cents": round(tuning * 100, 1),
        "demucs_used": demucs_applied,
        "key": key_info,
        "bar_offset": bar_offset,
        "beats": [round(float(t), 3) for t in beat_times],
        "segments": segments,
        "beat_chords": raw_beat_chords,
        "waveform": waveform,
    }
