"""
解析されたコード進行データを Standard MIDI File (.mid) に変換・出力するモジュール。
DAWで読み込んだ際にコードネームマーカーと、ベース音＋ボイシング和音の両方が配置されます。
"""

from __future__ import annotations

import io
from typing import Dict, Any, List
import mido

from analyzer import PITCH_NAMES


def build_midi_bytes(analysis: Dict[str, Any], transpose: int = 0, use_simple: bool = False) -> bytes:
    """
    解析結果辞書から MIDI ファイルのバイト列を生成する。
    transpose: 半音単位の移調量 (-11 〜 +11)
    use_simple: True の場合はトライアド・基本7thのみの簡易構成音にする
    """
    mid = mido.MidiFile(ticks_per_beat=480)
    ticks_per_beat = mid.ticks_per_beat

    bpm = float(analysis.get("bpm", 120.0))
    if bpm <= 0:
        bpm = 120.0
    tempo = mido.bpm2tempo(bpm)

    # Track 0: テンポ・拍子・コードマーカー
    meta_track = mido.MidiTrack()
    mid.tracks.append(meta_track)
    meta_track.append(mido.MetaMessage("track_name", name="Chord Markers & Tempo", time=0))
    meta_track.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))
    meta_track.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0))

    # Track 1: コード演奏トラック (Electric Piano / Acoustic Piano)
    chord_track = mido.MidiTrack()
    mid.tracks.append(chord_track)
    chord_track.append(mido.MetaMessage("track_name", name="Chords & Bass Voicing", time=0))
    # Program Change: 4 (Electric Piano 1) or 0 (Acoustic Grand Piano)
    chord_track.append(mido.Message("program_change", program=0, channel=0, time=0))

    def sec_to_ticks(sec: float) -> int:
        return int(round(mido.second2tick(sec, ticks_per_beat, tempo)))

    segments: List[Dict[str, Any]] = analysis.get("segments", [])

    # マーカーイベントとノートイベントを絶対tickで収集してからデルタタイムに変換する
    marker_events = []
    note_events = []  # (tick, type, note, velocity)

    for seg in segments:
        if int(seg.get("root_idx", -1)) < 0 or seg.get("chord") == "N.C.":
            continue
        start_tick = sec_to_ticks(float(seg["start"]))
        end_tick = max(start_tick + 60, sec_to_ticks(float(seg["end"])) - 10)

        root_idx = (int(seg["root_idx"]) + transpose) % 12
        intervals: List[int] = seg.get("intervals", [0, 4, 7])
        if use_simple and len(intervals) > 3:
            # テンション等を簡易化
            intervals = intervals[:4] if seg.get("quality") in ("7", "maj7", "min7", "m7b5") else intervals[:3]

        # 移調後のコード名
        root_name = PITCH_NAMES[root_idx]
        suffix = seg.get("simple_suffix", "") if use_simple else seg.get("suffix", "")
        bass_orig = seg.get("bass", seg.get("root", "C"))
        bass_idx = (PITCH_NAMES.index(bass_orig) + transpose) % 12 if bass_orig in PITCH_NAMES else root_idx
        bass_name = PITCH_NAMES[bass_idx]

        chord_label = f"{root_name}{suffix}"
        if bass_idx != root_idx:
            chord_label = f"{chord_label}/{bass_name}"

        marker_events.append((start_tick, chord_label))

        # MIDIノート番号の構築:
        # ベース音: C2 (36) 〜 B2 (47)
        bass_midi = 36 + bass_idx
        # コード構成音: C4 (60) 付近のボイシング
        chord_midis = [bass_midi]
        for iv in intervals:
            midi_note = 60 + root_idx + iv
            # 高くなりすぎる場合は1オクターブ下げる（クローズド〜ミディアムボイシング）
            if midi_note > 76:
                midi_note -= 12
            if midi_note not in chord_midis:
                chord_midis.append(midi_note)

        for n in chord_midis:
            vel = 88 if n == bass_midi else 74
            note_events.append((start_tick, "note_on", n, vel))
            note_events.append((end_tick, "note_off", n, 0))

    # マーカーイベントを書き出し
    marker_events.sort(key=lambda x: x[0])
    last_tick = 0
    for tick, text in marker_events:
        delta = max(0, tick - last_tick)
        meta_track.append(mido.MetaMessage("marker", text=text, time=delta))
        last_tick = tick
    meta_track.append(mido.MetaMessage("end_of_track", time=0))

    # ノートイベントを書き出し (同tickでは note_off を先に処理)
    note_events.sort(key=lambda x: (x[0], 0 if x[1] == "note_off" else 1))
    last_tick = 0
    for tick, msg_type, note, vel in note_events:
        delta = max(0, tick - last_tick)
        chord_track.append(mido.Message(msg_type, note=note, velocity=vel, channel=0, time=delta))
        last_tick = tick
    chord_track.append(mido.MetaMessage("end_of_track", time=0))

    buf = io.BytesIO()
    mid.save(file=buf)
    return buf.getvalue()
