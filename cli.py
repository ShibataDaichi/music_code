"""
コマンドライン（CLI）から直接YouTube URLやローカル音声ファイルを解析し、
コード進行テーブルの表示およびMIDIファイルの出力を行うスクリプト。
"""

from __future__ import annotations

import argparse
import os
import sys
import uuid
from pathlib import Path

from analyzer import analyze_audio_file
from downloader import download_from_url, convert_to_wav
from midi_export import build_midi_bytes


def format_time(sec: float) -> str:
    m = int(sec // 60)
    s = sec % 60
    return f"{m:02d}:{s:05.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="ChordScope CLI - 高精度コード進行・キー・BPM解析ツール")
    parser.add_argument("input", help="YouTube等のURL または ローカル音声ファイルのパス")
    parser.add_argument("--midi", "-m", help="出力するMIDIファイルパス (例: output.mid)", default=None)
    parser.add_argument("--transpose", "-t", type=int, default=0, help="移調する半音数 (-11 〜 +11)")
    args = parser.parse_args()

    cache_dir = Path(__file__).parent / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    job_id = f"cli_{uuid.uuid4().hex[:8]}"

    target = args.input.strip()
    if target.startswith(("http://", "https://")):
        print(f"[*] URLから音声を取得中: {target}")
        meta = download_from_url(target, cache_dir, job_id)
        wav_path = meta["audio_path"]
        title = meta["title"]
    else:
        if not os.path.exists(target):
            print(f"[!] ファイルが見つかりません: {target}", file=sys.stderr)
            sys.exit(1)
        title = Path(target).stem
        wav_path = str(cache_dir / f"{job_id}.wav")
        print(f"[*] 音声ファイルを変換中: {target}")
        convert_to_wav(target, wav_path)

    print("[*] CQTクロマグラム・ビート・キー・コード進行を解析中...")
    result = analyze_audio_file(wav_path)

    print("\n" + "=" * 68)
    print(f" 曲名       : {title}")
    print(f" 推定キー   : {result['key']['key']} (平行調: {result['key']['relative_key']}, 信頼度: {int(result['key']['confidence']*100)}%)")
    print(f" 推定BPM    : {result['bpm']} BPM (A4チューニング偏差: {result['tuning_cents']:+.1f} cents)")
    print(f" 再生時間   : {format_time(result['duration'])}")
    print("=" * 68)
    print(f"{'Time':<16} | {'Bar.Beat':<8} | {'Chord':<12} | {'Roman':<8} | {'Notes'}")
    print("-" * 68)

    for seg in result["segments"]:
        t_range = f"{format_time(seg['start'])}-{format_time(seg['end'])}"
        bar_beat = f"{seg['bar']}.{seg['beat_in_bar']}"
        notes_str = " ".join(seg["notes"])
        print(f"{t_range:<16} | {bar_beat:<8} | {seg['chord']:<12} | {seg['roman']:<8} | {notes_str}")

    if args.midi:
        midi_bytes = build_midi_bytes(result, transpose=args.transpose)
        out_path = Path(args.midi)
        out_path.write_bytes(midi_bytes)
        print("=" * 68)
        print(f"[+] MIDIファイルを出力しました: {out_path.resolve()}")


if __name__ == "__main__":
    main()
