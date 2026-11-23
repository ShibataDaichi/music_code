"""
曲のコード進行・キー・BPM解析ツール FastAPI バックエンドサーバー (Demucs AI 4ステム音源分離対応)
YouTube等のストリーミングURL・手元の音声ファイル・内蔵デモ音源の解析、
原曲・AI伴奏のみ（Vo/Dr除去）・ベースのみ（Bass Solo）・コード楽器のみ（Other Solo）・ドラム・ボーカルの
パート別ストリーミング再生＆WAV保存、およびMIDIファイル出力を提供します。
"""

from __future__ import annotations

import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Request
from fastapi.responses import FileResponse, Response, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from analyzer import analyze_audio_file
from downloader import download_from_url, convert_to_wav, generate_demo_song, DEMO_PRESETS
from midi_export import build_midi_bytes
from stem_separator import separate_stems_for_analysis

BASE_DIR = Path(__file__).parent.resolve()
CACHE_DIR = BASE_DIR / "cache"
STATIC_DIR = BASE_DIR / "static"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)

VALID_STEMS = ("backing", "bass", "other", "drums", "vocals")

app = FastAPI(
    title="ChordScope Studio - AI音源分離(Demucs)＆ベース抽出対応コード進行解析ツール",
    version="2.1.0",
)


@app.middleware("http")
async def disable_static_cache(request: Request, call_next):
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


JOBS_STORE: Dict[str, Dict[str, Any]] = {}


def _attach_stem_info(job_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
    backing_path = CACHE_DIR / f"{job_id}_backing.wav"
    bass_path = CACHE_DIR / f"{job_id}_bass.wav"
    has_stems = backing_path.exists() or bass_path.exists() or bool(data.get("analysis", {}).get("demucs_used"))
    data["has_backing"] = has_stems
    data["has_bass"] = bass_path.exists() or has_stems
    data["backing_audio_url"] = f"/api/audio/{job_id}/backing" if has_stems else None
    data["bass_audio_url"] = f"/api/audio/{job_id}/bass" if has_stems else None
    data["stem_urls"] = (
        {
            "mix": f"/api/audio/{job_id}",
            "backing": f"/api/audio/{job_id}/backing",
            "bass": f"/api/audio/{job_id}/bass",
            "other": f"/api/audio/{job_id}/other",
            "drums": f"/api/audio/{job_id}/drums",
            "vocals": f"/api/audio/{job_id}/vocals",
        }
        if has_stems
        else {"mix": f"/api/audio/{job_id}"}
    )
    return data


def _save_job(job_id: str, data: Dict[str, Any]) -> None:
    _attach_stem_info(job_id, data)
    JOBS_STORE[job_id] = data
    json_path = CACHE_DIR / f"{job_id}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def _load_job(job_id: str) -> Optional[Dict[str, Any]]:
    json_path = CACHE_DIR / f"{job_id}.json"
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            _attach_stem_info(job_id, data)
            JOBS_STORE[job_id] = data
            return data
    return JOBS_STORE.get(job_id)


class UrlAnalyzeRequest(BaseModel):
    url: str
    use_demucs: bool = True


class DemoAnalyzeRequest(BaseModel):
    preset: str = "royal_road"


@app.get("/", response_class=HTMLResponse)
async def index() -> FileResponse:
    return FileResponse(
        STATIC_DIR / "index.html",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
    )


@app.post("/api/analyze/url")
def api_analyze_url(req: UrlAnalyzeRequest) -> Dict[str, Any]:
    url = req.url.strip()
    if not url:
        raise HTTPException(status_code=400, detail="URLを入力してください。")

    job_id = uuid.uuid4().hex[:10]
    backing_wav = str(CACHE_DIR / f"{job_id}_backing.wav")
    try:
        meta = download_from_url(url=url, output_dir=CACHE_DIR, job_id=job_id)
        analysis = analyze_audio_file(
            meta["audio_path"],
            use_demucs=req.use_demucs,
            backing_wav_out=backing_wav if req.use_demucs else None,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"URLからの音源取得または解析に失敗しました: {str(exc)}",
        ) from exc

    job_data = {
        "job_id": job_id,
        "meta": {
            "title": meta["title"],
            "uploader": meta["uploader"],
            "thumbnail": meta["thumbnail"],
            "webpage_url": meta["webpage_url"],
            "video_id": meta["video_id"],
            "source_type": meta["source_type"],
        },
        "audio_url": f"/api/audio/{job_id}",
        "midi_url": f"/api/midi/{job_id}",
        "analysis": analysis,
    }
    _save_job(job_id, job_data)
    return job_data


@app.post("/api/analyze/upload")
async def api_analyze_upload(
    file: UploadFile = File(...),
    use_demucs: bool = Query(True),
) -> Dict[str, Any]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="ファイルが選択されていません。")

    job_id = uuid.uuid4().hex[:10]
    ext = Path(file.filename).suffix or ".tmp"
    raw_path = CACHE_DIR / f"{job_id}_upload{ext}"
    wav_path = CACHE_DIR / f"{job_id}.wav"
    backing_wav = str(CACHE_DIR / f"{job_id}_backing.wav")

    try:
        with open(raw_path, "wb") as out_f:
            shutil.copyfileobj(file.file, out_f)
        convert_to_wav(str(raw_path), str(wav_path))
        if raw_path.exists() and raw_path != wav_path:
            raw_path.unlink(missing_ok=True)

        analysis = analyze_audio_file(
            str(wav_path),
            use_demucs=use_demucs,
            backing_wav_out=backing_wav if use_demucs else None,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"音声ファイルの解析に失敗しました: {str(exc)}",
        ) from exc

    title = Path(file.filename).stem
    job_data = {
        "job_id": job_id,
        "meta": {
            "title": title,
            "uploader": "Local Audio File",
            "thumbnail": "",
            "webpage_url": "",
            "video_id": "",
            "source_type": "upload",
        },
        "audio_url": f"/api/audio/{job_id}",
        "midi_url": f"/api/midi/{job_id}",
        "analysis": analysis,
    }
    _save_job(job_id, job_data)
    return job_data


@app.post("/api/analyze/demo")
def api_analyze_demo(req: DemoAnalyzeRequest) -> Dict[str, Any]:
    preset_key = req.preset if req.preset in DEMO_PRESETS else "royal_road"
    job_id = f"demo_{preset_key}"

    existing = _load_job(job_id)
    wav_file = CACHE_DIR / f"{job_id}.wav"
    if existing and wav_file.exists():
        return existing

    try:
        meta = generate_demo_song(preset_key=preset_key, output_dir=CACHE_DIR, job_id=job_id)
        analysis = analyze_audio_file(meta["audio_path"], use_demucs=False)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"デモ音源の生成・解析に失敗しました: {str(exc)}") from exc

    job_data = {
        "job_id": job_id,
        "meta": {
            "title": meta["title"],
            "uploader": meta["uploader"],
            "thumbnail": "",
            "webpage_url": "",
            "video_id": "",
            "source_type": "demo",
        },
        "audio_url": f"/api/audio/{job_id}",
        "midi_url": f"/api/midi/{job_id}",
        "analysis": analysis,
    }
    _save_job(job_id, job_data)
    return job_data


@app.post("/api/jobs/{job_id}/reanalyze_demucs")
def api_reanalyze_with_demucs(job_id: str) -> Dict[str, Any]:
    """
    キャッシュ済みの楽曲に対して Demucs AI 4ステム音源分離を実行し、
    ベース単体・伴奏単体・コード楽器単体などの各ステム音源の生成と最高精度のコード再解析を行う。
    """
    job = _load_job(job_id)
    wav_path = CACHE_DIR / f"{job_id}.wav"
    if not job or not wav_path.exists():
        raise HTTPException(status_code=404, detail="対象の音声データが見つかりません。")

    backing_wav = str(CACHE_DIR / f"{job_id}_backing.wav")
    analysis = analyze_audio_file(
        str(wav_path),
        use_demucs=True,
        backing_wav_out=backing_wav,
    )
    job["analysis"] = analysis
    _save_job(job_id, job)
    return job


@app.get("/api/audio/{job_id}")
def api_get_audio(job_id: str, download: bool = Query(False)) -> FileResponse:
    wav_path = CACHE_DIR / f"{job_id}.wav"
    if not wav_path.exists():
        raise HTTPException(status_code=404, detail="音声ファイルが見つかりません。")
    headers = {}
    if download:
        job = _load_job(job_id)
        title = job["meta"]["title"] if job else job_id
        safe_name = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in title)[:40] or job_id
        headers["Content-Disposition"] = f'attachment; filename="{safe_name}_mix.wav"'
    return FileResponse(wav_path, media_type="audio/wav", headers=headers)


@app.get("/api/audio/{job_id}/{stem}")
def api_get_stem_audio(
    job_id: str,
    stem: str,
    download: bool = Query(False),
) -> FileResponse:
    if stem not in VALID_STEMS:
        raise HTTPException(status_code=400, detail=f"未対応のステム名です: {stem}")

    stem_path = CACHE_DIR / f"{job_id}_{stem}.wav"
    raw_wav_path = CACHE_DIR / f"{job_id}.wav"

    # もしステムWAVがまだ生成されていない場合は、原曲WAVからオンデマンドで4ステム分離を実行して保存する
    if not stem_path.exists():
        if not raw_wav_path.exists():
            raise HTTPException(status_code=404, detail="元となる音声ファイルが見つかりません。")
        backing_wav = str(CACHE_DIR / f"{job_id}_backing.wav")
        separate_stems_for_analysis(
            audio_path=str(raw_wav_path),
            backing_wav_out=backing_wav,
        )

    if not stem_path.exists():
        raise HTTPException(status_code=404, detail=f"AI分離ステム ({stem}) が見つかりません。")

    headers = {}
    if download:
        job = _load_job(job_id)
        title = job["meta"]["title"] if job else job_id
        safe_name = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in title)[:40] or job_id
        headers["Content-Disposition"] = f'attachment; filename="{safe_name}_{stem}.wav"'
    return FileResponse(stem_path, media_type="audio/wav", headers=headers)


@app.get("/api/midi/{job_id}")
def api_get_midi(
    job_id: str,
    transpose: int = Query(0, ge=-12, le=12),
    simple: bool = Query(False),
) -> Response:
    job = _load_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="解析データが見つかりません。")

    midi_bytes = build_midi_bytes(job["analysis"], transpose=transpose, use_simple=simple)
    safe_name = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in job["meta"]["title"])[:40] or job_id
    filename = f"{safe_name}_chords.mid"

    return Response(
        content=midi_bytes,
        media_type="audio/midi",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/api/jobs")
def api_list_jobs() -> List[Dict[str, Any]]:
    items = []
    seen_titles = set()
    for json_file in sorted(CACHE_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                d = json.load(f)
                title = d["meta"]["title"]
                if title in seen_titles:
                    continue
                seen_titles.add(title)
                items.append({
                    "job_id": d["job_id"],
                    "title": title,
                    "uploader": d["meta"]["uploader"],
                    "source_type": d["meta"]["source_type"],
                    "demucs_used": d["analysis"].get("demucs_used", False),
                    "key": d["analysis"]["key"]["key"],
                    "bpm": d["analysis"]["bpm"],
                    "duration": d["analysis"]["duration"],
                })
        except Exception:
            continue
    return items


@app.get("/api/jobs/{job_id}")
def api_get_job(job_id: str) -> Dict[str, Any]:
    job = _load_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="指定されたジョブが見つかりません。")
    return job


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
