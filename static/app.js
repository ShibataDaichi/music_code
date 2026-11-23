/**
 * ChordScope Studio フロントエンドアプリケーション (v7)
 * - AI 4ステム音源分離 (Demucs): 原曲 / 伴奏のみ / ベースのみ抽出 / コード楽器のみ / ドラムのみ / ボーカルのみ のリアルタイム切替＆WAV保存
 * - 1段4小節（4 Bars per Row）のコード譜ビュー＆小節内スマート整理機能
 * - 小節頭（拍）シフト微調整機能（1拍単位で小節線を即座に再計算）
 * - 波形描画・再生同期ハイライト・2オクターブ鍵盤構成音表示・移調・MIDI出力
 */

const PITCH_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"];

const STEM_LABELS = {
  mix: "原曲フルミックス",
  backing: "伴奏のみ(Vo/Dr除去)",
  bass: "ベースのみ(Bass Solo)",
  other: "コード楽器のみ(Gu/Key)",
  drums: "ドラムのみ(Drums)",
  vocals: "ボーカルのみ(Vocals)",
};

const state = {
  jobData: null,
  activeSegments: [], // beatShift 適用後の現在有効なセグメント配列
  transpose: 0,       // 半音単位の移調量 (-11 〜 +11)
  beatShift: 0,       // 拍単位の小節頭オフセット調整 (-3 〜 +3)
  currentStem: "mix", // "mix" | "backing" | "bass" | "other" | "drums" | "vocals"
  mode: "rich",       // "rich" (7th/テンション含む) | "simple" (シンプル三和音)
  viewMode: "4bar",   // "4bar" (1段4小節譜面ビュー) | "cards" (詳細カード一覧)
  cleanBar: true,     // 小節内スマート整理（1小節最大1〜2コードに集約）
  hideNC: false,      // すべて N.C. の小節を省略
  activeSegIndex: -1,
  rafId: null,
};

const els = {
  urlInput: document.getElementById("urlInput"),
  btnAnalyzeUrl: document.getElementById("btnAnalyzeUrl"),
  historySelect: document.getElementById("historySelect"),
  dropZone: document.getElementById("dropZone"),
  fileInput: document.getElementById("fileInput"),
  statusBanner: document.getElementById("statusBanner"),
  statusText: document.getElementById("statusText"),
  sourceBadge: document.getElementById("sourceBadge"),
  trackTitle: document.getElementById("trackTitle"),
  trackSub: document.getElementById("trackSub"),
  metricKey: document.getElementById("metricKey"),
  metricRelative: document.getElementById("metricRelative"),
  metricBpm: document.getElementById("metricBpm"),
  metricTuning: document.getElementById("metricTuning"),
  btnTransDown: document.getElementById("btnTransDown"),
  btnTransUp: document.getElementById("btnTransUp"),
  transposeValue: document.getElementById("transposeValue"),
  capoHint: document.getElementById("capoHint"),
  btnModeRich: document.getElementById("btnModeRich"),
  btnModeSimple: document.getElementById("btnModeSimple"),
  btnView4Bar: document.getElementById("btnView4Bar"),
  btnViewCards: document.getElementById("btnViewCards"),
  btnBeatShiftLeft: document.getElementById("btnBeatShiftLeft"),
  btnBeatShiftRight: document.getElementById("btnBeatShiftRight"),
  beatShiftValue: document.getElementById("beatShiftValue"),
  btnRunDemucs: document.getElementById("btnRunDemucs"),
  stemBar: document.getElementById("stemBar"),
  audioStemToggle: document.getElementById("audioStemToggle"),
  btnDownloadStem: document.getElementById("btnDownloadStem"),
  chkCleanBar: document.getElementById("chkCleanBar"),
  chkHideNC: document.getElementById("chkHideNC"),
  prevChordDisplay: document.getElementById("prevChordDisplay"),
  prevRomanDisplay: document.getElementById("prevRomanDisplay"),
  currentBarBeat: document.getElementById("currentBarBeat"),
  currentChordDisplay: document.getElementById("currentChordDisplay"),
  currentRomanDisplay: document.getElementById("currentRomanDisplay"),
  currentBassDisplay: document.getElementById("currentBassDisplay"),
  currentNotesDisplay: document.getElementById("currentNotesDisplay"),
  nextChordDisplay: document.getElementById("nextChordDisplay"),
  nextRomanDisplay: document.getElementById("nextRomanDisplay"),
  keyboardChordLabel: document.getElementById("keyboardChordLabel"),
  pianoContainer: document.getElementById("pianoContainer"),
  waveformWrapper: document.getElementById("waveformWrapper"),
  waveformCanvas: document.getElementById("waveformCanvas"),
  btnPlayPause: document.getElementById("btnPlayPause"),
  playIcon: document.getElementById("playIcon"),
  btnStop: document.getElementById("btnStop"),
  currentTimeText: document.getElementById("currentTimeText"),
  totalTimeText: document.getElementById("totalTimeText"),
  speedSelect: document.getElementById("speedSelect"),
  chkAutoScroll: document.getElementById("chkAutoScroll"),
  audioPlayer: document.getElementById("audioPlayer"),
  chordSheetContainer:
    document.getElementById("chordSheetContainer") ||
    document.getElementById("chordGrid"),
  btnCopyChords: document.getElementById("btnCopyChords"),
  btnExportMidi: document.getElementById("btnExportMidi"),
};

function formatTime(sec, includeTenths = true) {
  if (!Number.isFinite(sec) || sec < 0) return includeTenths ? "00:00.0" : "00:00";
  const m = Math.floor(sec / 60);
  if (includeTenths) {
    const s = (sec % 60).toFixed(1);
    return `${String(m).padStart(2, "0")}:${s.padStart(4, "0")}`;
  } else {
    const s = Math.floor(sec % 60);
    return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
  }
}

function initPianoKeyboard() {
  els.pianoContainer.innerHTML = "";
  const whiteNotes = [0, 2, 4, 5, 7, 9, 11];
  const blackMap = {
    1: 0,
    3: 1,
    6: 3,
    8: 4,
    10: 5,
  };

  const totalWhiteKeys = 14;
  for (let oct = 0; oct < 2; oct++) {
    for (let wIdx = 0; wIdx < whiteNotes.length; wIdx++) {
      const pc = whiteNotes[wIdx];
      const keyIdx = oct * 12 + pc;
      const div = document.createElement("div");
      div.className = "piano-key-white";
      div.dataset.pc = String(pc);
      div.dataset.keyIdx = String(keyIdx);
      div.textContent = PITCH_NAMES[pc];
      els.pianoContainer.appendChild(div);
    }
  }

  for (let oct = 0; oct < 2; oct++) {
    for (const [pcStr, whiteOffset] of Object.entries(blackMap)) {
      const pc = Number(pcStr);
      const keyIdx = oct * 12 + pc;
      const globalWhiteIndex = oct * 7 + whiteOffset;
      const leftPercent = ((globalWhiteIndex + 1) / totalWhiteKeys) * 100 - 2.6;

      const div = document.createElement("div");
      div.className = "piano-key-black";
      div.style.left = `${leftPercent}%`;
      div.dataset.pc = String(pc);
      div.dataset.keyIdx = String(keyIdx);
      div.textContent = PITCH_NAMES[pc];
      els.pianoContainer.appendChild(div);
    }
  }
}

function rebuildSegmentsFromBeatShift() {
  if (!state.jobData) {
    state.activeSegments = [];
    return;
  }
  const { segments, beat_chords } = state.jobData.analysis;
  if (state.beatShift === 0 || !beat_chords || !beat_chords.length) {
    state.activeSegments = segments;
    return;
  }

  const baseFirstBeatPos = (beat_chords[0].bar - 1) * 4 + (beat_chords[0].beat_in_bar - 1);
  const rebuilt = [];

  beat_chords.forEach((bc, i) => {
    const pos = Math.max(0, baseFirstBeatPos + i + state.beatShift);
    const barNum = Math.floor(pos / 4) + 1;
    const beatInBar = (pos % 4) + 1;

    if (
      rebuilt.length > 0 &&
      rebuilt[rebuilt.length - 1].chord === bc.chord &&
      rebuilt[rebuilt.length - 1].bar === barNum
    ) {
      const prev = rebuilt[rebuilt.length - 1];
      prev.end = bc.end;
      prev.beat_end = bc.beat_index + 1;
      prev.beats_count = (prev.beats_count || 1) + 1;
      prev.duration = Number((prev.end - prev.start).toFixed(3));
      prev.confidence = Number(((prev.confidence + bc.confidence) / 2).toFixed(3));
    } else {
      rebuilt.push({
        index: rebuilt.length,
        start: bc.start,
        end: bc.end,
        duration: Number((bc.end - bc.start).toFixed(3)),
        beat_start: bc.beat_index,
        beat_end: bc.beat_index + 1,
        beats_count: 1,
        bar: barNum,
        beat_in_bar: beatInBar,
        chord: bc.chord,
        simple_chord: bc.simple_chord,
        root: bc.root,
        root_idx: bc.root_idx,
        quality: bc.quality,
        suffix: bc.suffix,
        simple_suffix: bc.simple_suffix || bc.suffix,
        bass: bc.bass,
        notes: bc.notes,
        intervals: bc.intervals,
        roman: bc.roman,
        confidence: bc.confidence,
      });
    }
  });

  state.activeSegments = rebuilt;
}

function getTransposedSegmentInfo(seg) {
  if (!seg) return null;
  if (seg.root_idx < 0 || seg.chord === "N.C.") {
    return {
      chordName: "N.C.",
      rootIdx: -1,
      rootName: "N.C.",
      bassIdx: -1,
      bassName: "—",
      notes: [],
      noteIndices: [],
      roman: "N.C.",
      bar: seg.bar,
      beat_in_bar: seg.beat_in_bar,
      start: seg.start,
      end: seg.end,
      beats_count: seg.beats_count || 1,
      index: seg.index,
    };
  }

  const t = state.transpose;
  const rootIdx = ((seg.root_idx + t) % 12 + 12) % 12;
  const rootName = PITCH_NAMES[rootIdx];

  const origBassIdx = PITCH_NAMES.indexOf(seg.bass);
  const bassIdx = origBassIdx >= 0 ? ((origBassIdx + t) % 12 + 12) % 12 : rootIdx;
  const bassName = PITCH_NAMES[bassIdx];

  const suffix = state.mode === "simple" ? seg.simple_suffix : seg.suffix;
  let chordName = `${rootName}${suffix}`;
  if (bassIdx !== rootIdx && state.mode === "rich") {
    chordName += `/${bassName}`;
  }

  const intervals =
    state.mode === "simple" && seg.intervals.length > 3
      ? seg.intervals.slice(0, 3)
      : seg.intervals;

  const noteIndices = intervals.map((iv) => (rootIdx + iv) % 12);
  const notes = noteIndices.map((idx) => PITCH_NAMES[idx]);

  return {
    chordName,
    rootIdx,
    rootName,
    bassIdx,
    bassName,
    notes,
    noteIndices,
    roman: seg.roman,
    bar: seg.bar,
    beat_in_bar: seg.beat_in_bar,
    start: seg.start,
    end: seg.end,
    beats_count: seg.beats_count || Math.max(1, (seg.beat_end || 1) - (seg.beat_start || 0)),
    index: seg.index,
  };
}

function consolidateBarSegments(barSegs) {
  if (!state.cleanBar || barSegs.length <= 2) {
    const merged = [];
    barSegs.forEach((s) => {
      const info = getTransposedSegmentInfo(s);
      if (merged.length > 0 && merged[merged.length - 1].info.chordName === info.chordName) {
        const prev = merged[merged.length - 1];
        prev.end = s.end;
        prev.beats_count += info.beats_count;
        prev.segIndices.push(s.index);
      } else {
        merged.push({
          start: s.start,
          end: s.end,
          beats_count: info.beats_count,
          info,
          segIndices: [s.index],
        });
      }
    });
    return merged;
  }

  const firstHalf = barSegs.filter((s) => s.beat_in_bar <= 2);
  const secondHalf = barSegs.filter((s) => s.beat_in_bar >= 3);

  const pickDominant = (list) => {
    if (list.length === 0) return barSegs[0];
    return list.reduce((best, cur) => {
      const bScore = (best.beats_count || 1) * (best.confidence || 0.5);
      const cScore = (cur.beats_count || 1) * (cur.confidence || 0.5);
      return cScore > bScore ? cur : best;
    }, list[0]);
  };

  const dom1 = pickDominant(firstHalf);
  const dom2 = pickDominant(secondHalf.length ? secondHalf : firstHalf);
  const info1 = getTransposedSegmentInfo(dom1);
  const info2 = getTransposedSegmentInfo(dom2);

  const allIndices = barSegs.map((s) => s.index);

  if (info1.chordName === info2.chordName) {
    return [
      {
        start: barSegs[0].start,
        end: barSegs[barSegs.length - 1].end,
        beats_count: 4,
        info: info1,
        segIndices: allIndices,
      },
    ];
  } else {
    const firstIndices = firstHalf.length ? firstHalf.map((s) => s.index) : [barSegs[0].index];
    const secondIndices = secondHalf.length ? secondHalf.map((s) => s.index) : [barSegs[barSegs.length - 1].index];
    const splitTime = secondHalf.length ? secondHalf[0].start : (barSegs[0].start + barSegs[barSegs.length - 1].end) / 2;
    return [
      {
        start: barSegs[0].start,
        end: splitTime,
        beats_count: 2,
        info: info1,
        segIndices: firstIndices,
      },
      {
        start: splitTime,
        end: barSegs[barSegs.length - 1].end,
        beats_count: 2,
        info: info2,
        segIndices: secondIndices,
      },
    ];
  }
}

function updatePianoHighlight(segInfo) {
  const allKeys = els.pianoContainer.querySelectorAll("[data-key-idx]");
  allKeys.forEach((k) => {
    k.classList.remove("active-root", "active-tone");
  });

  if (!segInfo || segInfo.rootIdx < 0) {
    els.keyboardChordLabel.textContent = "N.C. (No Chord)";
    return;
  }

  els.keyboardChordLabel.textContent = `${segInfo.chordName} [Bass: ${segInfo.bassName}] (${segInfo.notes.join(" - ")})`;

  allKeys.forEach((k) => {
    const pc = Number(k.dataset.pc);
    const keyIdx = Number(k.dataset.keyIdx);

    if (keyIdx < 12 && pc === segInfo.bassIdx) {
      k.classList.add("active-root");
    } else if (segInfo.noteIndices.includes(pc)) {
      if (pc === segInfo.rootIdx && keyIdx < 12) {
        k.classList.add("active-root");
      } else if (keyIdx >= 12 || pc === segInfo.rootIdx) {
        k.classList.add(pc === segInfo.rootIdx ? "active-root" : "active-tone");
      }
    }
  });
}

function updateStemButtonsUI() {
  if (!els.audioStemToggle) return;
  const isDemo = state.jobData && state.jobData.meta && state.jobData.meta.source_type === "demo";
  const btns = els.audioStemToggle.querySelectorAll("[data-stem]");
  btns.forEach((btn) => {
    const stem = btn.dataset.stem;
    btn.classList.toggle("active", stem === state.currentStem);
    btn.disabled = !state.jobData || (isDemo && stem !== "mix");
  });

  if (els.btnDownloadStem) {
    els.btnDownloadStem.disabled = !state.jobData;
    const label = STEM_LABELS[state.currentStem] || "原曲";
    els.btnDownloadStem.textContent = `⬇ ${label} (.wav) を保存`;
  }
}

function renderSummary() {
  if (!state.jobData) return;
  const { meta, analysis } = state.jobData;

  const isDemucs = Boolean(analysis.demucs_used);
  els.sourceBadge.textContent = isDemucs
    ? "✨ AI DEMUCS 4-STEM"
    : (meta.source_type || "AUDIO").toUpperCase();
  els.trackTitle.textContent = meta.title || "Untitled Track";
  const engineLabel = isDemucs ? "AI 4ステム分離・ベース抽出対応" : "M/S信号処理";
  els.trackSub.textContent = `${meta.uploader || ""} • 長さ ${formatTime(analysis.duration)} • [${engineLabel}]`;

  if (els.btnRunDemucs) {
    if (!isDemucs && meta.source_type !== "demo") {
      els.btnRunDemucs.classList.remove("hidden");
    } else {
      els.btnRunDemucs.classList.add("hidden");
    }
  }

  updateStemButtonsUI();

  const origKeyRootIdx = analysis.key.root_idx;
  const transKeyRootIdx = ((origKeyRootIdx + state.transpose) % 12 + 12) % 12;
  const transKeyName = `${PITCH_NAMES[transKeyRootIdx]} ${analysis.key.scale}`;

  const relOffset = analysis.key.scale === "Major" ? 9 : 3;
  const relScale = analysis.key.scale === "Major" ? "Minor" : "Major";
  const transRelName = `${PITCH_NAMES[(transKeyRootIdx + relOffset) % 12]} ${relScale}`;

  els.metricKey.textContent = transKeyName;
  els.metricRelative.textContent = `平行調: ${transRelName} (信頼度 ${Math.round(analysis.key.confidence * 100)}%)`;
  els.metricBpm.textContent = `${analysis.bpm} BPM`;
  const sign = analysis.tuning_cents >= 0 ? "+" : "";
  els.metricTuning.textContent = `A4偏差: ${sign}${analysis.tuning_cents} cents`;

  const t = state.transpose;
  els.transposeValue.textContent = t === 0 ? "±0" : t > 0 ? `+${t}` : `${t}`;
  if (t === 0) {
    els.capoHint.textContent = "原曲キー表示";
  } else if (t < 0) {
    els.capoHint.textContent = `ギター Capo ${Math.abs(t)} 相当のコード形`;
  } else {
    els.capoHint.textContent = `${t}半音キー上げ表示`;
  }

  if (els.beatShiftValue) {
    const bs = state.beatShift;
    els.beatShiftValue.textContent = bs === 0 ? "0拍" : bs > 0 ? `+${bs}拍` : `${bs}拍`;
  }
}

function renderChordSheet() {
  if (!state.jobData || !els.chordSheetContainer) return;
  rebuildSegmentsFromBeatShift();
  const segments = state.activeSegments;
  els.chordSheetContainer.innerHTML = "";

  if (state.viewMode === "cards") {
    const grid = document.createElement("div");
    grid.className = "chord-grid";
    segments.forEach((seg, idx) => {
      const info = getTransposedSegmentInfo(seg);
      if (state.hideNC && info.chordName === "N.C.") return;
      const card = document.createElement("div");
      card.className = "chord-card";
      if (idx === state.activeSegIndex) card.classList.add("active");
      card.dataset.segIndices = JSON.stringify([idx]);

      card.innerHTML = `
        <div class="card-top">
          <span>Bar ${seg.bar}.${seg.beat_in_bar} (${formatTime(seg.start)})</span>
          <span class="card-roman">${info.roman}</span>
        </div>
        <div class="card-chord-name">${info.chordName}</div>
        <div class="card-notes">${info.notes.length ? `Bass: ${info.bassName} | ${info.notes.join(" · ")}` : "無音 / トーク"}</div>
      `;
      card.addEventListener("click", () => seekAndPlay(seg.start));
      grid.appendChild(card);
    });
    els.chordSheetContainer.appendChild(grid);
    return;
  }

  const barsMap = new Map();
  segments.forEach((seg) => {
    if (!barsMap.has(seg.bar)) {
      barsMap.set(seg.bar, []);
    }
    barsMap.get(seg.bar).push(seg);
  });

  let barNumbers = Array.from(barsMap.keys()).sort((a, b) => a - b);

  let leadingNcBars = [];
  while (barNumbers.length > 0) {
    const firstB = barNumbers[0];
    const bSegs = barsMap.get(firstB);
    if (bSegs.every((s) => s.chord === "N.C.")) {
      leadingNcBars.push(barNumbers.shift());
    } else {
      break;
    }
  }

  if (leadingNcBars.length > 0 && !state.hideNC) {
    const firstNcSeg = barsMap.get(leadingNcBars[0])[0];
    const lastNcSegs = barsMap.get(leadingNcBars[leadingNcBars.length - 1]);
    const lastNcSeg = lastNcSegs[lastNcSegs.length - 1];
    const banner = document.createElement("div");
    banner.className = "nc-intro-banner";
    banner.innerHTML = `
      <span>🔇 冒頭 無音・MC区間 (${formatTime(firstNcSeg.start, false)} 〜 ${formatTime(lastNcSeg.end, false)} / ${leadingNcBars.length}小節分 N.C.)</span>
      <span>クリックで冒頭から再生 ▶</span>
    `;
    banner.addEventListener("click", () => seekAndPlay(firstNcSeg.start));
    els.chordSheetContainer.appendChild(banner);
  }

  if (state.hideNC) {
    barNumbers = barNumbers.filter((bNum) => {
      const bSegs = barsMap.get(bNum);
      return bSegs.some((s) => s.chord !== "N.C.");
    });
  }

  for (let i = 0; i < barNumbers.length; i += 4) {
    const rowBarNums = barNumbers.slice(i, i + 4);
    const firstBarSegs = barsMap.get(rowBarNums[0]);
    const rowStartSec = firstBarSegs[0].start;
    const displayStartBar = i + 1;
    const displayEndBar = i + rowBarNums.length;

    const rowEl = document.createElement("div");
    rowEl.className = "sheet-row";

    const labelEl = document.createElement("div");
    labelEl.className = "sheet-row-label";
    labelEl.innerHTML = `
      <span class="row-bar-range">${displayStartBar}–${displayEndBar}小節</span>
      <span class="row-time-start">${formatTime(rowStartSec, false)}</span>
    `;
    rowEl.appendChild(labelEl);

    const barsWrap = document.createElement("div");
    barsWrap.className = "sheet-row-bars";

    rowBarNums.forEach((bNum, offsetIdx) => {
      const displayBarNum = i + offsetIdx + 1;
      const rawSegs = barsMap.get(bNum);
      const slots = consolidateBarSegments(rawSegs);

      const barCell = document.createElement("div");
      barCell.className = "bar-cell";
      barCell.dataset.bar = String(bNum);

      const barHeader = document.createElement("div");
      barHeader.className = "bar-header";
      barHeader.innerHTML = `
        <span class="bar-num">${displayBarNum}小節</span>
        <span>${formatTime(rawSegs[0].start, false)}</span>
      `;
      barCell.appendChild(barHeader);

      const slotsWrap = document.createElement("div");
      slotsWrap.className = "bar-slots";

      slots.forEach((slot) => {
        const slotEl = document.createElement("div");
        slotEl.className = "chord-slot";
        if (slot.info.chordName === "N.C.") {
          slotEl.classList.add("slot-nc");
        }
        slotEl.style.flex = `${Math.max(1, slot.beats_count)} 1 0%`;
        slotEl.dataset.segIndices = JSON.stringify(slot.segIndices);

        const notesStr = slot.info.notes.length ? slot.info.notes.join("·") : "";
        slotEl.innerHTML = `
          <div class="slot-top-row">
            <span class="slot-chord-name">${slot.info.chordName}</span>
            <span class="slot-roman">${slot.info.roman !== "N.C." ? slot.info.roman : ""}</span>
          </div>
          <div class="slot-notes">${notesStr}</div>
        `;

        slotEl.addEventListener("click", () => seekAndPlay(slot.start));
        slotsWrap.appendChild(slotEl);
      });

      barCell.appendChild(slotsWrap);
      barsWrap.appendChild(barCell);
    });

    rowEl.appendChild(barsWrap);
    els.chordSheetContainer.appendChild(rowEl);
  }
}

function seekAndPlay(targetSec) {
  els.audioPlayer.currentTime = targetSec;
  updateActiveMonitor(targetSec, true);
  if (els.audioPlayer.paused) {
    els.audioPlayer.play().catch(() => {});
  }
}

function drawWaveform() {
  const canvas = els.waveformCanvas;
  const rect = els.waveformWrapper.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  if (canvas.width !== Math.floor(rect.width * dpr) || canvas.height !== Math.floor(rect.height * dpr)) {
    canvas.width = Math.floor(rect.width * dpr);
    canvas.height = Math.floor(rect.height * dpr);
  }

  const ctx = canvas.getContext("2d");
  ctx.save();
  ctx.scale(dpr, dpr);

  const w = rect.width;
  const h = rect.height;
  ctx.clearRect(0, 0, w, h);

  if (!state.jobData) {
    ctx.fillStyle = "#475569";
    ctx.font = "12px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("音源が読み込まれると波形とコード境界がここに表示されます", w / 2, h / 2 + 4);
    ctx.restore();
    return;
  }

  const { waveform, duration } = state.jobData.analysis;
  const segments = state.activeSegments;
  const currentTime = els.audioPlayer.currentTime || 0;
  const progressRatio = duration > 0 ? Math.min(1, Math.max(0, currentTime / duration)) : 0;
  const playX = progressRatio * w;

  const ribbonHeight = 20;
  const waveAreaHeight = h - ribbonHeight;

  segments.forEach((seg, idx) => {
    const x1 = (seg.start / duration) * w;
    const x2 = (seg.end / duration) * w;
    const bw = Math.max(1, x2 - x1);

    ctx.fillStyle = idx === state.activeSegIndex ? "rgba(56, 189, 248, 0.32)" : idx % 2 === 0 ? "#131926" : "#182032";
    ctx.fillRect(x1, waveAreaHeight, bw, ribbonHeight);

    ctx.strokeStyle = "#2a344d";
    ctx.beginPath();
    ctx.moveTo(x1, 0);
    ctx.lineTo(x1, h);
    ctx.stroke();

    if (bw > 28) {
      const info = getTransposedSegmentInfo(seg);
      ctx.fillStyle = idx === state.activeSegIndex ? "#ffffff" : "#94a3b8";
      ctx.font = "bold 10px -apple-system, BlinkMacSystemFont, sans-serif";
      ctx.textAlign = "left";
      ctx.fillText(info.chordName, x1 + 4, h - 6);
    }
  });

  const nPoints = waveform.length;
  const barStep = w / nPoints;
  const midY = waveAreaHeight / 2;

  const activeWaveColor = state.currentStem === "bass" ? "#34d399" : "#38bdf8";
  for (let i = 0; i < nPoints; i++) {
    const x = i * barStep;
    const amp = waveform[i] * (waveAreaHeight * 0.42);
    ctx.fillStyle = x <= playX ? activeWaveColor : "#475569";
    ctx.fillRect(x, midY - amp, Math.max(1, barStep - 0.5), amp * 2);
  }

  ctx.strokeStyle = "#fbbf24";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(playX, 0);
  ctx.lineTo(playX, h);
  ctx.stroke();

  ctx.restore();
}

function updateActiveMonitor(currentTime, forceScroll = false) {
  els.currentTimeText.textContent = formatTime(currentTime);
  drawWaveform();

  if (!state.jobData) return;
  const segments = state.activeSegments;
  if (!segments || segments.length === 0) return;

  let idx = segments.findIndex((s) => currentTime >= s.start && currentTime < s.end);
  if (idx === -1) {
    idx = currentTime >= segments[segments.length - 1].end ? segments.length - 1 : 0;
  }

  if (idx !== state.activeSegIndex || forceScroll) {
    state.activeSegIndex = idx;

    const prevSeg = idx > 0 ? getTransposedSegmentInfo(segments[idx - 1]) : null;
    const currSeg = getTransposedSegmentInfo(segments[idx]);
    const nextSeg = idx + 1 < segments.length ? getTransposedSegmentInfo(segments[idx + 1]) : null;

    els.prevChordDisplay.textContent = prevSeg ? prevSeg.chordName : "—";
    els.prevRomanDisplay.textContent = prevSeg ? prevSeg.roman : "";

    els.currentBarBeat.textContent = `Bar ${currSeg.bar} : Beat ${currSeg.beat_in_bar}`;
    els.currentChordDisplay.textContent = currSeg.chordName;
    els.currentRomanDisplay.textContent = `Degree: ${currSeg.roman}`;
    if (els.currentBassDisplay) {
      els.currentBassDisplay.textContent = `Bass: ${currSeg.bassName}`;
    }
    els.currentNotesDisplay.textContent = currSeg.notes.length
      ? `構成音: ${currSeg.notes.join(" - ")}`
      : "構成音: 無音 / トーク区間";

    els.nextChordDisplay.textContent = nextSeg ? nextSeg.chordName : "END";
    els.nextRomanDisplay.textContent = nextSeg ? nextSeg.roman : "";

    updatePianoHighlight(currSeg);

    const barCells = els.chordSheetContainer.querySelectorAll(".bar-cell");
    barCells.forEach((cell) => {
      if (Number(cell.dataset.bar) === currSeg.bar) {
        cell.classList.add("bar-active");
      } else {
        cell.classList.remove("bar-active");
      }
    });

    const interactiveEls = els.chordSheetContainer.querySelectorAll("[data-seg-indices]");
    interactiveEls.forEach((el) => {
      const indices = JSON.parse(el.dataset.segIndices || "[]");
      const isMatch = indices.includes(idx);
      if (isMatch) {
        el.classList.add(state.viewMode === "4bar" ? "slot-active" : "active");
        if (els.chkAutoScroll.checked || forceScroll) {
          el.scrollIntoView({ behavior: "smooth", block: "nearest" });
        }
      } else {
        el.classList.remove("slot-active", "active");
      }
    });
  }
}

function tickLoop() {
  if (!els.audioPlayer.paused) {
    updateActiveMonitor(els.audioPlayer.currentTime);
    state.rafId = requestAnimationFrame(tickLoop);
  }
}

async function refreshHistoryList() {
  if (!els.historySelect) return;
  try {
    const res = await fetch("/api/jobs");
    if (!res.ok) return;
    const jobs = await res.json();
    const currentId = state.jobData ? state.jobData.job_id : "";
    els.historySelect.innerHTML = `<option value="">過去の解析曲を選択 (${jobs.length}件)...</option>`;
    jobs.forEach((j) => {
      const opt = document.createElement("option");
      opt.value = j.job_id;
      opt.textContent = `${j.title} (${j.key} / ${j.bpm} BPM)`;
      if (j.job_id === currentId) opt.selected = true;
      els.historySelect.appendChild(opt);
    });
  } catch (_) {}
}

function getStemAudioUrl(jobData, stem) {
  if (!jobData) return "";
  if (stem === "mix") return jobData.audio_url;
  return `/api/audio/${jobData.job_id}/${stem}`;
}

function applyJobData(data) {
  state.jobData = data;
  state.transpose = 0;
  state.beatShift = 0;
  state.currentStem = "mix";
  state.activeSegIndex = -1;

  els.audioPlayer.pause();
  els.audioPlayer.src = data.audio_url;
  els.audioPlayer.playbackRate = Number(els.speedSelect.value || 1.0);
  els.totalTimeText.textContent = formatTime(data.analysis.duration);
  els.currentTimeText.textContent = "00:00.0";

  els.btnPlayPause.disabled = false;
  els.btnStop.disabled = false;
  els.btnCopyChords.disabled = false;
  els.btnExportMidi.disabled = false;

  renderSummary();
  renderChordSheet();
  updateActiveMonitor(0, true);
  refreshHistoryList();
}

function setLoading(isLoading, message = "音源を取得・AIステム分離＆CQTクロマグラム解析中...") {
  if (isLoading) {
    els.statusText.textContent = message;
    els.statusBanner.classList.remove("hidden");
    els.btnAnalyzeUrl.disabled = true;
  } else {
    els.statusBanner.classList.add("hidden");
    els.btnAnalyzeUrl.disabled = false;
  }
}

async function loadDemoPreset(presetKey) {
  setLoading(true, `内蔵デモ音源 (${presetKey}) を読み込み中...`);
  try {
    const res = await fetch("/api/analyze/demo", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ preset: presetKey }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "デモの解析に失敗しました");
    applyJobData(data);
  } catch (err) {
    alert(err.message);
  } finally {
    setLoading(false);
  }
}

async function loadJobById(jobId) {
  if (!jobId) return;
  setLoading(true, "キャッシュ済みの解析データを読み込んでいます...");
  try {
    const res = await fetch(`/api/jobs/${jobId}`);
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "読み込みに失敗しました");
    applyJobData(data);
  } catch (err) {
    alert(err.message);
  } finally {
    setLoading(false);
  }
}

async function analyzeFromUrl() {
  const url = els.urlInput.value.trim();
  if (!url) {
    alert("YouTube等のURLを入力してください。");
    return;
  }
  setLoading(true, "URLから音声ストリームを取得し、Meta Demucs AI 4ステム分離（ベース・伴奏抽出）＋コード解析を実行しています...");
  try {
    const res = await fetch("/api/analyze/url", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url, use_demucs: true }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "URLの解析に失敗しました");
    applyJobData(data);
  } catch (err) {
    alert(err.message);
  } finally {
    setLoading(false);
  }
}

async function analyzeUploadedFile(file) {
  if (!file) return;
  setLoading(true, `「${file.name}」を Meta Demucs AI で4ステム分離し、ベース抽出＆コード進行を解析しています...`);
  try {
    const formData = new FormData();
    formData.append("file", file);
    const res = await fetch("/api/analyze/upload?use_demucs=true", {
      method: "POST",
      body: formData,
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "ファイルの解析に失敗しました");
    applyJobData(data);
  } catch (err) {
    alert(err.message);
  } finally {
    setLoading(false);
  }
}

function bindEvents() {
  els.btnAnalyzeUrl.addEventListener("click", analyzeFromUrl);
  els.urlInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") analyzeFromUrl();
  });

  if (els.historySelect) {
    els.historySelect.addEventListener("change", (e) => {
      if (e.target.value) loadJobById(e.target.value);
    });
  }

  document.querySelectorAll("[data-demo]").forEach((btn) => {
    btn.addEventListener("click", () => {
      loadDemoPreset(btn.dataset.demo);
    });
  });

  els.dropZone.addEventListener("click", () => els.fileInput.click());
  els.fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files[0]) {
      analyzeUploadedFile(e.target.files[0]);
    }
  });
  els.dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    els.dropZone.classList.add("dragover");
  });
  els.dropZone.addEventListener("dragleave", () => {
    els.dropZone.classList.remove("dragover");
  });
  els.dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    els.dropZone.classList.remove("dragover");
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      analyzeUploadedFile(e.dataTransfer.files[0]);
    }
  });

  els.btnPlayPause.addEventListener("click", () => {
    if (!state.jobData) return;
    if (els.audioPlayer.paused) {
      els.audioPlayer.play();
    } else {
      els.audioPlayer.pause();
    }
  });

  els.btnStop.addEventListener("click", () => {
    els.audioPlayer.pause();
    els.audioPlayer.currentTime = 0;
    updateActiveMonitor(0, true);
  });

  els.audioPlayer.addEventListener("play", () => {
    els.playIcon.textContent = "⏸";
    cancelAnimationFrame(state.rafId);
    state.rafId = requestAnimationFrame(tickLoop);
  });

  els.audioPlayer.addEventListener("pause", () => {
    els.playIcon.textContent = "▶";
    cancelAnimationFrame(state.rafId);
    updateActiveMonitor(els.audioPlayer.currentTime);
  });

  els.audioPlayer.addEventListener("ended", () => {
    els.playIcon.textContent = "▶";
    cancelAnimationFrame(state.rafId);
  });

  els.speedSelect.addEventListener("change", () => {
    els.audioPlayer.playbackRate = Number(els.speedSelect.value);
  });

  if (els.btnRunDemucs) {
    els.btnRunDemucs.addEventListener("click", async () => {
      if (!state.jobData) return;
      setLoading(
        true,
        `「${state.jobData.meta.title}」を Meta Demucs (htdemucs) で4ステムAI分離・ベース抽出＆再解析しています (約15〜25秒)...`
      );
      try {
        const res = await fetch(`/api/jobs/${state.jobData.job_id}/reanalyze_demucs`, {
          method: "POST",
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "Demucs再解析に失敗しました");
        applyJobData(data);
      } catch (err) {
        alert(err.message);
      } finally {
        setLoading(false);
      }
    });
  }

  // AI 4ステム（原曲 / 伴奏 / ベースのみ / コード楽器のみ / ドラム / ボーカル）切替
  if (els.audioStemToggle) {
    els.audioStemToggle.querySelectorAll("[data-stem]").forEach((btn) => {
      btn.addEventListener("click", () => {
        if (!state.jobData) return;
        const targetStem = btn.dataset.stem;
        if (targetStem === state.currentStem) return;

        const wasPlaying = !els.audioPlayer.paused;
        const savedTime = els.audioPlayer.currentTime || 0;
        const savedRate = els.audioPlayer.playbackRate || 1.0;

        state.currentStem = targetStem;
        updateStemButtonsUI();
        drawWaveform();

        const targetUrl = getStemAudioUrl(state.jobData, targetStem);
        if (targetStem !== "mix" && !state.jobData.has_bass) {
          setLoading(true, `AI音源分離 (Demucs) で「${STEM_LABELS[targetStem]}」トラックを抽出しています (初回のみ約15秒)...`);
        }

        const onReady = () => {
          els.audioPlayer.removeEventListener("loadedmetadata", onReady);
          els.audioPlayer.removeEventListener("error", onError);
          setLoading(false);
          if (state.jobData) state.jobData.has_bass = true;
          els.audioPlayer.playbackRate = savedRate;
          els.audioPlayer.currentTime = savedTime;
          if (wasPlaying) {
            els.audioPlayer.play().catch(() => {});
          }
        };

        const onError = () => {
          els.audioPlayer.removeEventListener("loadedmetadata", onReady);
          els.audioPlayer.removeEventListener("error", onError);
          setLoading(false);
          alert("パート音源の読み込みに失敗しました。");
        };

        els.audioPlayer.addEventListener("loadedmetadata", onReady);
        els.audioPlayer.addEventListener("error", onError);
        els.audioPlayer.src = targetUrl;
        els.audioPlayer.load();
      });
    });
  }

  // 選択中のパート音源 (.wav) のダウンロード保存
  if (els.btnDownloadStem) {
    els.btnDownloadStem.addEventListener("click", () => {
      if (!state.jobData) return;
      const baseUrl = getStemAudioUrl(state.jobData, state.currentStem);
      window.location.href = `${baseUrl}?download=true`;
    });
  }

  if (els.btnBeatShiftLeft) {
    els.btnBeatShiftLeft.addEventListener("click", () => {
      if (!state.jobData) return;
      state.beatShift = ((state.beatShift - 1 + 4) % 4);
      if (state.beatShift > 2) state.beatShift -= 4;
      state.activeSegIndex = -1;
      renderSummary();
      renderChordSheet();
      updateActiveMonitor(els.audioPlayer.currentTime, false);
    });
  }

  if (els.btnBeatShiftRight) {
    els.btnBeatShiftRight.addEventListener("click", () => {
      if (!state.jobData) return;
      state.beatShift = ((state.beatShift + 1 + 4) % 4);
      if (state.beatShift > 2) state.beatShift -= 4;
      state.activeSegIndex = -1;
      renderSummary();
      renderChordSheet();
      updateActiveMonitor(els.audioPlayer.currentTime, false);
    });
  }

  if (els.chkCleanBar) {
    els.chkCleanBar.addEventListener("change", () => {
      state.cleanBar = els.chkCleanBar.checked;
      renderChordSheet();
      updateActiveMonitor(els.audioPlayer.currentTime, false);
    });
  }

  if (els.chkHideNC) {
    els.chkHideNC.addEventListener("change", () => {
      state.hideNC = els.chkHideNC.checked;
      renderChordSheet();
      updateActiveMonitor(els.audioPlayer.currentTime, false);
    });
  }

  if (els.btnView4Bar) {
    els.btnView4Bar.addEventListener("click", () => {
      state.viewMode = "4bar";
      els.btnView4Bar.classList.add("active");
      if (els.btnViewCards) els.btnViewCards.classList.remove("active");
      renderChordSheet();
      updateActiveMonitor(els.audioPlayer.currentTime, false);
    });
  }

  if (els.btnViewCards) {
    els.btnViewCards.addEventListener("click", () => {
      state.viewMode = "cards";
      els.btnViewCards.classList.add("active");
      if (els.btnView4Bar) els.btnView4Bar.classList.remove("active");
      renderChordSheet();
      updateActiveMonitor(els.audioPlayer.currentTime, false);
    });
  }

  els.waveformWrapper.addEventListener("click", (e) => {
    if (!state.jobData) return;
    const rect = els.waveformWrapper.getBoundingClientRect();
    const ratio = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    const targetSec = ratio * state.jobData.analysis.duration;
    els.audioPlayer.currentTime = targetSec;
    updateActiveMonitor(targetSec, true);
  });

  els.btnTransDown.addEventListener("click", () => {
    if (!state.jobData) return;
    state.transpose = Math.max(-11, state.transpose - 1);
    renderSummary();
    renderChordSheet();
    updateActiveMonitor(els.audioPlayer.currentTime, false);
  });

  els.btnTransUp.addEventListener("click", () => {
    if (!state.jobData) return;
    state.transpose = Math.min(11, state.transpose + 1);
    renderSummary();
    renderChordSheet();
    updateActiveMonitor(els.audioPlayer.currentTime, false);
  });

  els.btnModeRich.addEventListener("click", () => {
    state.mode = "rich";
    els.btnModeRich.classList.add("active");
    els.btnModeSimple.classList.remove("active");
    renderChordSheet();
    updateActiveMonitor(els.audioPlayer.currentTime, false);
  });

  els.btnModeSimple.addEventListener("click", () => {
    state.mode = "simple";
    els.btnModeSimple.classList.add("active");
    els.btnModeRich.classList.remove("active");
    renderChordSheet();
    updateActiveMonitor(els.audioPlayer.currentTime, false);
  });

  els.btnExportMidi.addEventListener("click", () => {
    if (!state.jobData) return;
    const simpleFlag = state.mode === "simple" ? "true" : "false";
    const url = `${state.jobData.midi_url}?transpose=${state.transpose}&simple=${simpleFlag}`;
    window.location.href = url;
  });

  els.btnCopyChords.addEventListener("click", async () => {
    if (!state.jobData) return;
    const segments = state.activeSegments;
    const barsMap = new Map();
    segments.forEach((seg) => {
      if (!barsMap.has(seg.bar)) barsMap.set(seg.bar, []);
      barsMap.get(seg.bar).push(seg);
    });

    let barNumbers = Array.from(barsMap.keys()).sort((a, b) => a - b);
    if (state.hideNC) {
      barNumbers = barNumbers.filter((b) => barsMap.get(b).some((s) => s.chord !== "N.C."));
    }

    const lines = [];
    for (let i = 0; i < barNumbers.length; i += 4) {
      const chunk = barNumbers.slice(i, i + 4);
      const cells = chunk.map((bNum) => {
        const slots = consolidateBarSegments(barsMap.get(bNum));
        return slots.map((s) => s.info.chordName).join(" ");
      });
      lines.push(`| ${cells.map((c) => c.padEnd(12, " ")).join(" | ")} |`);
    }

    const header = `Title: ${state.jobData.meta.title}\nKey: ${els.metricKey.textContent} / BPM: ${state.jobData.analysis.bpm}\n\n`;
    await navigator.clipboard.writeText(header + lines.join("\n"));
    const origText = els.btnCopyChords.textContent;
    els.btnCopyChords.textContent = "コピーしました！";
    setTimeout(() => {
      els.btnCopyChords.textContent = origText;
    }, 1800);
  });

  window.addEventListener("keydown", (e) => {
    if (e.code === "Space" && document.activeElement !== els.urlInput) {
      e.preventDefault();
      els.btnPlayPause.click();
    }
  });

  window.addEventListener("resize", () => drawWaveform());
}

initPianoKeyboard();
bindEvents();
drawWaveform();
(async () => {
  try {
    const res = await fetch("/api/jobs");
    const jobs = await res.json();
    if (jobs && jobs.length > 0) {
      await loadJobById(jobs[0].job_id);
      return;
    }
  } catch (_) {}
  loadDemoPreset("royal_road");
})();
