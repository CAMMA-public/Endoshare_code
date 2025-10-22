#!/usr/bin/env python3

import shutil
import threading
import time
import random
from datetime import datetime
from pathlib import Path
from typing import List

import cv2

from . import mutils, vutils


def mk_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d%H%M%S")


def process_video(
    video_in: List[Path],
    video_out: Path,
    logger,
    progress_bar_handle,
    curr_progress: int,
    max_progress: int
):
    # ── 0) preliminary analysis (fake progress with restarts) ──
    analysis_duration = 3.0  # seconds to spend in fake analysis
    analysis_start = time.time()
    while True:
        p = 0
        # one 0→100 pass with random increments
        while p < 100:
            p = min(p + random.randint(1, 3), 100)
            progress_bar_handle.emit(
                p, 100,
                "Step 0/4: Analyzing video…",
                False
            )
            time.sleep(1)
            # bail out early if overall duration exceeded
            if time.time() - analysis_start >= analysis_duration:
                break
        if time.time() - analysis_start >= analysis_duration:
            break
        # restart animation
        progress_bar_handle.emit(
            0, 100,
            "Step 0/4: Analyzing video: restarting…",
            False
        )
    # ensure final indication of completion before moving on
    progress_bar_handle.emit(
        100, 100,
        "Step 0/4: Analysis complete ✔",
        False
    )

    # ── 1) estimate total segments ─────────────────────────
    estimate_dir = video_out.parent / f"estimate_{mk_timestamp()}"
    estimate_dir.mkdir()
    total_segments = 0
    try:
        for v in video_in:
            tmp = estimate_dir / f"frames_{v.stem}"
            tmp.mkdir()
            vutils.VideoWorker(None).extract_frames(str(v), tmp)
            try:
                segs = mutils.pipeline(tmp)
            except ZeroDivisionError as e:
                logger.error(f"[Phase 0] pipeline empty for {v.name}: {e}")
                segs = []
            total_segments += len(segs)
            shutil.rmtree(tmp)
    finally:
        shutil.rmtree(estimate_dir)
    
    total_segments = max(1, total_segments)

    # ── 2) real work ───────────────────────────────────────
    tmp_dir   = video_out.parent / f"tmp_{mk_timestamp()}"
    tmp_dir.mkdir()
    logfile   = tmp_dir / "report.log"
    worker    = vutils.VideoWorker(logfile)
    frame_dir = tmp_dir / "frames"
    frame_dir.mkdir()

    processed = 0
    segment_paths: List[Path] = []

    try:
        for vid_idx, v in enumerate(video_in):
            # ── Phase 1: Extract frames ────────────────────────
            logger.info(f"Step 1/4: Extracting frames {vid_idx+1}/{len(video_in)}: {v.name}")
            for f in frame_dir.iterdir():
                if f.is_file(): f.unlink()
                else:          shutil.rmtree(f)

            extract_thread = threading.Thread(
                target=lambda: worker.extract_frames(str(v), frame_dir),
                daemon=True
            )
            extract_thread.start()

            # fake‐progress for extraction (0→100, one pass)
            phase1 = 0
            while extract_thread.is_alive():
                phase1 = min(phase1 + random.randint(5, 15), 99)
                progress_bar_handle.emit(
                    phase1, 100,
                    f"Step 1/4: Extracting {vid_idx+1}/{len(video_in)}: {phase1}%",
                    False
                )
                time.sleep(0.1)
            extract_thread.join()
            progress_bar_handle.emit(
                100, 100,
                f"Step 1/4: Extracted {vid_idx+1}/{len(video_in)} ✔",
                False
            )

            # ── Step 2: Preparing segmentation ──────────────
            logger.info(f"Step 2/4: Preparing segmentation for {v.name}")
            segment_times: List = []
            # 2A) start pipeline in background
            def do_pipeline():
                nonlocal segment_times
                try:
                    segment_times = mutils.pipeline(frame_dir)
                except ZeroDivisionError as e:
                    logger.error(f"[Phase 2] pipeline empty for {v.name}: {e}")
                    segment_times = []
                # write to log
                with open(logfile, "a") as logf:
                    logf.write(str(segment_times))

            pipe_thread = threading.Thread(target=do_pipeline, daemon=True)
            pipe_thread.start()

            # 2B) spin fake 0→100 loops until pipeline actually completes
            while pipe_thread.is_alive():
                # one 0→100 pass
                p = 0
                while p < 100 and pipe_thread.is_alive():
                    p = min(p + random.randint(1, 3), 100)
                    progress_bar_handle.emit(
                        p, 100,
                        f"Step 2/4: Preparing segmentation: {p}%",
                        False
                    )
                    time.sleep(1)
                # if we hit 100 but pipeline still running, restart
                if pipe_thread.is_alive():
                    progress_bar_handle.emit(
                        0, 100,
                        "Step 2/4: Preparing segmentation: restarting…",
                        False
                    )
            pipe_thread.join()
            progress_bar_handle.emit(
                100, 100,
                "Step 2/4: Preparation complete ✔",
                False
            )

            # ── Phase 3: Cut/black‐out segments ─────────────────
            logger.info(f"Step 3/4: Segmenting {v.name} …")
            seg_dir = tmp_dir / f"segments{vid_idx}"
            seg_dir.mkdir()
            cap2 = cv2.VideoCapture(str(v))
            w = cap2.get(cv2.CAP_PROP_FRAME_WIDTH)
            h = cap2.get(cv2.CAP_PROP_FRAME_HEIGHT)
            cap2.release()

            for seg_idx, (sensitive, st, nd) in enumerate(segment_times):
                out_seg = seg_dir / (
                    video_out.stem + f".p{seg_idx:04d}" + video_out.suffix
                )
                if not sensitive:
                    worker.kf_cut(v, str(out_seg), st, nd, tbn=10000)
                else:
                    worker.mk_black_video(nd - st, str(out_seg), w, h)

                segment_paths.append(out_seg)
                processed += 1
                progress_bar_handle.emit(
                    curr_progress + processed,
                    curr_progress + total_segments,
                    f"Step 3/4: Segment {processed}/{total_segments}",
                    False
                )

            # clean up frames
            for f in frame_dir.iterdir():
                if f.is_file(): f.unlink()
                else:          shutil.rmtree(f)

        # ── Phase 4: Merge ──────────────────────────────────
        logger.info("Step 4/4: Merging all segments…")
        merge_thread = threading.Thread(
            target=lambda: worker.merge(segment_paths, str(video_out)),
            daemon=True
        )
        merge_thread.start()

        # fake‐progress for merging
        m = 0
        while merge_thread.is_alive():
            m = min(m + random.randint(2, 5), 99)
            progress_bar_handle.emit(
                m, 100,
                f"Step 4/4: Merging: {m}%",
                False
            )
            time.sleep(0.1)
        merge_thread.join()
        progress_bar_handle.emit(
            100, 100,
            "Step 4/4: Merge complete ✔",
            False
        )

    finally:
        logger.info(f"Cleaning up {tmp_dir}")
        shutil.rmtree(tmp_dir)
