from PyQt5.QtCore import QThread, pyqtSignal
import os
import sys
import secrets
import math
import shutil
from pathlib import Path
import subprocess
import csv
import time
from copy import deepcopy

import numpy as np
import cv2
import tensorflow as tf
from loguru import logger
from tqdm import tqdm
from vidgear.gears import WriteGear

from ..utils.resources import FFMPEG_BIN, resource_path
from ..utils.types import ProcessingMode, ProcessingInterrupted
from ..processing import deid
from .video_browser import VIDEO_EXTENSIONS
from uuid import uuid4


LOG_PERSIST = "PERSIST"
try:
    logger.level(LOG_PERSIST, no=25)  # custom level between INFO and WARNING
except ValueError:
    pass

def extract_vpt_args(rt: dict):
    return {
        "fps": rt["fps"],
        "resolution": rt["resolution"],
        "mode": rt["mode"],
        "purge_after": rt.get("purge_after", False),
    }

##########################Video Copy Thread for updating the video dictionary about the location; no need to save video################

class VideoCopyThread(QThread):
    update_progress = pyqtSignal(int, int, str, bool)

    def __init__(self, video_files, selected_folder):
        super().__init__()
        self.video_files = video_files  
        self.selected_folder = selected_folder
        self.video_dict = {}  # Dictionary to store the mapping of original file names to new names

    def run(self):
        total_videos = len(self.video_files)
        logger.info("total_videos:", total_videos)
        for i, video_file in enumerate(self.video_files):
            self.video_dict[video_file] = os.path.join(self.selected_folder, video_file)
            progress = int(((i + 1) / total_videos) * 100)
            self.update_progress.emit(i + 1, total_videos, f"Arranging file {video_file}... ({progress}%)", True)
        self.update_progress.emit(total_videos, total_videos, "Arranging completed successfully!" , True)

    def get_video_dict(self):
        return self.video_dict
    


#####################################Video Process Thread for OOB detection, merging and deidentification#######################


class VideoProcessThread(QThread):
    update_progress = pyqtSignal(int, int, str, bool)
    update_color = pyqtSignal(str, str)
    error           = pyqtSignal(str)

    def __init__(self, video_in_root_dir, shared_folder, local_folder, 
                 fps,
                 resolution,
                 mode,
                 purge_after=False
                 ):
        super().__init__()
        
        self.video_in_root_dir = video_in_root_dir 
        self.destination_folder = ""
        
        # Diagnose resolution of the checkpoint path
        ckpt_rel = os.path.join("ckpt", "oobnet_weights.h5")
        resolved = resource_path(ckpt_rel)
        if not os.path.isfile(resolved):
            # list what’s actually in the expected folder for further clarity
            parent = os.path.dirname(resolved)
            try:
                contents = os.listdir(parent)
            except Exception as e:
                contents = f"<could not list {parent}: {e}>"
            raise FileNotFoundError(
                f"Checkpoint resolution failed. Tried: {resolved}\n"
                f"Directory contents of {parent}: {contents}"
            )
        self.ckpt_path = resolved

        self.device = "/cpu:0"
        self.out_final = shared_folder  ## needs to be changed with hone settings
        self.name_translation_filename = os.path.join(local_folder, "./patientID_log.csv") ## needs to be changed from settings
        self.patient_name = ""
        self.crf = 20   ## needs to be changed from settings
        self.fps = fps
        self.resolution = resolution
        self.processing_mode = mode
        self.buffer_size=2048
        self.default_output_folder = local_folder
        self.purge_after = purge_after

    def preprocess(self, image, shape=[64, 64]):
        with tf.device(self.device):
            image = tf.cast(image, tf.float32)
            image = tf.image.resize(image, shape)
            image = tf.reshape(image, shape + [3])
            image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
            return tf.expand_dims(image, 0)

    def build_model(self, input_shape=[64, 64]):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Input(input_shape + [3], dtype=tf.float32))
        model.add(
            tf.keras.applications.MobileNetV2(
                input_shape=input_shape + [3], alpha=1.0, include_top=False, weights=None
            )
        )
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.LayerNormalization())
        model.add(tf.keras.layers.Dropout(0))
        model.add(tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, 0)))
        model.add(tf.keras.layers.LSTM(units=640, return_sequences=True, stateful=True))
        model.add(tf.keras.layers.LayerNormalization())
        model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
        return model
    
    def terminate(self):
        """
        When the user hits Terminate:
          • In ADVANCED mode → soft‑stop so we don’t crash WriteGear.
          • In NORMAL mode   → fall back to the old hard kill.
        """
        if self.processing_mode == ProcessingMode.ADVANCED:
            # soft stop
            self.requestInterruption()

            # close WriteGear if it’s up
            vg = getattr(self, "_vg", None)
            if vg:
                try: vg.close()
                except: pass

            # close tqdm if it’s up
            pb = getattr(self, "_pbar", None)
            if pb:
                try: pb.close()
                except: pass

            # don’t call super().terminate() here
        else:
            # normal mode: do exactly what you had before
            super().terminate()

    def run_fast_inference(
        self,
        video_in_root_dir,
        video_out_root_dir,
        text_root_dir,
        ckpt_path,
        buffer_size,
        device,
        curr_progress,
        max_progress,
    ):
        video_names = list(video_in_root_dir.values())

        self.update_progress.emit(curr_progress, max_progress, "Processing started for " + self.patient_name , False)
        start_time = time.time()
        
        file_name, file_ext = os.path.splitext(video_names[0])
        out_name = file_name.split(".")[0]
        out_ext = file_ext[1:]
        out_video_path = os.path.join(
            video_out_root_dir, self.patient_name + "."+ out_ext
        )

        deid.process_video([Path(p) for p in video_names], Path(out_video_path), logger, self.update_progress, curr_progress, max_progress)
        end_time = time.time()

        # Video duration
        video      = cv2.VideoCapture(out_video_path)
        framecount = video.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        fps        = video.get(cv2.CAP_PROP_FPS) or 0
        if fps > 0:
            video_duration = framecount / fps
        else:
            logger.warning(f"FPS is zero for output video '{out_video_path}', skipping duration/speed calc.")
            video_duration = 0.0

        elapsed = end_time - start_time
        logger.log(LOG_PERSIST, f"total time spent: {elapsed:.2f} sec")
        if elapsed > 0:
            speed = video_duration / elapsed if video_duration > 0 else None
            if speed is not None:
                logger.log(LOG_PERSIST, f"processing speed: {speed:.2f}× real time")
            else:
                logger.log(LOG_PERSIST, "processing speed: N/A (could not compute)")
        else:
            logger.log(LOG_PERSIST, "processing speed: N/A (zero elapsed time)")

        # Emit the signal to update the progress bar in the main GUI thread
        self.update_progress.emit(curr_progress+len(video_names), max_progress, "Processing completed for " + self.patient_name , False)
        # Emit the signal to update the color of the patient in the name_list 
        self.update_color.emit(self.patient_name, "green")

    def run_advanced_inference(
        self,
        video_in_root_dir,
        video_out_root_dir,
        text_root_dir,
        ckpt_path,
        buffer_size,
        device,
        curr_progress,
        max_progress,
    ):
        videos_duration = 0
        write_out_video = True
        # build the model
        counter = 0
        with tf.device(device):
            model = self.build_model()
            model.load_weights(ckpt_path)
        init_once = True
        
        video_names = list(video_in_root_dir.values())
       
        self.update_progress.emit(curr_progress, max_progress, "Processing started for " + self.patient_name , False)
        start_time = time.time()
        rescaled_size = None
        total_chunks = 0
        for in_video_path in video_names:
            cap = cv2.VideoCapture(in_video_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            total_chunks += math.ceil(frame_count / buffer_size)
        self._total_units = total_chunks
        self._processed_units = 0

        
        for i, in_video_path in enumerate(video_names):
            # ── interruption check before each file ───────────────
            if self.isInterruptionRequested():
                logger.info("Advanced inference interrupted before starting next video.")
                # gracefully close writer and progress bar
                video_out.close()
                pbar.close()
                raise ProcessingInterrupted()

            logger.info(f"Processing video {i+1} in advanced mode ...")

            try:
                video_in = cv2.VideoCapture(in_video_path)
                assert video_in.isOpened()
                fps_in = video_in.get(cv2.CAP_PROP_FPS)
                frame_count = int(video_in.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = frame_count / fps_in
            except OSError:
                logger.error("Could not open/read file")
            if write_out_video and init_once:
                init_once = False
                os.makedirs(video_out_root_dir, exist_ok=True)
                
                width = int(video_in.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(video_in.get(cv2.CAP_PROP_FRAME_HEIGHT))
                orig_image_buffer = np.zeros(
                    (buffer_size, height, width, 3), dtype=np.uint8
                )
                
                file_name, file_ext = os.path.splitext(in_video_path)
                out_name = file_name.split(".")[0]
                out_ext = file_ext[1:]
                #out_name, out_ext = os.path.basename(in_video_path).split(".")
                out_video_path = os.path.join(
                    video_out_root_dir, self.patient_name + "." + out_ext
                )
                fps = video_in.get(cv2.CAP_PROP_FPS)
                logger.info(f"fps: {self.fps}, resolution: {self.resolution}p")
                
                ####Need to add to log################
                ##### Need to add resolution ###################

                w = int(video_in.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(video_in.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = video_in.get(cv2.CAP_PROP_FPS)

                # choose 0.10 for a balanced quality/size or 0.15 for high quality
                bpp = 0.10  
                bitrate_k = round(w * h * fps * bpp / 1000)

                output_params = {
                    "-pix_fmt": "yuv420p",
                    "-input_framerate": self.fps,
                }

                if sys.platform == "darwin":
                    output_params.update({
                        "-vcodec": "h264_videotoolbox",
                        "-b:v": f"{bitrate_k}k",
                        "-profile:v": "high",
                        "-tune": "zerolatency",
                    })
                else:
                    output_params.update({
                        "-vcodec": "libx264",
                        "-preset": "ultrafast",
                        "-crf": self.crf,
                        "-tune": "zerolatency",
                    })

                if self.resolution > 0:
                    rescaled_width = np.round(width*(self.resolution/height))
                    if rescaled_width%2 != 0:
                        rescaled_width += 1
                    rescaled_size = (int(rescaled_width), self.resolution)
                video_out = WriteGear(output=out_video_path, logging=False, compression_mode=True, **output_params) 

            video_nframes = int(video_in.get(cv2.CAP_PROP_FRAME_COUNT))
            pred_history = []
            image_buffer = []
            image_count = 0
            pbar = tqdm(total=video_nframes // buffer_size)
            self._vg   = video_out
            self._pbar = pbar


            target_fps = self.fps
            frame_interval = fps / target_fps if fps > target_fps else target_fps / fps
            frame_index = 0
            if fps_in >= target_fps:
                while True:
                    # ── interruption check inside frame loop ─────────────
                    if self.isInterruptionRequested():
                        logger.info("Advanced inference interrupted during frame loop.")
                        video_out.close()
                        pbar.close()
                        raise ProcessingInterrupted()
                    ok, frame = video_in.read()
                    counter += 1
                    if ok:
                        preprocess_image = self.preprocess(
                            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        image_buffer.append(preprocess_image)
                        if write_out_video:
                            orig_image_buffer[image_count] = frame
                        image_count += 1
                        if len(image_buffer) == buffer_size:
                            with tf.device(device):
                                image_batch = tf.concat(image_buffer, axis=0)
                                prediction = model(image_batch)
                            preds = np.round(prediction.numpy()[0, :, 0]).astype(np.uint8)
                            orig_image_buffer[preds.astype(bool)] = np.zeros_like(orig_image_buffer[preds.astype(bool)])
                            
                            for frame in orig_image_buffer:
                                if rescaled_size is not None:
                                    frame = cv2.resize(frame, rescaled_size, interpolation=cv2.INTER_AREA)
                                # Adjust for FPS-change.
                                if frame_index % frame_interval < 1:
                                    video_out.write(frame)
                                frame_index += 1
                            pred_history += preds.tolist()
                            image_buffer = []
                            image_count = 0
                            pbar.update(1)
                            progress = int((pbar.n / pbar.total * 100) if pbar.total != 0 else 0)
                            ####Need to add to log################
                            # Emit the signal to update the progress bar in the main GUI thread
                            self._processed_units += 1
                            self.update_progress.emit(self._processed_units,
                                                    self._total_units,
                                                    f"Processing {self.patient_name} ({i+1}/{len(video_names)})…",
                                                    False)
                    else:
                        # at end of file, also obey interruption
                        if self.isInterruptionRequested():
                            break
                        if len(image_buffer) > 0:
                            with tf.device(self.device):
                                image_batch = tf.concat(image_buffer, axis=0)
                                prediction = model(image_batch)
                            preds = np.round(prediction.numpy()[0, :, 0]).astype(np.uint8)
                            orig_image_buffer_write = deepcopy(
                                orig_image_buffer[:image_count]
                            )
                            orig_image_buffer_write[preds.astype(bool)] = (
                                orig_image_buffer_write[preds.astype(bool)]
                                .mean(1)
                                .mean(1)
                                .reshape(preds.sum(), 1, 1, 3)
                            )
                           
                            for frame in orig_image_buffer_write:
                                if rescaled_size is not None:
                                    frame = cv2.resize(frame, rescaled_size, interpolation=cv2.INTER_AREA)
                                video_out.write(frame)
                            pred_history += preds.tolist()
                        break
            else:
                while True:
                    if self.isInterruptionRequested():
                        logger.info("Advanced inference interrupted during frame loop.")
                        video_out.close()
                        pbar.close()
                        raise ProcessingInterrupted()
                    ok, frame = video_in.read()
                    counter += 1
                    if ok:
                        preprocess_image = self.preprocess(
                            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        image_buffer.append(preprocess_image)
                        if write_out_video:
                            orig_image_buffer[image_count] = frame
                        image_count += 1
                        if len(image_buffer) == buffer_size:
                            with tf.device(device):
                                image_batch = tf.concat(image_buffer, axis=0)
                                prediction = model(image_batch)
                            preds = np.round(prediction.numpy()[0, :, 0]).astype(np.uint8)
                            orig_image_buffer[preds.astype(bool)] = np.zeros_like(orig_image_buffer[preds.astype(bool)])
                            #orig_image_buffer[preds.astype(bool)] = (
                            #    orig_image_buffer[preds.astype(bool)]
                            #    .mean(1)
                            #    .mean(1)
                            #    .reshape(preds.sum(), 1, 1, 3)
                            #)
                            
                            for frame in orig_image_buffer:
                                if rescaled_size is not None:
                                    frame = cv2.resize(frame, rescaled_size, interpolation=cv2.INTER_AREA)
                                # Don't adjust for FPS-change...
                                video_out.write(frame)
                            pred_history += preds.tolist()
                            image_buffer = []
                            image_count = 0
                            pbar.update(1)
                            progress = int((pbar.n / pbar.total * 100) if pbar.total != 0 else 0)
                            ####Need to add to log################
                            # Emit the signal to update the progress bar in the main GUI thread
                            self._processed_units += 1
                            self.update_progress.emit(self._processed_units,
                                                    self._total_units,
                                                    f"Processing {self.patient_name} ({i+1}/{len(video_names)})…",
                                                    False)
                    else:
                        if len(image_buffer) > 0:
                            with tf.device(self.device):
                                image_batch = tf.concat(image_buffer, axis=0)
                                prediction = model(image_batch)
                            preds = np.round(prediction.numpy()[0, :, 0]).astype(np.uint8)
                            orig_image_buffer_write = deepcopy(
                                orig_image_buffer[:image_count]
                            )
                            orig_image_buffer_write[preds.astype(bool)] = (
                                orig_image_buffer_write[preds.astype(bool)]
                                .mean(1)
                                .mean(1)
                                .reshape(preds.sum(), 1, 1, 3)
                            )
                           
                            for frame in orig_image_buffer_write:
                                if rescaled_size is not None:
                                    frame = cv2.resize(frame, rescaled_size, interpolation=cv2.INTER_AREA)
                                video_out.write(frame)
                            pred_history += preds.tolist()
                        break
            
            # ── per‑video cleanup if interrupted ────────────────
            if self.isInterruptionRequested():
                try: video_out.close()
                except: pass
                pbar.close()
                return

            framecount = video_in.get(cv2.CAP_PROP_FRAME_COUNT)
            fps = video_in.get(cv2.CAP_PROP_FPS)
            duration = (framecount/fps)/1000
            videos_duration += duration
            video_in.release()
            pbar.update(1)
            self._processed_units += 1
            self.update_progress.emit(self._processed_units,
                                    self._total_units,
                                    f"Processing {self.patient_name} ({i+1}/{len(video_names)})…",
                                    False)
            progress = int((pbar.n / pbar.total * 100) if pbar.total != 0 else 0)
            self._vg   = None
            self._pbar = None


            ####Need to add to log################
            # Emit the signal to update the progress bar in the main GUI thread

            self.update_progress.emit(curr_progress+len(video_names), max_progress, f"Processing for {self.patient_name}, file {out_name}...", False)

        end_time = time.time()

        ####Need to add to log################
        logger.log(LOG_PERSIST, f"total time spent: {end_time - start_time:.2f} sec")
        elapsed = end_time - start_time
        if elapsed > 0:
            speed = videos_duration / elapsed
            logger.log(LOG_PERSIST, f"processing speed: {speed:.2f} video_duration/processing_time")
        else:
            logger.log(LOG_PERSIST, "processing speed: N/A (zero elapsed time)")

        video_out.close()
        pbar.close()
        # Emit the signal to update the progress bar in the main GUI thread
        self.update_progress.emit(curr_progress+len(video_names), max_progress, "Processing completed for " + self.patient_name , False)
        # Emit the signal to update the color of the patient in the name_list 
        self.update_color.emit(self.patient_name, "green")

    def setup_name_translation_file(self, name_translation_filename):
        """Creates a log file to record original to randomized video names.
        If no filename is specified, will create a log file named
        'patientID_log.csv'. If the file already exists, it will append
        new entries to it."""
        
        name_translation_file_path = Path(name_translation_filename)
        
        # Check if log file exists
        if name_translation_file_path.exists():
            return name_translation_file_path
        
        # If not, create the log file and write header
        with open(name_translation_file_path, mode='w', newline='') as name_translation_file:
            name_translation_writer = csv.writer(name_translation_file)
            name_translation_writer.writerow(["original", "anonymized"])
        
        return name_translation_file_path


    def name_generator(self,uuid=False, prefix="", start=0, width=3):
        """Returns functions that generate names. If uuid is True, will
        return a function that generates uuid4s. Otherwise, will return a
        function to generate incrementing names with a prefix added to the
        incrementing numbers that start at 'start' and padded to be 'width'
        wide, eg prefix = "video" then video001, video002, etc."""

        def incrementing_name():
            nonlocal n
            n+=1
            return prefix + f"{n:0{width}}"

        def uuid_name():
            # This is random. I thought, we don't want random ...
            return uuid4().hex[:-25]

        if uuid:
            return uuid_name
        else:
            # subtract 1 from start because incrementing name will increment it
            # by one before initial use
            n = start-1
            return incrementing_name


    def seq_width(self,num):
        """Returns one more than order of magnitude (base-10) for num which
        is equivalent to the number of digits-wide a sequential
        representation would need to be. E.g.: num = 103 return 3 so
        sequences would be 000, 001, ... 102, 103."""
        return math.floor(math.log10(num)) + 1


    def shuffle(self,some_list):
        """Returns the items in some_list in shuffled order."""
        # do a defensive copy so that the original list doesn't get
        # consumed by the 'pop()'
        items = some_list.copy()
        while items:
            yield items.pop(secrets.randbelow(len(items)))


    def randomize_paths(self,vid_paths, outdir, sequentialize):
        """Returns a dict of orig_name: random_name. When sequentialize is
        true, random_names will be like video000, video001, etc, otherwise
        returns a uuid4."""
        if sequentialize:
            generate_name = self.name_generator(prefix="video", width=self.seq_width(len(vid_paths)))
        else:
            generate_name = self.name_generator(uuid=True)
        orig_to_random = {}
        # shuffle the filenames so that sequentially generated filenames
        # don't mimic the order of the filenames in the input directory
        for orig_path in self.shuffle(vid_paths):
            randomized_path = Path(outdir).joinpath(generate_name() + orig_path.suffix)
            orig_to_random[orig_path] = randomized_path

        return orig_to_random


    def transpose_paths(self,paths, outdir):
        """Returns dict mapping each path in paths to a path from joining
        outdir with the basename of path."""
        return {path: Path(outdir).joinpath(path.name) for path in paths}


    def is_video_path(self,path):
        """Checks if path has a video extension and is a file."""
        #vid_exts = (".mp4", ".avi")
        return path.suffix.lower() in VIDEO_EXTENSIONS and path.is_file()


    def get_video_paths(self,vid_dir):
        """Yield files with video extensions in vid_dir"""
        return [path for path in Path(vid_dir).rglob("*") if self.is_video_path(path)]
    

    def strip_metadata(self, input_vid, output_vid):
        """Strips metadata from input_vid and places stripped video in
        output_vid. If successful returns output_vid's path, otherwise
        returns 'FAILED'."""
        command = [
            FFMPEG_BIN,
            "-nostdin",
            # set input video
            "-i",
            str(input_vid),
            # select all video streams
            "-map",
            "0:v",
            # select all audio streams if present
            "-map",
            "0:a?",
            # just copy streams, do not transcode (much faster and lossless)
            "-c",
            "copy",
            # strip global metadata for the video container
            "-map_metadata",
            "-1",
            # strip metadata for video stream
            "-map_metadata:s:v",
            "-1",
            # strip metadata for audio stream
            "-map_metadata:s:a",
            "-1",
            # remove any chapter information
            "-map_chapters",
            "-1",
            # remove any disposition info
            "-disposition",
            "0",
            str(output_vid),
        ]

        try:
            subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
                text=True,
            )
        except subprocess.CalledProcessError as perr:
            logger.error(f"ffmpeg failed to strip '{input_vid}' with output: {perr.stderr}")
            # File failed to process so delete it is ffmpeg made an
            # incomplete one
            if output_vid.is_file():
                output_vid.unlink()
            return "FAILED"

        return output_vid if output_vid.is_file() else None

    def anonymize(self, video_in_root_dir, video_out_root_dir, name_translation_filename):
        """Will strip metadata and optionally randomize filenames from a
        directory of videos."""

        vid_paths = self.get_video_paths(video_in_root_dir)
        outdir = Path(video_out_root_dir)
        outdir.mkdir(exist_ok=True)

        name_translation_file_path = self.setup_name_translation_file(name_translation_filename)

        vid_map = self.randomize_paths(vid_paths, outdir, sequentialize=False)
        final_name = None

        for orig_path, new_path in vid_map.items():
            # strip metadata then save into the csv log file:
            # orig_path,output (either new_path or "FAILED" if it was not successful)
            final_name = self.strip_metadata(orig_path, new_path)

            # Extract file names without extensions
            orig_name = orig_path.stem
            new_name = new_path.stem
            
            # Append to log file
            with open(name_translation_file_path, mode='a', newline='') as name_translation_file:
                name_translation_writer = csv.writer(name_translation_file)
                name_translation_writer.writerow([orig_name, new_name])
            logger.info(f"Anonymized into {new_path}.")

        return final_name

    
    def run(self):
        # ─── 0) Pre‑flight: verify every video can be opened ─────────────────
        all_paths = [
            p
            for vid_map in self.video_in_root_dir.values()
            for p in vid_map.values()
        ]
        for path in all_paths:
            cmd = [
                FFMPEG_BIN, "-v", "error",
                "-i", path,
                "-f", "null", "-"   # decode but throw away frames
            ]
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            if proc.returncode != 0:
                # grab the first few lines of ffmpeg’s stderr to keep it concise
                err_lines = proc.stderr.strip().splitlines()[:5]
                snippet   = "\n".join(err_lines)
                self.error.emit(
                    f"Corrupt file detected: “{Path(path).name}”\n\n"
                    f"{snippet}\n\n"
                    "Processing aborted."
                )
                return

        name_translation_file_path = self.setup_name_translation_file(self.name_translation_filename)
        ###############Iteration happens for #of Patients############################
        n_all_videos = sum([len(videos_iter) for videos_iter in self.video_in_root_dir.values()])
        curr_n_videos = 0
        for patient_id, videos_iter in self.video_in_root_dir.items():
            if self.isInterruptionRequested():
                break
            temp_folder = self.default_output_folder
            os.makedirs(temp_folder, exist_ok=True)
            if not self.destination_folder:
                previous_patient_id = patient_id
                self.destination_folder = os.path.join(temp_folder, patient_id)
                os.makedirs(self.destination_folder, exist_ok=True)
            elif patient_id != previous_patient_id:
                previous_patient_id = patient_id
                self.destination_folder = os.path.join(temp_folder, patient_id)
                os.makedirs(self.destination_folder, exist_ok=True)
            else:
                self.destination_folder = os.path.join(self.destination_folder, patient_id)
                os.makedirs(self.destination_folder, exist_ok=True)
            self.patient_name = patient_id

            try:
                for path in videos_iter.values():
                    cap = cv2.VideoCapture(path)
                    if not cap.isOpened():
                        cap.release()
                        raise RuntimeError(f"Cannot open “{Path(path).name}”")
                    cap.release()
                    
                if self.processing_mode == ProcessingMode.ADVANCED:
                    self.run_advanced_inference(
                        video_in_root_dir=videos_iter,
                        video_out_root_dir=self.destination_folder,
                        text_root_dir=self.destination_folder,
                        ckpt_path=self.ckpt_path,
                        buffer_size=64,
                        device=self.device,
                        curr_progress=curr_n_videos,
                        max_progress=n_all_videos,
                    )
                elif self.processing_mode == ProcessingMode.NORMAL:
                    self.run_fast_inference(
                        video_in_root_dir=videos_iter,
                        video_out_root_dir=self.destination_folder,
                        text_root_dir=self.destination_folder,
                        ckpt_path=self.ckpt_path,
                        buffer_size=64,
                        device=self.device,
                        curr_progress=curr_n_videos,
                        max_progress=n_all_videos,
                    )
                curr_n_videos += len(videos_iter)
                anonymized_video = self.anonymize(self.destination_folder, self.out_final, self.name_translation_filename)
                if self.purge_after:
                    orig_folder = Path(self.default_output_folder) / self.patient_name
                    if orig_folder.exists():
                        try:
                            shutil.rmtree(orig_folder)
                            logger.info(f"Purged archive folder {orig_folder}")
                        except Exception as e:
                            logger.warning(f"Failed to purge archive folder {orig_folder}: {e}")
            
            except ProcessingInterrupted:
                logger.info(f"Processing aborted by user at patient {patient_id}")
                return
            except Exception as exc:
                # log full traceback
                logger.error(f"Error processing patient {patient_id}", exc_info=True)
                # notify UI
                self.error.emit(str(exc))
                return



