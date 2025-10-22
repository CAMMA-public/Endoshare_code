import os

from PyQt5.QtCore import pyqtSignal
from ..utils.resources import resource_path

class VideoProcessThread(QThread):
    update_progress = pyqtSignal(int, int, str, bool)
    update_color = pyqtSignal(str, str)

    def __init__(self, video_in_root_dir, shared_folder, local_folder, fps, resolution):
        super().__init__()
        
        self.video_in_root_dir = video_in_root_dir 
        self.destination_folder = ""
        self.ckpt_path = resource_path(os.path.join("ckpt", "oobnet_weights.h5"))
        self.device = "/cpu:0"
        self.out_final = shared_folder  ## needs to be changed with hone settings
        self.log_filename = os.path.join(local_folder, "./patientID_log.csv") ## needs to be changed from settings
        self.patient_name = ""
        self.crf = 20   ## needs to be changed from settings
        self.fps = fps
        self.resolution = resolution
        self.write_out_video=True
        self.default_output_folder = local_folder

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

    def run_inference(self,
        video_in_root_dir,
        video_out_root_dir,
        text_root_dir,
        ckpt_path,
        buffer_size,
        device="/cpu:0",
    ):
        # build the model
        counter = 0
        with tf.device(device):
            model = self.build_model()
            model.load_weights(ckpt_path)

        video_names = list(video_in_root_dir.values())
       
        self.update_progress.emit(0, len(video_names), "Processing started for " + self.patient_name , False)
        start_time = time()
        for i, in_video_path in enumerate(video_names):
            process(in_video_path, video_out_root_dir, text_root_dir, buffer_size)

            self.update_progress.emit(i + 1, len(video_names), f"Processing for {self.patient_name}, file {out_name}...", False)

        end_time = time()

        ####Need to add to log################
        logger.info(f"total time spent: {end_time - start_time:.2f} sec")

        video_out.close()
        pbar.close()
        # Emit the signal to update the progress bar in the main GUI thread
        self.update_progress.emit(len(video_names), len(video_names), "Processing completed for " + self.patient_name , False)
        # Emit the signal to update the color of the patient in the name_list 
        self.update_color.emit(self.patient_name, "green")

    def setup_log(self,log_filename):
        """Creates a log file to record original to randomized video names.
        If no filename is specified, will create a log file named
        'patientID_log.csv'. If the file already exists, it will append
        new entries to it."""
        
        log_file_path = Path(log_filename)
        
        # Check if log file exists
        if log_file_path.exists():
            return log_file_path
        
        # If not, create the log file and write header
        with open(log_file_path, mode='w', newline='') as log_file:
            log_writer = csv.writer(log_file)
            log_writer.writerow(["original", "anonymized"])
        
        return log_file_path


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
        return floor(log10(num)) + 1


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
        vid_exts = (".mp4", ".avi")
        return path.suffix.casefold() in vid_exts and path.is_file()


    def get_video_paths(self,vid_dir):
        """Yield files with video extensions in vid_dir"""
        return [path for path in Path(vid_dir).rglob("*") if self.is_video_path(path)]
    

    def strip_metadata(self, input_vid, output_vid):
        """Strips metadata from input_vid and places stripped video in
        output_vid. If successful returns output_vid's path, otherwise
        returns 'FAILED'."""
        command = [
            "ffmpeg",
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
            "-loglevel",
            "debug",
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
            logger.error(f"ffmpeg failed to strip '{input_vid}' with output:")
            logger.error(perr.stdout, "And error messages:", perr.stderr, sep="\n")
            logger.error("Script will continue to process the other videos.")
            # File failed to process so delete it is ffmpeg made an
            # incomplete one
            if output_vid.is_file():
                output_vid.unlink()
            return "FAILED"

        return output_vid if output_vid.is_file() else None

    def anonymize(self, video_in_root_dir, video_out_root_dir, log_filename):
        """Will strip metadata and optionally randomize filenames from a
        directory of videos."""

        vid_paths = self.get_video_paths(video_in_root_dir)
        logger.info("vid_paths", vid_paths)
        outdir = Path(video_out_root_dir)
        outdir.mkdir(exist_ok=True)

        log_file_path = self.setup_log(log_filename)

        vid_map = self.randomize_paths(vid_paths, outdir, sequentialize=False)

        for orig_path, new_path in vid_map.items():
            # strip metadata then save into the csv log file:
            # orig_path,output (either new_path or "FAILED" if it was not successful)
            final_name = self.strip_metadata(orig_path, new_path)

            # Extract file names without extensions
            orig_name = orig_path.stem
            new_name = new_path.stem
            
            # Append to log file
            with open(log_file_path, mode='a', newline='') as log_file:
                log_writer = csv.writer(log_file)
                log_writer.writerow([orig_name, new_name])

        return final_name
    
    def run(self):
        
        log_file_path = self.setup_log(self.log_filename)

        ###############Iteration happens for #of Patients############################
        for patient_id, videos_iter in self.video_in_root_dir.items():
            logger.info(f"Patient ID: {patient_id}")
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

                    
            self.run_inference(
                video_in_root_dir=videos_iter,
                video_out_root_dir=self.destination_folder,
                text_root_dir=self.destination_folder,
                ckpt_path=self.ckpt_path,
                device=self.device,
                buffer_size=64,
            )

            anonymized_video = self.anonymize(self.destination_folder, self.out_final, self.log_filename)
            
            logger.info("Anonymized video:", anonymized_video)

