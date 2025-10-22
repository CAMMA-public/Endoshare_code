import numpy as np
import tensorflow as tf
import cv2
from .model import build_model
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from ..utils.resources import resource_path
import os


def _find_bundled_ckpt():
    # inside a .app bundle: use frozen logic
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).parent
        resources = exe_dir.parent / "Resources"
        # look both in Resources/ckpt and Resources/Resources/ckpt
        for candidate in (resources / "ckpt", resources / "Resources" / "ckpt"):
            p = candidate / "oobnet_weights.h5"
            if p.exists():
                return str(p)
        # fallback
        return str(resources / "ckpt" / "oobnet_weights.h5")
    # running from source tree: use resource_path to point into resources/ckpt
    return resource_path(os.path.join("ckpt", "oobnet_weights.h5"))

WEIGHTS_PATH = _find_bundled_ckpt()


def preprocess(img):
    return tf.expand_dims(
        tf.keras.applications.mobilenet_v2.preprocess_input(
            tf.cast(img, tf.float32),
        ),
        axis=0
    )

def find_sensitive(video_frame_dir):
    m = build_model()
    m.load_weights(WEIGHTS_PATH)
    frame_paths = sorted(Path(video_frame_dir).glob("*"))
    prediction_buffer = []
    n = len(frame_paths)
    for j, fp in enumerate(frame_paths):
        print("{} / {}".format(j, n))
        frame = cv2.cvtColor(
            cv2.imread(str(fp)),
            cv2.COLOR_BGR2RGB
        )
        prediction = np.round(
            np.squeeze(m(preprocess(frame)).numpy())
        )
        prediction_buffer.append(prediction)
    return prediction_buffer

def mk_plot(arr):
    plt.pcolormesh(arr)

def delete_isolated_non_sensitive(arr):
    n = len(arr)
    # nothing to do if fewer than 2 frames
    if n < 2:
        return
    # fix isolated at the start/end
    if arr[0] == 0 and arr[1] == 1:
        arr[0] = 1
    if arr[-1] == 0 and arr[-2] == 1:
        arr[-1] = 1
    # fix isolated in the middle
    for j in range(1, n - 1):
        if arr[j] == 0 and arr[j - 1] == 1 and arr[j + 1] == 1:
            arr[j] = 1

def find_segments(arr):
    delete_isolated_non_sensitive(arr)
    segments = []
    curr_run_start = 0
    curr_value = arr[0]
    for j, v in enumerate(np.concatenate([arr, [-1]])):
        if v != curr_value:
            curr_run_end = j if v else j - 1
            if curr_run_end != curr_run_start:
                segments.append((curr_value, curr_run_start, curr_run_end))
                curr_run_start = curr_run_end
                curr_value = v
    return segments

def pipeline(video_frame_dir):
    return find_segments(find_sensitive(video_frame_dir))

if __name__ == "__main__":
    r = find_sensitive("tmp_20240111151241/frames")
    print(find_segments(r))
    u = np.expand_dims(np.array(r), axis=0)
    mk_plot(u)
    plt.show()
    print("done")
