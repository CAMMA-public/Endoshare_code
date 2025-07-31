import cv2
import subprocess as sp
from pathlib import Path
import sys
import os


def resource_path(relative_path):
    """
    Locate a bundled resource in Contents/Resources (or nested),
    or fall back to the source-tree path.
    """
    base = os.path.abspath(os.path.dirname(__file__))
    if getattr(sys, "frozen", False):
        bundle_dir = os.path.dirname(sys.executable)
        res_root   = os.path.abspath(os.path.join(bundle_dir, "..", "Resources"))
        res_nested = os.path.join(res_root, "Resources")
        res_macos  = bundle_dir
        for candidate in (res_root, res_nested, res_macos):
            p = os.path.join(candidate, relative_path)
            if os.path.exists(p):
                return p
        return os.path.join(res_root, relative_path)
    return os.path.join(base, relative_path)


# point to your bundled ARM64 ffmpeg (which also contains ffprobe)
FFMPEG_BIN  = resource_path("Externals/ffmpeg")
FFPROBE_BIN = FFMPEG_BIN


class VideoWorker:
  def __init__(self, logfile=None):
    self._logfile = logfile

  def log(self, s):
    if self._logfile:
      with open(self._logfile, "a") as f:
        f.write("_" * 40 + "\n" * 2 + s)

  def extract_frames(
    self,
    video_in,
    dir_out,
    frame_dim=(64, 64),
    fps=1
  ):
    cmd = [
      FFMPEG_BIN,
      "-i",
      "{}".format(video_in),
      "-filter:v",
      "fps={},scale={}:{}".format(
        fps, frame_dim[0], frame_dim[1]
      ),
      "{}/%05d.png".format(dir_out)
    ]
    self.log(" ".join(cmd))
    sp.run(cmd)

  def kf_cut(self, video_in, video_out, t1, t2, tbn=10000):
    duration = t2 - t1
    cmd = [
      FFMPEG_BIN,
      "-ss",
      "{}".format(t1),
      "-i",
      "{}".format(video_in),
      "-c",
      "copy",
      "-t",
      "{}".format(duration),
      "-video_track_timescale",
      "{}".format(tbn),
      video_out
    ]
    self.log(" ".join(cmd))
    sp.run(cmd)

  def non_kf_cut(self, video_in, video_out, t1, t2, tbn=10000):
    duration = t2 - t1
    cmd = [
      FFMPEG_BIN,
      "-ss",
      "{}".format(t1),
      "-i",
      "{}".format(video_in),
      "-c:v",
      "libx264",
      "-profile:v",
      "main",
      "-t",
      "{}".format(duration),
      "-video_track_timescale",
      "{}".format(tbn),
      video_out
    ]
    self.log(" ".join(cmd))
    sp.run(cmd)

  def list_kf(self, video_in):
    cmd_1 = [
      FFPROBE_BIN,
      "-loglevel",
      "error",
      "-select_streams",
      "v:0",
      "-show_entries",
      "packet=pts_time,flags",
      "-of",
      "csv=print_section=0",
      video_in
    ]
    self.log(" ".join(cmd_1))
    proc_1 = sp.Popen(cmd_1, stdout=sp.PIPE, stderr=sp.STDOUT)
    cmd_2 = "awk -F',' '/K/ {{print $1}}'"
    self.log(" ".join(cmd_2))
    proc_2 = sp.Popen(cmd_2, stdin=proc_1.stdout, stdout=sp.PIPE, shell=True)
    out, err = proc_2.communicate()
    raw = out.decode()
    # splitlines() skips trailing newline, and we filter out any empty strings
    lines = [line for line in raw.splitlines() if line.strip()]
    # convert only the valid, non‑empty lines
    try:
        ts = [float(t) for t in lines]
    except ValueError as e:
        # if something sneaks through, log and skip
        self.log(f"Warning: could not parse keyframe timestamps: {lines!r}\n{e}")
        ts = []
    return ts

  def cut(self, video_in, video_out, t1, t2, keyframes, tmp_dir="."):
    if not keyframes:
        # fallback: just do a non‑keyframe cut for the whole segment
        return self.non_kf_cut(video_in, video_out, t1, t2)
    if t1 in keyframes:
      self.kf_cut(video_in, video_out, t1, t2)
    elif t1 > keyframes[-1]:
      self.non_kf_cut(video_in, video_out, t1, t2)
    else:
      tkf = [kf for kf in keyframes if kf > t1][0]
      out_path = Path(video_out)
      left = tmp_dir / out_path.with_stem(out_path.stem + ".left").name
      right = tmp_dir / out_path.with_stem(out_path.stem + ".right").name
      self.non_kf_cut(video_in, str(left), t1, tkf)
      self.kf_cut(video_in, str(right), tkf, t2)
      self.merge([str(left), str(right)], video_out)
      left.unlink()
      right.unlink()

  def merge(self, video_list, video_out, tmpfile=None):
    if tmpfile is None:
        # fallback: place next to the final output
        tmpfile = Path(video_out).parent / "tmp_concat.txt"
    else:
        tmpfile = Path(tmpfile)

    txt_video_list = [f"file '{v}'\n" for v in video_list]
    with open(tmpfile, "w") as f:
        f.writelines(txt_video_list)
    cmd = [
      FFMPEG_BIN,
      "-f",
      "concat",
      "-safe",
      "0",
      "-i", str(tmpfile),
      "-c",
      "copy",
      str(video_out)
    ]
    self.log(" ".join(cmd))
    sp.run(cmd, check=True)
    try: tmpfile.unlink()
    except: pass

  def mk_black_video(self, duration, video_out, width, height, ts=10000):
    cmd = [
      FFMPEG_BIN,
      "-t",
      "{}".format(duration),
      "-f",
      "lavfi",
      "-i", f"color=c=black:s={int(width)}x{int(height)}",
      "-c:v",
      "libx264",
      "-profile:v",
      "main",
      "-video_track_timescale",
      "{}".format(ts),
      "-tune",
      "stillimage",
      "-pix_fmt",
      "yuv420p",
      video_out,
    ]
    self.log(" ".join(cmd))
    sp.run(cmd)

  def reencode(self, video_in, video_out):
    cap = cv2.VideoCapture(video_in)
    rec = cv2.VideoWriter(
      video_out,
      cv2.VideoWriter_fourcc(*"mp4v"),
      25,
      (1920, 1080)
    )
    frame_count = 0
    while True:
      r, f = cap.read()
      if not r:
        break
      rec.write(f)
      frame_count += 1
      if frame_count % 250 == 0:
        print(frame_count / 250)
    cap.release()
    rec.release()

if __name__ == "__main__":
  # mk_black_video(10, "black.mp4", 10000)
  print("done")
