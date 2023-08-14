import cv2
import time
import pathlib
import subprocess
import shutil

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    # 5秒ごとに一回撮像する
    seconds_per_infer = 5
    duration = 40 # 40秒
    FRAMES = int(cap.get(cv2.CAP_PROP_FPS)) * seconds_per_infer
    ret = True
    frames = []
    i = 0
    while ret:
        i += 1
        i %= FRAMES
        if i % FRAMES != 0:
            continue
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        OUT_FPS = int(cap.get(cv2.CAP_PROP_FPS))
        if len(frames) == OUT_FPS * duration:
            out_file_path = str(time.time()).replace(".", '-')
            out_file_path = f"{out_file_path}.mp4"
            codec = cv2.VideoWriter.fourcc(*'mp4v')
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            video_writer = cv2.VideoWriter(filename=out_file_path, fourcc=codec, fps=OUT_FPS, frameSize=(width, height))
            for f in frames:
                video_writer.write(f)
            video_writer.release()
            frames = []
            out_file_path = pathlib.Path(out_file_path)
            copy_out_file_path = pathlib.Path(f"demo/video/{out_file_path.name}")
            vis_path = copy_out_file_path.parent.parent / "vis.py"
            shutil.move(out_file_path, copy_out_file_path)
            print(out_file_path.name)
            subprocess.Popen(f"python {vis_path} --video {out_file_path}", shell=True)
            break
    cap.release()
    cv2.destroyAllWindows()
