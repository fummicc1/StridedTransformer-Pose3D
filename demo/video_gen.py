import cv2
import time

if __name__ == "__main__":
    cap = cv2.VideoCapture(1)
    FPS = int(cap.get(cv2.CAP_PROP_FPS)) * 5
    ret = True
    frames = []
    i = 0
    while ret:
        i += 1
        i %= FPS
        if i % FPS != 0:
            continue
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        cv2.imshow("webcam", frame)
        if len(frames) == 200:
            out_file_path = f"{time.time()}.mp4"
            codec = cv2.VideoWriter.fourcc(*'mp4v')
            OUT_FPS = 5
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            video_writer = cv2.VideoWriter(filename=out_file_path, fourcc=codec, fps=OUT_FPS, frameSize=(width, height))
            for f in frames:
                video_writer.write(f)
            frames = []
    cap.release()
    cv2.destroyAllWindows()
