import cv2
from pathlib import Path


def read_video(video_path):
    """
    Read all frames from a video file into memory.

    Args:
        video_path (str): Path to the input video file.

    Returns:
        list: List of video frames as numpy arrays.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    return frames


def save_video(ouput_video_frames, output_video_path):
    """
    Save a sequence of frames as a video file.

    Creates necessary directories if they don't exist and writes frames using XVID codec.

    Args:
        ouput_video_frames (list): List of frames to save.
        output_video_path (str): Path where the video should be saved.
    """
    # If folder doesn't exist, create it
    output_path = Path(output_video_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Use appropriate codec based on file extension
    if output_path.suffix.lower() == ".mp4":
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    else:
        fourcc = cv2.VideoWriter_fourcc(*"XVID")

    out = cv2.VideoWriter(
        output_video_path,
        fourcc,
        24.0,
        (ouput_video_frames[0].shape[1], ouput_video_frames[0].shape[0]),
    )
    for frame in ouput_video_frames:
        out.write(frame)
    out.release()
