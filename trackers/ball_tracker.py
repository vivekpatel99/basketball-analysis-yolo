import pandas as pd
from ultralytics import YOLO
import supervision as sv
from utils.stubs import read_stub, save_stub
import numpy as np


class BallTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect_frames(self, frames, batch_size=20):
        detections = []
        for i in range(0, len(frames), batch_size):
            batch = frames[i : i + batch_size]
            results = self.model.predict(batch, conf=0.5, iou=0.7)
            detections.extend(results)
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        """
        Get player tracking results for a sequence of frames with optional caching.

        Args:
            frames (list): List of video frames to process.
            read_from_stub (bool): Whether to attempt reading cached results.
            stub_path (str): Path to the cache file.

        Returns:
            list: List of dictionaries containing player tracking information for each frame,
                where each dictionary maps player IDs to their bounding box coordinates.
        """
        tracks = read_stub(read_from_stub, stub_path)
        if tracks is not None:
            if len(tracks) == len(frames):
                return tracks
        detections = self.detect_frames(frames)
        tracks = []
        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}
            max_confidence = 0
            chosen_bbox = None

            # Convert to supervision Detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)
            tracks.append({})
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                confidence = frame_detection[2]
                # track_id = frame_detection[4]
                if cls_id == cls_names_inv["Ball"]:
                    if max_confidence < confidence:
                        chosen_bbox = bbox
                        max_confidence = confidence

            if chosen_bbox is not None:
                tracks[frame_num][1] = {"bbox": chosen_bbox}

        save_stub(stub_path, tracks)
        return tracks

    def remove_wrong_detections(self, ball_position):
        MAX_ALLOWED_DISTANCE = 25
        LAST_GOOD_DETECTION_FRAM_IDX = -1
        for i in range(len(ball_position)):
            current_bbox = ball_position[i].get(1, {}).get("bbox", [])
            if len(current_bbox) == 0:
                continue
            if LAST_GOOD_DETECTION_FRAM_IDX == -1:
                LAST_GOOD_DETECTION_FRAM_IDX = i
                continue

            last_good_bbox = (
                ball_position[LAST_GOOD_DETECTION_FRAM_IDX].get(1, {}).get("bbox", [])
            )
            frame_gap = i - LAST_GOOD_DETECTION_FRAM_IDX
            adjusted_max_distance = MAX_ALLOWED_DISTANCE * frame_gap

            norm_distance = (
                np.linalg.norm(
                    np.array(current_bbox[:2]) - np.array(last_good_bbox[:2])
                )
                > adjusted_max_distance
            )
            if norm_distance:
                ball_position[i][1]["bbox"] = []
            else:
                LAST_GOOD_DETECTION_FRAM_IDX = i

        return ball_position

    def interpolate_ball_position(self, ball_position):
        ball_position = [x.get(1, {}).get("bbox", []) for x in ball_position]
        df_ball_position = pd.DataFrame(ball_position, columns=["x1", "y1", "x2", "y2"])
        df_ball_position = df_ball_position.interpolate(method="linear")
        df_ball_position = df_ball_position.bfill()

        ball_position = [{1: {"bbox": x}} for x in df_ball_position.to_numpy().tolist()]
        return ball_position
