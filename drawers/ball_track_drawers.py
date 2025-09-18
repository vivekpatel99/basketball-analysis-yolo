from drawers._utils import draw_triangle


class BallTrackDrawers:
    def __init__(self):
        self.ball_color = (0, 255, 0)

    def draw(self, frames, tracks):
        video_frames = []
        for frame_num, frame in enumerate(frames):
            frame = frame.copy()
            ball_track = tracks[frame_num]
            frame = self.draw_ball_track(frame, ball_track)
            video_frames.append(frame)
        return video_frames

    def draw_ball_track(self, frame, ball_tracks):
        for _, ball_track in ball_tracks.items():
            if ball_track["bbox"] is not None:
                frame = draw_triangle(frame, ball_track["bbox"], self.ball_color)
        return frame
