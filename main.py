from pathlib import Path

from drawers import PlayerTracksDrawer
from drawers.ball_track_drawers import BallTrackDrawers
from team_assigner import TeamAssigner
from trackers import BallTracker, PlayerTracker
from utils.utils import read_video, save_video


def main():
    video_path = Path.cwd() / "input_videos" / "video_1.mp4"
    frames = read_video(video_path)
    player_tracker = PlayerTracker(Path.cwd() / "trained_models/player_detector.pt")
    ball_tracker = BallTracker(Path.cwd() / "trained_models/player_detector.pt")

    player_tracks = player_tracker.get_object_tracks(
        frames,
        read_from_stub=True,
        stub_path=Path.cwd() / "stubs" / "player_tracks.pkl",
    )
    ball_tracks = ball_tracker.get_object_tracks(
        frames, read_from_stub=True, stub_path=Path.cwd() / "stubs" / "ball_tracks.pkl"
    )
    ball_tracks = ball_tracker.remove_wrong_detections(ball_tracks)
    ball_tracks = ball_tracker.interpolate_ball_position(ball_tracks)

    team_assigner = TeamAssigner()
    player_teams = team_assigner.get_player_teams_across_frames(
        frames,
        player_tracks,
        read_from_stub=True,
        stub_path=Path.cwd() / "stubs" / "player_teams.pkl",
    )
    print(player_teams)
    player_tracks_drawer = PlayerTracksDrawer()
    ball_tracks_drawer = BallTrackDrawers()
    frames = ball_tracks_drawer.draw(frames, ball_tracks)
    frames = player_tracks_drawer.draw(frames, player_tracks, None, ball_tracks)

    save_video(frames, Path.cwd() / "output_videos" / "output.mp4")


if __name__ == "__main__":
    main()
