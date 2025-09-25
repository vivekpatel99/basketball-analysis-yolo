from drawers._utils import draw_ellipse, draw_traingle


class PlayerTracksDrawer:
    def __init__(self, team_1_color=[255, 245, 238], team_2_color=[128, 0, 0]):
        """
        Initialize the PlayerTracksDrawer with specified team colors.

        Args:
            team_1_color (list, optional): RGB color for Team 1. Defaults to [255, 245, 238].
            team_2_color (list, optional): RGB color for Team 2. Defaults to [128, 0, 0].
        """
        self.default_player_team_id = (
            1  # if something goes wrong, we don't get player ID then default to team 1
        )
        self.team_1_color = team_1_color
        self.team_2_color = team_2_color

    def draw(self, video_frames, tracks, player_assignment, ball_aquisition):
        output_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()
            player_tracks = tracks[frame_num]
            player_assignment_frame = player_assignment[frame_num]
            player_id_has_ball = ball_aquisition[frame_num]

            frame = self.draw_player_tracks(
                frame,
                player_tracks,
                player_assignment_frame=player_assignment_frame,
                player_id_has_ball=player_id_has_ball,
            )
            output_frames.append(frame)

        return output_frames

    def draw_player_tracks(
        self, frame, player_tracks, player_assignment_frame, player_id_has_ball
    ):
        for track_id, track in player_tracks.items():
            team_id = player_assignment_frame.get(track_id, self.default_player_team_id)
            if team_id == self.default_player_team_id:
                color = self.team_1_color
            else:
                color = self.team_2_color
            frame = draw_ellipse(frame, track["bbox"], color, track_id)
            if track_id == player_id_has_ball:
                frame = draw_traingle(frame, track["bbox"], color)

        return frame
