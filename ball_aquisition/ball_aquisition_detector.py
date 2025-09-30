from utils.bbox_utils import get_center_of_bbox, measure_distance_between_points


class BallAquisitionDetector:
    def __init__(self):
        self.possenssion_threshold = 0.5
        self.min_frames = 11
        self.containment_threshold = 0.5

    def get_key_basketball_player_assignment_points(self, player_bbox, ball_center):
        """
        ================================================================================
        COMPREHENSIVE BOUNDING BOX (BBOX) VISUALIZATION GUIDE
        ================================================================================

        🎯 COORDINATE SYSTEM BASICS:
        - Origin (0,0) is at TOP-LEFT corner
        - X increases going RIGHT →
        - Y increases going DOWN ↓

        📦 BBOX FORMAT: [x1, y1, x2, y2]
        - (x1, y1) = TOP-LEFT corner (smaller coordinates)
        - (x2, y2) = BOTTOM-RIGHT corner (larger coordinates)
        - x1 < x2 and y1 < y2 (always!)

        📐 VISUAL REPRESENTATION:

        Image/Frame (0,0) at top-left
        ┌─────────────────────────────────┐
        │ (0,0)                           │
        │  ┌─────────────────────────┐    │
        │  │ (x1,y1)                 │    │
        │  │     ┌─────────┐         │    │
        │  │     │  OBJECT │         │    │
        │  │     │         │         │    │
        │  │     └─────────┘         │    │
        │  │                 (x2,y2) │    │
        │  └─────────────────────────┘    │
        │                                 │
        └─────────────────────────────────┘

        🔢 CALCULATIONS:
        - Width = x2 - x1
        - Height = y2 - y1
        - Center X = (x1 + x2) / 2
        - Center Y = (y1 + y2) / 2
        - Area = width × height

        📍 KEY POINTS:
        - Top-left: (x1, y1)
        - Top-right: (x2, y1)
        - Bottom-left: (x1, y2)
        - Bottom-right: (x2, y2)
        - Center: ((x1+x2)/2, (y1+y2)/2)

        🎯 COMMON OPERATIONS:

        1. CHECK IF POINT IS INSIDE BBOX:
           point_inside = (x1 <= px <= x2) and (y1 <= py <= y2)

        2. CHECK IF TWO BBOXES OVERLAP:
           overlap = not (x2_1 < x1_2 or x2_2 < x1_1 or y2_1 < y1_2 or y2_2 < y1_1)

        3. GET INTERSECTION BBOX:
           ix1 = max(x1_1, x1_2)
           iy1 = max(y1_1, y1_2)
           ix2 = min(x2_1, x2_2)
           iy2 = min(y2_1, y2_2)

        4. CALCULATE IOU (Intersection over Union):
           intersection_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
           union_area = area1 + area2 - intersection_area
           iou = intersection_area / union_area

        ⚠️  COMMON MISTAKES TO AVOID:
        - Don't confuse (x1,y1) with (x2,y2) - always check which is smaller
        - Remember Y increases downward, not upward like math graphs
        - Width/Height can be 0 if x1=x2 or y1=y2 (invalid bbox)
        - Always validate: x1 < x2 and y1 < y2

        ================================================================================
        """
        ball_center_x, ball_center_y = ball_center

        player_x1, player_y1, player_x2, player_y2 = player_bbox
        width = player_x2 - player_x1
        height = player_y2 - player_y1

        output_points = []
        # check if ball is in player's bounding box (horizontally, x-axis)
        if ball_center_y > player_y1 and ball_center_y < player_y2:
            output_points.append((player_x1, ball_center_y))
            output_points.append((player_x2, ball_center_y))

        # check if ball is in player's bounding box (vertically, y-axis)
        if ball_center_x > player_x1 and ball_center_x < player_x2:
            output_points.append((ball_center_x, player_y1))
            output_points.append((ball_center_x, player_y2))

        output_points += [
            (player_x1, player_y1),  # top-left corner
            (player_x2, player_y1),  # top-right corner
            (player_x1, player_y2),  # bottom-left corner
            (player_x2, player_y2),  # bottom-right corner
            # top-center
            (player_x1 + (width // 2), player_y1),
            # bottom-center
            (player_x1 + (width // 2), player_y2),
            # left-center
            (player_x1, player_y1 + (height // 2)),
            # right-center
            (player_x2, player_y1 + (height // 2)),
        ]

        return output_points

    def find_minimum_distance_to_ball_center(self, ball_center, player_bbox):
        """4:00:00"""
        key_points = self.get_key_basketball_player_assignment_points(
            player_bbox, ball_center
        )
        return min(
            measure_distance_between_points(ball_center, point) for point in key_points
        )

    def calculate_ball_containment_ratio(self, player_bbox, ball_center):
        px1, py1, px2, py2 = player_bbox
        bx1, by1, bx2, by2 = ball_center

        ball_area = (bx2 - bx1) * (by2 - by1)

        intersection_x1 = max(px1, bx1)
        intersection_y1 = max(py1, by1)
        intersection_x2 = min(px2, bx2)
        intersection_y2 = min(py2, by2)

        intersection_area = max(0, intersection_x2 - intersection_x1) * max(
            0, intersection_y2 - intersection_y1
        )
        return intersection_area / ball_area

    def find_best_candidate_for_for_posseession(
        self, ball_center, player_tracks_frame, ball_bbox
    ):
        """There might be multiple players near to the ball, we have to find the closest one to the ball center and also has the highest ball containment ratio"""
        high_containment_players = []
        regular_distance_players = []

        for player_id, player_info in player_tracks_frame.items():
            player_bbox = player_info.get("bbox", [])
            if len(player_bbox) == 0:
                continue
            ball_containment_ratio = self.calculate_ball_containment_ratio(
                player_bbox, ball_bbox
            )
            min_distance_to_ball_center = self.find_minimum_distance_to_ball_center(
                ball_center, player_bbox
            )
            if ball_containment_ratio > self.containment_threshold:
                high_containment_players.append(
                    (player_id, ball_containment_ratio, ball_containment_ratio)
                )
            else:
                regular_distance_players.append(
                    (player_id, min_distance_to_ball_center)
                )

        # First priority high containment players
        if high_containment_players:
            return max(high_containment_players, key=lambda x: x[1])[0]

        if regular_distance_players:
            best_candidate = min(regular_distance_players, key=lambda x: x[1])[0]
            if best_candidate[1] < self.possenssion_threshold:
                # return the best candidate with the highest containment ratio
                return best_candidate[0]

        # ball is not in any player's bounding box
        return None

    def detect_ball_passession(self, player_tracks, ball_tracks):
        """ """
        num_frames = len(player_tracks)
        possession_list = [None] * num_frames
        consecutive_possession_count = {}

        for frame_num in range(num_frames):
            ball_info = ball_tracks[frame_num].get(1, {})
            if not ball_info:
                continue

            ball_bbox = ball_info.get("bbox", [])
            if not ball_bbox:
                continue

            ball_center = get_center_of_bbox(ball_bbox)
            player_id = self.find_best_candidate_for_for_posseession(
                ball_center, player_tracks[frame_num], ball_bbox
            )
            if player_id is not None:
                number_of_consecutive_frames = (
                    consecutive_possession_count.get(player_id, 0) + 1
                )
                consecutive_possession_count[player_id] = {
                    player_id: number_of_consecutive_frames
                }

                if consecutive_possession_count[player_id] >= self.min_frames:
                    possession_list[frame_num] = player_id

            else:
                consecutive_possession_count = {}

        return possession_list
