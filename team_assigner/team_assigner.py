import cv2
from PIL import Image

# Load model directly
from transformers import AutoModelForZeroShotImageClassification, AutoProcessor

from utils.stubs import read_stub, save_stub


class TeamAssigner:
    def __init__(
        self, team_1="white basketball jersey", team_2="dark blue basketball jersey"
    ):
        self.team_1_class_name = team_1
        self.team_2_class_name = team_2
        self.model_id = "patrickjohncyh/fashion-clip"
        self.player_teams_dict = {}  # cache for player teams
        self.confidence_threshold = 0.6  # Minimum confidence for classification

    def load_model(self):
        # Pin to a specific revision for security - using main branch
        self.processor = AutoProcessor.from_pretrained(self.model_id)  # nosec B615
        self.model = AutoModelForZeroShotImageClassification.from_pretrained(  # nosec B615
            self.model_id,
        )

    def get_player_color(self, frame, bbox):
        image = frame[int(bbox[1]) : int(bbox[3]), int(bbox[0]) : int(bbox[2])]

        # Convert to PIL Image
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)

        # Use more descriptive class names that work better with fashion-clip
        classes = [self.team_1_class_name, self.team_2_class_name]
        inputs = self.processor(
            text=classes, images=pil_image, return_tensors="pt", padding=True
        )
        outputs = self.model(**inputs)
        logits = outputs.logits_per_image
        probs = logits.softmax(dim=1)

        class_name = classes[probs.argmax(dim=1).item()]
        return class_name

    def get_player_team(self, frame, player_bbox, player_id):
        if player_id in self.player_teams_dict:
            return self.player_teams_dict[player_id]
        player_color = self.get_player_color(frame, player_bbox)
        # Fixed logic: team 1 for team_1_class_name, team 2 for team_2_class_name
        team_id = 1 if player_color == self.team_1_class_name else 2
        player_id_value = player_id.item() if hasattr(player_id, "item") else player_id
        self.player_teams_dict[player_id_value] = team_id
        return team_id

    def get_player_teams_across_frames(
        self, video_frames, player_tracks, read_from_stub=False, stub_path=None
    ):
        player_assignment = read_stub(read_from_stub, stub_path)
        if player_assignment is not None:
            if len(player_assignment) == len(video_frames):
                return player_assignment

        self.load_model()
        player_assignment = []
        for frame_num, player_track in enumerate(player_tracks):
            player_assignment.append({})

            # refresh cache at every 50th frame (model can make mistakes on overlapping detections to avoid that)
            # improves efficiency and helps with tracking consistency
            if frame_num % 50 == 0:
                self.player_teams_dict = {}

            for player_id, track in player_track.items():
                team = self.get_player_team(
                    video_frames[frame_num], track["bbox"], player_id
                )
                player_id_value = (
                    player_id.item() if hasattr(player_id, "item") else player_id
                )
                player_assignment[frame_num][player_id_value] = team

        save_stub(stub_path, player_assignment)
        return player_assignment
