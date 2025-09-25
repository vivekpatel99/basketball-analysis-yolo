import cv2
from PIL import Image

# Load model directly
from transformers import AutoModelForZeroShotImageClassification, AutoProcessor

from utils.stubs import read_stub, save_stub


class TeamAssigner:
    def __init__(self, team_1="white shirt", team_2="dark blue shirt"):
        self.team_1_class_name = team_1
        self.team_2_class_name = team_2
        self.model_id = "patrickjohncyh/fashion-clip"
        self.player_teams_dict = {}  # cache for player teams

    def load_model(self):
        # Pin to a specific revision for security - using main branch
        self.processor = AutoProcessor.from_pretrained(self.model_id, revision="main")  # nosec B615
        self.model = AutoModelForZeroShotImageClassification.from_pretrained(
            self.model_id,
            revision="main",  # nosec B615
        )

    def get_player_color(self, frame, bbox):
        image_cropped = frame[int(bbox[1]) : int(bbox[3]), int(bbox[0]) : int(bbox[2])]
        image_cropped = cv2.cvtColor(image_cropped, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image_cropped)
        classes = [self.team_1_class_name, self.team_2_class_name]
        inputs = self.processor(
            text=classes, images=image, return_tensors="pt", padding=True
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
        team_id = 2 if player_color == self.team_1_class_name else 1
        self.player_teams_dict[player_id.item()] = team_id
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
            # improves efficiency
            if frame_num % 50 == 0:
                self.player_teams_dict = {}

            for player_id, track in player_track.items():
                team = self.get_player_team(
                    video_frames[frame_num], track["bbox"], player_id
                )
                player_assignment[frame_num][player_id.item()] = team

        save_stub(stub_path, player_assignment)
        return player_assignment
