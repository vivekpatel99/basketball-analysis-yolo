from pathlib import Path
import pickle  # nosec B403 - Used for caching internal tracking data


def read_stub(read_from_stub: bool, stub_path: str):
    stub_path = Path(stub_path)
    if read_from_stub:
        if stub_path.exists():
            with stub_path.open("rb") as f:
                return pickle.load(f)  # nosec B301 - Loading cached internal tracking data
    return None


def save_stub(stub_path: str, data: dict):
    stub_path = Path(stub_path)
    stub_path.parent.mkdir(parents=True, exist_ok=True)
    with stub_path.open("wb") as f:
        pickle.dump(data, f)
