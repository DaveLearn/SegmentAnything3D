from pathlib import Path
import os

def get_sam_checkpoint():
    models_dir = Path.home() / ".frame-seg-init" / "models"
    checkpoint_file = str(models_dir / "sam_vitl.pth")
    checkpoint_url = (
            "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth"
    )
    # check if we have the checkpoint in ./models. If not, download it
    if not os.path.exists(checkpoint_file):
        print(f"Downloading model from {checkpoint_url}")
        models_dir.mkdir(parents=True, exist_ok=True)
        # download model using streaming=True to download large files
        import requests

        r = requests.get(checkpoint_url, stream=True)
        with open(checkpoint_file, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

    return checkpoint_file

if __name__ == "__main__":
    checkpoint_file = get_sam_checkpoint()
    print(f"Sam checkpoint in located at {checkpoint_file}")
