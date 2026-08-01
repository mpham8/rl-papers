from pathlib import Path

import torch


def save_checkpoint(model, save_dir, filename):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / filename
    torch.save(model.state_dict(), save_path)
    print(f"saved checkpoint to {save_path}")
    return save_path
