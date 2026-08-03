from pathlib import Path
from model import LLMBackbone, RewardModel, SupervisedFineTuningModel

import torch


def save_checkpoint(model, save_dir, filename):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / filename
    torch.save(model.state_dict(), save_path)
    print(f"saved checkpoint to {save_path}")
    return save_path


def load_sft_model(model_name, data_type, device, checkpoint_dir, checkpoint_file="sft.pt"):

    checkpoint_path = Path(checkpoint_dir) / checkpoint_file
    model = SupervisedFineTuningModel(model_name, data_type, device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()
    return model


def load_rm_model(model_name, data_type, device, checkpoint_dir, checkpoint_file="rm.pt"):
    checkpoint_path = Path(checkpoint_dir) / checkpoint_file
    backbone = LLMBackbone(model_name, data_type, device)
    model = RewardModel(backbone, device, data_type)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()
    return model
