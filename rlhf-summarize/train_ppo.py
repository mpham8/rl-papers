from functools import partial
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

import torch.nn.functional as F
from tqdm import tqdm


from data import collate_ppo_batch, load_rlhf_splits
from model import RewardModel, PPOModel
from utils import load_rm_model, load_sft_model, save_checkpoint


def train_ppo(config=None):
    with open(Path(__file__).parent / "config.yaml") as f:
        cfg = config or yaml.safe_load(f)

    #load data
    splits = load_rlhf_splits(config=cfg)
    train_data = splits["ppo"]["train"]
    val_data = splits["ppo"]["validation"]
    print(f"ppo train: {len(train_data)} examples")
    print(f"ppo validation: {len(val_data)} examples")

    model_name = cfg["MODEL_NAME"]
    sft_model_checkpoint_folder = cfg["SFT_SAVE_PATH"]
    reward_model_checkpoint_folder = cfg["RM_SAVE_PATH"]
    sft_model_checkpoint_file = "sft.pt"
    reward_model_checkpoint_file = "rm.pt"
    data_type = torch.bfloat16
    device = "cuda"
    max_length = cfg.get("PPO_MAX_LENGTH", cfg.get("RM_MAX_LENGTH", 638))
    max_new_tokens = cfg["PPO_MAX_NEW_TOKENS"]
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    sft_model = load_sft_model(model_name, data_type, device, sft_model_checkpoint_folder, sft_model_checkpoint_file)
    reward_model = load_rm_model(model_name, data_type, device, reward_model_checkpoint_folder, reward_model_checkpoint_file)
    ppo_model = PPOModel.from_sft_model(sft_model, device, data_type)

    optimizer = torch.optim.Adam(ppo_model.parameters(), lr=cfg["RM_LR"], betas=(cfg["RM_ADAM_BETA1"], cfg["RM_ADAM_BETA2"]))


    #data loader
    collate = partial(
        collate_ppo_batch,
        tokenizer=tokenizer,
        device=device,
        max_length=max_length,
        max_new_tokens=max_new_tokens,
    )
    train_loader = DataLoader(
        train_data,
        batch_size=cfg["PPO_BATCH"],
        shuffle=True,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=cfg["PPO_BATCH"],
        shuffle=False,
        collate_fn=collate,
    )

    iters = 0
    while iters < cfg['PPO_TOTAL_ITERS']:
        
        #rollout token by token
        for step in range(cfg['PPO_HORIZON']):

            pass
        
        #compute advantage estimate

        #PPO update
        for minibatch in minibatches:
            #do update
            pass