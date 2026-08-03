from functools import partial
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

import torch.nn.functional as F
from tqdm import tqdm


from data import collate_rm_batch, load_rlhf_splits
from model import RewardModel
from utils import load_sft_model, save_checkpoint


def train_rm(config=None):
    with open(Path(__file__).parent / "config.yaml") as f:
        cfg = config or yaml.safe_load(f)

    #load data
    splits = load_rlhf_splits(config=cfg)
    train_data = splits["rm"]["train"]
    val_data = splits["rm"]["validation"]
    print(f"rm train: {len(train_data)} examples")
    print(f"rm validation: {len(val_data)} examples")

    model_name = cfg["MODEL_NAME"]
    sft_model_checkpoint_folder = cfg["SFT_SAVE_PATH"]
    sft_model_checkpoint_file = "sft.pt"
    data_type = torch.bfloat16
    device = "cuda"
    max_length = cfg.get("RM_MAX_LENGTH")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    sft_model = load_sft_model(model_name, data_type, device, sft_model_checkpoint_folder, sft_model_checkpoint_file)
    reward_model = RewardModel.from_sft_model(sft_model, device, data_type)
    optimizer = torch.optim.Adam(reward_model.parameters(), lr=cfg["RM_LR"], betas=(cfg["RM_ADAM_BETA1"], cfg["RM_ADAM_BETA2"]))

    total_steps = cfg["RM_EPOCHS"] * (len(train_data) // cfg["RM_BATCH"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=cfg["RM_LR"] * cfg["RM_LR_DECAY"],
    )


    #data loader
    collate = partial(
        collate_rm_batch,
        tokenizer=tokenizer,
        device=device,
        max_length=max_length,
    )
    train_loader = DataLoader(
        train_data,
        batch_size=cfg["RM_BATCH"],
        shuffle=True,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=cfg["RM_BATCH"],
        shuffle=False,
        collate_fn=collate,
    )

    #training loop
    for epoch in range(cfg["RM_EPOCHS"]):

        reward_model.train()
        total_loss = 0.0
        n_batches = 0

        train_pbar = tqdm(train_loader, desc=f"rm train {epoch + 1}/{cfg['RM_EPOCHS']}")
        for batch in train_pbar:
            optimizer.zero_grad()
            
            chosen_rewards = reward_model(batch["chosen_input_ids"], batch["chosen_attention_mask"])
            rejected_rewards = reward_model(batch["rejected_input_ids"], batch["rejected_attention_mask"])
            loss =  -F.logsigmoid(chosen_rewards - rejected_rewards).mean()

            loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            n_batches += 1
            train_pbar.set_postfix(loss=f"{loss.item():.4f}")

            if n_batches % 100 == 0:
                save_checkpoint(reward_model, Path(cfg["RM_SAVE_PATH"]), f"rm_batch{n_batches}.pt")


        reward_model.eval()
        val_loss_total = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"rm val {epoch + 1}/{cfg['RM_EPOCHS']}", leave=False):
                chosen_rewards = reward_model(batch["chosen_input_ids"], batch["chosen_attention_mask"])
                rejected_rewards = reward_model(batch["rejected_input_ids"], batch["rejected_attention_mask"])
                loss =  -F.logsigmoid(chosen_rewards - rejected_rewards).mean()
                val_loss_total += loss.item()
                val_batches += 1

        train_loss = total_loss / max(n_batches, 1)
        val_loss = val_loss_total / max(val_batches, 1)
        print(f"epoch={epoch + 1}/{cfg['RM_EPOCHS']}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

    
    save_checkpoint(reward_model, Path(cfg["RM_SAVE_PATH"]), "rm.pt")

    return reward_model



