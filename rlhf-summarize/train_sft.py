from functools import partial
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from data import collate_sft_batch, load_rlhf_splits
from model import SupervisedFineTuningModel
from utils import save_checkpoint


def train_sft(config=None):
    with open(Path(__file__).parent / "config.yaml") as f:
        cfg = config or yaml.safe_load(f)

    splits = load_rlhf_splits(config=cfg)
    train_data = splits["sft"]["train"]
    val_data = splits["sft"]["validation"]
    print(f"sft train: {len(train_data)} examples")
    print(f"sft validation: {len(val_data)} examples")

    model_name = cfg["MODEL_NAME"]
    data_type = torch.bfloat16
    device = "cuda"
    max_length = cfg.get("SFT_MAX_LENGTH", 562)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    sft_model = SupervisedFineTuningModel(model_name, data_type, device)
    optimizer = torch.optim.Adam(sft_model.parameters(), lr=cfg["SFT_LR"], betas=(cfg["SFT_ADAM_BETA1"], cfg["SFT_ADAM_BETA2"]))

    total_steps = cfg["SFT_EPOCHS"] * (len(train_data) // cfg["SFT_BATCH"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=cfg["SFT_LR"] * cfg["SFT_LR_DECAY"],
    )

    #data loader
    collate = partial(
        collate_sft_batch,
        tokenizer=tokenizer,
        device=device,
        max_length=max_length,
    )
    train_loader = DataLoader(
        train_data,
        batch_size=cfg["SFT_BATCH"],
        shuffle=True,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=cfg["SFT_BATCH"],
        shuffle=False,
        collate_fn=collate,
    )

    save_dir = Path(cfg["SFT_SAVE_PATH"])

    #train loop
    for epoch in range(cfg["SFT_EPOCHS"]):
        sft_model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            optimizer.zero_grad()

            loss = sft_model.sft_loss(
                batch["input_ids"], batch["attention_mask"], batch["labels"]
            )

            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            n_batches += 1


        sft_model.eval()
        val_loss_total = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                loss = sft_model.sft_loss(
                    batch["input_ids"], batch["attention_mask"], batch["labels"]
                )
                val_loss_total += loss.item()
                val_batches += 1
        
        train_loss = total_loss / max(n_batches, 1)
        val_loss = val_loss_total / max(val_batches, 1)
        print(f"epoch={epoch + 1}/{cfg['SFT_EPOCHS']}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")
   
        save_checkpoint(sft_model, save_dir, f"sft_epoch{epoch + 1}.pt")

    save_checkpoint(sft_model, save_dir, "sft.pt")

    return sft_model
