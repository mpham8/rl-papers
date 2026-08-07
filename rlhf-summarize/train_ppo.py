from functools import partial
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

from agent import (
    build_prefix_minibatch,
    compute_generalized_advantages,
    ref_log_prob_last,
    select_action,
    train_step,
)
from data import (
    collate_ppo_batch,
    collate_pretrain_batch,
    load_pretrain_data,
    load_rlhf_splits,
    next_pretrain_batch,
)
from model import PPOModel
from utils import (
    compute_rm_offset,
    load_rm_model,
    load_rm_offset,
    load_sft_model,
    save_checkpoint,
    save_rm_offset,
)


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

    ref_model = load_sft_model(model_name, data_type, device, sft_model_checkpoint_folder, sft_model_checkpoint_file)
    reward_model = load_rm_model(model_name, data_type, device, reward_model_checkpoint_folder, reward_model_checkpoint_file)
    ppo_model = PPOModel.from_sft_model(ref_model, device, data_type)
    for param in ref_model.parameters():
        param.requires_grad = False
    ref_model.eval()
    reward_model.eval()

    rm_offset = load_rm_offset(reward_model_checkpoint_folder, cfg["RM_OFFSET_FILE"], device)
    if rm_offset is None:
        rm_offset = compute_rm_offset(reward_model, splits["sft"]["train"], tokenizer, device, max_length, batch_size=cfg.get("RM_BATCH", 8))
        save_rm_offset(rm_offset, reward_model_checkpoint_folder, cfg["RM_OFFSET_FILE"])
    print(f"rm offset: {rm_offset:.4f}")

    optimizer = torch.optim.Adam(ppo_model.parameters(), lr=cfg["PPO_LR"])


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

    pretrain_gamma = cfg["PPO_PRETRAIN_GAMMA"] 
    pretrain_example_iter = None
    pretrain_collate = None
    pretrain_batch_size = cfg["PPO_PRETRAIN_BATCH"]
    pretrain_stream_factory = None
    if pretrain_gamma > 0:
        pretrain_collate = partial(
            collate_pretrain_batch,
            tokenizer=tokenizer,
            device=device,
            max_length=max_length,
        )
        pretrain_stream_factory = lambda: load_pretrain_data(cfg)
        pretrain_example_iter = iter(pretrain_stream_factory())
        print(
            f"pretrain mix enabled: gamma={pretrain_gamma}  "
            f"dataset={cfg.get('PPO_PRETRAIN_DATASET', 'HuggingFaceFW/fineweb-edu')}"
        )

    iters = 0
    rollout = 0
    T = cfg["PPO_MAX_NEW_TOKENS"]
    train_iter = iter(train_loader)
    save_dir = Path(cfg["PPO_SAVE_PATH"])
    pbar = tqdm(total=cfg["PPO_TOTAL_ITERS"], desc="ppo")

    while iters < cfg['PPO_TOTAL_ITERS']:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        prompt_ids = batch["input_ids"]
        prompt_mask = batch["attention_mask"]

        rewards_T = torch.zeros(cfg["PPO_BATCH"], T, device=device)
        log_probs_T = torch.zeros(cfg["PPO_BATCH"], T, device=device)
        values_T = torch.zeros(cfg["PPO_BATCH"], T + 1, device=device)
        actions_T = torch.zeros(cfg["PPO_BATCH"], T, dtype=torch.long, device=device)

        #rollout token by token (no torch.compile — seq len changes each step)
        with torch.inference_mode():
            ppo_model.backbone.backbone.gradient_checkpointing_disable()
            rollout_bar = tqdm(range(T), desc="rollout", leave=False)
            for step in rollout_bar:
                if step == 0:
                    input_ids, attention_mask = prompt_ids, prompt_mask
                else:
                    generated_mask = torch.ones(cfg["PPO_BATCH"], step, dtype=prompt_mask.dtype, device=device)
                    input_ids = torch.cat([prompt_ids, actions_T[:, :step]], dim=1)
                    attention_mask = torch.cat([prompt_mask, generated_mask], dim=1)

                actions, log_probs, values = select_action(ppo_model, input_ids, attention_mask)
                actions_T[:, step] = actions
                values_T[:, step] = values
                log_probs_T[:, step] = log_probs
                ref_log_probs = ref_log_prob_last(ref_model, input_ids, attention_mask, actions)
                rewards_T[:, step] = -cfg["PPO_KL_BETA"] * (log_probs - ref_log_probs)
                rollout_bar.set_postfix(avg_reward=f"{rewards_T[:, step].mean().item():.4f}")

            generated_mask = torch.ones(cfg["PPO_BATCH"], T, dtype=prompt_mask.dtype, device=device)
            full_input_ids = torch.cat([prompt_ids, actions_T], dim=1)
            full_attention_mask = torch.cat([prompt_mask, generated_mask], dim=1)

            _, _, bootstrap_values = select_action(ppo_model, full_input_ids, full_attention_mask)
            values_T[:, T] = bootstrap_values
            ppo_model.backbone.backbone.gradient_checkpointing_enable()

        with torch.no_grad():
            rm_scores = reward_model(full_input_ids, full_attention_mask)
        rewards_T[:, T - 1] += rm_scores - rm_offset

        avg_reward = rewards_T.sum(dim=1).mean().item()
        avg_rm = (rm_scores - rm_offset).mean().item()
        print(f"rollout={rollout + 1}  avg_reward={avg_reward:.4f}  avg_rm={avg_rm:.4f}")

        gae, returns = compute_generalized_advantages(rewards_T, values_T, cfg["PPO_GAMMA"], cfg["PPO_LAMBDA"], T, cfg["PPO_BATCH"])
   

        NT = cfg["PPO_BATCH"] * T
        clip_eps = cfg["PPO_EPS"]
        pad_id = tokenizer.pad_token_id
        update_steps = cfg["PPO_EPOCHS"] * NT // cfg["PPO_MINIBATCH"]
        ppo_model.train()
        for i in tqdm(range(0, cfg["PPO_EPOCHS"] * NT, cfg["PPO_MINIBATCH"]), desc="ppo update", leave=False):
            minibatch_idx = torch.randperm(NT, device=device)[: cfg["PPO_MINIBATCH"]]
            t_idx = minibatch_idx // cfg["PPO_BATCH"]
            n_idx = minibatch_idx % cfg["PPO_BATCH"]

            input_ids_mb, attention_mask_mb = build_prefix_minibatch(prompt_ids, actions_T, n_idx, t_idx, pad_id, device)

            pretrain_input_ids = None
            pretrain_attention_mask = None
            pretrain_batch, pretrain_example_iter = next_pretrain_batch(
                pretrain_example_iter,
                pretrain_collate,
                pretrain_batch_size,
                pretrain_stream_factory,
            )
            pretrain_input_ids = pretrain_batch["input_ids"]
            pretrain_attention_mask = pretrain_batch["attention_mask"]
           
            train_step(ppo_model, optimizer, input_ids_mb, attention_mask_mb, actions_T[n_idx, t_idx], gae[t_idx, n_idx], returns[t_idx, n_idx], log_probs_T[n_idx, t_idx], clip_eps, cfg["PPO_C1"], pretrain_input_ids=pretrain_input_ids, pretrain_attention_mask=pretrain_attention_mask, pretrain_gamma=pretrain_gamma)
       
       
       

        rollout += 1
        save_checkpoint(ppo_model, save_dir, f"ppo_rollout{rollout}.pt")

        iters += cfg["PPO_BATCH"]
        pbar.update(cfg["PPO_BATCH"])
        pbar.set_postfix(rollout=rollout, avg_reward=f"{avg_reward:.4f}", update_steps=update_steps)

    pbar.close()
    save_checkpoint(ppo_model, save_dir, "ppo.pt")
