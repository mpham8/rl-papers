"""
Run evals: hellaswag, mmlu, rouge-l, rm win rate vs sft
"""

import argparse
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from datasets import load_dataset
from rouge_score import rouge_scorer
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from data import (
    _pad_sequences,
    collate_ppo_batch,
    load_rlhf_splits,
)
from model import PPOModel
from utils import load_base_model, load_ppo_model, load_rm_model, load_sft_model


def _policy_logits(model, input_ids, attention_mask):
    if isinstance(model, PPOModel):
        logits, _ = model(input_ids, attention_mask, last_token_only=False)
    else:
        logits = model(input_ids, attention_mask)
    return logits


def _continuation_mean_log_prob(model, context_ids, continuation_ids, device):
    input_ids = torch.tensor([context_ids + continuation_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    cont_start = len(context_ids)

    logits = _policy_logits(model, input_ids, attention_mask)
    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
    targets = input_ids[:, 1:]
    token_log_probs = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)

    start = max(cont_start - 1, 0)
    cont_log_probs = token_log_probs[:, start:]
    if cont_log_probs.numel() == 0:
        return torch.tensor(float("-inf"), device=device)
    return cont_log_probs.mean()


def _score_choices(model, context, choices, tokenizer, device):
    context_ids = tokenizer.encode(context, add_special_tokens=False)
    scores = []
    for choice in choices:
        continuation = choice if choice.startswith(" ") else " " + choice
        continuation_ids = tokenizer.encode(continuation, add_special_tokens=False)
        scores.append(_continuation_mean_log_prob(model, context_ids, continuation_ids, device).item())
    return scores


def eval_hellaswag(model, tokenizer, device, limit=None):
    dataset = load_dataset("hellaswag", split="validation")
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))

    correct = 0
    for example in tqdm(dataset, desc="hellaswag"):
        scores = _score_choices(model, example["ctx"], example["endings"], tokenizer, device)
        if int(example["label"]) == max(range(len(scores)), key=scores.__getitem__):
            correct += 1
    return correct / max(len(dataset), 1)


def eval_mmlu(model, tokenizer, device, limit=None):
    dataset = load_dataset("cais/mmlu", "all", split="validation")
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))

    correct = 0
    for example in tqdm(dataset, desc="mmlu"):
        question = example["question"]
        choices = example["choices"]
        context = question + "\n\n" + "\n".join(f"{chr(65 + i)}. {choice}" for i, choice in enumerate(choices)) + "\nAnswer:"
        scores = _score_choices(model, context, choices, tokenizer, device)
        if int(example["answer"]) == max(range(len(scores)), key=scores.__getitem__):
            correct += 1
    return correct / max(len(dataset), 1)


def _policy_logits_last(model, input_ids, attention_mask):
    if isinstance(model, PPOModel):
        logits, _ = model(input_ids, attention_mask, last_token_only=True)
    else:
        hidden = model.backbone(input_ids, attention_mask)
        logits = model.policy_head.lin(hidden[:, -1, :])
    return logits


@torch.inference_mode()
def _generate_summaries(model, input_ids, attention_mask, max_new_tokens, eos_token_id):
    input_ids = input_ids.clone()
    attention_mask = attention_mask.clone()
    batch_size = input_ids.size(0)
    generated = torch.full((batch_size, max_new_tokens), eos_token_id, dtype=torch.long, device=input_ids.device)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)
    step = -1

    for step in range(max_new_tokens):
        logits = _policy_logits_last(model, input_ids, attention_mask)
        next_token = logits.argmax(dim=-1)
        generated[:, step] = next_token
        finished |= next_token.eq(eos_token_id)
        if finished.all():
            break
        next_token = next_token.masked_fill(finished, eos_token_id)
        input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones((batch_size, 1), dtype=attention_mask.dtype, device=input_ids.device)],
            dim=1,
        )

    return generated[:, : step + 1]


def _decode_summaries(tokenizer, generated_ids):
    summaries = []
    for row in generated_ids:
        ids = row.tolist()
        while ids and ids[-1] == tokenizer.eos_token_id:
            ids.pop()
        text = tokenizer.decode(ids, skip_special_tokens=True)
        if not text.startswith(" "):
            text = " " + text
        summaries.append(text)
    return summaries


def eval_rouge_l(model, dataset, tokenizer, device, max_length, max_new_tokens, batch_size):
    references = {
        example["post_id"]: example["reference_summary"]
        for example in dataset
    }
    collate = partial(
        collate_ppo_batch,
        tokenizer=tokenizer,
        device=device,
        max_length=max_length,
        max_new_tokens=max_new_tokens,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_scores = []

    for batch in tqdm(loader, desc="rouge-l"):
        generated_ids = _generate_summaries(
            model,
            batch["input_ids"],
            batch["attention_mask"],
            max_new_tokens,
            tokenizer.eos_token_id,
        )
        predictions = _decode_summaries(tokenizer, generated_ids)

        for post_id, prediction in zip(batch["post_ids"], predictions):
            reference = references[post_id]
            if not reference.startswith(" "):
                reference = " " + reference
            rouge_scores.append(scorer.score(reference, prediction)["rougeL"].fmeasure)

    return sum(rouge_scores) / max(len(rouge_scores), 1)


@torch.inference_mode()
def eval_rm_win_rate(sft_model, ppo_model, reward_model, dataset, tokenizer, device, max_length, max_new_tokens, batch_size):
    collate = partial(
        collate_ppo_batch,
        tokenizer=tokenizer,
        device=device,
        max_length=max_length,
        max_new_tokens=max_new_tokens,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)
    wins = 0
    ties = 0
    total = 0

    for batch in tqdm(loader, desc="rm win rate"):
        prompt_ids = batch["input_ids"]
        prompt_mask = batch["attention_mask"]

        sft_ids = _generate_summaries(sft_model, prompt_ids, prompt_mask, max_new_tokens, tokenizer.eos_token_id)
        ppo_ids = _generate_summaries(ppo_model, prompt_ids, prompt_mask, max_new_tokens, tokenizer.eos_token_id)

        pad_id = tokenizer.pad_token_id
        for i in range(prompt_ids.size(0)):
            sft_seq = torch.cat([prompt_ids[i], sft_ids[i]])
            ppo_seq = torch.cat([prompt_ids[i], ppo_ids[i]])
            ids_list = [sft_seq, ppo_seq]
            input_ids, attention_mask = _pad_sequences(ids_list, pad_id, device)
            sft_score, ppo_score = reward_model(input_ids, attention_mask)
            if ppo_score > sft_score:
                wins += 1
            elif ppo_score == sft_score:
                ties += 1
            total += 1

    return {
        "win_rate": wins / max(total, 1),
        "tie_rate": ties / max(total, 1),
        "n": total,
    }


def _eval_model_suite(model, name, tokenizer, device, cfg, rouge_dataset, rouge_batch_size, limits):
    print(f"\n=== {name} ===")
    results = {
        "hellaswag_acc": eval_hellaswag(model, tokenizer, device, limit=limits["hellaswag"]),
        "mmlu_acc": eval_mmlu(model, tokenizer, device, limit=limits["mmlu"]),
    }
    if rouge_dataset is not None:
        results["rouge_l"] = eval_rouge_l(
            model,
            rouge_dataset,
            tokenizer,
            device,
            cfg["PPO_MAX_LENGTH"],
            cfg["PPO_MAX_NEW_TOKENS"],
            rouge_batch_size,
        )
    print(f"hellaswag_acc={results['hellaswag_acc']:.4f}")
    print(f"mmlu_acc={results['mmlu_acc']:.4f}")
    if "rouge_l" in results:
        print(f"rouge_l={results['rouge_l']:.4f}")
    return results


def run_eval(
    config=None,
    sft_checkpoint="sft.pt",
    ppo_checkpoint=None,
    base=False,
    rouge_split="validation",
    limits=None,
):
    with open(Path(__file__).parent / "config.yaml") as f:
        cfg = config or yaml.safe_load(f)

    limits = limits or {}
    limits.setdefault("hellaswag", None)
    limits.setdefault("mmlu", None)
    limits.setdefault("rouge", None)
    limits.setdefault("rm_win_rate", None)

    device = "cuda"
    dtype = torch.bfloat16
    model_name = cfg["MODEL_NAME"]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    splits = load_rlhf_splits(config=cfg)
    rouge_dataset = splits["sft"][rouge_split]
    if limits["rouge"] is not None:
        rouge_dataset = rouge_dataset.select(range(min(limits["rouge"], len(rouge_dataset))))

    rm_dataset = splits["ppo"][rouge_split]
    if limits["rm_win_rate"] is not None:
        rm_dataset = rm_dataset.select(range(min(limits["rm_win_rate"], len(rm_dataset))))

    rouge_batch_size = cfg.get("EVAL_BATCH", cfg.get("PPO_BATCH", 8))
    rm_batch_size = cfg.get("EVAL_BATCH", cfg.get("PPO_BATCH", 8))

    if base:
        eval_model = load_base_model(model_name, dtype, device)
        results = {"base": _eval_model_suite(eval_model, "base", tokenizer, device, cfg, rouge_dataset, rouge_batch_size, limits)}
    else:
        sft_model = load_sft_model(model_name, dtype, device, cfg["SFT_SAVE_PATH"], sft_checkpoint)
        results = {"sft": _eval_model_suite(sft_model, "sft", tokenizer, device, cfg, rouge_dataset, rouge_batch_size, limits)}

    if ppo_checkpoint is not None:
        ppo_model = load_ppo_model(
            model_name,
            dtype,
            device,
            cfg["SFT_SAVE_PATH"],
            cfg["PPO_SAVE_PATH"],
            sft_file=sft_checkpoint,
            ppo_file=ppo_checkpoint,
        )
        results["ppo"] = _eval_model_suite(ppo_model, "ppo", tokenizer, device, cfg, rouge_dataset, rouge_batch_size, limits)

        if not base:
            reward_model = load_rm_model(model_name, dtype, device, cfg["RM_SAVE_PATH"])
            rm_stats = eval_rm_win_rate(
                sft_model,
                ppo_model,
                reward_model,
                rm_dataset,
                tokenizer,
                device,
                cfg["PPO_MAX_LENGTH"],
                cfg["PPO_MAX_NEW_TOKENS"],
                rm_batch_size,
            )
            results["rm_win_rate"] = rm_stats
            print(f"\n=== rm win rate (ppo vs sft) ===")
            print(f"win_rate={rm_stats['win_rate']:.4f}  tie_rate={rm_stats['tie_rate']:.4f}  n={rm_stats['n']}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate SFT / PPO models")
    parser.add_argument("--base", action="store_true", help="evaluate pretrained model before SFT")
    parser.add_argument("--sft-checkpoint", default="sft.pt")
    parser.add_argument("--ppo-checkpoint", default=None, help="e.g. ppo.pt or ppo_rollout21.pt")
    parser.add_argument("--rouge-split", default="validation", choices=["validation", "train"])
    parser.add_argument("--hellaswag-limit", type=int, default=None)
    parser.add_argument("--mmlu-limit", type=int, default=None)
    parser.add_argument("--rouge-limit", type=int, default=256)
    parser.add_argument("--rm-win-rate-limit", type=int, default=256)
    args = parser.parse_args()

    run_eval(
        base=args.base,
        sft_checkpoint=args.sft_checkpoint,
        ppo_checkpoint=args.ppo_checkpoint,
        rouge_split=args.rouge_split,
        limits={
            "hellaswag": args.hellaswag_limit,
            "mmlu": args.mmlu_limit,
            "rouge": args.rouge_limit,
            "rm_win_rate": args.rm_win_rate_limit,
        },
    )


if __name__ == "__main__":
    main()
