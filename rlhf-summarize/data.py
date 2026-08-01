"""
Load summarize_from_feedback and build SFT / RM / PPO splits.

Split posts from comparisons.train by post_id (never by row):
  SFT  12.75% of posts -> (post, ref_summary); train 88.5% / val 11.5%
  RM   45.30% of posts -> full comparison rows; train 65.0% / val 35.0%
  PPO  41.95% of posts -> posts only; train 65.8% / val 34.2%

Held-out eval uses the axis subset only (validation for iteration, test at the end).
comparisons.validation (valid1/valid2) is not used for training.
"""

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml
from datasets import Dataset, DatasetDict, load_dataset


def load_config(config=None):
    if config is not None:
        return config
    with open(Path(__file__).parent / "config.yaml") as f:
        return yaml.safe_load(f)


@dataclass(frozen=True)
class PostSplits:
    sft_train: list[str]
    sft_val: list[str]
    rm_train: list[str]
    rm_val: list[str]
    ppo_train: list[str]
    ppo_val: list[str]


def _split_posts(post_ids: list[str], train_frac: float, rng: np.random.Generator):
    shuffled = post_ids.copy()
    rng.shuffle(shuffled)
    n_train = int(len(shuffled) * train_frac)
    return shuffled[:n_train], shuffled[n_train:]


def split_post_ids(post_ids: list[str], config=None, seed: int | None = None) -> PostSplits:
    """Assign each post_id to exactly one pipeline stage, then train/val within stage."""
    cfg = load_config(config)
    seed = cfg["SEED"] if seed is None else seed
    rng = np.random.default_rng(seed)
    shuffled = post_ids.copy()
    rng.shuffle(shuffled)

    n = len(shuffled)
    sft_n = int(n * cfg["SFT_POST_FRAC"])
    rm_n = int(n * cfg["RM_POST_FRAC"])

    sft_posts = shuffled[:sft_n]
    rm_posts = shuffled[sft_n : sft_n + rm_n]
    ppo_posts = shuffled[sft_n + rm_n :]

    sft_train, sft_val = _split_posts(sft_posts, cfg["SFT_TRAIN_FRAC"], rng)
    rm_train, rm_val = _split_posts(rm_posts, cfg["RM_TRAIN_FRAC"], rng)
    ppo_train, ppo_val = _split_posts(ppo_posts, cfg["PPO_TRAIN_FRAC"], rng)

    return PostSplits(
        sft_train=sft_train,
        sft_val=sft_val,
        rm_train=rm_train,
        rm_val=rm_val,
        ppo_train=ppo_train,
        ppo_val=ppo_val,
    )


def _ref_summary(rows: list[dict]) -> str | None:
    for row in rows:
        for summary in row["summaries"]:
            if summary["policy"] == "ref":
                return summary["text"]
    return None


def _format_query(info: dict) -> str:
    return (
        f"SUBREDDIT: r/{info['subreddit']}\n\n"
        f"TITLE: {info['title']}\n\n"
        f"POST: {info['post']}\n\n"
        "TL;DR:"
    )


def _group_comparisons_by_post(comparisons) -> dict[str, list[dict]]:
    posts: dict[str, list[dict]] = defaultdict(list)
    for row in comparisons:
        posts[row["info"]["id"]].append(row)
    return posts


def build_sft_dataset(post_ids: set[str], posts: dict[str, list[dict]]) -> Dataset:
    examples = []
    for post_id in post_ids:
        ref_summary = _ref_summary(posts[post_id])
        if ref_summary is None:
            continue
        info = posts[post_id][0]["info"]
        examples.append(
            {
                "post_id": post_id,
                "query": _format_query(info),
                "reference_summary": ref_summary,
            }
        )
    return Dataset.from_list(examples)


def build_rm_dataset(post_ids: set[str], posts: dict[str, list[dict]]) -> Dataset:
    examples = []
    for post_id in post_ids:
        for row in posts[post_id]:
            chosen_idx = row["choice"]
            rejected_idx = 1 - chosen_idx
            examples.append(
                {
                    "post_id": post_id,
                    "query": _format_query(row["info"]),
                    "chosen_summary": row["summaries"][chosen_idx]["text"],
                    "rejected_summary": row["summaries"][rejected_idx]["text"],
                    "choice": chosen_idx,
                }
            )
    return Dataset.from_list(examples)


def build_ppo_dataset(post_ids: set[str], posts: dict[str, list[dict]]) -> Dataset:
    examples = []
    for post_id in post_ids:
        info = posts[post_id][0]["info"]
        examples.append(
            {
                "post_id": post_id,
                "query": _format_query(info),
            }
        )
    return Dataset.from_list(examples)


def _tokenize_query_summary(query, summary, tokenizer, max_length):
    query_ids = tokenizer.encode(query, add_special_tokens=False)
    if not summary.startswith(" "):
        summary = " " + summary
    summary_ids = tokenizer.encode(summary, add_special_tokens=False)
    eos_id = tokenizer.eos_token_id

    max_query_len = max_length - len(summary_ids) - 1
    if max_query_len < 1:
        return None
    if len(query_ids) > max_query_len:
        query_ids = query_ids[-max_query_len:]

    return query_ids + summary_ids + [eos_id]


def _pad_sequences(sequences, pad_id, device):
    batch_size = len(sequences)
    seq_len = max(len(ids) for ids in sequences)
    input_ids = torch.full((batch_size, seq_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, seq_len), dtype=torch.long)

    for i, ids in enumerate(sequences):
        input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        attention_mask[i, : len(ids)] = 1

    return input_ids.to(device), attention_mask.to(device)


def collate_rm_batch(examples, tokenizer, device, max_length):
    chosen_ids_list = []
    rejected_ids_list = []

    for example in examples:
        chosen_ids = _tokenize_query_summary(
            example["query"], example["chosen_summary"], tokenizer, max_length
        )
        rejected_ids = _tokenize_query_summary(
            example["query"], example["rejected_summary"], tokenizer, max_length
        )
        if chosen_ids is None or rejected_ids is None:
            continue
        chosen_ids_list.append(chosen_ids)
        rejected_ids_list.append(rejected_ids)

    if not chosen_ids_list:
        raise ValueError("batch is empty after tokenization")

    pad_id = tokenizer.pad_token_id
    chosen_input_ids, chosen_attention_mask = _pad_sequences(chosen_ids_list, pad_id, device)
    rejected_input_ids, rejected_attention_mask = _pad_sequences(rejected_ids_list, pad_id, device)

    return {
        "chosen_input_ids": chosen_input_ids,
        "chosen_attention_mask": chosen_attention_mask,
        "rejected_input_ids": rejected_input_ids,
        "rejected_attention_mask": rejected_attention_mask,
    }


def collate_sft_batch(examples, tokenizer, device, max_length):
    input_ids_list = []
    labels_list = []

    for example in examples:
        query_ids = tokenizer.encode(example["query"], add_special_tokens=False)
        summary = example["reference_summary"]
        if not summary.startswith(" "):
            summary = " " + summary
        summary_ids = tokenizer.encode(summary, add_special_tokens=False)

        input_ids = _tokenize_query_summary(
            example["query"], example["reference_summary"], tokenizer, max_length
        )
        if input_ids is None:
            continue
        query_len = len(input_ids) - len(summary_ids) - 1
        labels = [-100] * query_len + summary_ids + [tokenizer.eos_token_id]
        input_ids_list.append(input_ids)
        labels_list.append(labels)

    if not input_ids_list:
        raise ValueError("batch is empty after tokenization")

    batch_size = len(input_ids_list)
    seq_len = max(len(ids) for ids in input_ids_list)
    pad_id = tokenizer.pad_token_id

    input_ids = torch.full((batch_size, seq_len), pad_id, dtype=torch.long)
    labels = torch.full((batch_size, seq_len), -100, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, seq_len), dtype=torch.long)

    for i, (ids, labs) in enumerate(zip(input_ids_list, labels_list)):
        input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        labels[i, : len(labs)] = torch.tensor(labs, dtype=torch.long)
        attention_mask[i, : len(ids)] = 1

    return {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
        "labels": labels.to(device),
    }


def load_rlhf_splits(config=None, seed: int | None = None) -> dict:
    """
    Returns:
        sft, rm, ppo: DatasetDict(train/validation) built from comparisons.train
        axis: DatasetDict(validation/test) for held-out eval
        post_splits: PostSplits with post_id assignments (for debugging/repro)
    """
    cfg = load_config(config)
    comparisons = load_dataset(
        "openai/summarize_from_feedback", "comparisons", trust_remote_code=True
    )
    posts = _group_comparisons_by_post(comparisons["train"])
    post_splits = split_post_ids(list(posts.keys()), config=cfg, seed=seed)

    sft = DatasetDict(
        {
            "train": build_sft_dataset(set(post_splits.sft_train), posts),
            "validation": build_sft_dataset(set(post_splits.sft_val), posts),
        }
    )
    rm = DatasetDict(
        {
            "train": build_rm_dataset(set(post_splits.rm_train), posts),
            "validation": build_rm_dataset(set(post_splits.rm_val), posts),
        }
    )
    ppo = DatasetDict(
        {
            "train": build_ppo_dataset(set(post_splits.ppo_train), posts),
            "validation": build_ppo_dataset(set(post_splits.ppo_val), posts),
        }
    )
    axis = load_dataset("openai/summarize_from_feedback", "axis", trust_remote_code=True)
    axis = DatasetDict({"validation": axis["validation"], "test": axis["test"]})

    return {
        "sft": sft,
        "rm": rm,
        "ppo": ppo,
        "axis": axis,
        "post_splits": post_splits,
    }
