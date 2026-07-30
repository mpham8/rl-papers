# rl-papers

Implementating some RL papers I read. I use [PufferLib 4.0](https://github.com/PufferAI/PufferLib) for environment sim - their native Ocean envs are so insanely fast (w/ 50k parallel agents 30M+ sps on Cartpole). PufferLib 4.0 deprecated a lot of the Python and Gymnasium features in favor of C implementation, but I like fast iteration with Python and Torch (understandably trading off slower perf than PufferLib in C), so in each project I wrap PufferLib's `pufferlib._C`, the PyBind11 bindings for the native C/CUDA vector env, in a small `env.py` file with familiar Gymnasium env syntax and zero-copy views of the GPU for nn modelling with Torch.

## Papers

| Paper | Algorithm | Implementation | Description |
|-------|-----------|----------------|-------------|
| [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602) (Mnih et al., 2013) | DQN | [`dqn-cartpole/`](dqn-cartpole/) | DQN to balance Cartpole (scores 200 after 5 sec of training on RTX 3090) |
| [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) (Schulman et al., 2017) | PPO | [`ppo-pong/`](ppo-pong/) | PPO to play Pong (scores 21 on RTX 3090) |
| [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783) (Mnih et al., 2016) | A2C | [`a2c-breakout/`](a3c-breakout/) | A2C to play Breakout (scores 330 after 100 min of training on RTX 3090) |
| [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/abs/1801.01290) (Haarnoja et al., 2018) | SAC-v1 | [`sac-drone/`](sac-drone/) | SAC to hover drone (scores ~475, perf ~0.7 after 15 min of training on RTX 3090) |
| [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) (Ouyang et al., 2022) | RLHF (PPO) | [`rlhf-gemma/`](rlhf-summarize/) | Reproduces the InstructGPT RLHF pipeline - SFT → reward model → PPO on [Gemma 3 270M](https://huggingface.co/google/gemma-3-270m) trained on [openai/summarize_from_feedback dataset](https://huggingface.co/datasets/openai/summarize_from_feedback) for summary alignment. Pre-training using (PPO-ptx term) with [fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu). (Benchmarked ROUGE-L: X%→X%, RM win-rate vs. SFT: X%, MMLU regression: X%→X%, HellaSwag regression: X%→X%.) |


## Running

(If doing an RL env) After installing PufferLib, build the desired Ocean env in the PufferLib repo root with:
```bash
./build.sh [ocean environment] --float
```
navigate to the corresponding folder in this repo for the project then run:
```bash
python3 run.py
```

