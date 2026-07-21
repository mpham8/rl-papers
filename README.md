# rl-papers

Implementating some RL papers I read. I use [PufferLib 4.0](https://github.com/PufferAI/PufferLib) for environment sim - their native Ocean envs are so insanely fast (w/ 50k parallel agents 30M+ sps on Cartpole). 

PufferLib 4.0 deprecated a lot of the Python and Gymnasium features in favor of faster C implementation, but I like fast iteration with Python and Torch (understandably trading off slower perf than native PufferLib C), so in each project I wrap PufferLib's `pufferlib._C`, the PyBind11 bindings for the native C/CUDA vector env, in a small `env.py` file with familiar Gymnasium env syntax and zero-copy views of the GPU for nn modelling with Torch.

## Papers

| Paper | Algorithm | Implementation |
|-------|-----------|----------------|
| [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602) (Mnih et al., 2013) | DQN | [`dqn-cartpole/`](dqn-cartpole/) |
| [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) (Schulman et al., 2017) | PPO | [`ppo-pong/`](ppo-pong/) |


## Running

After installing PufferLib, build the Ocean env for your project from the PufferLib repo root with:
```bash
./build.sh [ocean environment] --float
```
navigate to the corresponding folder in this repo for the project then run:
```bash
python3 run.py
```

