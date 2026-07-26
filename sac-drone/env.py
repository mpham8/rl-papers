import sys

import torch
from pufferlib import pufferl, _C


class _CudaPtr:
    '''wraps raw CUDA pointer so torch.as_tensor can directly read'''
    def __init__(self, ptr, shape, typestr='<f4'):
        self.__cuda_array_interface__ = {
            'data': (ptr, False),
            'shape': shape,
            'typestr': typestr,
            'version': 2,
        }


class PufferEnv:
    def __init__(self, env_name, total_agents, action_repeat=1, alive_bonus=0.0):
        self.env_name = env_name
        self.total_agents = total_agents
        self.action_repeat = action_repeat
        self.alive_bonus = alive_bonus

        self.vec = self._make_vec()
        self.obs_size = self.vec.obs_size
        #num_atns is the action dim; act_sizes is all 1s for continuous envs
        self.num_actions = self.vec.num_atns

        n, obs = self.total_agents, self.obs_size
        self.vec_state = torch.as_tensor(
            _CudaPtr(self.vec.gpu_obs_ptr, (n, obs)), device='cuda')
        self.vec_reward = torch.as_tensor(
            _CudaPtr(self.vec.gpu_rewards_ptr, (n,)), device='cuda')
        self.vec_terminal = torch.as_tensor(
            _CudaPtr(self.vec.gpu_terminals_ptr, (n,)), device='cuda')

    def _make_vec(self):
        sys.argv = ['puffer']
        args = pufferl.load_config(self.env_name)
        args['vec']['num_buffers'] = 1
        args['vec']['total_agents'] = self.total_agents
        #single-task hover only: dense positive reward, no suicide attractor —
        #the validation task for this SAC implementation. Race needs a curriculum
        #on top of this (pufferlib's own config never trains race without hover)
        args['env']['hover_frac'] = 1.0
        args['env']['race_frac'] = 0.0
        args['env']['sphere_frac'] = 0.0
        args['env']['cube_frac'] = 0.0
        args['env']['flag_frac'] = 0.0
        return _C.create_vec(args, 1)

    def reset(self):
        self.vec.reset()
        return self.vec_state.clone()

    def step(self, actions):
        #gpu_step blindly copies total_agents * num_atns floats off this pointer,
        #so an undersized buffer reads out of bounds instead of raising
        assert actions.shape == (self.total_agents, self.num_actions), \
            f'expected actions {(self.total_agents, self.num_actions)}, got {tuple(actions.shape)}'
        actions = actions.to(torch.float32).contiguous()

        #hold the action for action_repeat env steps; the env auto-resets done
        #agents mid-repeat, so stop accumulating their reward at first terminal
        #(post-reset rewards belong to the next episode, not this transition)
        total_reward = torch.zeros(self.total_agents, device='cuda')
        done = torch.zeros(self.total_agents, device='cuda')
        for _ in range(self.action_repeat):
            self.vec.gpu_step(actions.data_ptr())
            torch.cuda.synchronize()
            total_reward += self.vec_reward * (1 - done)
            done = torch.maximum(done, self.vec_terminal)

        #survival incentive: without it, a bootstrapped critic values death (0)
        #above any negative-value continuation and the policy learns to fly
        #out of bounds on purpose
        total_reward += self.alive_bonus * (1 - done)

        return (
            self.vec_state.clone(),
            total_reward,
            done,
        )

    def log(self):
        return self.vec.log()

    def close(self):
        self.vec.close()
