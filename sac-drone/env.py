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
    def __init__(self, env_name, total_agents):
        self.env_name = env_name
        self.total_agents = total_agents

        self.vec = self._make_vec()
        self.obs_size = self.vec.obs_size
        self.num_actions = self.vec.act_sizes[0]

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
        return _C.create_vec(args, 1)

    def reset(self):
        self.vec.reset()
        return self.vec_state.clone()

    def step(self, actions):
        self.vec.gpu_step(actions.to(torch.float32).contiguous().data_ptr())
        torch.cuda.synchronize()
        return (
            self.vec_state.clone(),
            self.vec_reward.clone(),
            self.vec_terminal.clone(),
        )

    def log(self):
        return self.vec.log()

    def close(self):
        self.vec.close()
