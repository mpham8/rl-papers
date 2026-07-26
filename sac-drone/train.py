import os
import time
from pathlib import Path
import copy

import torch
import wandb
import yaml

from agent import select_action, train_step
from env import PufferEnv
from model import SoftValueFunction, SoftQFunction, PolicyFunction
from replaybuffer import ReplayBuffer, NStepAccumulator



def train(config=None):
    with open(Path(__file__).parent / 'config.yaml') as f:
        cfg = config or yaml.safe_load(f)

    wandb_run = None
    if cfg.get('WANDB_ENABLED', True):
        init_kwargs = {
            'project': cfg.get('WANDB_PROJECT', 'sac-drone'),
            'config': cfg,
        }
        if cfg.get('WANDB_ENTITY'):
            init_kwargs['entity'] = cfg['WANDB_ENTITY']
        if cfg.get('WANDB_RUN_NAME'):
            init_kwargs['name'] = cfg['WANDB_RUN_NAME']
        wandb_run = wandb.init(**init_kwargs)

    #load puffer env
    env = PufferEnv('drone', cfg['TOTAL_AGENTS'], action_repeat=cfg['ACTION_REPEAT'],
                    alive_bonus=cfg['ALIVE_BONUS'])

    #torch optimized
    compiled_select_action = torch.compile(select_action)
    
    num_states = env.obs_size
    num_actions = env.num_actions

    replay = ReplayBuffer(cfg['BUFFER_SIZE'], num_states, num_actions)
    nstep = NStepAccumulator(cfg['NSTEP'], cfg['GAMMA'], replay)

    global_step = 0
    update = 0

    model_v = SoftValueFunction(num_states, cfg['HIDDEN_SIZE']).cuda()
    model_q1 = SoftQFunction(num_states, num_actions, cfg['HIDDEN_SIZE']).cuda()
    model_q2 = SoftQFunction(num_states, num_actions, cfg['HIDDEN_SIZE']).cuda()
    model_p = PolicyFunction(num_states, num_actions, cfg['HIDDEN_SIZE'], cfg['LOG_STD_MIN'], cfg['LOG_STD_MAX']).cuda()
    model_vtarget = copy.deepcopy(model_v)

    for param in model_vtarget.parameters():
        param.requires_grad = False

    optimizer_v = torch.optim.Adam(model_v.parameters(), lr = cfg['LR'])
    optimizer_q1 = torch.optim.Adam(model_q1.parameters(), lr = cfg['LR'])
    optimizer_q2 = torch.optim.Adam(model_q2.parameters(), lr = cfg['LR'])
    optimizer_p = torch.optim.Adam(model_p.parameters(), lr = cfg['LR'])

    start = time.time()
    states_t = env.reset()
    while global_step < cfg['TOTAL_ITERS']:
        #select action
        actions_t = select_action(model_p, states_t)

        #step through action a
        states_next, rewards_t, terminals_t = env.step(actions_t)

        #add transition to replay buffer (n-step accumulator emits once it has NSTEP steps)
        nstep.add_batch(states_t, actions_t, rewards_t, states_next, terminals_t)
        states_t = states_next
   
        global_step += cfg['TOTAL_AGENTS']

        if replay.size >= cfg['MINIBATCH']:
            for i in range(cfg['TARGET_GRAD_STEPS']):
                #sample minibatch
                batch = replay.sample(cfg['MINIBATCH'])
                
                #update
                loss_v, loss_q1, loss_q2, loss_p = train_step(
                    model_v, model_q1, model_q2, model_p, model_vtarget,
                    optimizer_v, optimizer_q1, optimizer_q2, optimizer_p,
                    batch, cfg,
                )
                update += 1

                if update % cfg['LOG_EVERY'] == 0:
                    logs = env.log()
                    score = logs.get('score', float('nan'))
                    perf = logs.get('perf', float('nan'))
                    n_eps = logs.get('n', 0)
                    sps = global_step / (time.time() - start)
                    print(f'update={update:5d}  steps={global_step:10d}  '
                        f'loss_v={loss_v:.3f}  loss_q1={loss_q1:.3f}  loss_q2={loss_q2:.3f}  '
                        f'loss_p={loss_p:.3f}  replay={replay.size}/{cfg["BUFFER_SIZE"]}  '
                        f'episodes={n_eps:.0f}  score={score:.1f}  perf={perf:.3f}  sps={sps:.0f}')
                    if wandb_run is not None:
                        wandb.log({
                            'loss/v': loss_v,
                            'loss/q1': loss_q1,
                            'loss/q2': loss_q2,
                            'loss/policy': loss_p,
                            'replay/size': replay.size,
                            'env/episodes': n_eps,
                            'env/score': score,
                            'env/perf': perf,
                            'perf/sps': sps,
                            'global_step': global_step,
                        }, step=update)

                if update % cfg['SAVE_EVERY'] == 0:
                    for save_path, model in (
                        (cfg['POLICY_FCN_SAVE_PATH'], model_p),
                        (cfg['VALUE_FCN_SAVE_PATH'], model_v),
                        (cfg['VALUE_TARGET_FCN_SAVE_PATH'], model_vtarget),
                        (cfg['Q1_FCN_SAVE_PATH'], model_q1),
                        (cfg['Q2_FCN_SAVE_PATH'], model_q2),
                    ):
                        save_dir = os.path.dirname(save_path)
                        save_stem, _ = os.path.splitext(os.path.basename(save_path))
                        if save_dir:
                            os.makedirs(save_dir, exist_ok=True)
                        ckpt_path = (
                            os.path.join(save_dir, f'{save_stem}_update{update:05d}.pt')
                            if save_dir else f'{save_stem}_update{update:05d}.pt'
                        )
                        torch.save(model.state_dict(), ckpt_path)
                        print(f'saved checkpoint to {ckpt_path}')

    
    total_time = time.time() - start
    print(f'total training time: {total_time:.2f} seconds')
    if wandb_run is not None:
        wandb.log({'train/total_time_s': total_time}, step=update)
        wandb.finish()
    for save_path, model in (
        (cfg['POLICY_FCN_SAVE_PATH'], model_p),
        (cfg['VALUE_FCN_SAVE_PATH'], model_v),
        (cfg['VALUE_TARGET_FCN_SAVE_PATH'], model_vtarget),
        (cfg['Q1_FCN_SAVE_PATH'], model_q1),
        (cfg['Q2_FCN_SAVE_PATH'], model_q2),
    ):
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f'saved checkpoint to {save_path}')
    env.close()
