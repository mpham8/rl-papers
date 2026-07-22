import os
import time
from pathlib import Path

import torch
import yaml

from agent import select_action, train_step, compute_return_advantage
from env import PufferEnv
from model import PolicyFunctionApproximation
from model import ValueFunctionApproximation



def train(config=None):
    with open(Path(__file__).parent / 'config.yaml') as f:
        cfg = config or yaml.safe_load(f)

    #load puffer env
    env = PufferEnv('breakout', cfg['TOTAL_AGENTS'])

    #torch optimized
    compiled_select_action = torch.compile(select_action)
    compiled_train_step = torch.compile(train_step)
    compiled_compute_return_advantage = torch.compile(compute_return_advantage)
    
    num_states = env.obs_size
    num_actions = env.num_actions    

    global_step = 0
    update = 0

    model_p = PolicyFunctionApproximation(num_states, cfg['HIDDEN_SIZE'], num_actions).cuda()
    model_v = ValueFunctionApproximation(num_states, cfg['HIDDEN_SIZE']).cuda()
    optimizer_p = torch.optim.Adam(model_p.parameters(), lr=cfg['LR'])
    optimizer_v = torch.optim.Adam(model_v.parameters(), lr=cfg['LR'])


    start = time.time()
    states_t = env.reset()
    while global_step < cfg['TOTAL_ITERS']:
        actions_T = torch.zeros(cfg['T_MAX'], cfg['TOTAL_AGENTS'], dtype=torch.long).cuda()
        states_T = torch.zeros(cfg['T_MAX'], cfg['TOTAL_AGENTS'], num_states).cuda()
        rewards_T = torch.zeros(cfg['T_MAX'], cfg['TOTAL_AGENTS']).cuda()
        terminals_T = torch.zeros(cfg['T_MAX'], cfg['TOTAL_AGENTS']).cuda()
        values_T = torch.zeros(cfg['T_MAX'], cfg['TOTAL_AGENTS']).cuda()

        
        #T_MAX rollouts
        with torch.no_grad():
            for t in range(cfg['T_MAX']):
                
                #sample action, step thru
                actions_T[t], values_T[t] = compiled_select_action(model_p, model_v, states_t)
                states_t, rewards_T[t], terminals_T[t] = env.step(actions_T[t])
                states_T[t, :, :] = states_t

                global_step += cfg['TOTAL_AGENTS']
            
        #bootstrap
        values_bootstrap = model_v(states_t).squeeze(-1) * (1 - terminals_T[cfg['T_MAX'] - 1])

        #calculate advantages
        advantages, returns = compute_return_advantage(rewards_T, values_bootstrap, values_T, terminals_T, cfg)

        #linear annealing from LR to 0
        lr = cfg['LR'] * (1.0 - min(global_step / cfg['TOTAL_ITERS'], 1.0))
        for optimizer in (optimizer_p, optimizer_v):
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        loss_p, loss_v = train_step(model_p, model_v, optimizer_p, optimizer_v, states_T, actions_T, advantages, returns, cfg)
        update += 1

        #logging
        if update % cfg['LOG_EVERY'] == 0:
            logs = env.log()
            score = logs.get('score', float('nan'))
            n_eps = logs.get('n', 0)
            sps = global_step / (time.time() - start)
            print(f'update={update:5d}  steps={global_step:10d}  '
                  f'loss_p={loss_p.item():.3f}  loss_v={loss_v.item():.3f}  '
                  f'episodes={n_eps:.0f}  score={score:.1f}  sps={sps:.0f}')
           

        #saving model weights
        if update % cfg['SAVE_EVERY'] == 0:
            for save_path, model in (
                (cfg['POLICY_FCN_SAVE_PATH'], model_p),
                (cfg['VALUE_FCN_SAVE_PATH'], model_v),
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
    for save_path, model in (
        (cfg['POLICY_FCN_SAVE_PATH'], model_p),
        (cfg['VALUE_FCN_SAVE_PATH'], model_v),
    ):
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f'saved checkpoint to {save_path}')
    env.close()
