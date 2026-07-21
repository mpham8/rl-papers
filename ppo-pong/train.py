import os
import time
from pathlib import Path

import torch
import yaml

from agent import select_action, train_step, compute_return_advantage
from env import PufferEnv
from model import L_clip_vf_s


def train(config=None):
    with open(Path(__file__).parent / 'config.yaml') as f:
        cfg = config or yaml.safe_load(f)

    #load puffer env
    env = PufferEnv('pong', cfg['TOTAL_AGENTS'])

    #torch optimized
    compiled_select_action = torch.compile(select_action)
    compiled_train_step = torch.compile(train_step)
    compiled_compute_return_advantage = torch.compile(compute_return_advantage)
    
    num_states = env.obs_size
    num_actions = env.num_actions    

    global_step = 0
    update = 0

    L_nn = L_clip_vf_s(num_states, num_actions, cfg['HIDDEN_SIZE']).cuda()
    optimizer = torch.optim.Adam(L_nn.parameters(), lr = 1e-3) #TODO: change so lr decreases over time

    start = time.time()
    states_t = env.reset()
    while global_step < cfg['TOTAL_ITERS']:
        states_T = torch.zeros(cfg['HORIZON'], cfg['TOTAL_AGENTS'], num_states).cuda()
        rewards_T = torch.zeros(cfg['HORIZON'], cfg['TOTAL_AGENTS']).cuda()
        terminals_T = torch.zeros(cfg['HORIZON'], cfg['TOTAL_AGENTS']).cuda()
        log_prob_T = torch.zeros(cfg['HORIZON'], cfg['TOTAL_AGENTS']).cuda()
        values_T = torch.zeros(cfg['HORIZON'] + 1, cfg['TOTAL_AGENTS']).cuda()
        actions_T = torch.zeros(cfg['HORIZON'], cfg['TOTAL_AGENTS'], dtype=torch.long).cuda()


        # ==== run policy pi_old for T timesteps ====
        with torch.no_grad():
            for t in range(cfg['HORIZON']):
                #select action
                actions, log_prob, values_t = compiled_select_action(L_nn, states_t)
                
                #store transitions
                states_T[t, :, :] = states_t
                actions_T[t, :] = actions
                log_prob_T[t, :] = log_prob
                values_T[t, :] = values_t.squeeze(-1)
                
                #step through, observe reward
                states_t, rewards_T[t], terminals_T[t] = env.step(actions)

            #bootstrap T+1 value
            _, bootstrap_values = L_nn(states_t)
            values_T[cfg['HORIZON'], :] = bootstrap_values.squeeze(-1)
    

        # ==== compute advantage estimates and value targets====
        gae, values_target = compiled_compute_return_advantage(rewards_T, values_T, terminals_T, cfg)

        # ==== optimize surrogate L wrt theta, minibatches until K * N * T samples used ====
        NT = cfg['TOTAL_AGENTS'] * cfg['HORIZON']
        for i in range(0, cfg['EPOCHS'] * NT, cfg['MINIBATCH']):
            #sample mini batch
            minibatch_idx = torch.randperm(NT, device='cuda')[:cfg['MINIBATCH']]
            t_idx = minibatch_idx // cfg['TOTAL_AGENTS']
            n_idx = minibatch_idx % cfg['TOTAL_AGENTS']
            loss = compiled_train_step(L_nn, optimizer, states_T, actions_T, t_idx, n_idx, gae, values_target, log_prob_T, cfg)

            update += 1
        global_step += NT

        #logging
        if update % cfg['LOG_EVERY'] == 0:
            logs = env.log()
            score = logs.get('score', float('nan'))
            n_eps = logs.get('n', 0)
            sps = global_step / (time.time() - start)
            print(f'update={update:5d}  steps={global_step:10d}  '
                  f'loss={loss.item():.3f}  '
                  f'episodes={n_eps:.0f}  score={score:.1f}  sps={sps:.0f}')

        #saving model weights
        if update % cfg['SAVE_EVERY'] == 0:
            save_path = cfg['SAVE_PATH']
            save_dir = os.path.dirname(save_path)
            save_stem, _ = os.path.splitext(os.path.basename(save_path))
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
            ckpt_path = os.path.join(save_dir, f'{save_stem}_update{update:05d}.pt') if save_dir else f'{save_stem}_update{update:05d}.pt'
            torch.save(L_nn.state_dict(), ckpt_path)
            print(f'saved checkpoint to {ckpt_path}')

    
    total_time = time.time() - start
    print(f'total training time: {total_time:.2f} seconds')
    save_dir = os.path.dirname(cfg['SAVE_PATH'])
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    torch.save(L_nn.state_dict(), cfg['SAVE_PATH'])
    print(f'saved checkpoint to {cfg["SAVE_PATH"]}')
    env.close()
