import os
import time
from pathlib import Path

import torch
import yaml

from torch.distributions import Categorical

from agent import select_action, train_step, compute_advantage_gae
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
    compiled_compute_advantage_gae = torch.compile(compute_advantage_gae)
    
    num_states = env.obs_size
    num_actions = env.num_actions    

    global_step = 0
    update = 0

    L_nn = L_clip_vf_s(num_states, num_actions, cfg['HIDDEN_SIZE']).cuda()

    start = time.time()
    states_t = env.reset()
    while global_step < cfg['TOTAL_ITERS']:
        states_T = torch.zeros(cfg['HORIZON'], cfg['TOTAL_AGENTS'], num_states).cuda()
        rewards_T = torch.zeros(cfg['HORIZON'], cfg['TOTAL_AGENTS']).cuda()
        terminals_T = torch.zeros(cfg['HORIZON'], cfg['TOTAL_AGENTS']).cuda()
        log_prob_T = torch.zeros(cfg['HORIZON'], cfg['TOTAL_AGENTS']).cuda()
        values_T = torch.zeros(cfg['HORIZON'] + 1, cfg['TOTAL_AGENTS']).cuda()
        gae = torch.zeros(cfg['HORIZON'], cfg['TOTAL_AGENTS']).cuda()



        for t in range(cfg['HORIZON']):

            #run policy thru T steps
            compiled_select_action()

            policy_logits, values_t = L_nn(states_t)
            dist = Categorical(logits = policy_logits)
            policy_actions = dist.sample()
            log_prob = dist.log_prob(policy_actions)
            
            states_T[t, :, :] = states_t
            log_prob_T[t, :] = log_prob
            values_T[t, :] = values_t.squeeze(-1)
            states_t, rewards_T[t], terminals_T[t] = env.step(policy_actions)


            # print("policy_actions size:", policy_actions.size())
            # print("rewards size:", rewards_t.size())
            # print("rewards device:", rewards_t.device)

            # print("states_T_next size:", states_T.size())
            # print("rewards_T size:", rewards_T.size())
            # print("terminals_T size:", terminals_T.size())
            # print("log_prob_T size:", log_prob_T.size())

        with torch.no_grad():
            _, bootstrap_values = L_nn(states_t)
        values_T[cfg['HORIZON'], :] = bootstrap_values.squeeze(-1)
   


       
            

        #compute advantage estimate
        compiled_compute_advantage_gae()

        running_gae = torch.zeros(cfg['TOTAL_AGENTS']).cuda()
        for t in range(cfg['HORIZON'] - 1, -1, -1):
            delta = rewards_T[t, :] + cfg['GAMMA'] * values_T[t + 1, :] * (1 - terminals_T[t, :]) - values_T[t, :]
            running_gae = delta + cfg['GAMMA'] * cfg['LAMBDA'] * running_gae * (1 - terminals_T[t, :])
            gae[t] = running_gae


        #optimize surrogate L wrt theta, minibatches until K * N * T samples used
        NT = cfg['TOTAL_AGENTS'] * cfg['HORIZON']
        for i in range(0, cfg['EPOCHS'] * NT, cfg['MINIBATCH']):
            #sample mini batch
            minibatch_idx = torch.randperm(NT)[:cfg['MINIBATCH']]
            t_idx = minibatch_idx // cfg['MINIBATCH'] #idx 0 
            n_idx = minibatch_idx % cfg['MINIBATCH'] #idx 1
            loss = compiled_train_step()

            update += 1
        global_step += NT

        #logging
        if update % cfg['LOG_EVERY'] == 0:
            logs = env.log()
            score = logs.get('score', float('nan'))
            n_eps = logs.get('n', 0)
            sps = global_step / (time.time() - start)
            print(f'iter={iter:5d}  steps={global_step:10d}  eps={epsilon:.3f}  '
                f'loss={loss.item():.3f}'
                f'episodes={n_eps:.0f}  score={score:.1f}  sps={sps:.0f}')

        #saving
        if update % cfg['SAVE_EVERY'] == 0:
            save_path = cfg['SAVE_PATH']
            save_dir = os.path.dirname(save_path)
            save_stem, _ = os.path.splitext(os.path.basename(save_path))
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
            ckpt_path = os.path.join(save_dir, f'{save_stem}_iter{iter:05d}.pt') if save_dir else f'{save_stem}_iter{iter:05d}.pt'
            torch.save(L_nn.state_dict(), ckpt_path)
            print(f'saved checkpoint to {ckpt_path}')

        break
    
    total_time = time.time() - start
    print(f'total training time: {total_time:.2f} seconds')
    save_dir = os.path.dirname(cfg['SAVE_PATH'])
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    torch.save(L_nn.state_dict(), cfg['SAVE_PATH'])
    print(f'saved checkpoint to {cfg["SAVE_PATH"]}')
    env.close()
