import os
import time
from pathlib import Path

import torch
import yaml

from agent import select_action, train_step
from env import PufferEnv
from model import QNetwork
from replaybuffer import ReplayBuffer


def init_phi(s, frame_stack):
    '''Stack the initial observation cfg['FRAME_STACK'] times (DQN warm-start).'''
    return s.repeat(1, frame_stack)


def next_phi(phi, s_n, term, frame_stack):
    '''Roll the frame stack forward; reset stacks for terminated agents.'''
    obs_size = s_n.shape[1]
    phi_n = torch.cat([phi[:, obs_size:], s_n], dim=1)
    reset_phi = s_n.repeat(1, frame_stack)
    done = term.bool().unsqueeze(1)
    return torch.where(done, reset_phi, phi_n)


def train(config=None):
    with open(Path(__file__).parent / 'config.yaml') as f:
        cfg = config or yaml.safe_load(f)

    #load puffer env
    env = PufferEnv('cartpole', cfg['TOTAL_AGENTS'])

    #torch optimized
    compiled_select_action = torch.compile(select_action)
    compiled_train_step = torch.compile(train_step)

    qnet = QNetwork(env.obs_size * cfg['FRAME_STACK'], env.num_actions, cfg['HIDDEN_SIZE']).cuda()
    q_target = QNetwork(env.obs_size * cfg['FRAME_STACK'], env.num_actions, cfg['HIDDEN_SIZE']).cuda()
    q_target.load_state_dict(qnet.state_dict())
    q_target.eval()
    for p in q_target.parameters():
        p.requires_grad = False
    optimizer = torch.optim.Adam(qnet.parameters(), lr=cfg['LR'])
    replay = ReplayBuffer(cfg['HISTORY'], env.obs_size * cfg['FRAME_STACK'])

    s = env.reset()
    phi = init_phi(s, cfg['FRAME_STACK'])

    global_step = 0
    update_count = 0
    start = time.time()
    loss = torch.tensor(float('nan'), device='cuda')
    for iter in range(cfg['TOTAL_ITERS']):
        #eps-greedy policy (declining schedule)
        epsilon = cfg['EPS_END'] + (cfg['EPS_START'] - cfg['EPS_END']) * max(0, 1 - iter / cfg['EPS_DECAY_ITERS'])
        a = compiled_select_action(qnet, phi, epsilon, env.num_actions, cfg['TOTAL_AGENTS'])

        #step thru w/ action, observe next state/reward
        s_n, r, term = env.step(a)

        #phi
        phi_n = next_phi(phi, s_n, term, cfg['FRAME_STACK'])

        #store transition in replay buffer
        replay.add_batch(phi, a, r, phi_n, term)

        #update according to update frequency
        if iter % cfg['UPDATE_EVERY'] == 0:
            if replay.size >= cfg['MINIBATCH']:
                batch = replay.sample(cfg['MINIBATCH'])
                loss = compiled_train_step(qnet, q_target, optimizer, *batch, cfg['GAMMA'])
                update_count += 1
                
                #sync qtarget with q according to sync frequency
                if update_count % cfg['TARGET_SYNC_EVERY'] == 0:
                    q_target.load_state_dict(qnet.state_dict())

        phi = phi_n
        global_step += cfg['TOTAL_AGENTS']

        #logging
        if iter % cfg['LOG_EVERY'] == 0:
            logs = env.log()
            score = logs.get('score', float('nan'))
            n_eps = logs.get('n', 0)
            sps = global_step / (time.time() - start)
            print(f'iter={iter:5d}  steps={global_step:10d}  eps={epsilon:.3f}  '
                  f'loss={loss.item():.3f}  replay={replay.size}/{cfg['HISTORY']}  '
                  f'episodes={n_eps:.0f}  score={score:.1f}  sps={sps:.0f}')

        #saving
        if iter % cfg['SAVE_EVERY'] == 0:
            save_path = cfg['SAVE_PATH']
            save_dir = os.path.dirname(save_path)
            save_stem, _ = os.path.splitext(os.path.basename(save_path))
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
            ckpt_path = os.path.join(save_dir, f'{save_stem}_iter{iter:05d}.pt') if save_dir else f'{save_stem}_iter{iter:05d}.pt'
            torch.save(qnet.state_dict(), ckpt_path)
            print(f'saved checkpoint to {ckpt_path}')

    total_time = time.time() - start
    print(f'total training time: {total_time:.2f} seconds')
    torch.save(qnet.state_dict(), cfg['SAVE_PATH'])
    print(f'saved checkpoint to {cfg["SAVE_PATH"]}')
    env.close()
