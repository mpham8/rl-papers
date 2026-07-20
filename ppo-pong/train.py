import os
import time
from pathlib import Path

import torch
import yaml

from agent import select_action, train_step
from env import PufferEnv


def train(config=None):
    with open(Path(__file__).parent / 'config.yaml') as f:
        cfg = config or yaml.safe_load(f)

    #load puffer env
    env = PufferEnv('cartpole', cfg['TOTAL_AGENTS'])

    #torch optimized
    compiled_select_action = torch.compile(select_action)
    compiled_train_step = torch.compile(train_step)

 
 
    for iter in range(cfg['TOTAL_ITERS']):



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
