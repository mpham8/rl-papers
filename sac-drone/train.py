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
    env = PufferEnv('drone', cfg['TOTAL_AGENTS'])

    #torch optimized
    compiled_select_action = torch.compile(select_action)
    compiled_train_step = torch.compile(train_step)
    
    num_states = env.obs_size
    num_actions = env.num_actions    

    global_step = 0
    update = 0

    model_p = PolicyFunctionApproximation(num_states, cfg['HIDDEN_SIZE'], num_actions).cuda()
    optimizer_p = torch.optim.Adam(model_p.parameters(), lr=cfg['LR'])


    start = time.time()
    states_t = env.reset()
    while global_step < cfg['TOTAL_ITERS']:
        

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
