import torch



def select_action(model_p, states):
    with torch.no_grad():
        actions, _, _, _, _ = model_p.sample(states)
    return actions


def train_step(model_v, model_q1, model_q2, model_p, model_vtarget, optimizer_v, optimizer_q1, optimizer_q2, optimizer_p, batch, cfg):
    states, actions, rewards, states_next, terminals = batch
    
    #update soft value function
    action_current_policy, log_prob, _, _, _ = model_p.sample(states)

    q1_current_policy = model_q1(states, action_current_policy)
    q2_current_policy = model_q2(states, action_current_policy)
    q = torch.min(q1_current_policy, q2_current_policy)
    
    v = model_v(states)
    loss_v = 0.5 * (v - (q - cfg['ALPHA'] * log_prob).detach()).pow(2).mean()

    optimizer_v.zero_grad()
    loss_v.backward()
    optimizer_v.step()


    #update soft q function
    #rewards is an n-step discounted sum, so the bootstrap sits n steps out
    with torch.no_grad():
        v_target_next = model_vtarget(states_next)
        q_target = cfg['REWARD_SCALE'] * rewards + cfg['GAMMA'] ** cfg['NSTEP'] * v_target_next * (1-terminals)

    q1_buffer = model_q1(states, actions)
    q2_buffer = model_q2(states, actions)
    loss_q1 = 0.5 * (q1_buffer - q_target).pow(2).mean()
    loss_q2 = 0.5 * (q2_buffer - q_target).pow(2).mean()

    optimizer_q1.zero_grad()
    loss_q1.backward()
    optimizer_q1.step()

    optimizer_q2.zero_grad()
    loss_q2.backward()
    optimizer_q2.step()


    #update policy function
    action_current_policy, log_prob, _, _, _ = model_p.sample(states)
    q1_current_policy = model_q1(states, action_current_policy)
    q2_current_policy = model_q2(states, action_current_policy)
    q = torch.min(q1_current_policy, q2_current_policy)

    loss_p = (cfg['ALPHA'] * log_prob - q).mean()

    optimizer_p.zero_grad()
    loss_p.backward()
    optimizer_p.step()


    #update value target
    with torch.no_grad():
        for tp, p in zip(model_vtarget.parameters(), model_v.parameters()):
            tp.data.copy_(cfg['TAU'] * p.data + (1 - cfg['TAU']) * tp.data)

    return loss_v.item(), loss_q1.item(), loss_q2.item(), loss_p.item()

