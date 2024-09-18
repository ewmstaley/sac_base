'''
Copyright © 2024 The Johns Hopkins University Applied Physics Laboratory LLC
 
Permission is hereby granted, free of charge, to any person obtaining a copy 
of this software and associated documentation files (the “Software”), to 
deal in the Software without restriction, including without limitation the 
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or 
sell copies of the Software, and to permit persons to whom the Software is 
furnished to do so, subject to the following conditions:
 
The above copyright notice and this permission notice shall be included in 
all copies or substantial portions of the Software.
 
THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, 
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR 
IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''

from copy import deepcopy
import itertools
import numpy as np
import torch
from torch.optim import Adam
import time
import os
from sac_base.buffer import ReplayBuffer
from sac_base.logger import Local_Logger
import sac_base.core as core


class SACWorker:

    def __init__(self,
        env_fn, 
        make_policy_fn,
        make_q_fn,
        seed=0, 
        replay_size=int(1e6), 
        gamma=0.99, 
        polyak=0.995, 
        lr=1e-3,
        alpha=0.2, 
        alpha_is_tuned=False,
        batch_size=100, 
        start_steps=10000, 
        update_after=1000, 
        update_every=50, 
        num_test_episodes=10, 
        max_ep_len=1000, 
        save_freq=1,
        logs_dir="./logs/",
        save_dir="./saved_models/",
        **kwargs
    ):
        if logs_dir[-1] != "/": logs_dir += "/"
        if save_dir[-1] != "/": save_dir += "/"

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.env, self.test_env = env_fn(), env_fn()

        try:
            self.env.seed(seed)
            self.test_env.seed(seed)
        except:
            self.env.reset(seed=seed)
            self.test_env.reset(seed=seed)

        obs_dim = self.env.observation_space.shape
        act_dim = self.env.action_space.shape[0]
        self.Hbar = float(-act_dim)

        # Action limit for clamping: critically, assumes all dimensions share the same bound!
        act_limit = self.env.action_space.high[0]

        # Create actor-critic module and target networks
        self.ac = core.ActorCritic(self.env.observation_space, self.env.action_space, make_policy_fn, make_q_fn)
        self.ac_targ = deepcopy(self.ac)

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.ac_targ.parameters():
            p.requires_grad = False
            
        # List of parameters for both Q-networks (save this for convenience)
        self.q_params = itertools.chain(self.ac.q1.parameters(), self.ac.q2.parameters())

        # Experience buffer
        self.replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)
        self.batch_size = batch_size

        self.alpha_is_tuned = alpha_is_tuned
        self.gamma = gamma
        self.polyak = polyak

        self.max_ep_len = max_ep_len
        self.update_after = update_after
        self.update_every = update_every
        self.start_steps = start_steps

        self.test_history = []
        self.best_test = -100000000
        self.latest_batch = None
        self.total_steps = 0

        # Set up optimizers for policy and q-function
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=lr)
        self.q_optimizer = Adam(self.q_params, lr=lr)
        log_alpha = np.log(alpha)
        if alpha_is_tuned:
            self.log_alpha = torch.nn.Parameter(torch.tensor(log_alpha))
            self.a_optimizer = Adam([self.log_alpha], lr=lr)
        else:
            self.log_alpha = log_alpha

        self.logger = Local_Logger(logs_dir)
        self.save_dir = save_dir


    # q updates ============================================================

    def compute_grad_q(self):
        self.q_optimizer.zero_grad()
        data = self.latest_batch
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        q1 = self.ac.q1(o,a)
        q2 = self.ac.q2(o,a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = self.ac.pi(o2)

            # Target Q-values
            q1_pi_targ = self.ac_targ.q1(o2, a2)
            q2_pi_targ = self.ac_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            alpha = self.log_alpha.exp() if self.alpha_is_tuned else np.exp(self.log_alpha)
            backup = r + self.gamma * (1 - d) * (q_pi_targ - alpha * logp_a2)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2
        loss_q.backward()

        return loss_q.item()


    def update_q_networks(self):
        self.q_optimizer.step()

    def polyak_q(self):
        with torch.no_grad():
            for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

    def freeze_q(self):
        for p in self.q_params:
            p.requires_grad = False

    def unfreeze_q(self):
        for p in self.q_params:
            p.requires_grad = True


    # pi updates (and alpha) ==================================================

    def compute_loss_pi(self):
        self.pi_optimizer.zero_grad()
        data = self.latest_batch
        o = data['obs']

        self.freeze_q()

        pi, logp_pi = self.ac.pi(o)
        q1_pi = self.ac.q1(o, pi)
        q2_pi = self.ac.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        alpha = self.log_alpha.exp() if self.alpha_is_tuned else np.exp(self.log_alpha)
        loss_pi = (alpha * logp_pi - q_pi).mean()
        loss_pi.backward()

        if self.alpha_is_tuned:
            with torch.no_grad():
                pi, logp_pi = self.ac.pi(o)
            self.a_optimizer.zero_grad()
            alpha = self.log_alpha.exp()
            loss_alpha = torch.mean(-alpha*logp_pi - alpha*self.Hbar)
            loss_alpha.backward()
            loss_alpha_item = loss_alpha.item()
        else:
            loss_alpha_item = None

        return loss_pi.item(), loss_alpha_item

    def update_pi_networks(self):
        self.pi_optimizer.step()
        self.unfreeze_q()

    def update_alpha(self):
        if self.alpha_is_tuned:
            self.a_optimizer.step()

    # saving ===============================================================
    def save(self, explicit_file=None):
        if explicit_file is not None:
            if explicit_file[-3:] != ".pt":
                explicit_file += ".pt"
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
            torch.save(self.ac.state_dict(), self.save_dir+explicit_file)
        else:
            # only save if we have best-ever performance
            perf = np.mean(self.test_history[-20:])
            if perf > self.best_test:
                self.best_test = perf
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
                torch.save(self.ac.state_dict(), self.save_dir+"model.pt")

    # gather data ==========================================================

    def sample_latest_batch(self):
        self.latest_batch = self.replay_buffer.sample_batch(self.batch_size)

    def start_collection(self):
        self.curr_o, self.curr_ret, self.curr_len = self.env.reset(), 0, 0

    def get_action(self, o, deterministic=False, ac_override=None):
        ac = self.ac if ac_override is None else ac_override
        return ac.act(torch.as_tensor(o, dtype=torch.float32), deterministic)

    def rollout_to_next_update(self, test_if_done=True):

        if self.total_steps < self.update_after:
            target_steps = self.update_after
        else:
            target_steps = self.total_steps + self.update_every
        
        while self.total_steps < target_steps:
            if self.total_steps < self.start_steps:
                a = self.env.action_space.sample()
            else:
                a = self.get_action(self.curr_o)

            o2, r, d, _ = self.env.step(a)
            self.curr_ret += r
            self.curr_len += 1

            d = False if self.curr_len==self.max_ep_len else d
            self.replay_buffer.store(self.curr_o, a, r, o2, d)
            self.curr_o = o2

            if d or (self.curr_len==self.max_ep_len):
                # print("Reward:", self.curr_ret)
                if test_if_done:
                    self.run_test_episode()
                self.logger.log_mean_scalar("train_reward", self.curr_ret, self.total_steps)
                self.logger.log_mean_scalar("train_length", self.curr_len, self.total_steps)
                self.curr_o, self.curr_ret, self.curr_len = self.env.reset(), 0, 0

            self.total_steps += 1

        self.logger.flush()


    def run_test_episode(self):
        o, d, ep_ret, ep_len = self.test_env.reset(), False, 0, 0
        while not(d or (ep_len == self.max_ep_len)):
            # Take deterministic actions at test time 
            o, r, d, _ = self.test_env.step(self.get_action(o, True))
            ep_ret += r
            ep_len += 1
        self.logger.log_mean_scalar("test_reward", ep_ret, self.total_steps)
        self.logger.log_mean_scalar("test_length", ep_len, self.total_steps)

        self.test_history.append(ep_ret)