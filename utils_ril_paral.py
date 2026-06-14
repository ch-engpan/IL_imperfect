import torch
from torch import autograd, nn
import torch.nn.functional as F
from torch.optim import Adam
import tqdm
import numpy as np
import pickle
import gymnasium as gym
from utils_gail import GAIL_Discrim
from copy import deepcopy as cp


class APL_Loss(nn.Module):   
    def __init__(self, alpha=0.5, beta=0.5):
        super().__init__()
        self.alpha = alpha 
        self.beta = beta 

    def forward(self, z, reduction=True):
        normalized_logistic = F.logsigmoid(z) / ( F.logsigmoid(z) + F.logsigmoid(-z) )
        sigmoid = torch.sigmoid(-z) 
        loss = self.alpha * normalized_logistic + self.beta * sigmoid
        return loss.mean() if reduction else loss   

    def reward(self, z, reduction=False):
        return self.forward(z, reduction)


class RIL_Discrim(GAIL_Discrim):
    def __init__(self, state_shape, action_shape, ):
        super().__init__(state_shape, action_shape)
        self.adversarial_loss = APL_Loss()
        self.label_expert = 1
        self.label_policy = -1

    def calculate_reward(self, x_s, x_a):
        # Reward = loss( -g(s,a))t
        with torch.no_grad():
            logits = self.forward(x_s, x_a)
            return self.adversarial_loss.reward(logits * self.label_policy)


# class RIL_Discrim(nn.Module):
#     def __init__(self, state_shape, action_shape, hidden_dim=100):
#         super(RIL_Discrim, self).__init__()
#         num_inputs = state_shape + action_shape
#         self.main = nn.Sequential(
#             nn.Linear(num_inputs, hidden_dim), nn.Tanh(),
#             nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
#             nn.Linear(hidden_dim, 1))
#         self.train()
#         self.adversarial_loss = APL_Loss()
#         self.label_expert = 1
#         self.label_policy = -1

#     def forward(self, state, action):
#         inputs = torch.cat([state, action], dim=1)
#         return self.main(inputs)

#     def reward(self, inputs):
#         return self.main(inputs)
    
#     def calculate_reward(self, x_s, x_a):
#         # Reward = loss( -g(s,a))
#         with torch.no_grad():
#             logits = self.forward(x_s, x_a)
#             return self.adversarial_loss.reward(logits * self.label_policy)



class RIL_CO():
    def __init__(self, demonstrations, demo_batch_size,  device=None,
                 
                 gen_algo=None, gen_train_timesteps=int(1.5e4), gen_callback=None,
                 discrim_A=None, discrim_B=None, 
                 n_disc_updates_per_round=100, lr_disc=1e-4,

                 scaler_state=None, scaler_action=None, data_augm=None  
                 ):
        
        self.demonstrations_A = cp(demonstrations)
        self.demonstrations_B = cp(demonstrations)

        
        ## randomly split demonstration-dataset into disjoint subsets. 
        split = 0.5 
        demon_size_A = int(demonstrations.traj_s.shape[0]*split)

        indices = torch.randperm(demonstrations.traj_s.shape[0])
        self.demonstrations_A.traj_s = cp(demonstrations.traj_s)[indices[:demon_size_A]]
        self.demonstrations_A.traj_a = cp(demonstrations.traj_a)[indices[:demon_size_A]]
        self.demonstrations_A.buffer_size = demon_size_A

        self.demonstrations_B.traj_s = cp(demonstrations.traj_s)[indices[demon_size_A:]]
        self.demonstrations_B.traj_a = cp(demonstrations.traj_a)[indices[demon_size_A:]]
        self.demonstrations_B.buffer_size = int(demonstrations.traj_s.shape[0] - demon_size_A)
        
        
        self.label_expert = discrim_A.label_expert
        self.label_policy = discrim_A.label_policy
        self.adversarial_loss = discrim_A.adversarial_loss
        self.ril_prior = 0.5
        self.b_size_multiplier = 5



        self.demo_batch_size = demo_batch_size
        self.device = device

        self.gen_algo = gen_algo
        self.gen_train_timesteps = gen_train_timesteps
        self.gen_callback = gen_callback

        self.discrim_A = discrim_A.to(self.device)
        self.optim_disc_A = Adam(self.discrim_A.parameters(), lr=lr_disc)
        self.discrim_B = discrim_B.to(self.device)
        self.optim_disc_B = Adam(self.discrim_B.parameters(), lr=lr_disc)

        self.n_disc_updates_per_round = n_disc_updates_per_round

        # print("max", scaler_state.data_max_)

        self.s_max = torch.tensor(scaler_state.data_max_, dtype=torch.float32).to(self.device)
        self.s_min = torch.tensor(scaler_state.data_min_, dtype=torch.float32).to(self.device)
        self.a_max = torch.tensor(scaler_action.data_max_, dtype=torch.float32).to(self.device)
        self.a_min = torch.tensor(scaler_action.data_min_, dtype=torch.float32).to(self.device)
        self.acc_agent_list = []
        self.acc_expert_list = []
        self.loss_disc_list = []

        self.data_augm = data_augm # "mixup" or "normal" or "both"

    def norm_s_a(self, x_s, x_a):
        x_s = (x_s - self.s_min) / (self.s_max - self.s_min)
        x_a = (x_a - self.a_min) / (self.a_max - self.a_min)
        # trasfer from torch.float64 to torch.float32
        return x_s.to(torch.float32), x_a.to(torch.float32)


    def optimizer_step(self, optimizer, loss):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    

    def update_RL(self, total_timesteps):
        # self.gen_algo.env.envs[0].scoring_model = self.discrim_A
        # self.gen_algo.env.envs[0].scoring_model.eval()
        self.discrim_A.eval()
        self.gen_algo.env.set_attr('scoring_model', self.discrim_A)

        self.gen_algo.learn(
                total_timesteps=total_timesteps,
                reset_num_timesteps=False,
                callback=self.gen_callback,
                progress_bar=False,
        )
    

    def mixup_state_action(self, x_s_expert, x_a_expert, x_s_agent, x_a_agent, alpha=1.0):
        lam = np.random.beta(alpha, alpha)
        batch_size = x_s_expert.size()[0]
        index = torch.randperm(batch_size).to(self.device)
        mixed_x_s = lam * x_s_expert + (1 - lam) * x_s_agent[index, :]
        mixed_x_a = lam * x_a_expert + (1 - lam) * x_a_agent[index, :]

        return mixed_x_s, mixed_x_a, lam


    def compute_grad_pen(self,
                         s_exp, a_exp,  
                         s_agt, a_agt,
                         gp_lambda=10,
                         network=None):
        
        # mixup
        s_mix, a_mix, lam = self.mixup_state_action(s_exp, a_exp, s_agt, a_agt )
        s_mix.requires_grad = True
        a_mix.requires_grad = True

        discrm_output = network.forward(s_mix, a_mix)
        ones = torch.ones(discrm_output.size()).to(self.device)
        grad_s = autograd.grad(
            outputs=discrm_output,
            inputs=s_mix,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        
        grad_a = autograd.grad(
            outputs=discrm_output,
            inputs=a_mix,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        
        grad_s_a = torch.cat([grad_s, grad_a], dim=1)
        grad_pen = gp_lambda * (grad_s_a.norm(2, dim=1) - 1).pow(2).mean()
        return grad_pen


    def co_train_loss(self, net_A, net_B, 
                              s_agt, a_agt, 
                              s_exp_A_1, a_exp_A_1, 
                              s_exp_B_2, a_exp_B_2):
        
        logits_agt = net_A.forward(s_agt, a_agt)
        policy_loss = self.adversarial_loss( logits_agt * self.label_policy)
        
        logits_exp = net_A.forward(s_exp_A_1, a_exp_A_1)
        expert_loss_tr = self.adversarial_loss( logits_exp * self.label_expert)

        # pseudo labeling. Get prediction from net 2 to select indices 
        with torch.no_grad():
            expert_d_p = (net_B.forward(s_exp_B_2, a_exp_B_2)).detach().data.squeeze()
            index_p = (expert_d_p < 0).nonzero()
        if index_p.size(0) > 1:
            index_p_sort = torch.argsort(expert_d_p[index_p], dim=0)[:self.demo_batch_size] # ascending 
            index_p = index_p[index_p_sort].squeeze()

            loss_p = self.adversarial_loss( net_A.forward(s_exp_B_2[index_p], a_exp_B_2[index_p]) * self.label_policy)

            policy_loss = (1-self.ril_prior) * policy_loss + self.ril_prior * loss_p

        grad_pen = self.compute_grad_pen( s_exp_A_1, a_exp_A_1,  
                                            s_agt, a_agt,
                                            gp_lambda=10, network=net_A)

        gail_loss = policy_loss + expert_loss_tr + grad_pen

        return gail_loss, [policy_loss, expert_loss_tr, grad_pen], logits_agt, logits_exp



    def update_discriminator(self, n_disc_updates_per_round):
        self.discrim_A.train()
        self.discrim_B.train()

        for iter_discr in range(n_disc_updates_per_round):

            # Sample Agent trajectories
            agent_trajs = self.gen_algo.replay_buffer.sample(self.demo_batch_size)
            # don't use reward signals here,
            states_agent, actions_agent, next_states, _, dones = agent_trajs.observations, agent_trajs.actions, agent_trajs.next_observations, agent_trajs.rewards, agent_trajs.dones
            states_agent, actions_agent = self.norm_s_a(states_agent, actions_agent)

            # Sample Expert trajectories
            states_expert_A_1, actions_expert_A_1, _ = self.demonstrations_A.sample_s_a(self.demo_batch_size) # for true-label (tr) term 
            # states_expert_A_1, actions_expert_A_1 = self.norm_s_a(states_expert_A_1, actions_expert_A_1) # demos are already normalized

            states_expert_A_2, actions_expert_A_2, _ = self.demonstrations_A.sample_s_a(int(self.demo_batch_size*self.b_size_multiplier)) # for pseudo-labels (p) term
            # states_expert_A_2, actions_expert_A_2 = self.norm_s_a(states_expert_A_2, actions_expert_A_2) # demos are already normalized

            states_expert_B_1, actions_expert_B_1, _ = self.demonstrations_B.sample_s_a(self.demo_batch_size)  # for true-label term
            # states_expert_B_1, actions_expert_B_1 = self.norm_s_a(states_expert_B_1, actions_expert_B_1) # demos are already normalized

            states_expert_B_2, actions_expert_B_2, _ = self.demonstrations_B.sample_s_a(int(self.demo_batch_size*self.b_size_multiplier))  # for pseudo-labels term
            # states_expert_B_2, actions_expert_B_2 = self.norm_s_a(states_expert_B_2, actions_expert_B_2) # demos are already normalized

            """ discriminator A """
            loss_A, loss_A_list, logits_agent, logits_expert = self.co_train_loss(self.discrim_A, self.discrim_B, 
                                                     states_agent, actions_agent, 
                                                     states_expert_A_1, actions_expert_A_1, 
                                                     states_expert_B_2, actions_expert_B_2)
            
            """ discriminator B """
            loss_B, loss_B_list, *_ = self.co_train_loss(self.discrim_B, self.discrim_A, 
                                                     states_agent, actions_agent, 
                                                     states_expert_B_1, actions_expert_B_1, 
                                                     states_expert_A_2, actions_expert_A_2)
            
            self.optimizer_step(self.optim_disc_A, loss_A)
            self.optimizer_step(self.optim_disc_B, loss_B)

            # calculate accuracy
            with torch.no_grad():
                acc_agent = (logits_agent < 0).float().mean().item()
                acc_expert = (logits_expert > 0).float().mean().item()
            self.acc_agent_list.append(acc_agent)
            self.acc_expert_list.append(acc_expert)
            self.loss_disc_list.append([loss_A.item(), loss_A_list[0].item(), loss_A_list[1].item(), loss_A_list[2].item()])



    def train(self, total_timesteps):
        n_rounds = total_timesteps // self.gen_train_timesteps
        assert n_rounds >= 1, (
            "No updates (need at least "
            f"{self.gen_train_timesteps} timesteps, have only "
            f"total_timesteps={total_timesteps})!"
                )
        # self.discrim_A.modified_reward_debug = 1
        for round in tqdm.tqdm(range(0, n_rounds), desc="round"):
            self.update_RL(self.gen_train_timesteps)
            # self.discrim_A.modified_reward_debug = self.discrim_A.modified_reward_debug * 10
            # self.update_RL(1000)
            self.update_discriminator(self.n_disc_updates_per_round)
            