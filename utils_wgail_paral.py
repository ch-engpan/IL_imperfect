import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
import tqdm
import numpy as np
import pickle
import gymnasium as gym
from utils_gail import GAIL_Discrim
import math
from copy import deepcopy as cp


class WGAIL_Discrim(GAIL_Discrim):
    def __init__(self, state_shape, action_shape, ):
        super().__init__(state_shape, action_shape)

    def calculate_reward(self, x_s, x_a):
        # Reward = loss( -g(s,a))
        with torch.no_grad():
            logits = self.forward(x_s, x_a)
            return -F.logsigmoid(logits)
        


class WGAIL():
    def __init__(self, demonstrations, demo_batch_size,  device=None,
                 
                 gen_algo=None, gen_train_timesteps=int(1.5e4), gen_callback=None,
                 discrim=None, n_disc_updates_per_round=100, lr_disc=1e-4,

                 scaler_state=None, scaler_action=None, data_augm=None  
                 ):
        self.demonstrations = demonstrations
        self.demo_batch_size = demo_batch_size
        self.device = device

        self.gen_algo = gen_algo
        self.gen_train_timesteps = gen_train_timesteps
        self.gen_callback = gen_callback

        self.discrim = discrim.to(self.device)
        self.n_disc_updates_per_round = n_disc_updates_per_round
        self.optim_disc = Adam(self.discrim.parameters(), lr=lr_disc)

        self.s_max = torch.tensor(scaler_state.data_max_, dtype=torch.float32).to(self.device)
        self.s_min = torch.tensor(scaler_state.data_min_, dtype=torch.float32).to(self.device)
        self.a_max = torch.tensor(scaler_action.data_max_, dtype=torch.float32).to(self.device)
        self.a_min = torch.tensor(scaler_action.data_min_, dtype=torch.float32).to(self.device)
        self.acc_agent_list = []
        self.acc_expert_list = []
        self.loss_disc_list = []
        self.confi_list = []

        self.data_augm = data_augm # "mixup" or "normal" or "both"


        self.expert_conf = torch.ones((self.demonstrations.buffer_size, 1)).to(device)
        self.disc_loss = nn.BCEWithLogitsLoss()
        self.disc_loss1 = nn.BCELoss()
        self.conf_beta = 1

        self.label_expert = torch.zeros(demo_batch_size, 1).to(device)
        self.label_agent = torch.ones(demo_batch_size, 1).to(device)

        self.acc_agent_list = []
        self.acc_expert_list = []
        self.loss_disc_list = []

        
    def norm_s_a(self, x_s, x_a):
        x_s = (x_s - self.s_min) / (self.s_max - self.s_min)
        x_a = (x_a - self.a_min) / (self.a_max - self.a_min)
        # trasfer from torch.float64 to torch.float32
        return x_s.to(torch.float32), x_a.to(torch.float32)
    


    def get_action_distribution(self, agent, state):
        with torch.no_grad():
            # print(agent.policy)
            # print("state ", state.shape)
            # print("state ", state)
            # actions_pi, log_prob = agent.actor.action_log_prob(state)
            # print("actions_pi ", actions_pi.shape)
            # print("actions_pi ", actions_pi)
            # print("log_prob ", log_prob.shape)
            # print("prob ", torch.exp(log_prob))

            mean_actions, log_std, _ = agent.actor.get_action_dist_params(state)
            actions_pi, log_prob  = agent.actor.action_log_prob(state)
            # print("--------------------")
            # print("")
            # print("mean_actions ", mean_actions.shape)
            # print("mean_actions ", mean_actions)
            # print("std ", torch.exp(log_std).shape)
            # print("std ", torch.exp(log_std))


            # print("")
            # print("actions_pi ", actions_pi.shape)
            # print("actions_pi ", actions_pi)
            # print("prob ", torch.exp(log_prob).shape)
            # print("prob ", torch.exp(log_prob))
            # print("")
            # print("--------------------")

            std_actions = torch.exp(log_std)

            
        return mean_actions, std_actions, log_std
    

    def update_conf(self, print_conf):
        # update the confidence of expert/demonstration data
        states_demon, actions_demon = self.demonstrations.traj_s, self.demonstrations.traj_a
        ac_mean, ac_std, ac_log_std = self.get_action_distribution(self.gen_algo, states_demon)
        ac_var = ac_std**2
        
        # lg_prob = -((ac-ac_mean)**2) / (2*ac_var) - torch.log(ac_std) - math.log(math.sqrt(2*math.pi))
        lg_prob = -((actions_demon-ac_mean)**2) / (2*ac_var) - ac_log_std - math.log(math.sqrt(2*math.pi))
        print("lg_prob ", lg_prob.shape)
        lg_prob = lg_prob.sum(-1, keepdim=True)
        print("lg_prob ", lg_prob.shape)
        # print("lg_prob ", lg_prob)
        # print("prob ======= ", torch.exp(lg_prob)[50:100])
        with torch.no_grad():
            D_logits = self.discrim.forward(states_demon, actions_demon)
            D_s_a = torch.sigmoid(D_logits)
        # print("D_s_a ======", D_s_a[50:100])

        # # same as sb3-common-distrubtions (line 239) implementation for numerical stability 
        # epsilon = 1e-6
        # print("torch.log(1 - actions_demon**2 + epsilon)", (torch.log(1 - actions_demon**2 + epsilon)).shape)
        # print("torch.sum(torch.log(1 - actions_demon**2 + epsilon), dim=1) ", (torch.sum(torch.log(1 - actions_demon**2 + epsilon), dim=1, keepdim=True)).shape)
        # print("lg_prob", lg_prob.shape)
        # lg_prob -= torch.sum(torch.log(1 - actions_demon**2 + epsilon), dim=1, keepdim=True)
        # print("----stable----")
        # print("lg_prob ", lg_prob.shape)
        # print("lg_prob ", lg_prob)
        # print("prob ======= ", torch.exp(lg_prob))

        prob = torch.exp(lg_prob)

        
        eq1 = (1 / D_s_a - 1)
        eq2 = (1 / D_s_a - 1) * prob
        D_s_a += 1e-6 # to avoid division by zero
        self.expert_conf = ((1 / D_s_a + 1e-6 - 1) * prob).pow(1 / (self.conf_beta+1))


        # check if there are nan or inf in self.expert_conf
        if torch.isnan(self.expert_conf).sum() > 0 or torch.isinf(self.expert_conf).sum() > 0:

            # get the index of the values of self.expert_conf if there are nan or inf
            nan_idx = torch.isnan(self.expert_conf)
            inf_idx = torch.isinf(self.expert_conf)

            # print D_s_a and prob  and self.expert_conf
            print("")
            print("nan_idx ", nan_idx)
            print("nan number ", nan_idx.sum()) 
            print("D_logits ", D_logits[nan_idx])
            print("D_s_a ", D_s_a[nan_idx])
            print("prob ", prob[nan_idx])
            print("eq1 ", eq1[nan_idx])
            print("eq2 ", eq2[nan_idx])
            print("self.expert_conf ", self.expert_conf[nan_idx])

            print("")
            print("inf_idx ", inf_idx)
            print("inf number ", inf_idx.sum())
            print("D_logits ", D_logits[inf_idx])
            print("D_s_a ", D_s_a[inf_idx])
            print("prob ", prob[inf_idx])
            print("eq1 ", eq1[inf_idx])
            print("eq2 ", eq2[inf_idx])
            print("self.expert_conf ", self.expert_conf[inf_idx])
            print("")


        if print_conf:

            print("")
            print(" ========== Before Normalization ===========")
            print("self.expert_conf mean", self.expert_conf.mean())
            print("self.expert_conf max", self.expert_conf.max())
            print("self.expert_conf max / num", self.expert_conf.max() / self.expert_conf.shape[0])
            print("self.expert_conf min", self.expert_conf.min())
            print("self.expert_conf num < 1e-4", (self.expert_conf <= 1e-4).sum())
            print("self.expert_conf num < 1", (self.expert_conf <= 1).sum())
            print("self.expert_conf num > 2", (self.expert_conf > 2).sum())

            print("")
            print("")

        self.expert_conf = self.expert_conf / (self.expert_conf.mean() + 1e-6) # normalize the expert_conf as the mean of expert_conf is 1
        if print_conf:

            print("")
            print(" =========== After Normalization ===========")
            print("self.expert_conf mean", self.expert_conf.mean())
            print("self.expert_conf max", self.expert_conf.max())
            print("self.expert_conf max / num", self.expert_conf.max() / self.expert_conf.shape[0])
            print("self.expert_conf min", self.expert_conf.min())
            print("self.expert_conf num < 1e-4", (self.expert_conf <= 1e-4).sum())
            print("self.expert_conf num < 1", (self.expert_conf <= 1).sum())
            print("self.expert_conf num > 2", (self.expert_conf > 2).sum())

            print("")
            print("")

        # limit values in self.expert_conf to 10:
        self.expert_conf = torch.min(self.expert_conf, torch.ones((self.demonstrations.buffer_size, 1)).to(self.device)*10)

        # print("print_conf ", print_conf)
        if print_conf:
            # print("lg_prob ", lg_prob[:3])
            # print("lg_prob ", lg_prob.shape)
            # print("lg_prob.sum(-1, keepdim=True)", lg_prob.sum(-1, keepdim=True)[:3])
            # print("prob ", prob[:3])
            # print("prob ", prob.shape)
            # print("")
            # print("D_s_a", D_s_a)
            # print("D_s_a", D_s_a.shape)
            # print("")
            # print("self.expert_conf ", self.expert_conf)
            # print("self.expert_conf ", self.expert_conf.shape)
            print("")
            print(" ========== After Normalization and Limitation ===========")
            print("self.expert_conf mean", self.expert_conf.mean())
            print("self.expert_conf max", self.expert_conf.max())
            print("self.expert_conf max / num", self.expert_conf.max() / self.expert_conf.shape[0])
            print("self.expert_conf min", self.expert_conf.min())
            print("self.expert_conf num < 1e-4", (self.expert_conf <= 1e-4).sum())
            print("self.expert_conf num < 1", (self.expert_conf <= 1).sum())
            print("self.expert_conf num > 2", (self.expert_conf > 2).sum())

            print("")
            print("")

            
    def mixup_state_action(self, x_s_expert, x_a_expert, x_s_agent, x_a_agent, alpha=1.0):
        lam = np.random.beta(alpha, alpha)
        batch_size = x_s_expert.size()[0]
        index = torch.randperm(batch_size).to(self.device)
        mixed_x_s = lam * x_s_expert + (1 - lam) * x_s_agent[index, :]
        mixed_x_a = lam * x_a_expert + (1 - lam) * x_a_agent[index, :]

        return mixed_x_s, mixed_x_a, lam
    
    def calculate_loss_mixup(self, logits_mixed, lam):
        loss_expert_w = self.weighted_loss(logits_mixed, self.label_expert)
        loss_agent = self.disc_loss(logits_mixed, self.label_agent) 
        # loss_agent_w = self.weighted_loss(logits_mixed, self.label_agent)

        mixup_loss = lam * loss_expert_w + (1 - lam) * loss_agent
        # mixup_loss = lam * loss_expert_w + (1 - lam) * loss_agent_w
        return mixup_loss
    

    def update_discriminator(self, n_disc_updates_per_round, round):
        self.discrim.train()

        for iter_discr in range(n_disc_updates_per_round):

            agent_trajs = self.gen_algo.replay_buffer.sample(self.demo_batch_size)
            # don't use reward signals here,
            states_agent, actions_agent, next_states, _, dones = agent_trajs.observations, agent_trajs.actions, agent_trajs.next_observations, agent_trajs.rewards, agent_trajs.dones
            states_agent, actions_agent = self.norm_s_a(states_agent, actions_agent)
            

            states_expert, actions_expert, idx_expert = self.demonstrations.sample_s_a(self.demo_batch_size)
            # states_expert, actions_expert = self.norm_s_a(states_expert, actions_expert) # demos are already normalized

            logits_agent = self.discrim.forward(states_agent, actions_agent)
            logits_expert = self.discrim.forward(states_expert, actions_expert)
            # Discriminator is to maximize E_{\pi} [log(1 - D)] + E_{exp} [log(D)].
            # loss_pi = -F.logsigmoid(-logits_agent).mean()
            # loss_exp = -F.logsigmoid(logits_expert).mean()

            # --- confi ---
            update_conf_start_iter = 10
            update_conf_stop_iter = 16

            if iter_discr == 0 and round <= update_conf_stop_iter and round >= update_conf_start_iter and round % 2 == 0:
                print_conf = True
                print(" ======================= round  ========== ", round)
                self.update_conf(print_conf)
            
            if round <= update_conf_start_iter and iter_discr == 0:
                self.expert_conf = torch.ones((self.demonstrations.buffer_size, 1)).to(self.device)

            conf_expert = (self.expert_conf[idx_expert, :])
            self.weighted_loss = nn.BCEWithLogitsLoss(weight=conf_expert.detach())

            if round <= 20 and iter_discr == 0:
                self.confi_list.append(np.array(self.expert_conf.detach().cpu()))



            # if (iter_discr) == 250 and iter_discr != 0 and round <= 10 and round > 5:

            #     conf_expert = (self.expert_conf[idx_expert, :])
            #     # self.expert_conf = torch.ones((self.demonstrations.buffer_size, 1)).to(self.device) ###################  TEST

            #     # print("conf_expert ", conf_expert.shape)
            #     # print("self.expert_conf ", self.expert_conf.shape)
            #     # print("idx_expert ", idx_expert.shape)
            #     # print("")
            #     self.weighted_loss = nn.BCEWithLogitsLoss(weight=conf_expert.detach())
            #     print("update conf_expert !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

            

            loss_agent = self.disc_loss(logits_agent, self.label_agent) # equals to loss_agent_manuel4

            # loss_agent_manuel1 = self.disc_loss1(torch.sigmoid(logits_agent), self.label_agent)
            # loss_agent_manuel2 = -torch.log(1-F.sigmoid(logits_agent)).mean()
            # loss_agent_manuel3 = -torch.log(F.sigmoid(-logits_agent)).mean()
            # loss_agent_manuel4 = -torch.log(F.sigmoid(logits_agent)).mean()
            # print("loss_agent ", lossF_agent)
            # print("loss_agent_manuel1 ", loss_agent_manuel1)
            # print("loss_agent_manuel2 ", loss_agent_manuel2)
            # print("loss_agent_manuel3 ", loss_agent_manuel3)
            # print("loss_agent_manuel4 ", loss_agent_manuel4)
            # print("\n")
            # loss_expert = self.disc_loss(logits_expert, self.label_expert)
            # loss_expert_manuel1 = self.disc_loss1(torch.sigmoid(logits_expert), self.label_expert)
            # loss_expert_manuel2 = -torch.log(1-F.sigmoid(logits_expert)).mean()
            # loss_expert_manuel3 = -torch.log(F.sigmoid(-logits_expert)).mean()
            # loss_expert_manuel4 = -torch.log(F.sigmoid(logits_expert)).mean()
            # print("loss_expert ", loss_expert)
            # print("loss_expert_manuel1 ", loss_expert_manuel1)
            # print("loss_expert_manuel2 ", loss_expert_manuel2)
            # print("loss_expert_manuel3 ", loss_expert_manuel3)
            # print("loss_expert_manuel4 ", loss_expert_manuel4)
            # print("\n")
            # print("==========")

            
            loss_expert_w = self.weighted_loss(logits_expert, self.label_expert)
            

            if self.data_augm == "normal":
                loss_disc = loss_agent + loss_expert_w
            elif self.data_augm == "mixup":
                mixed_x_s, mixed_x_a, lam = self.mixup_state_action(states_expert, actions_expert, states_agent, actions_agent)
                logits_mixed = self.discrim.forward(mixed_x_s, mixed_x_a)
                loss_disc = self.calculate_loss_mixup(logits_mixed, lam).mean()

            elif self.data_augm == "both":
                mixed_x_s, mixed_x_a, lam = self.mixup_state_action(states_expert, actions_expert, states_agent, actions_agent)
                logits_mixed = self.discrim.forward(mixed_x_s, mixed_x_a)
                loss_mix = self.calculate_loss_mixup(logits_mixed, lam).mean()
                
                loss_disc = 0.5*(loss_agent + loss_expert_w) + loss_mix

            self.optim_disc.zero_grad()
            loss_disc.backward()
            self.optim_disc.step()

            # calculate accuracy
            with torch.no_grad():
                acc_agent = (torch.sigmoid(logits_agent) > 0.5).float().mean().item()
                acc_expert = (torch.sigmoid(logits_expert) <= 0.5).float().mean().item()
            self.acc_agent_list.append(acc_agent)
            self.acc_expert_list.append(acc_expert)
            self.loss_disc_list.append([loss_disc.item(), loss_agent.item(), loss_expert_w.item()])
            # if iter_discr % 100 == 0:
                # print(f"------->>>>>>>>  round {round}, iter_discr {iter_discr}, acc_agent {acc_agent}, acc_expert {acc_expert}, loss_disc {loss_disc.item()}, loss_agent {loss_agent.item()}, loss_expert {loss_expert_w.item()}")
            

    def update_RL(self, total_timesteps):
        # self.gen_algo.env.envs[0].scoring_model = self.discrim
        # self.gen_algo.env.envs[0].scoring_model.eval()
        self.discrim.eval()
        self.gen_algo.env.set_attr('scoring_model', self.discrim)
        
        self.gen_algo.learn(
                total_timesteps=total_timesteps,
                reset_num_timesteps=False,
                callback=self.gen_callback,
                progress_bar=False,
                # progress_bar=True,
        )

    def train(self, total_timesteps, data_augm=None):
        n_rounds = total_timesteps // self.gen_train_timesteps
        # n_rounds = total_timesteps // 2000
        assert n_rounds >= 1, (
            "No updates (need at least "
            f"{self.gen_train_timesteps} timesteps, have only "
            f"total_timesteps={total_timesteps})!"
                )
        # self.discrim.modified_reward_debug = 1
        for round in tqdm.tqdm(range(0, n_rounds), desc="round"):
            self.update_RL(self.gen_train_timesteps)
            # self.discrim.modified_reward_debug = self.discrim.modified_reward_debug * 10
            # self.update_RL(2000)
            # self.cotrain_scoring_models(self.n_scoring_updates_per_round, data_augm = data_augm)
            self.update_discriminator(self.n_disc_updates_per_round, round)

    