import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
import tqdm
import numpy as np
import pickle
import gymnasium as gym
import random
from copy import deepcopy as cp

# class GAIL_Discrim(nn.Module):
#     def __init__(self, state_shape, action_shape, hidden_units=(100, 100),):
#         super().__init__()
#         self.state_shape = state_shape
#         self.action_shape = action_shape
#         self.hidden_units = hidden_units
#         self.layers = nn.ModuleList()

#         self.layers.append(nn.Linear(state_shape[0] + action_shape[0], hidden_units[0]))
#         for i in range(1, len(hidden_units)):
#             self.layers.append(nn.Linear(hidden_units[i-1], hidden_units[i]))
#         self.layers.append(nn.Linear(hidden_units[-1], 1))
    
#     def forward(self, states, actions, ):
#         x = torch.cat([states, actions], dim=1)
#         for layer in self.layers[:-1]:
#             x = nn.Tanh(layer(x))
#         logits = self.layers[-1](x)
#         return logits
    
#     def calculate_reward(self, states, actions, ):
#         # RL agent / Generator of (GAIL) is to maximize E_{\pi} [-log(1 - D)].
#         with torch.no_grad():
#             logits = self.forward(states, actions, )
#             return -F.logsigmoid(-logits)


class GAIL_Discrim(nn.Module):
    def __init__(self, state_shape, action_shape, ):
        super().__init__()

        bias_control = True
        self.relu = nn.LeakyReLU()
        self.state_layer = nn.Sequential(
            nn.Linear(state_shape, 64, bias=bias_control),
            nn.LeakyReLU(),
             nn.Linear(64, 64, bias=bias_control),
            nn.LeakyReLU()
        )
        self.action_layer = nn.Sequential(
            nn.Linear(action_shape, 64, bias=bias_control),
            nn.LeakyReLU(),
            nn.Linear(64, 64, bias=bias_control),
            nn.LeakyReLU()
        )

        self.fc3 = nn.Linear(128, 128, bias=bias_control)
        self.fc4 = nn.Linear(128, 128, bias=bias_control)
        self.fc5 = nn.Linear(128, 1, bias=bias_control)

    def forward(self, x_s, x_a,):

        x_state = self.state_layer(x_s)
        x_action = self.action_layer(x_a)
        x = torch.cat((x_state, x_action), dim=1)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.relu(x)
        x = self.fc5(x)
        return x

    def calculate_reward(self, x_s, x_a):
        # RL agent / Generator of (GAIL) is to maximize E_{\pi} [-log(1 - D)].
        with torch.no_grad():
            logits = self.forward(x_s, x_a)
            return -F.logsigmoid(-logits)



class GAIL():
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
    
    def mixup_state_action(self, x_s_expert, x_a_expert, x_s_agent, x_a_agent, alpha=1.0):
        lam = np.random.beta(alpha, alpha)
        batch_size = x_s_expert.size()[0]
        index = torch.randperm(batch_size).to(self.device)
        mixed_x_s = lam * x_s_expert + (1 - lam) * x_s_agent[index, :]
        mixed_x_a = lam * x_a_expert + (1 - lam) * x_a_agent[index, :]

        return mixed_x_s, mixed_x_a, lam
    
    def calculate_loss_mixup(self, logits_mixed, lam):
        mixup_loss = lam * -F.logsigmoid(logits_mixed) + (1 - lam) * -F.logsigmoid(-1*logits_mixed)

        return mixup_loss


    def update_discriminator(self, n_disc_updates_per_round):
        self.discrim.train()

        for iter_discr in range(n_disc_updates_per_round):

            agent_trajs = self.gen_algo.replay_buffer.sample(self.demo_batch_size)
            # don't use reward signals here,
            states_agent, actions_agent, next_states, _, dones = agent_trajs.observations, agent_trajs.actions, agent_trajs.next_observations, agent_trajs.rewards, agent_trajs.dones
            states_agent, actions_agent = self.norm_s_a(states_agent, actions_agent)

            states_expert, actions_expert, _ = self.demonstrations.sample_s_a(self.demo_batch_size)
            # states_expert, actions_expert = self.norm_s_a(states_expert, actions_expert) # demos are already normalized

            logits_agent = self.discrim.forward(states_agent, actions_agent)
            logits_expert = self.discrim.forward(states_expert, actions_expert)
            # Discriminator is to maximize E_{\pi} [log(1 - D)] + E_{exp} [log(D)].
            loss_pi = -F.logsigmoid(-logits_agent).mean()
            loss_exp = -F.logsigmoid(logits_expert).mean()


            if self.data_augm == "normal":
                loss_disc = loss_pi + loss_exp
        
            elif self.data_augm == "mixup":
                mixed_x_s, mixed_x_a, lam = self.mixup_state_action(states_expert, actions_expert, states_agent, actions_agent)
                logits_mixed = self.discrim.forward(mixed_x_s, mixed_x_a)
                loss_disc = self.calculate_loss_mixup(logits_mixed, lam).mean()

            elif self.data_augm == "both":
                mixed_x_s, mixed_x_a, lam = self.mixup_state_action(states_expert, actions_expert, states_agent, actions_agent)
                logits_mixed = self.discrim.forward(mixed_x_s, mixed_x_a)
                loss_mix = self.calculate_loss_mixup(logits_mixed, lam).mean()
                
                loss_disc = 0.5*(loss_pi + loss_exp) + loss_mix

            self.optim_disc.zero_grad()
            loss_disc.backward()
            self.optim_disc.step()

            # calculate accuracy
            with torch.no_grad():
                acc_agent = (logits_agent < 0).float().mean().item()
                acc_expert = (logits_expert > 0).float().mean().item()
            self.acc_agent_list.append(acc_agent)
            self.acc_expert_list.append(acc_expert)
            self.loss_disc_list.append([loss_mix.item(), loss_pi.item(), loss_exp.item()])


    def update_RL(self, total_timesteps):
        # self.gen_algo.env.envs[0].scoring_model = self.discrim
        # self.gen_algo.env.envs[0].scoring_model.eval()

        self.discrim.eval()
        # self.discrim.modified_reward_debug = self.discrim.modified_reward_debug * 10
        self.gen_algo.env.set_attr('scoring_model', self.discrim)

        self.gen_algo.learn(
                total_timesteps=total_timesteps,
                reset_num_timesteps=False,
                callback=self.gen_callback,
                progress_bar=False,
        )


    def train(self, total_timesteps):
        # self.discrim.modified_reward_debug = 1
        n_rounds = total_timesteps // self.gen_train_timesteps
        assert n_rounds >= 1, (
            "No updates (need at least "
            f"{self.gen_train_timesteps} timesteps, have only "
            f"total_timesteps={total_timesteps})!"
                )
        
        for round in tqdm.tqdm(range(0, n_rounds), desc="round"):
            self.update_RL(self.gen_train_timesteps)
            # self.update_RL(2000)  # fast debugging
            self.update_discriminator(self.n_disc_updates_per_round)


class Demonstration_Buffer_uu:
    def __init__(self, traj_s, traj_a, bagLabels, binLabels, priors_class, Pi_s, device, scaler_s, scaler_a, uuLoss_all, expertDemo_label_method = "threshold", opt_threshold=0.5, top_k_percent=0.2):
        self.traj_s = torch.tensor(traj_s, dtype=torch.float32).to(device)
        self.traj_a = torch.tensor(traj_a, dtype=torch.float32).to(device)
        self.bagLabels = torch.tensor(bagLabels, dtype=torch.float32).to(device) # (30000, 6)
        self.binLabels = torch.tensor(binLabels, dtype=torch.float32).to(device)
        # self.pseudo_binLabels = torch.zeros_like(self.binLabels[:, 0], dtype=torch.float32).to(device)
        self.uu_buffer_size = traj_s.shape[0]
        self.priors_class = priors_class
        self.Pi_s = Pi_s
        self.uuLoss_all = uuLoss_all

        self.states_opt = None
        self.actions_opt = None
        self.opt_buffer_size = 0

        self.TP_exp = -1
        self.TN_exp = -1
        self.FP_exp = -1
        self.FN_exp = -1
        self.unprecision_exp = -1
        self.recall_exp = -1

        true_labels = (self.binLabels[:, 0]).long()
        self.true_P_num = (true_labels == 1).sum().item()
        self.true_N_num = (true_labels == 0).sum().item()

        self.opt_threshold = opt_threshold
        self.top_k_percent = top_k_percent
        self.label_method = expertDemo_label_method

        # print("self.uuLoss_all =========== ", self.uuLoss_all   )

        if self.uuLoss_all:
            # print("self.bagLabels shape", self.bagLabels.shape)
            # print("self.bagLabels samples", self.bagLabels[-5:])
            # print("self.priors_class ", self.priors_class)
            # print("self.Pi_s ", self.Pi_s)

            # add additional all zero columns for bagLabels to make  (30000, 6) -> # (30000, 7), add after  all zero columns  6th column
            self.bagLabels = F.pad(self.bagLabels, (0, 1), "constant", 0)
            self.priors_class =  np.concatenate([self.priors_class/2, np.array([0.5])])
            self.Pi_s =  np.concatenate([self.Pi_s, np.array([0.0])])

            # print("self.bagLabels shape", self.bagLabels.shape)
            # print("self.bagLabels samples", self.bagLabels[-5:])
            # print("self.priors_class ", self.priors_class)
            # print("self.Pi_s ", self.Pi_s)

        self.scaler_s = scaler_s
        self.scaler_a = scaler_a
        self.device = device

    def sample_uu(self, batch_size):
        idxes = np.random.choice(self.uu_buffer_size, size=batch_size, replace=False)
        return (
            self.traj_s[idxes],
            self.traj_a[idxes],
            self.bagLabels[idxes],
            self.binLabels[idxes],
            self.pseudo_binLabels[idxes],
            torch.as_tensor(idxes, device=self.device)
        )
    def sample_pseudo_opt(self, batch_size):
        if self.states_opt is None or self.states_opt.shape[0] == 0:
            raise ValueError("No optimal demonstrations available. Please run update_opt_demo() first.")
        if self.states_opt.shape[0] < batch_size:
            # print("Warning: self.states_opt.shape[0] < batch_size, use smaller batch_size ", self.states_opt.shape[0], " < ", batch_size)
            idxes = np.random.choice(self.states_opt.shape[0], size=batch_size, replace=True)
        else:
            idxes = np.random.choice(self.states_opt.shape[0], size=batch_size, replace=False)
        return (
            self.states_opt[idxes],
            self.actions_opt[idxes],
            torch.as_tensor(idxes, device=self.device)
        )

    def update_opt_demo(self, net_model, relabel=True, first_label=False):

        net_model.eval()
        with torch.no_grad():
            opt_or_not, class_i, _ = net_model.forward(self.traj_s, self.traj_a)
            # idx_opt = opt_or_not[:, 0] > self.opt_threshold

            # pred_labels = (opt_or_not[:, 0] > self.opt_threshold).long()
            # self.label_method = "threshold"
            if first_label:
                pred_labels = (opt_or_not[:, 0] > 0.5).long()
                idx_opt = (opt_or_not[:, 0] >  0.5)
                self.states_opt = self.traj_s[idx_opt]
                self.actions_opt = self.traj_a[idx_opt]
                self.pred_labels_last = cp(pred_labels)

                # print(" pred_labels  shape", pred_labels.shape)
                # print(" pred_labels  samples", pred_labels[-20:])

                self.pseudo_binLabels = cp(pred_labels.float()) # update pseudo labels for all data
                # print ("self.pseudo_binLabels  shape", self.pseudo_binLabels.shape)
                # print ("self.pseudo_binLabels  samples", self.pseudo_binLabels[-20:])
                # print("self.binLabels[:, 0] shape", self.binLabels[:, 0].shape)
                # print("self.binLabels[:, 0] samples", self.binLabels[:, 0][-20:])
                # input(f"Press Enter to continue... Expert first label ")
                print("self.states_opt.shape==============", self.states_opt.shape)
                print("self.actions_opt.shape==============", self.actions_opt.shape)
                

            else:
                if self.label_method == "threshold":
                    # Case 1: threshold-based
                    pred_labels_thresh = (opt_or_not[:, 0] > self.opt_threshold).long()
                    pred_labels = pred_labels_thresh

                    # input(f"Press Enter to continue... Expert thresh -  {self.opt_threshold} ")

                elif self.label_method == "topk":

                    # input(f"Press Enter to continue... Expert topk - {self.top_k_percent} ", )

                    # Case 2: top 20% largest values
                    # values = opt_or_not[:, 0] # use sigmoid values
                    values = opt_or_not[:, 2] # use logits values, maybe different as sigmoid results

                    k = int(len(values) * self.top_k_percent) 
                    topk_vals, topk_idx = torch.topk(values, k)
                    # lowk_vals, lowk_idx = torch.topk(values, k, largest=False)

                    n = len(values)

                    # create label tensor (0 by default, 1 if in top 20%)
                    pred_labels_topk = torch.zeros_like(values, dtype=torch.long)
                    pred_labels_topk[topk_idx] = 1
                    pred_labels = pred_labels_topk

                    # # print distribution of top 20% values
                    # print("Top 20% values distribution:")
                    # print(f"  min: {topk_vals.min().item():.6f}")
                    # print(f"  max: {topk_vals.max().item():.6f}")
                    # print(f"  mean: {topk_vals.mean().item():.6f}")
                    # print(f"  std: {topk_vals.std().item():.6f}")

                    # # Distribution of each 20% quantile group
                    # values = opt_or_not[:, 2]
                    # sorted_vals, _ = torch.sort(values)
                    # step = n // 5
                    # print("Quantile bin distributions (each ~20% of values):")
                    # for i in range(5):
                    #     start = i * step
                    #     end = (i + 1) * step if i < 4 else n  # last bin takes the remainder
                    #     bin_vals = sorted_vals[start:end]
                    #     print(f"  {i*20}-{(i+1)*20}%:")
                    #     print(f"    min: {bin_vals.min().item():.6f}")
                    #     print(f"    max: {bin_vals.max().item():.6f}")
                    #     print(f"    mean: {bin_vals.mean().item():.6f}")
                    #     print(f"    std: {bin_vals.std().item():.6f}")

                    #     print(f"sigmoind min: {torch.sigmoid(bin_vals).min().item():}")
                    #     print(f"sigmoind max: {torch.sigmoid(bin_vals).max().item():}")
                    #     print(f"sigmoind mean: {torch.sigmoid(bin_vals).mean().item():}")
                    #     print(f"sigmoind std: {torch.sigmoid(bin_vals).std().item():}")
                else:
                    raise ValueError("Invalid expertDemo_label_method, should be 'threshold' or 'topk'")
                    
                if relabel:
                    idx_opt = (pred_labels == 1)
                    self.states_opt = self.traj_s[idx_opt]
                    self.actions_opt = self.traj_a[idx_opt]
                    # print("relabel self.states_opt.shape", self.states_opt.shape)
                    # input("Press Enter to continue... Expert fine")
                    self.pseudo_binLabels = cp(pred_labels.float())


            true_labels = (self.binLabels[:, 0]).long()

            self.TP_exp = ((pred_labels == 1) & (true_labels == 1)).sum().item()
            self.TN_exp = ((pred_labels == 0) & (true_labels == 0)).sum().item()
            self.FP_exp = ((pred_labels == 1) & (true_labels == 0)).sum().item()
            self.FN_exp = ((pred_labels == 0) & (true_labels == 1)).sum().item()
            
            # compare with self.pred_labels_last and count how many old P is labeled as N, and how many N is labeled as P
            self.P2N = ((self.pred_labels_last == 1) & (pred_labels == 0)).sum().item()
            self.N2P = ((self.pred_labels_last == 0) & (pred_labels == 1)).sum().item()
            
            self.pred_labels_last = cp(pred_labels)

            # recall: in all true P data, how much labeled as P
            self.recall_exp = self.TP_exp / (self.TP_exp + self.FN_exp) if (self.TP_exp + self.FN_exp) > 0 else -1

            # unprecision: in all predicted P data, how much are actually N data (wrongly labeled)
            self.unprecision_exp = self.FP_exp / (self.TP_exp + self.FP_exp) if (self.TP_exp + self.FP_exp) > 0 else -1


        self.opt_buffer_size = self.states_opt.shape[0]
        net_model.train()


        # print("self.states_opt.shape==============", self.states_opt.shape)
        # print("self.actions_opt.shape==============", self.actions_opt.shape)
    
    def update_opt_demo_trueLabels(self):
        true_labels = (self.binLabels[:, 0]).long()
        idx_opt = (true_labels == 1)
        self.states_opt = self.traj_s[idx_opt]
        self.actions_opt = self.traj_a[idx_opt]
        self.opt_buffer_size = self.states_opt.shape[0]
        pred_labels = cp(true_labels.float())
        self.pseudo_binLabels = cp(true_labels.float())
        print("self.states_opt.shape==============", self.states_opt.shape)
        print("self.actions_opt.shape==============", self.actions_opt.shape)

        self.TP_exp = ((pred_labels == 1) & (true_labels == 1)).sum().item()
        self.TN_exp = ((pred_labels == 0) & (true_labels == 0)).sum().item()
        self.FP_exp = ((pred_labels == 1) & (true_labels == 0)).sum().item()
        self.FN_exp = ((pred_labels == 0) & (true_labels == 1)).sum().item()
        self.recall_exp = self.TP_exp / (self.TP_exp + self.FN_exp) if (self.TP_exp + self.FN_exp) > 0 else -1
        self.unprecision_exp = self.FP_exp / (self.TP_exp + self.FP_exp) if (self.TP_exp + self.FP_exp) > 0 else -1


        # input("Press Enter to continue... Expert true labels")
      






    def sample_opt(self, batch_size):
        if self.states_opt.shape[0] < batch_size:
            # print("Warning: self.states_opt.shape[0] < batch_size, use smaller batch_size ", self.states_opt.shape[0], " < ", batch_size)
            idxes = np.random.choice(self.states_opt.shape[0], size=batch_size, replace=True)
        else:
            idxes = np.random.choice(self.states_opt.shape[0], size=batch_size, replace=False)
        return (
            self.states_opt[idxes],
            self.actions_opt[idxes],
            torch.as_tensor(idxes, device=self.device)
        )


class Demonstration_Buffer_gail:
    def __init__(self, traj_s, traj_a, scaler_s, scaler_a,device):
        self.device = device
        # self.traj_s = traj_s
        # self.traj_a = traj_a
        self.traj_s = torch.tensor(traj_s[:,0,:], dtype=torch.float32).to(self.device)
        self.traj_a = torch.tensor(traj_a[:,0,:], dtype=torch.float32).to(self.device)
        
        self.buffer_size = self.traj_s.shape[0]
        self.scaler_s = scaler_s
        self.scaler_a = scaler_a

    def sample_s_a(self, batch_size):
        idxes = np.random.choice(self.buffer_size, size=batch_size, replace=False)
        return (
            self.traj_s[idxes],
            self.traj_a[idxes],
            torch.as_tensor(idxes, device=self.device)     
        )


class Demonstration_Buffer:
    def __init__(self, path, device, opt_ratio_alpha="0.5", single_frame=True):
        opt_traj_s_set, opt_traj_a_set,  nonopt_traj_s_set, nonopt_traj_a_set, scaler_s, scaler_a = self.load_data(path, single_frame=single_frame)
        print("path ", path)
        print("opt_traj_s_set.shape", opt_traj_s_set.shape)
        self.opt_buffer_size = opt_traj_s_set.shape[0]
        self.nonopt_buffer_size = nonopt_traj_s_set.shape[0]
        self.device = device
        self.states_opt = torch.tensor(opt_traj_s_set, dtype=torch.float32).to(self.device)
        self.actions_opt = torch.tensor(opt_traj_a_set, dtype=torch.float32).to(self.device)
        self.states_nonopt = torch.tensor(nonopt_traj_s_set, dtype=torch.float32).to(self.device)
        self.actions_nonopt = torch.tensor(nonopt_traj_a_set, dtype=torch.float32).to(self.device)
        self.scaler_s = scaler_s
        self.scaler_a = scaler_a

        print("self.scaler_s min", self.scaler_s.data_min_)
        print("self.scaler_s max", self.scaler_s.data_max_)
        print("self.scaler_a min", self.scaler_a.data_min_)
        print("self.scaler_a max", self.scaler_a.data_max_)

        print("self.states_opt.shape", self.states_opt.shape) # torch.Size([18370, 1, 27])
        print("self.actions_opt.shape", self.actions_opt.shape) # torch.Size([18370, 1, 8])
        print("self.states_nonopt.shape", self.states_nonopt.shape) # torch.Size([11630, 1, 27])
        print("self.actions_nonopt.shape", self.actions_nonopt.shape) # torch.Size([11630, 1, 8])


        # print("self.states_opt samples", self.states_opt[228:233])
        # print("self.actions_opt samples", self.actions_opt[228:233])
        # print("self.states_nonopt samples", self.states_nonopt[228:233])
        # print("self.actions_nonopt samples", self.actions_nonopt[228:233])


    def load_data(self, save_trajs_path, single_frame=True):
        with open(save_trajs_path, 'rb') as f:
            loaded_data = pickle.load(f)
            
        opt_traj_s_set = loaded_data['opt_traj_s_set']
        opt_traj_a_set = loaded_data['opt_traj_a_set']
        nonopt_traj_s_set = loaded_data['nonopt_traj_s_set']
        nonopt_traj_a_set = loaded_data['nonopt_traj_a_set']
        scaler_s = loaded_data['scaler_s']
        scaler_a = loaded_data['scaler_a']
        opt_start_steps = loaded_data['opt_start_steps']

        if single_frame:
            opt_traj_s_set = opt_traj_s_set[:,0,:]
            opt_traj_a_set = opt_traj_a_set[:,0,:]
            nonopt_traj_s_set = nonopt_traj_s_set[:,0,:]
            nonopt_traj_a_set = nonopt_traj_a_set[:,0,:]

        return opt_traj_s_set, opt_traj_a_set, nonopt_traj_s_set, nonopt_traj_a_set, \
                scaler_s, scaler_a

    def sample_opt(self, batch_size):
        idxes = np.random.choice(self.opt_buffer_size, size=batch_size, replace=False)
        return (
            self.states_opt[idxes],
            self.actions_opt[idxes],
            torch.as_tensor(idxes, device=self.device) 
            
        )
    def sample_nonopt(self, batch_size):
        idxes = np.random.choice(self.nonopt_buffer_size, size=batch_size, replace=False)
        return (
            self.states_nonopt[idxes],
            self.actions_nonopt[idxes],
            torch.as_tensor(idxes, device=self.device)
        )
    
    def remove_P(self, idxes):
        """
        Remove the optimal trajectories corresponding to `idxes` from the buffer.

        Parameters
        ----------
        idxes : array-like (list, np.ndarray, or torch.Tensor)
            Indices of the optimal demonstrations to discard.
        """
        if idxes is None or len(idxes) == 0:           # nothing to do
            return

        # Make sure we have a 1-D Long tensor on the correct device.
        idxes = torch.as_tensor(idxes, dtype=torch.long, device=self.device)
        # idxes = torch.unique(idxes)                    # drop duplicates
        # idxes = idxes[(idxes >= 0) & (idxes < self.opt_buffer_size)]  # clamp to valid range
        if idxes.numel() == 0:
            return

        # Boolean mask: True for rows we keep, False for rows we drop.
        keep_mask = torch.ones(self.opt_buffer_size, dtype=torch.bool, device=self.device)
        keep_mask[idxes] = False

        # Slice both tensors and update the size counter.
        self.states_opt   = self.states_opt[keep_mask]
        self.actions_opt  = self.actions_opt[keep_mask]
        self.opt_buffer_size = self.states_opt.shape[0]
    
    def remove_N(self, idxes):
        """
        Remove the non-optimal trajectories corresponding to `idxes` from the buffer.
        """
        if idxes is None or len(idxes) == 0:           # nothing to do
            return

        # Make sure we have a 1-D Long tensor on the correct device.
        idxes = torch.as_tensor(idxes, dtype=torch.long, device=self.device)
        if idxes.numel() == 0:
            return

        # Boolean mask: True for rows we keep, False for rows we drop.
        keep_mask = torch.ones(self.nonopt_buffer_size, dtype=torch.bool, device=self.device)
        keep_mask[idxes] = False

        # Slice both tensors and update the size counter.
        self.states_nonopt   = self.states_nonopt[keep_mask]
        self.actions_nonopt  = self.actions_nonopt[keep_mask]
        self.nonopt_buffer_size = self.states_nonopt.shape[0]

        
    def add_N2P(self, states_N2P, actions_N2P):
        """
        Add the non-optimal trajectories to the optimal trajectories.
        """
        self.states_opt = torch.cat((self.states_opt, states_N2P), dim=0)
        self.actions_opt = torch.cat((self.actions_opt, actions_N2P), dim=0)
        self.opt_buffer_size = self.states_opt.shape[0]
    
    def add_P2N(self, states_P2N, actions_P2N):
        """
        Add the optimal trajectories to the non-optimal trajectories.
        """
        self.states_nonopt = torch.cat((self.states_nonopt, states_P2N), dim=0)
        self.actions_nonopt = torch.cat((self.actions_nonopt, actions_P2N), dim=0)
        self.nonopt_buffer_size = self.states_nonopt.shape[0]




class Custom_Env(gym.Wrapper):
    def __init__(self, env_name, scoring_model, input_scaler_s, input_scaler_a,  normalize, env=None,   modify_reward=True):
        # Create the original environment
        if env is None:
            env = gym.make(env_name)
        super().__init__(env)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if modify_reward:
            self.scoring_model = scoring_model
            self.scoring_model.eval()
            self.scoring_model.to(self.device)

        self.input_scaler_s = input_scaler_s
        self.input_scaler_a = input_scaler_a
        
        self.old_obs = None
        self.modify_reward = modify_reward

        # mujoco (for GAIL, WGAIL, and RIL): normalize = True , demonstraions are not yet normalized, so we need to normalize them here
        # set normalize = True for mujoco, set normalize = False for robomimic
        self.normalize = normalize # not normalize for robomimic, normalize for mujoco

        self.custom_reward_sum = 0
        self.real_reward_sum = 0
        self.step_count = 0

        
        
    def set_modify_reward(self, value):
        self.modify_reward = value

    def reset(self, **kwargs):
        self.custom_reward_sum = 0
        self.real_reward_sum = 0
        self.step_count = 0
        return self.env.reset()
    

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated

        if not self.modify_reward:
            return obs, reward, terminated, truncated, info # can save computation if not modifying reward
        
        # Modify the reward calculation here (customize as needed)
        if self.old_obs is not None:
            modified_reward, reward_pred_orig = self._custom_reward(self.old_obs, action, reward)

        else:
            modified_reward = 0 # the first step has no reward
        self.old_obs = obs
        # modified_reward = self.scoring_model.modified_reward_debug

        self.custom_reward_sum += modified_reward
        self.real_reward_sum += reward
        self.step_count += 1
        
        if done:
            self.old_obs = None
            # print("custom_reward_sum ", self.custom_reward_sum, "    real_reward_sum ", self.real_reward_sum, "   step len", self.step_count)
            self.custom_reward_sum = 0
            self.real_reward_sum = 0
            self.step_count = 0

        if self.modify_reward:
            return obs, modified_reward, terminated, truncated, info
        else:
            # reward = -10
            return obs, reward, terminated, truncated, info

    def _custom_reward(self, obs, action, reward):
        if self.normalize:
            obs = self.input_scaler_s.transform(obs.reshape(1, -1))
            action = self.input_scaler_a.transform(action.reshape(1, -1))
        else:
            obs = obs.reshape(1, -1)
            action = action.reshape(1, -1)
            
        obs = torch.from_numpy(obs[0]).float().unsqueeze(0).to(self.device)
        action = torch.from_numpy(action[0]).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            opt_or_not = self.scoring_model.calculate_reward(obs, action)
        # print("opt_or_not ", opt_or_not)
            
        # r_select = "log_log"
        # r_select = "log"
        r_select = "no_change"
            
        if r_select == "log_log":
            reward_pred_orig = np.array(opt_or_not.cpu()[0][0])
            reward_pred = np.max((1e-8, reward_pred_orig))
            reward_pred = np.min((1 - 1e-8, reward_pred))
            reward_pred = np.log(reward_pred ) - np.log(1 - reward_pred)
            reward_pred = np.max((-2, reward_pred))
            reward_pred = np.min((2, reward_pred))
        elif r_select == "log":
            reward_pred_orig = np.array(opt_or_not.cpu()[0][0])
            reward_pred = np.max((1e-8, reward_pred_orig))
            reward_pred = np.min((1 - 1e-8, reward_pred))
            reward_pred = np.log(reward_pred)
            reward_pred = np.max((-2, reward_pred))
            reward_pred = np.min((2, reward_pred))
        elif r_select == "no_change":
            reward_pred_orig = np.array(opt_or_not.cpu()[0][0])
            # reward_pred_orig = np.array(opt_or_not.cpu()[0])
            reward_pred = reward_pred_orig
            # reward_pred = np.max((1e-8, reward_pred_orig))
            # reward_pred = np.min((1 - 1e-8, reward_pred))
        
        # modified_reward = reward * 0  # Modify the reward for testing
        
        return reward_pred, reward_pred_orig


