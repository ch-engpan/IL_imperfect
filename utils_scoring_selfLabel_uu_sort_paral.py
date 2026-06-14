import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
import tqdm
import numpy as np
import pickle
import gymnasium as gym
from copy import deepcopy as cp
from stable_baselines3.common.vec_env import SubprocVecEnv
from collections import deque
import random
# from utils_gail import ReplayBuffer

class ReplayBuffer:
    def __init__(self, max_size, state_shape, action_shape, device):
        # Initialize tensors to store states and actions
        self.max_size = max_size
        # create empty tensors with the specified shapes

        self.state_tensor = torch.empty((0, 1, *state_shape), dtype=torch.float32).to(device)
        self.action_tensor = torch.empty((0, 1, *action_shape), dtype=torch.float32).to(device)
        self.buffer_full = False

    def add(self, states, actions):
        """Add multiple new state-action pairs (as tensors) to the buffer."""
        # Convert states and actions into tensors if they aren't already
        states_tensor = torch.tensor(states, dtype=torch.float32)
        actions_tensor = torch.tensor(actions, dtype=torch.float32)
        # states_tensor = states_tensor.squeeze(1)
        # actions_tensor = actions_tensor.squeeze(1)

        batch_size = states_tensor.size(0)
        self.state_tensor = torch.cat((self.state_tensor, states_tensor), dim=0)
        self.action_tensor = torch.cat((self.action_tensor, actions_tensor), dim=0)

        # Add the new states and actions
        if len(self.state_tensor) <= self.max_size:
            pass
            
        else:
            if not self.buffer_full:
                self.buffer_full = True
            # If the buffer is full, remove the oldest elements
            self.state_tensor = self.state_tensor[-self.max_size:]
            self.action_tensor = self.action_tensor[-self.max_size:]


    def sample(self, batch_size):
        """Return a random sample of states and actions from the buffer."""
        buffer_size = len(self.state_tensor)
        indices = random.sample(range(buffer_size), batch_size)

        # Use tensor indexing to efficiently retrieve states and actions
        states = self.state_tensor[indices]
        actions = self.action_tensor[indices]

        # Return states, actions, and a placeholder (underscore) for compatibility
        return states, actions  # The underscore is just a placeholder, can be replaced as needed

    def size(self):
        return len(self.state_tensor)
    

def inverse_sigmoid(x):
    return torch.log(x / (1 - x))




class RL_Scoring_selfLabel():
    def __init__(self, demonstrations, batch_size, device=None, 
                 rl_algo=None, rl_train_timesteps=int(1.5e4), rl_callback=None,
                 scoring_model_A=None,  scoring_model_B=None, n_scoring_updates_per_round=100, lr_scoring=1e-4,
                 scaler_state=None, scaler_action=None, sort_ratio=0.5, sort_loss=False, data_augm=None, loss_type = None,
                 demon_normalized = True, normalize = True,
                 original_method=None, expertDemo_fine=True, expertDemo_N2P=True, expertDemo_useN=True,
                 agentReplay_N2P=True, agentReplay_useP=True,
                 uuLoss_Nagent=False, uuLoss_all=False, agentReplay_fine=False,
                 early_stop=False,
                 exp_loss = "uu_loss", # "uu_loss" or "PN_loss"
                 ):

        self.original_method = original_method
        self.expertDemo_fine = expertDemo_fine
        self.expertDemo_N2P = expertDemo_N2P
        self.expertDemo_useN = expertDemo_useN
        self.agentReplay_N2P = agentReplay_N2P
        self.agentReplay_useP = agentReplay_useP

        self.uuLoss_Nagent = uuLoss_Nagent
        self.uuLoss_all = uuLoss_all
        self.exp_loss = exp_loss
        self.agentReplay_fine = agentReplay_fine
        self.early_stop_active = early_stop


        self.normalize = normalize # not normalize for robomimic, normalize for mujoco


        # False:  mujoco (for GAIL, WGAIL, and RIL) demonstraions are NOT yet normalized, 
        # True:   mujoco (for uu scoring - prelabeled, opt_ratio_alpha != 0.0 or 1.0) demonstrations are ALREADY normalized
        # False:  mujoco (for uu scoring - un-prelabeled, opt_ratio_alpha == 0.0 or 1.0) demonstrations are NOT yet normalized
        self.demon_normalized = demon_normalized
        self.demonstrations = demonstrations
        self.batch_size = batch_size
        self.device = device

        self.rl_algo = rl_algo
        self.rl_train_timesteps = rl_train_timesteps
        self.rl_callback = rl_callback

        self.scoring_model_A = scoring_model_A.net.to(self.device)
        self.scoring_model_B = scoring_model_B.net.to(self.device)

        self.n_scoring_updates_per_round = n_scoring_updates_per_round
        self.optim_scoring_A = Adam(self.scoring_model_A.parameters(), lr=lr_scoring)
        self.optim_scoring_B = Adam(self.scoring_model_B.parameters(), lr=lr_scoring)

        # self.ce_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.ce_loss = torch.nn.BCELoss(reduction='none')
        self.loss_bce_mean = torch.nn.BCELoss()
        self.ce_loss.to(self.device)

        self.ce_loss_mean = torch.nn.CrossEntropyLoss(reduction='mean')
        self.ce_loss_mean.to(self.device)

        self.s_max = torch.tensor(scaler_state.data_max_, dtype=torch.float32).to(self.device)
        self.s_min = torch.tensor(scaler_state.data_min_, dtype=torch.float32).to(self.device)
        self.a_max = torch.tensor(scaler_action.data_max_, dtype=torch.float32).to(self.device)
        self.a_min = torch.tensor(scaler_action.data_min_, dtype=torch.float32).to(self.device)

        self.s_shape = scaler_state.data_max_.shape
        self.a_shape = scaler_action.data_max_.shape

        self.acc_agent_list = []
        self.acc_expert_list = []
        self.loss_disc_list = []

        self.data_augm = data_augm # "mixup" or "normal" or "both"

        # print("scoring_model_A priors_corr", self.scoring_model_A.priors_corr)
        # print("scoring_model_A Pi", self.scoring_model_A.Pi)

        if not self.uuLoss_Nagent and not self.uuLoss_all:
            # for calculate UU_loss
            self.scoring_model_A.priors_corr = [0.5, 0.5] # bag size ratio = 0.5
            self.scoring_model_A.Pi = [1, 0] # optimal ratio = 1 for bag of expert, and optimal ratio = 0 for bag of agent
            # check if self.scoring_model_A.prior_test exists

            self.bag_agent = torch.tensor([[0, 1]]).float().to(self.device)
            print("bag_agent shape", self.bag_agent.shape)
            self.bag_expert = torch.tensor([[1, 0]]).float().to(self.device)
            self.bag_agent = self.bag_agent.repeat((self.batch_size // self.bag_agent.shape[0], 1))
            print("bag_agent shape", self.bag_agent.shape)
            self.bag_expert = self.bag_expert.repeat((self.batch_size // self.bag_expert.shape[0], 1))

        else:
            if self.uuLoss_all:
                self.scoring_model_A.priors_corr =  np.concatenate([self.scoring_model_A.priors_corr/2, np.array([0.5])])
                self.scoring_model_A.Pi = np.concatenate([self.scoring_model_A.Pi,  np.array([0.0])])

            self.bag_agent = np.array([0 for k in range(len(self.scoring_model_A.Pi))])
            self.bag_agent[-1] = 1.0
            self.bag_agent = torch.tensor(self.bag_agent, dtype=torch.float32).to(self.device)
            print("self.bag_agent  ", self.bag_agent)
            self.bag_agent = self.bag_agent.repeat((self.batch_size, 1))

            # self.bag_agent2 = np.array([6 for k in range(self.batch_size)])
            # self.bag_agent2 = torch.tensor(self.bag_agent2, dtype=torch.float32).to(self.device)
            # print("self.bag_agent2  ", self.bag_agent2.shape)

        print ("self.uuLoss_Nagent ", self.uuLoss_Nagent)
        print ("self.uuLoss_all ", self.uuLoss_all)

        # print("scoring_model_A priors_corr", self.scoring_model_A.priors_corr)
        # print("scoring_model_A Pi", self.scoring_model_A.Pi)

        if  hasattr(self.scoring_model_A, 'prior_test'): # old version has prior_test, but new version has Pi_test
            self.scoring_model_A.Pi_test = self.scoring_model_A.prior_test




        self.agent_replay_buffer_opt = ReplayBuffer(max_size=1024*5, state_shape=self.s_shape, action_shape=self.a_shape, device=self.device)
        self.agent_replay_buffer_nonopt = ReplayBuffer(max_size=1024*5, state_shape=self.s_shape, action_shape=self.a_shape, device=self.device)

        self.exp_N2P_num = []
        self.exp_P2N_num = []
        self.expDemo_opt_num = []
        self.expDemo_nonopt_num = []
        self.agt_N2P_num = []

        self.TP_exp = []
        self.true_exp_P_num = []
        self.true_exp_N_num = []
        self.TN_exp = []
        self.FP_exp = []
        self.FN_exp = []
        self.recall_exp = []
        self.unprecision_exp = []

        self.relabel_earlyStop = False
        self.relabel_earlyStop_list = []

        
        # Track replay buffer sizes during training
        self.agent_replay_buffer_opt_size = []
        self.agent_replay_buffer_nonopt_size = []

        co_train = False
        if co_train:
            self.demonstrations_A = cp(demonstrations)
            self.agent_replay_buffer_opt_A = ReplayBuffer(max_size=1024*5, state_shape=self.s_shape, action_shape=self.a_shape, device=self.device)
            self.agent_replay_buffer_nonopt_A = ReplayBuffer(max_size=1024*5, state_shape=self.s_shape, action_shape=self.a_shape, device=self.device)
            
            self.demonstrations_B = cp(demonstrations)
            self.agent_replay_buffer_opt_B = ReplayBuffer(max_size=1024*5, state_shape=self.s_shape, action_shape=self.a_shape, device=self.device)
            self.agent_replay_buffer_nonopt_B = ReplayBuffer(max_size=1024*5, state_shape=self.s_shape, action_shape=self.a_shape, device=self.device)



    def norm_s_a(self, x_s, x_a):
        x_s = (x_s - self.s_min) / (self.s_max - self.s_min)
        x_a = (x_a - self.a_min) / (self.a_max - self.a_min)
        # trasfer from torch.float64 to torch.float32
        return x_s.to(torch.float32), x_a.to(torch.float32)
    
    # def mixup_var(self, x, y, alpha=1.0):
    #     lam = np.random.beta(alpha, alpha)
    #     batch_size = x.size()[0]
    #     index = torch.randperm(batch_size).to(self.device)
    #     mixed_x = lam * x + (1 - lam) * x[index, :]
    #     y_a, y_b = y, y[index]
    #     return mixed_x, y_a, y_b, lam

    # def mixup_state_action(self, x_s, x_a, y, alpha=1.0):
    #     lam = np.random.beta(alpha, alpha)
    #     batch_size = x_s.size()[0]
    #     index = torch.randperm(batch_size).to(self.device)
    #     mixed_x_s = lam * x_s + (1 - lam) * x_s[index, :]
    #     mixed_x_a = lam * x_a + (1 - lam) * x_a[index, :]
    #     y_i, y_j = y, y[index]
    #     return mixed_x_s, mixed_x_a, y_i, y_j, lam
    
    def mixup_state_action(self, x_s_expert, x_a_expert, x_s_agent, x_a_agent, alpha=1.0):
        lam = np.random.beta(alpha, alpha)
        if x_s_expert.size()[0] != x_s_agent.size()[0] or x_a_expert.size()[0] != x_a_agent.size()[0]:
            raise ValueError("x_s_expert and x_a_expert should have the same size, but got x_s_expert.size()[0] = {} and x_a_expert.size()[0] = {}".format(x_s_expert.size()[0], x_s_agent.size()[0]))

        batch_size = x_s_expert.size()[0]
        index = torch.randperm(batch_size).to(self.device)
        mixed_x_s = lam * x_s_expert + (1 - lam) * x_s_agent[index, :, :]
        mixed_x_a = lam * x_a_expert + (1 - lam) * x_a_agent[index, :, :]

        return mixed_x_s, mixed_x_a, lam


    def discrm_predict(self, model, state, action):
        # prediction
        opt_or_not, pred_bag, *_ = model.forward(state, action) 
        opt_or_not = opt_or_not[:,2]  # pick up logit 
        return opt_or_not, pred_bag, state, action
    

    def sample_and_prediction(self, model):
        # Sample a batch of transitions
        trajs = self.rl_algo.replay_buffer.sample(self.batch_size)
        state, action, next_state, _, done = trajs.observations, trajs.actions, trajs.next_observations, trajs.rewards, trajs.dones
        # samples from agent replay buffer is not normalized
        if self.normalize:
            state, action = self.norm_s_a(state, action)
        state = state.unsqueeze(1)  
        action = action.unsqueeze(1)
        return self.discrm_predict(model, state, action)

    
        # if data_augm == "normal":
        #     return opt_or_not, state, action

        # elif data_augm == "mixup":
        #     state_mix, action_mix, opt_or_not_i, opt_or_not_j, lam = self.mixup_state_action(state, action, opt_or_not)
        #     opt_or_not_mix, *_ = model.forward(state_mix, action_mix)
        #     return opt_or_not_mix, opt_or_not_i, opt_or_not_j, lam
    

    def calculate_loss_normal(self, opt_or_not, opt_or_not_target):
        opt_or_not_target = opt_or_not_target # check if have to transfer to 0 or 1 
        train_loss = self.loss_func(opt_or_not, opt_or_not_target)
        return train_loss
    
    # def calculate_loss_mixup(self, opt_or_not, opt_or_not_target_i, opt_or_not_target_j, lam):
    #     train_loss = lam * self.loss_func(opt_or_not, opt_or_not_target_i) + (1 - lam) * self.loss_func(opt_or_not, opt_or_not_target_j)
    #     return train_loss
    
    def calculate_loss_mixup(self, logits_mixed, lam):
        mixup_loss = lam * -F.logsigmoid(logits_mixed) + (1 - lam) * -F.logsigmoid(-1*logits_mixed)
        return mixup_loss


    def optimizer_step(self, optimizer, loss):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    def cotrain_scoring_models(self, n_scoring_updates_per_round, data_augm ="normal"):
        self.scoring_model_A.train()
        self.scoring_model_B.train()

        for i in range(n_scoring_updates_per_round):
            # Update scoring model A using B's prediction
            # opt_or_not_A, opt_or_not_A_i, opt_or_not_A_j, lam_A = self.sample_and_prediction(self.scoring_model_A, data_augm)
            # sample agent data
            opt_p_agent_A, pred_bag_agent_A, states_agent_A, actions_agent_A = self.sample_and_prediction(self.scoring_model_A,)

            # Update scoring model B using A's prediction
            # opt_or_not_B, opt_or_not_B_i, opt_or_not_B_j, lam_B = self.sample_and_prediction(self.scoring_model_B, data_augm)
            opt_p_agent_B, pred_bag_agent_B, states_agent_B, actions_agent_B = self.sample_and_prediction(self.scoring_model_B,)

            # close the gradient
            opt_or_not_A_label = opt_or_not_A.detach()
            opt_or_not_B_label = opt_or_not_B.detach()
            if data_augm == "normal":
                train_loss_A = self.calculate_loss_normal(opt_or_not_A, opt_or_not_B_label) # Update scoring model A using B's prediction
                train_loss_B = self.calculate_loss_normal(opt_or_not_B, opt_or_not_A_label) # Update scoring model B using A's prediction
            elif data_augm == "mixup":
                opt_or_not_A_i_label = opt_or_not_A_i.detach()
                opt_or_not_A_j_label = opt_or_not_A_j.detach()
                opt_or_not_B_i_label = opt_or_not_B_i.detach()
                opt_or_not_B_j_label = opt_or_not_B_j.detach()
 
                train_loss_A = self.calculate_loss_mixup(opt_or_not_A, opt_or_not_B_i_label, opt_or_not_B_j_label, lam_A)
                train_loss_B = self.calculate_loss_mixup(opt_or_not_B, opt_or_not_A_i_label, opt_or_not_A_j_label, lam_B)
            
            self.optimizer_step(self.optim_scoring_A, train_loss_A)
            self.optimizer_step(self.optim_scoring_B, train_loss_B)

            # print("opt_or_not_A grad ", opt_or_not_A.requires_grad)
            # print("opt_or_not_B grad ", opt_or_not_B.requires_grad)
            # print("opt_or_not_A_label grad ", opt_or_not_A_label.requires_grad) 
            # print("opt_or_not_B_label grad ", opt_or_not_B_label.requires_grad)
    
    def calculate_loss_data_augm(self, loss_pi, loss_exp, bag_loss_pi, bag_loss_exp, scoring_model, states_expert, actions_expert, states_agent, actions_agent):
        if self.data_augm == "normal":
            loss_disc = loss_pi + loss_exp
            bag_loss_disc = bag_loss_pi + bag_loss_exp
        
        else:
            mixed_x_s, mixed_x_a, lam = self.mixup_state_action(states_expert, actions_expert, states_agent, actions_agent)
            opt_p, pred_bag_mixed, *_  = scoring_model.forward(mixed_x_s, mixed_x_a)
            

            # calculate number of pred_bag_mixed where the first element is larger than 0.9
            # num = torch.sum(pred_bag_mixed[:,0] > 0.9).item() 
            # print("num 0.9", num)
            # num = torch.sum(pred_bag_mixed[:,0] > 0.99).item()
            # print("num 0.99", num)


            logits_mixed = opt_p[:,2] # pick up logit
            loss_mix = self.calculate_loss_mixup(logits_mixed, lam).mean() # mixup_loss for binary classification

            bag_loss_mix = lam * self.ce_loss(pred_bag_mixed, self.bag_expert) + (1 - lam) * self.ce_loss(pred_bag_mixed, self.bag_agent)
            # print("bag_loss_mix (BAG)", bag_loss_mix)
            bag_loss_mix = bag_loss_mix.mean()
            # print("bag_loss_mix (BAG MEAN)", bag_loss_mix)
            # print("loss_mix      ", loss_mix)
            # mixup_loss = lam * -F.logsigmoid(logits_mixed) + (1 - lam) * -F.logsigmoid(-1*logits_mixed)

            if self.data_augm == "mixup":
                loss_disc = loss_mix
                bag_loss_disc = bag_loss_mix

            elif self.data_augm == "both":
                loss_disc = 0.5*(loss_pi + loss_exp) + loss_mix
                # print("bag_loss_pi ", bag_loss_pi.shape)
                # print("bag_loss_exp ", bag_loss_exp.shape)
                # print("bag_loss_mix ", bag_loss_mix.shape)
                # print("bag_loss_mix ", bag_loss_mix)
                # print("pred_bag_mixed", pred_bag_mixed )        
                
                # print("\n")
                bag_loss_disc = 0.5*(bag_loss_pi + bag_loss_exp) + bag_loss_mix

        return loss_disc, bag_loss_disc, loss_mix, bag_loss_mix
    

    
    def update_single_scoring_model_uuloss(self, n_scoring_updates_per_round):
        self.scoring_model_A.train()

        for i in range(n_scoring_updates_per_round):

            # Sample agent data
            opt_p_agent, pred_bag_agent, states_agent, actions_agent = self.sample_and_prediction(self.scoring_model_A,)  # data is normalized
            

            # top 50% smallest values
            agent_topk_k = 0.5
            label_method = "threshold"
            if self.agentReplay_fine and self.round / self.n_rounds >  0.5:

                if label_method == "threshold":
                    low_idxs = (opt_p_agent < 0)  # logits < 0
                    opt_p_agent = opt_p_agent[low_idxs]
                    states_agent = states_agent[low_idxs]
                    actions_agent = actions_agent[low_idxs]
                    pred_bag_agent = pred_bag_agent[low_idxs]
                    # print("states_agent.shape after threshold", states_agent.shape)
                    # print("actions_agent.shape after threshold", actions_agent.shape)
                    # print("pred_bag_agent.shape after threshold", pred_bag_agent.shape)
                    # input("Press Enter to continue... Agent threshold")

                elif label_method == "topk":
                    k = int(agent_topk_k*self.batch_size)
                    _, lowk_idxs = torch.topk(opt_p_agent, k, largest=False)
                    opt_p_agent = opt_p_agent[lowk_idxs]
                    states_agent = states_agent[lowk_idxs]
                    actions_agent = actions_agent[lowk_idxs]
                    pred_bag_agent = pred_bag_agent[lowk_idxs]
                    # print("states_agent.shape after topk", states_agent.shape)
                    # input("Press Enter to continue... Agent topk")

                N2P_num = self.batch_size - opt_p_agent.shape[0]


                # Make states_agent and actions_agent have the same size as BATCH_SIZE
                states_agent_orig = cp(states_agent)
                actions_agent_orig = cp(actions_agent)
                indices = torch.randperm(states_agent.shape[0]) # shuffle the data
                states_agent_orig = states_agent_orig[indices]
                actions_agent_orig = actions_agent_orig[indices]
                if self.batch_size//states_agent.shape[0] > 1:
                    for _ in range(self.batch_size//states_agent.shape[0]-1):
                        states_agent = torch.cat((states_agent, states_agent_orig), 0)
                        actions_agent = torch.cat((actions_agent, actions_agent_orig), 0)
                        # print("states_agent.shape 1", states_agent.shape)
                diff = self.batch_size - states_agent.shape[0]
                # print("diff", diff)
                if diff > 0:
                    states_agent = torch.cat((states_agent, states_agent_orig[:diff]), 0)
                    actions_agent = torch.cat((actions_agent, actions_agent_orig[:diff]), 0)
                # print("states_agent.shape 2", states_agent.shape)
                # print("\n")
                # input("Press Enter to continue... Agent make up to batch size")
            else:
                if label_method == "threshold":
                    low_idxs = (opt_p_agent < 0)  # logits < 0
                    # count how many are below the threshold
                    N2P_num = self.batch_size - low_idxs.sum().item()

                elif label_method == "topk":
                    k = int(agent_topk_k*self.batch_size)
                    N2P_num = self.batch_size - k

                N2P_num = -N2P_num # label that agent fine is inactive

                # N2P_num = -1 # dummy value


            if self.uuLoss_all:
                bag_loss_agent = self.ce_loss_mean(pred_bag_agent, self.bag_agent[:pred_bag_agent.shape[0]])
                loss_agent = bag_loss_agent
            elif self.uuLoss_Nagent:
                loss_agent = -F.logsigmoid(-opt_p_agent).mean()

            # Sample uu expert data
            states_expert, actions_expert, bagLabels, binLabels, pseudo_binLabels, exp_demo_opt_idx = self.demonstrations.sample_uu(self.batch_size)  # data is normalized
            if (not self.demon_normalized) and self.normalize:
                states_expert, actions_expert = self.norm_s_a(states_expert, actions_expert)

            opt_p_expert, pred_bag_expert, *_ = self.scoring_model_A.forward(states_expert, actions_expert)
            
            

            if self.exp_loss == "uu_loss":
                loss_expert = self.ce_loss_mean(pred_bag_expert, bagLabels) # bag_loss_expert
                # input("Press Enter to continue... UUUUUU loss")

            elif self.exp_loss == "PN_loss":
                # loss_expert based on pseudo labels
                loss_expert = self.loss_bce_mean((opt_p_expert[:,0]), pseudo_binLabels) # PN loss based on pseudo labels
                # input("Press Enter to continue... PN loss")

 

            # Sample (self-labeled) opt expert data - calculate mixup loss
            states_expert_opt, actions_expert_opt, _ = self.demonstrations.sample_opt(self.batch_size)  # data is normalized
            if (not self.demon_normalized) and self.normalize:
                states_expert_opt, actions_expert_opt = self.norm_s_a(states_expert_opt, actions_expert_opt)



            if self.data_augm == "both" or self.data_augm == "mixup":
                mixed_x_s, mixed_x_a, lam = self.mixup_state_action(states_expert_opt, actions_expert_opt, states_agent, actions_agent)
                opt_p, pred_bag_mixed, *_  = self.scoring_model_A.forward(mixed_x_s, mixed_x_a)
                logits_mixed = opt_p[:,2] # pick up logit
                loss_mix = self.calculate_loss_mixup(logits_mixed, lam).mean()
                loss_step = (loss_agent + loss_expert) * 0.5 + loss_mix
                loss_third = loss_mix
            elif self.data_augm == "normal":
                # loss_third  treat states_expert_opt as optimal
                expert_s_pseudo_opt, expert_a_pseudo_opt, _ = self.demonstrations.sample_pseudo_opt(self.batch_size)  # data is normalized
                if (not self.demon_normalized) and self.normalize:
                    expert_s_pseudo_opt, expert_a_pseudo_opt = self.norm_s_a(expert_s_pseudo_opt, expert_a_pseudo_opt)
                opt_p_expert_pseudo_opt, *_ = self.scoring_model_A.forward(expert_s_pseudo_opt, expert_a_pseudo_opt)
                opt_p_expert_pseudo_opt = opt_p_expert_pseudo_opt[:,2] # pick up logit
                loss_third = -F.logsigmoid(opt_p_expert_pseudo_opt).mean()
                loss_step = loss_agent + loss_expert + loss_third
                # input("NNNN Press Enter to continue... normal loss")

            self.optimizer_step(self.optim_scoring_A, loss_step)

            # input("NNNN Press Enter to continue...")


            with torch.no_grad():
                acc_agent = (opt_p_agent < 0).float().mean().item()
                
                opt_p_expert = opt_p_expert[:,2] # pick up logit
                true_labels = binLabels[:, 0]
                true_labels = true_labels.long()           
                pred_labels = (opt_p_expert > 0).long()
                acc_expert = (pred_labels == true_labels).float().mean().item()

            #     # Confusion matrix components
            #     TP_exp = ((pred_labels == 1) & (true_labels == 1)).sum().item()
            #     TN_exp = ((pred_labels == 0) & (true_labels == 0)).sum().item()
            #     FP_exp = ((pred_labels == 1) & (true_labels == 0)).sum().item()
            #     FN_exp = ((pred_labels == 0) & (true_labels == 1)).sum().item()

            # # recall: in all true P data, how much labeled as P
            # recall_exp = TP_exp / (TP_exp + FN_exp) if (TP_exp + FN_exp) > 0 else -1

            # # unprecision: in all predicted P data, how much are actually N data (wrongly labeled)
            # unprecision_exp = FP_exp / (TP_exp + FP_exp) if (TP_exp + FP_exp) > 0 else -1

            self.acc_agent_list.append(acc_agent)
            self.acc_expert_list.append(acc_expert)
            self.loss_disc_list.append([loss_third.item(), loss_agent.item(), loss_expert.item()])


            self.expDemo_opt_num.append(self.demonstrations.opt_buffer_size)
            self.TP_exp.append(self.demonstrations.TP_exp)
            self.TN_exp.append(self.demonstrations.TN_exp)
            self.FP_exp.append(self.demonstrations.FP_exp)
            self.FN_exp.append(self.demonstrations.FN_exp)
            self.true_exp_P_num.append(self.demonstrations.true_P_num)
            self.true_exp_N_num.append(self.demonstrations.true_N_num)
            self.recall_exp.append(self.demonstrations.recall_exp)
            self.unprecision_exp.append(self.demonstrations.unprecision_exp)
            self.agt_N2P_num.append(N2P_num)

            self.exp_N2P_num.append(self.demonstrations.N2P)
            self.exp_P2N_num.append(self.demonstrations.P2N)

            self.expDemo_nonopt_num.append(-1) # dummy value
            # self.agt_N2P_num.append(-1) # dummy value
            # self.exp_N2P_num.append(-1) # dummy value
            # self.exp_P2N_num.append(-1) # dummy value
            self.agent_replay_buffer_opt_size.append(-1) # dummy value
            self.agent_replay_buffer_nonopt_size.append(-1) # dummy value

            if self.relabel_earlyStop:
                self.relabel_earlyStop_list.append(1)
            else:
                self.relabel_earlyStop_list.append(0)

        # update_opt_demo
        # if self.round / self.n_rounds >  0.2:
        if self.round / 200 >  0.01:

            # detect early stopping criteria
            # Early stop threshold (can be set as an attribute or parameter)
            window_len_stop = 5

            # Only check if we haven't already early-stopped
            if not self.relabel_earlyStop and self.early_stop_active:
                if self.round - int(0.2*self.n_rounds) > window_len_stop:
                    last_N2P = self.exp_N2P_num[-window_len_stop:]
                    last_P2N = self.exp_P2N_num[-window_len_stop:]
                    # if all(v == 0 for v in last_N2P) and all(v == 0 for v in last_P2N):
                    if all(v <= 30 for v in last_N2P) and all(v <= 30 for v in last_P2N):
                        self.relabel_earlyStop = True
            
            if self.expertDemo_fine:
                if not self.relabel_earlyStop:
                    relabel = True
                else:
                    relabel = False
            else:
                relabel = False

            if self.round / self.n_rounds <  0.2: # warm up period, do not relabel
                relabel = False
                self.relabel_earlyStop = False

            self.demonstrations.update_opt_demo(self.scoring_model_A, relabel=relabel)
            self.scoring_model_A.train()












    def update_single_scoring_model(self, n_scoring_updates_per_round):
        self.scoring_model_A.train()

        for i in range(n_scoring_updates_per_round):
            
            # sample agent data
            opt_p_agent, pred_bag_agent, states_agent, actions_agent = self.sample_and_prediction(self.scoring_model_A,)  # data is normalized

            
            bag_loss_agent = self.ce_loss(pred_bag_agent[:, 1], self.bag_agent[:, 1])


            # input("NNNN Press Enter to continue...")

            # sample expert data -
            states_expert, actions_expert, exp_demo_opt_idx = self.demonstrations.sample_opt(self.batch_size)  # data is normalized
            states_expert_nonopt, actions_expert_nonopt, exp_demo_nonopt_idx = self.demonstrations.sample_nonopt(self.batch_size)  # data is normalized

            if (not self.demon_normalized) and self.normalize:
                states_expert, actions_expert = self.norm_s_a(states_expert, actions_expert)
                states_expert_nonopt, actions_expert_nonopt = self.norm_s_a(states_expert_nonopt, actions_expert_nonopt)
                # print(" not self.demon_normalized")

            opt_p_expert, pred_bag_expert, *_ = self.scoring_model_A.forward(states_expert, actions_expert) 
            opt_p_expert_nonopt, pred_bag_expert_nonopt, *_ = self.scoring_model_A.forward(states_expert_nonopt, actions_expert_nonopt)

            # print("opt_p_EXP", opt_p_expert)
            # print("pred_bag_EXP ", pred_bag_expert)
            
            opt_p_expert = opt_p_expert[:,2] # pick up logit
            opt_p_expert_nonopt = opt_p_expert_nonopt[:,2] # pick up logit

            bag_loss_expert = self.ce_loss(pred_bag_expert[:, 0], self.bag_expert[:, 0])
            # print("\n")
            # print("-F.logsigmoid(opt_p_expert)          EXP  ", -F.logsigmoid(opt_p_expert))
            # print("bag_loss_expert   EXP", bag_loss_expert)
            # print("\n")

            # opt_p_expert is sigmoid output
            # opt_p_expert = inverse_sigmoid(opt_p_expert) 

            original_method = self.original_method
            
            expertDemo_fine = self.expertDemo_fine
            
            expertDemo_N2P = self.expertDemo_N2P
            expertDemo_useN = self.expertDemo_useN

            agentReplay_N2P = self.agentReplay_N2P
            agentReplay_useP = self.agentReplay_useP


            if self.round / self.n_rounds < 0.2 or original_method: # early stage, do not do fine-grained 
                loss_agent = -F.logsigmoid(-opt_p_agent).mean()
  
                self.exp_N2P_num.append(0)
                self.exp_P2N_num.append(0)
                self.agt_N2P_num.append(0)
                
                # Track replay buffer sizes (no change in early stage)
                self.agent_replay_buffer_opt_size.append(self.agent_replay_buffer_opt.size())
                self.agent_replay_buffer_nonopt_size.append(self.agent_replay_buffer_nonopt.size())

                states_agent_N = cp(states_agent)
                actions_agent_N = cp(actions_agent)

            else:
                selfLabel = True
                expert_P2N = 0.3
                expert_N2P = 0.95
                agent_N2P = 0.95


                if selfLabel:
                # if i > 0.2*n_scoring_updates_per_round:

                    # use torch.sigmoid(opt_p_agent) to pick up the data which has opt_p_agent < 0.3

                    # Expert
                    expert_P2N_indices = torch.where(torch.sigmoid(opt_p_expert) < expert_P2N)[0]
                    # print("expert_P2N_indices.shape[0] ", expert_P2N_indices.shape[0])
                    if expert_P2N_indices.shape[0] > 0 and expertDemo_fine:
                        states_expert_P2N = cp(states_expert[expert_P2N_indices])
                        actions_expert_P2N = cp(actions_expert[expert_P2N_indices])
                        self.demonstrations.add_P2N(states_expert_P2N, actions_expert_P2N)

                        print("expert_P2N_indices ", expert_P2N_indices)
                        print("expert_P2N_indices shape ", expert_P2N_indices.shape)

                        print("states_expert shape ", states_expert.shape)
                        print("states_expert[expert_P2N_indices] shape ", states_expert[expert_P2N_indices].shape)

                        print("exp_demo_opt_idx shape", exp_demo_opt_idx.shape)
                        print("exp_demo_opt_idx [expert_P2N_indices] shape", exp_demo_opt_idx[expert_P2N_indices].shape)

                        print("exp_demo_opt_idx", exp_demo_opt_idx)
                        print("exp_demo_opt_idx [expert_P2N_indices]", exp_demo_opt_idx[expert_P2N_indices])
                        print("")
                        self.demonstrations.remove_P(exp_demo_opt_idx[expert_P2N_indices])
                        

                    expert_N2P_indices = torch.where(torch.sigmoid(opt_p_expert_nonopt) > expert_N2P)[0]
                    # print("expert_N2P_indices.shape[0] ", expert_N2P_indices.shape[0])
                    if expert_N2P_indices.shape[0] > 0 and  expertDemo_fine and expertDemo_N2P:
                        states_expert_N2P = cp(states_expert_nonopt[expert_N2P_indices])
                        actions_expert_N2P = cp(actions_expert_nonopt[expert_N2P_indices])
                        self.demonstrations.add_N2P(states_expert_N2P, actions_expert_N2P)
                        print("exp_demo_nonopt_idx", exp_demo_nonopt_idx)
                        print("exp_demo_nonopt_idx [expert_N2P_indices]", exp_demo_nonopt_idx[expert_N2P_indices])
                        self.demonstrations.remove_N(exp_demo_nonopt_idx[expert_N2P_indices])
                        

                    # Agent 
                    agent_N2P_indices = torch.where(torch.sigmoid(opt_p_agent) > agent_N2P)[0]
                    agent_N_indices = torch.where(torch.sigmoid(opt_p_agent) <= agent_N2P)[0]

                    if not agentReplay_N2P:
                        agent_N2P_indices = torch.where(torch.sigmoid(opt_p_agent) > 2)[0]  # create empty indices
                        agent_N_indices =  torch.where(torch.sigmoid(opt_p_agent) <= 2)[0] # create full indices

                    # print("agent_N2P_indices.shape[0] ", agent_N2P_indices.shape[0])
                    if agent_N2P_indices.shape[0] > 0:
                        s_agent_N2P = states_agent[agent_N2P_indices]
                        a_agent_N2P = actions_agent[agent_N2P_indices]
                        self.agent_replay_buffer_opt.add(s_agent_N2P, a_agent_N2P)
                
                    if agent_N_indices.shape[0] > 0:
                        s_agent_N = states_agent[agent_N_indices]
                        a_agent_N = actions_agent[agent_N_indices]
                        self.agent_replay_buffer_nonopt.add(s_agent_N, a_agent_N)

                self.exp_N2P_num.append(expert_N2P_indices.shape[0])
                self.exp_P2N_num.append(expert_P2N_indices.shape[0])
                self.agt_N2P_num.append(agent_N2P_indices.shape[0])
                
                # Track replay buffer sizes
                self.agent_replay_buffer_opt_size.append(self.agent_replay_buffer_opt.size())
                self.agent_replay_buffer_nonopt_size.append(self.agent_replay_buffer_nonopt.size())


                if not self.agent_replay_buffer_nonopt.buffer_full :
                    loss_agent_nonopt_N = -F.logsigmoid(-opt_p_agent[agent_N_indices]).mean()
                    states_agent_N = s_agent_N
                    actions_agent_N = a_agent_N
                else:
                    if agent_N_indices.shape[0] > 0 and agent_N_indices.shape[0] < self.batch_size: # some new N data, some from buffer
                        residual_num = self.batch_size - agent_N_indices.shape[0]
                        s_agent_nonopt, a_agent_nonopt = self.agent_replay_buffer_nonopt.sample(residual_num)
                        # print("s_agent_nonopt shape ", s_agent_nonopt.shape)
                        opt_p_agent_residual, *_ = self.discrm_predict(self.scoring_model_A, s_agent_nonopt, a_agent_nonopt)
                        loss_agent_nonopt_N = -F.logsigmoid(- torch.cat((opt_p_agent_residual, opt_p_agent[agent_N_indices]), dim=0) ).mean() # list plus
                        states_agent_N = torch.cat((s_agent_nonopt, states_agent[agent_N_indices]), dim=0)
                        actions_agent_N = torch.cat((a_agent_nonopt, actions_agent[agent_N_indices]), dim=0)
                    
                    elif agent_N_indices.shape[0] == self.batch_size: # all new N data
                        loss_agent_nonopt_N = -F.logsigmoid(-opt_p_agent).mean()
                        states_agent_N = cp(states_agent)
                        actions_agent_N = cp(actions_agent)

                    elif agent_N_indices.shape[0] == 0: # no new N data
                        s_agent_nonopt, a_agent_nonopt = self.agent_replay_buffer_nonopt.sample(self.batch_size)
                        opt_p_agent_residual, *_ = self.discrm_predict(self.scoring_model_A, s_agent_nonopt, a_agent_nonopt)
                        loss_agent_nonopt_N = -F.logsigmoid(-opt_p_agent_residual).mean()
                        states_agent_N = s_agent_nonopt
                        actions_agent_N = a_agent_nonopt

                    
                if (not self.agent_replay_buffer_opt.buffer_full) or (not agentReplay_useP):
                    loss_agent_opt_P = 0
                    loss_agent = loss_agent_nonopt_N
                else:
                    if agent_N2P_indices.shape[0] > 0 and agent_N2P_indices.shape[0] < self.batch_size:
                        residual_num = self.batch_size - agent_N2P_indices.shape[0]
                        s_agent_opt, a_agent_opt = self.agent_replay_buffer_opt.sample(residual_num)
                        opt_p_agent_residual, *_ = self.discrm_predict(self.scoring_model_A, s_agent_opt, a_agent_opt)
                        # loss_agent_opt_P = -F.logsigmoid( (opt_p_agent_residual + opt_p_agent[agent_N2P_indices])).mean()
                        loss_agent_opt_P = -F.logsigmoid( torch.cat((opt_p_agent_residual, opt_p_agent[agent_N2P_indices]), dim=0) ).mean() # list plus
                    elif agent_N2P_indices.shape[0] == self.batch_size:
                        loss_agent_opt_P = -F.logsigmoid( opt_p_agent).mean()
                    elif agent_N2P_indices.shape[0] == 0:
                        s_agent_opt, a_agent_opt = self.agent_replay_buffer_opt.sample(self.batch_size)
                        opt_p_agent_residual, *_ = self.discrm_predict(self.scoring_model_A, s_agent_opt, a_agent_opt)
                        loss_agent_opt_P = -F.logsigmoid( opt_p_agent_residual).mean()
                    
                    loss_agent = 0.5 * (loss_agent_nonopt_N + loss_agent_opt_P)
            
            
            
            self.expDemo_opt_num.append(self.demonstrations.opt_buffer_size)
            self.expDemo_nonopt_num.append(self.demonstrations.nonopt_buffer_size)

            # # Checked, almost same as bag_loss_agent.mean(), but bag_loss_agent will round the small values <xx e-08 to 0
            # print("-F.logsigmoid(-opt_p_agent)", -F.logsigmoid(-opt_p_agent)[:100])
            # print("bag_loss_agent ", bag_loss_agent[:100])

            # error similar, around: 1 e-07
            # print("-F.logsigmoid(-opt_p_exp)", -F.logsigmoid(opt_p_expert)[:100])
            # print("bag_loss_exp ", bag_loss_expert[:100])

            # if i <= 0.1*n_scoring_updates_per_round:
            #     loss_agent = -F.logsigmoid(-opt_p_agent).mean()

            # elif i > 0.1*n_scoring_updates_per_round:
            #     pass

            loss_expert = -F.logsigmoid(opt_p_expert).mean()
            loss_expert_nonopt = -F.logsigmoid(- opt_p_expert_nonopt).mean()


            # loss = loss_agent + loss_expert
            loss, bag_loss, loss_mix, _ = self.calculate_loss_data_augm(loss_agent, loss_expert, bag_loss_agent.mean(), bag_loss_expert.mean(), self.scoring_model_A, states_expert, actions_expert, states_agent_N, actions_agent_N)
   

            
            if original_method:
                loss_step = 0.5 * (loss_expert + loss_agent) + loss_mix  # origin method
            else:
                if expertDemo_useN:
                    loss_step = 0.5 * (0.5 * (loss_expert + loss_expert_nonopt) + loss_agent)  + loss_mix
                else:
                    loss_step = 0.5 * (loss_expert + loss_agent) + loss_mix

            # if self.loss_type == "uu_loss":
            #     self.optimizer_step(self.optim_scoring_A, bag_loss)
            # else:
            #     self.optimizer_step(self.optim_scoring_A, loss)
            self.optimizer_step(self.optim_scoring_A, loss_step)

            # calculate accuracy  --- need to change if use Fine-grained GAIL !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            with torch.no_grad():
                acc_agent = (opt_p_agent < 0).float().mean().item()
                acc_expert = (opt_p_expert > 0).float().mean().item()
            self.acc_agent_list.append(acc_agent)
            self.acc_expert_list.append(acc_expert)
            self.loss_disc_list.append([loss.item(), loss_agent.item(), loss_expert.item()])


    def update_RL(self, total_timesteps):
        # test_print = self.rl_algo.env.get_attr('test_print')  # Retrieve all environments
        # print("test_print ", test_print)
        # self.rl_algo.env.set_attr('test_print', test_print[0]+1)  # Set the attribute for all environments
        self.scoring_model_A.eval()
        self.rl_algo.env.set_attr('scoring_model', self.scoring_model_A)
        
        # self.rl_algo.env.envs[0].scoring_model = self.scoring_model_A
        # self.rl_algo.env.envs[0].scoring_model.eval()
        self.rl_algo.learn(
                total_timesteps=total_timesteps,
                reset_num_timesteps=False,
                callback=self.rl_callback,
                progress_bar=False,
                # progress_bar=True,
     
        )
        print("num_timesteps ", self.rl_algo.num_timesteps)
        print("")

    def train(self, total_timesteps, data_augm=None):
        n_rounds = total_timesteps // self.rl_train_timesteps
        self.n_rounds = n_rounds
        assert n_rounds >= 1, (
            "No updates (need at least "
            f"{self.rl_train_timesteps} timesteps, have only "
            f"total_timesteps={total_timesteps})!"
                )
        # n_rounds = 5 # for short debugging
        for round in tqdm.tqdm(range(0, n_rounds), desc="round"):
            self.round = round
            self.update_RL(self.rl_train_timesteps)
            # self.update_RL(2000)  # fast debugging
            print("update_RL ")
            # self.cotrain_scoring_models(self.n_scoring_updates_per_round, data_augm = data_augm)

            if self.uuLoss_Nagent or self.uuLoss_all:
                self.update_single_scoring_model_uuloss(self.n_scoring_updates_per_round)
            else:
                self.update_single_scoring_model(self.n_scoring_updates_per_round)
            print("update_single_scoring_model ")



