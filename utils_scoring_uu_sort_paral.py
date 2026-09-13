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


def categorical_cross_entropy_from_probs(probabilities, targets, eps=1e-8):
    """Cross-entropy for normalized probabilities and one-hot or soft targets."""
    log_probabilities = torch.log(probabilities.clamp_min(eps))
    return -(targets * log_probabilities).sum(dim=-1).mean()

def inverse_sigmoid(x):
    return torch.log(x / (1 - x))

class RL_Scoring():
    def __init__(self, demonstrations, batch_size, device=None, 
                 rl_algo=None, rl_train_timesteps=int(1.5e4), rl_callback=None,
                 scoring_model_A=None,  scoring_model_B=None, n_scoring_updates_per_round=100, lr_scoring=1e-4,
                 scaler_state=None, scaler_action=None, sort_ratio=0.5, sort_loss=False, data_augm=None, loss_type = None,
                 demon_normalized = True, normalize = True):
        

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

        # self.scoring_model_A = scoring_model_A.net.to(self.device)
        # self.scoring_model_B = scoring_model_B.net.to(self.device)
        self.scoring_model_A = scoring_model_A.net.to(self.device)
        self.scoring_model_B = scoring_model_B.net.to(self.device)

        self.n_scoring_updates_per_round = n_scoring_updates_per_round
        self.optim_scoring_A = Adam(self.scoring_model_A.parameters(), lr=lr_scoring)
        self.optim_scoring_B = Adam(self.scoring_model_B.parameters(), lr=lr_scoring)

        # self.ce_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.ce_loss = torch.nn.BCELoss(reduction='none')
        self.ce_loss.to(self.device)

        self.ce_loss_mean = categorical_cross_entropy_from_probs


        self.s_max = torch.tensor(scaler_state.data_max_, dtype=torch.float32).to(self.device)
        self.s_min = torch.tensor(scaler_state.data_min_, dtype=torch.float32).to(self.device)
        self.a_max = torch.tensor(scaler_action.data_max_, dtype=torch.float32).to(self.device)
        self.a_min = torch.tensor(scaler_action.data_min_, dtype=torch.float32).to(self.device)

        self.acc_agent_list = []
        self.acc_expert_list = []
        self.loss_disc_list = []

        self.data_augm = data_augm # "mixup" or "normal" or "both"

        self.sort_loss = sort_loss # if True, sort the loss and pick up the 50% (sort_ratio) of data which has smallest loss
        self.sort_ratio = sort_ratio

        # for calculate UU_loss
        self.scoring_model_A.priors_corr = [0.5, 0.5] # bag size ratio = 0.5
        self.scoring_model_A.Pi = [1, 0] # optimal ratio = 1 for bag of expert, and optimal ratio = 0 for bag of agent
        # check if self.scoring_model_A.prior_test exists

        if  hasattr(self.scoring_model_A, 'prior_test'): # old version has prior_test, but new version has Pi_test
            self.scoring_model_A.Pi_test = self.scoring_model_A.prior_test
    
        self.bag_agent = torch.tensor([[0, 1]]).float().to(self.device)
        self.bag_expert = torch.tensor([[1, 0]]).float().to(self.device)
        self.bag_agent = self.bag_agent.repeat((self.batch_size // self.bag_agent.shape[0], 1))
        self.bag_expert = self.bag_expert.repeat((self.batch_size // self.bag_expert.shape[0], 1))

        self.loss_type = loss_type


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
        # print("x_s_agent  ", x_s_agent.shape)
        # print("x_s_expert ", x_s_expert.shape)
        mixed_x_s = lam * x_s_expert + (1 - lam) * x_s_agent[index, :, :]
        mixed_x_a = lam * x_a_expert + (1 - lam) * x_a_agent[index, :, :]

        return mixed_x_s, mixed_x_a, lam
    
    

    def sample_and_prediction(self, model):
        # Sample a batch of transitions
        trajs = self.rl_algo.replay_buffer.sample(self.batch_size)
        state, action, next_state, _, done = trajs.observations, trajs.actions, trajs.next_observations, trajs.rewards, trajs.dones
        if self.normalize:
            state, action = self.norm_s_a(state, action)
        state = state.unsqueeze(1)  
        action = action.unsqueeze(1) 

        # prediction
        opt_or_not, pred_bag, *_ = model.forward(state, action) 
        # print("opt_p_AGENT ", opt_or_not)
        # print("pred_bag_AGENT ", pred_bag)
        opt_or_not = opt_or_not[:,2]  # pick up logit 

        return opt_or_not, pred_bag, state, action
    
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
            opt_or_not_A, opt_or_not_A_i, opt_or_not_A_j, lam_A = self.sample_and_prediction(self.scoring_model_A, data_augm)

            # Update scoring model B using A's prediction
            opt_or_not_B, opt_or_not_B_i, opt_or_not_B_j, lam_B = self.sample_and_prediction(self.scoring_model_B, data_augm)

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

        return loss_disc, bag_loss_disc
    
    def update_single_scoring_model(self, n_scoring_updates_per_round):
        self.scoring_model_A.train()


        for i in range(n_scoring_updates_per_round):
            
            # sample agent data
            
            opt_p_agent, pred_bag_agent, states_agent, actions_agent = self.sample_and_prediction(self.scoring_model_A,)
            # print("\n")
            # print("self.bag_agent shape ", self.bag_agent.shape)
            # print("pred_bag_agent shape ", pred_bag_agent.shape)
            # print("pred_bag_AGENT ", pred_bag_agent)        
            
            bag_loss_agent = self.ce_loss(pred_bag_agent[:, 1], self.bag_agent[:, 1])
            # print("\n")
            # print("pred_bag_agent         AGENT  ", pred_bag_agent)
            # print("-torch.log(1-pred_bag_agent)         AGENT  ", -torch.log(1-pred_bag_agent))
            # print("F.sigmoid(opt_p_agent)               AGENT  ", F.sigmoid(opt_p_agent))
            # print("-torch.log(1-F.sigmoid(opt_p_agent)) AGENT  ", -torch.log(1-F.sigmoid(opt_p_agent)))
            # print("-F.logsigmoid(-opt_p_agent)          AGENT  ", -F.logsigmoid(-opt_p_agent))
            # print("bag_loss_agent  ", bag_loss_agent)

            # print("\n")
            # print("\n")

            # print("bah_loss_agent shape ", bag_loss_agent.shape)
            # print("bah_loss_agent ", bag_loss_agent)
            # print("binary loss_agent", -F.logsigmoid(-opt_p_agent))
            # print("binary loss_agent shape", (-F.logsigmoid(-opt_p_agent)).shape)
            # print(" bah_loss_agent none -- > mean ", bag_loss_agent.mean()) 
            # print(" bah_loss_agent mean ", self.ce_loss_mean(pred_bag_agent, self.bag_agent)) 
            
            # opt_p_agent is sigmoid output
            # opt_p_agent = inverse_sigmoid(opt_p_agent) # discrmdiscrmdiscrm

            # sample expert data -
            states_expert, actions_expert, _ = self.demonstrations.sample_opt(self.batch_size)  # data is normalized
            if not self.demon_normalized and self.normalize:
                states_expert, actions_expert = self.norm_s_a(states_expert, actions_expert)
                # print(" not self.demon_normalized")
            

            opt_p_expert, pred_bag_expert, *_ = self.scoring_model_A.forward(states_expert, actions_expert) 
            # print("opt_p_EXP", opt_p_expert)
            # print("pred_bag_EXP ", pred_bag_expert)
            
            opt_p_expert = opt_p_expert[:,2] # pick up logit
            bag_loss_expert = self.ce_loss(pred_bag_expert[:, 0], self.bag_expert[:, 0])
            # print("\n")
            # print("-F.logsigmoid(opt_p_expert)          EXP  ", -F.logsigmoid(opt_p_expert))
            # print("bag_loss_expert   EXP", bag_loss_expert)
            # print("\n")

            # opt_p_expert is sigmoid output
            # opt_p_expert = inverse_sigmoid(opt_p_expert) 
            
            if i <= 0.1*n_scoring_updates_per_round or not self.sort_loss:
                loss_agent = -F.logsigmoid(-opt_p_agent).mean()

            elif i > 0.1*n_scoring_updates_per_round and self.sort_loss:
                # pick up the 10% of data which has smallest loss
                if self.loss_type == "uu_loss":
                    loss_agent_all = bag_loss_agent ## check the shape ????
                else:
                    loss_agent_all = -F.logsigmoid(-opt_p_agent)

                # sort the loss
                loss_agent_all_sorted, indices = torch.sort(loss_agent_all, descending=False)

                # pick up the 50% of data which has smallest loss
                loss_agent = -F.logsigmoid(-opt_p_agent[indices[:int(self.sort_ratio*self.batch_size)]]).mean()
                states_agent = states_agent[indices[:int(self.sort_ratio*self.batch_size)]]
                actions_agent = actions_agent[indices[:int(self.sort_ratio*self.batch_size)]]

                # # Repeat states_agent so that the size is the same as states_expert -----
                # print("\n")
                # print("states_expert.shape", states_expert.shape)
                # print("states_agent.shape", states_agent.shape)
                # print("states_expert.shape[0]//states_agent.shape[0]-1   ", states_expert.shape[0]//states_agent.shape[0]-1)
                states_agent_orig = cp(states_agent)
                actions_agent_orig = cp(actions_agent)
                indices = torch.randperm(states_agent.shape[0]) # shuffle the data
                states_agent_orig = states_agent_orig[indices]
                actions_agent_orig = actions_agent_orig[indices]

                if states_expert.shape[0]//states_agent.shape[0] > 1:
                    for _ in range(states_expert.shape[0]//states_agent.shape[0]-1):
                        states_agent = torch.cat((states_agent, states_agent_orig), 0)
                        actions_agent = torch.cat((actions_agent, actions_agent_orig), 0)
                        # print("states_agent.shape", states_agent.shape)

                diff = states_expert.shape[0] - states_agent.shape[0]
                # print("diff", diff)
                if diff > 0:
                    states_agent = torch.cat((states_agent, states_agent_orig[:diff]), 0)
                    actions_agent = torch.cat((actions_agent, actions_agent_orig[:diff]), 0)
                # print("states_agent.shape", states_agent.shape)
                # print("\n")

                # only pick up the 50% of data which has smallest loss -- need check !!! added on 26/06/2024
                bag_loss_agent = bag_loss_agent[indices[:int(self.sort_ratio*self.batch_size)]]

            loss_expert = -F.logsigmoid(opt_p_expert).mean()

            # loss = loss_agent + loss_expert
            loss, bag_loss = self.calculate_loss_data_augm(loss_agent, loss_expert, bag_loss_agent.mean(), bag_loss_expert.mean(), self.scoring_model_A, states_expert, actions_expert, states_agent, actions_agent)
            if self.loss_type == "uu_loss":
                self.optimizer_step(self.optim_scoring_A, bag_loss)
            else:
                self.optimizer_step(self.optim_scoring_A, loss)

            # calculate accuracy
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
        assert n_rounds >= 1, (
            "No updates (need at least "
            f"{self.rl_train_timesteps} timesteps, have only "
            f"total_timesteps={total_timesteps})!"
                )
        
        for round in tqdm.tqdm(range(0, n_rounds), desc="round"):
            self.update_RL(self.rl_train_timesteps)
            # self.update_RL(2000)    
            print("update_RL ")
            # self.cotrain_scoring_models(self.n_scoring_updates_per_round, data_augm = data_augm)
            self.update_single_scoring_model(self.n_scoring_updates_per_round)
            print("update_single_scoring_model ")



class Custom_Env(gym.Wrapper):
    def __init__(self, env_name, scoring_model, input_scaler_s, input_scaler_a,  normalize, env=None, modify_reward=True):
        # Create the original environment
        if env is None:
            env = gym.make(env_name)

        super().__init__(env)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scoring_model = scoring_model.net
        self.scoring_model.eval()
        self.scoring_model.to(self.device)
        # self.test_print = 9999

        self.input_scaler_s = input_scaler_s
        self.input_scaler_a = input_scaler_a
        
        self.old_obs = None
        self.modify_reward = modify_reward

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
        
        # Modify the reward calculation here (customize as needed)
        if self.modify_reward:
            if self.old_obs is not None:
                modified_reward, modified_reward_orig = self._custom_reward(self.old_obs, action, reward)

            else:
                modified_reward = 0 # the first step has no reward
        else:
            modified_reward = -1 # dummy reward
        self.old_obs = obs


        # modified_reward = -1
        # modified_reward = 0
        
        self.custom_reward_sum += modified_reward
        self.real_reward_sum += reward
        self.step_count += 1

        if done:
            self.old_obs = None
            # print("custom_reward_sum ", self.custom_reward_sum, "    real_reward_sum ", self.real_reward_sum, "   step len", self.step_count)
            self.custom_reward_sum = 0
            self.real_reward_sum = 0
            self.step_count = 0

        # print("self.modify_reward ", self.modify_reward )

        if self.modify_reward:
            return obs, modified_reward, terminated, truncated, info
        else:
            # reward = -10
            return obs, reward, terminated, truncated, info

    def _custom_reward(self, obs, action, reward):
        obs_0 = obs
        action_0 = action

        if self.normalize:
            obs = self.input_scaler_s.transform(obs.reshape(1, -1))
            action = self.input_scaler_a.transform(action.reshape(1, -1))

        else:
            obs = obs.reshape(1, -1)
            action = action.reshape(1, -1)

        obs = torch.from_numpy(obs[0]).float().unsqueeze(0).to(self.device)
        action = torch.from_numpy(action[0]).float().unsqueeze(0).to(self.device)


        obs = obs.unsqueeze(0) # scorescorescore
        action = action.unsqueeze(0)  # scorescorescore

        with torch.no_grad():

            opt_or_not, class_i, nn_infos = self.scoring_model.forward(obs, action)  # scorescorescore
            opt_or_not_old = opt_or_not 
            # opt_or_not = self.scoring_model.forward(obs, action) # discrmdiscrmdiscrm

            # RL agent / Generator of (GAIL) is to maximize E_{\pi} [-log(1 - D)].
            # opt_or_not = np.max((1e-6, opt_or_not.cpu()[0][0]))
            opt_or_not = opt_or_not[:,2] # scorescorescore
            # make sure opt_or_not is numerical stable larger than e-6 and smaller than 1-e-6
            # opt_or_not = torch.max(torch.tensor(1e-6).to(self.device), opt_or_not)
            # opt_or_not = torch.min(torch.tensor(1 - 1e-6).to(self.device), opt_or_not)
            opt_or_not_temp = opt_or_not

            # opt_or_not = inverse_sigmoid(opt_or_not) # scorescorescore
            opt_or_not = -F.logsigmoid(-opt_or_not)

            # check if opt_or_not is numerical stable

            if torch.isnan(opt_or_not.cpu()[0]) or torch.isnan(opt_or_not[0])   or torch.isinf(opt_or_not):
                print("")
                print("")
                print("")
                print("nan ")
                print("obs 0 ", obs_0, "  action 0 ", action_0)
                print("obs ", obs, "  action  ", action)
                print("opt_or_not old ", opt_or_not_old)
                print("opt_or_not_temp ", opt_or_not_temp)
                print("opt_or_not ", opt_or_not)
                print("opt_or_not ", opt_or_not.cpu())
                print("opt_or_not ", opt_or_not.cpu()[0])
                print("nn_infos ", nn_infos)
                print("")
                print("")
                print("")
                opt_or_not[0] = 0

            if  torch.isinf(opt_or_not.cpu()[0]) or torch.isnan(opt_or_not[0])  or torch.isinf(opt_or_not):
                print("")
                print("")
                print("")
                print("inf ")
                print("opt_or_not old ", opt_or_not_old)
                print("opt_or_not_temp ", opt_or_not_temp)
                print("opt_or_not ", opt_or_not)
                print("opt_or_not ", opt_or_not.cpu())
                print("opt_or_not ", opt_or_not.cpu()[0])
                print("nn_infos ", nn_infos)
                print("")
                print("")
                print("")
                opt_or_not[0] = 10
            



        # print("opt_or_not ", opt_or_not)
            
        # r_select = "log_log"
        # r_select = "log"
        r_select = "no_change"
            
        if r_select == "log_log":
            reward_pred_orig = np.array(opt_or_not.cpu()[0][0])
            reward_pred = np.max((1e-6, reward_pred_orig))
            reward_pred = np.min((1 - 1e-6, reward_pred))
            reward_pred = np.log(reward_pred ) - np.log(1 - reward_pred)
            reward_pred = np.max((-2, reward_pred))
            reward_pred = np.min((2, reward_pred))
        elif r_select == "log":
            reward_pred_orig = np.array(opt_or_not.cpu()[0][0])
            reward_pred = np.max((1e-6, reward_pred_orig))
            reward_pred = np.min((1 - 1e-6, reward_pred))
            reward_pred = np.log(reward_pred)
            reward_pred = np.max((-2, reward_pred))
            reward_pred = np.min((2, reward_pred))
        elif r_select == "no_change":
            reward_pred_orig = np.array(opt_or_not.cpu()[0])
            reward_pred = reward_pred_orig
            # reward_pred = np.max((1e-6, reward_pred_orig))
            # reward_pred = np.min((1 - 1e-6, reward_pred))
        
        # modified_reward = reward * 0  # Modify the reward for testing
        return reward_pred, reward_pred_orig


        

