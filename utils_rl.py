from copy import deepcopy as cp
import numpy as np
import gymnasium as gym
import torch
import math

def True_or_False(string):
    if string == "True":
        return True
    else:
        return False

def eval_model_multiEnvs(loaded_model, env, n_eval_episodes=10, max_ep_length=2*50000):
    """
    Evaluate a vectorized model over multiple parallel envs.

    :param loaded_model: a Stable-Baselines3 model supporting .predict
    :param env: a VecEnv with num_envs parallel envs
    :param n_eval_episodes: total number of episodes to collect (across all envs)
    :param max_ep_length: maximum steps per episode
    :returns:
        mean_reward: float
        std_reward: float
        traj_s_list: list of state-trajectories (each a list of observations)
        traj_a_list: list of action-trajectories
        traj_r_list: list of reward-trajectories
        acc_reward_list: list of cumulative rewards per episode
        steps_list: list of episode lengths
    """
    num_envs = env.num_envs

    # global accumulators
    steps_list, acc_reward_list = [], []
    traj_s_list, traj_a_list, traj_r_list = [], [], []

    # per-env buffers
    buf = [{
        "s": [], "a": [], "r": [],
        "acc_reward": 0.0, "steps": 0
    } for _ in range(num_envs)]

    # initial observations: shape (num_envs, obs_dim, …)
    obs = env.reset()

    # keep going until we have enough episodes
    while len(acc_reward_list) < n_eval_episodes:
        # get actions for all envs
        actions, _states = loaded_model.predict(obs, deterministic=True)

        # step all envs at once
        next_obs, rewards, dones, infos = env.step(actions)

        # for each sub-env, record and check for termination
        for i in range(num_envs):
            # record state→action→reward
            buf[i]["s"].append(cp(obs[i]))
            buf[i]["a"].append(actions[i])
            buf[i]["r"].append(rewards[i])
            buf[i]["acc_reward"] += rewards[i]
            buf[i]["steps"] += 1

            # if this env finished, or hit max length, flush to global lists
            if dones[i] or buf[i]["steps"] >= max_ep_length:
                steps_list.append(buf[i]["steps"])
                acc_reward_list.append(buf[i]["acc_reward"])
                traj_s_list.append(buf[i]["s"])
                traj_a_list.append(buf[i]["a"])
                traj_r_list.append(buf[i]["r"])

                # reset only this env’s buffer
                buf[i] = {"s": [], "a": [], "r": [], "acc_reward": 0.0, "steps": 0}

                if len(acc_reward_list)  == n_eval_episodes:
                    break

        # advance
        obs = next_obs

    # compute aggregate stats
    mean_reward = np.mean(acc_reward_list)
    std_reward = np.std(acc_reward_list)

    # print(f"Evaluated over {n_eval_episodes} episodes: mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")
    # print("acc_reward_list", acc_reward_list , "shape", np.array(acc_reward_list).shape)
    # print("traj_s_list shape", np.array(traj_s_list).shape)
    # aa=bb
    return mean_reward, std_reward, traj_s_list, traj_a_list, traj_r_list, acc_reward_list, steps_list




def eval_model(loaded_model, env, n_eval_episodes=10, max_ep_length = 2*50000):
    # Evaluate the  loaded model for 10 episodes in the specified environment
    steps_list = []
    acc_reward_list = []
    traj_s_list = []
    traj_a_list = []
    traj_r_list = []

    for ep in range(n_eval_episodes):
        done = False
        traj_s = []
        traj_a = []
        traj_r = []
        gt_rewards = []
        r = 0
        ob = env.reset()
        steps = 0
        acc_reward = 0

        while True:
            # action = agent.act(ob, r, done)
            action, _states = loaded_model.predict(ob, deterministic=True)
            # print("step ", steps,)
            # print("state ", ob) 
            # print("action ", action)
            # print("r ", r)
            # print("------------------")
            # print("action ", action)
            ob_old = cp(ob)
            ob, r, done, info = env.step(action)
            # print(r)
            ob_processed = cp(ob_old)

            # print("ob_processed.shape ", ob_processed.shape)
            # print(ob_processed)
            # print("action.shape ", action.shape)
            # print(action)
            # print("r.shape ", r.shape)
            # print(r)

            traj_s.append(ob_processed[0])
            traj_a.append(action[0])
            traj_r.append(r[0])
            # print("len", len(traj))
            gt_rewards.append(r[0])
            steps += 1
            acc_reward += r[0]

            # print("traj_s len", len(traj_s))
            # print("done", done)
            
            if done or len(traj_s) >= max_ep_length:
                steps_list.append(steps)
                acc_reward_list.append(acc_reward)

                if len(traj_s) >= max_ep_length:
                    print("max ep_length reached")
                break
        
        traj_s_list.append(traj_s)
        traj_a_list.append(traj_a)
        traj_r_list.append(traj_r)

    mean_reward = np.mean(acc_reward_list)
    std_reward = np.std(acc_reward_list)
    return mean_reward, std_reward, traj_s_list, traj_a_list, traj_r_list, acc_reward_list, steps_list



# Get RL params ---------------------------------------------------
def get_TRPO_RL_params(RL_params, RL_alg, env_name):
    print("RL_alg", RL_alg)
    print("env_name", env_name[:-3])
    RL_params_alg = RL_params[RL_alg]
    RL_params_env = RL_params_alg[env_name[:-3]]
    policy = RL_params_env["policy"]
    batch_size = RL_params_env["batch_size"]
    cg_damping = RL_params_env["cg_damping"]
    cg_max_steps = RL_params_env["cg_max_steps"]
    gae_lambda = RL_params_env["gae_lambda"]
    gamma = RL_params_env["gamma"]
    learning_rate = RL_params_env["learning_rate"]
    n_critic_updates = RL_params_env["n_critic_updates"]
    n_steps = RL_params_env["n_steps"]
    normalize_advantage = RL_params_env["normalize_advantage"]
    if normalize_advantage == "True":
        normalize_advantage = True
    else:
        normalize_advantage = False
    sub_sampling_factor = RL_params_env["sub_sampling_factor"]
    target_kl = RL_params_env["target_kl"]

    return policy, batch_size, cg_damping, cg_max_steps, gae_lambda, gamma, learning_rate, n_critic_updates, n_steps, normalize_advantage, sub_sampling_factor, target_kl    

"""        "HalfCheetah": {
            "policy": "MlpPolicy",
            "learning_starts": 10000,
            "use_sde":"False",
            "normalize":"False"

        }"""

def get_SAC_RL_params(RL_params, RL_alg, env_name):
    print("RL_alg", RL_alg)
    print("env_name", env_name[:-3])
    RL_params_alg = RL_params[RL_alg]
    RL_params_env = RL_params_alg[env_name[:-3]]

    policy = RL_params_env["policy"]
    learning_starts = RL_params_env["learning_starts"]
    gamma = RL_params_env["gamma"]

    use_sde = RL_params_env["use_sde"]
    use_sde = True_or_False(use_sde)

    normalize = RL_params_env["normalize"]
    normalize = True_or_False(normalize)


    return policy, gamma, learning_starts, use_sde, normalize


def get_TD3_RL_params(RL_params, RL_alg, env_name):
    RL_params_alg = RL_params[RL_alg]
    RL_params_env = RL_params_alg[env_name[:-3]]
    policy = RL_params_env["policy"]
    batch_size = RL_params_env["batch_size"]
    gradient_steps = RL_params_env["gradient_steps"]
    learning_rate = RL_params_env["learning_rate"]
    learning_starts = RL_params_env["learning_starts"]
    train_freq = RL_params_env["train_freq"]
    normalize = RL_params_env["normalize"]

    return policy, batch_size, gradient_steps, learning_rate, learning_starts, train_freq, normalize


def get_PPO_RL_params(RL_params, RL_alg, env_name):
    RL_params_alg = RL_params[RL_alg]
    RL_params_env = RL_params_alg[env_name[:-3]]
    policy = RL_params_env["policy"]
    batch_size = RL_params_env["batch_size"]
    clip_range = RL_params_env["clip_range"]
    ent_coef = RL_params_env["ent_coef"]
    gae_lambda = RL_params_env["gae_lambda"]
    gamma = RL_params_env["gamma"]
    learning_rate = RL_params_env["learning_rate"]
    max_grad_norm = RL_params_env["max_grad_norm"]
    n_epochs = RL_params_env["n_epochs"]
    n_steps = RL_params_env["n_steps"]
    # normalize = RL_params_env["normalize"]
    vf_coef = RL_params_env["vf_coef"]

    return policy, batch_size, clip_range, ent_coef, gae_lambda, gamma, learning_rate, max_grad_norm, n_epochs, n_steps, vf_coef


# Atari ---------------------------------------------------
# def get_PPO_RL_params(RL_params, RL_alg, env_name):
#     RL_params_alg = RL_params[RL_alg]
#     RL_params_env = RL_params_alg[env_name[:-3]]
#     policy = RL_params_env["policy"]
#     batch_size = RL_params_env["batch_size"]
#     clip_range = RL_params_env["clip_range"]
#     ent_coef = RL_params_env["ent_coef"]
#     frame_stack = RL_params_env["frame_stack"]
#     learning_rate = RL_params_env["learning_rate"]
#     n_epochs = RL_params_env["n_epochs"]
#     n_steps = RL_params_env["n_steps"]
#     normalize = RL_params_env["normalize"]
#     vf_coef = RL_params_env["vf_coef"]

#     return policy, batch_size, clip_range, ent_coef, frame_stack, learning_rate, n_epochs, n_steps, normalize, vf_coef





def get_DQN_RL_params(RL_params, RL_alg, env_name):
    RL_params_alg = RL_params[RL_alg]
    RL_params_env = RL_params_alg[env_name[:-3]]
    batch_size = RL_params_env["batch_size"]
    buffer_size = RL_params_env["buffer_size"]
    exploration_final_eps = RL_params_env["exploration_final_eps"]
    exploration_fraction = RL_params_env["exploration_fraction"]
    frame_stack = RL_params_env["frame_stack"]
    gradient_steps = RL_params_env["gradient_steps"]
    learning_rate = RL_params_env["learning_rate"]
    learning_starts = RL_params_env["learning_starts"]
    optimize_memory_usage = RL_params_env["optimize_memory_usage"]
    optimize_memory_usage = True_or_False(optimize_memory_usage)
    policy = RL_params_env["policy"]
    target_update_interval = RL_params_env["target_update_interval"]
    train_freq = RL_params_env["train_freq"]

    return batch_size, buffer_size, exploration_final_eps, exploration_fraction, frame_stack, gradient_steps, learning_rate, learning_starts, optimize_memory_usage, policy, target_update_interval, train_freq





# Create the HalfCheetah environment --------------------------------------
class Custom_Env_multiCl(gym.Wrapper):
    def __init__(self, env_name, scoring_model_0, scoring_model_1, scoring_model_2, modify_reward=True, render_mode=None):
        # Create the original HalfCheetah environment
        env = gym.make(env_name, render_mode=render_mode)
        super().__init__(env)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scoring_model_0 = scoring_model_0.net
        self.scoring_model_0.eval()
        self.scoring_model_0.to(self.device)
        self.input_scaler_s_0 = scoring_model_0.input_scaler_s
        self.input_scaler_a_0 = scoring_model_0.input_scaler_a


        self.scoring_model_1 = scoring_model_1.net
        self.scoring_model_1.eval()
        self.scoring_model_1.to(self.device)
        self.input_scaler_s_1 = scoring_model_1.input_scaler_s
        self.input_scaler_a_1 = scoring_model_1.input_scaler_a


        self.scoring_model_2 = scoring_model_2.net
        self.scoring_model_2.eval()
        self.scoring_model_2.to(self.device)
        self.input_scaler_s_2 = scoring_model_2.input_scaler_s
        self.input_scaler_a_2 = scoring_model_2.input_scaler_a

        # scoring_model.net.prior_test = 0.5 # set the prior (ratio of pos samples in testing) to 0.5
        self.old_obs = None
        self.modify_reward = modify_reward

        self.traj_s = []
        self.traj_a = []
        self.traj_true_r = []
        self.traj_pred_r = []
        
    def set_modify_reward(self, value):
        self.modify_reward = value

    def step(self, action):
        # print(self.modify_reward)
        # Execute the action in the environment
        # print(self.env.step(action))

        obs, reward, terminated, truncated, info = self.env.step(action)
        # print("obs", obs.shape)
        # print("reward", reward)
        # print("action", action.shape)
        done = terminated or truncated
        # if done:
        #     print("Done ==================")
        
        # Modify the reward calculation here (customize as needed)
        if self.old_obs is not None:
            modified_reward, reward_pred_orig = self._custom_reward(self.old_obs, action, reward)

            # if reward_pred_orig < 0.9:
            #     print("reward_pred (pred as bad)", reward_pred_orig)
            #     print("reward (True)", reward)
            #     print("")
            # if reward_pred_orig > 0.5:
            #     print("reward_pred (pred as  good)", reward_pred_orig)
            #     print("reward (True)", reward)
            #     print("")

        else:
            modified_reward = 0 # the first step has no reward
        self.old_obs = obs
        # modified_reward = 0

        self.traj_s.append(self.old_obs)
        self.traj_a.append(action)
        self.traj_true_r.append(reward)
        self.traj_pred_r.append(modified_reward)

        if done:
            self.old_obs = None
        if self.modify_reward:
            return obs, modified_reward, terminated, truncated, info
        else:
            # reward = 0
            # print("pred reward", modified_reward)
            return obs, reward, terminated, truncated, info

    def _custom_reward(self, obs, action, reward):
        obs_cp = cp(obs)
        action_cp = cp(action)


        obs = self.input_scaler_s_0.transform(obs_cp.reshape(1, -1))
        action = self.input_scaler_a_0.transform(action_cp.reshape(1, -1))
        obs = torch.from_numpy(obs[0]).float().unsqueeze(0).to(self.device)
        action = torch.from_numpy(action[0]).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            opt_or_not_0, class_i = self.scoring_model_0.forward(obs, action)


        obs = self.input_scaler_s_1.transform(obs_cp.reshape(1, -1))
        action = self.input_scaler_a_1.transform(action_cp.reshape(1, -1))
        obs = torch.from_numpy(obs[0]).float().unsqueeze(0).to(self.device)
        action = torch.from_numpy(action[0]).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            opt_or_not_1, class_i = self.scoring_model_1.forward(obs, action)


        obs = self.input_scaler_s_2.transform(obs_cp.reshape(1, -1))
        action = self.input_scaler_a_2.transform(action_cp.reshape(1, -1))
        obs = torch.from_numpy(obs[0]).float().unsqueeze(0).to(self.device)
        action = torch.from_numpy(action[0]).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            opt_or_not_2, class_i = self.scoring_model_2.forward(obs, action)
        

        # opt_or_not = (opt_or_not_0 + opt_or_not_1 + opt_or_not_2) / 3.0
            
        # Vote
        opt_count = 0
        # print("opt_or_not_0", np.array(opt_or_not_0.cpu()[0][0]))
        # print("opt_or_not_1", opt_or_not_1)
        # print("opt_or_not_2", opt_or_not_2)
        if np.array(opt_or_not_0.cpu()[0][0]) > 0.5:
            opt_count += 1
        if np.array(opt_or_not_1.cpu()[0][0]) > 0.5:
            opt_count += 1
        if np.array(opt_or_not_2.cpu()[0][0]) > 0.5:
            opt_count += 1

        if opt_count >= 2:
            opt_or_not = 1
        else:
            opt_or_not = 0


        # print("opt_or_not", opt_or_not)
        # print("class_i", class_i)
            
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
            # reward_pred_orig = np.array(opt_or_not.cpu()[0][0])
            reward_pred_orig = opt_or_not
            reward_pred = reward_pred_orig
            reward_pred = np.max((1e-8, reward_pred_orig))
            reward_pred = np.min((1 - 1e-8, reward_pred))
        
        # Custom reward calculation based on the observation and original reward
        # Modify this function according to your specific reward logic
        # For example, you can use a different reward function based on the observation
        # modified_reward = reward * 0  # Modify the reward as an example
        
        return reward_pred, reward_pred_orig


# Create the HalfCheetah environment --------------------------------------
class Custom_Env_multiCl_multiFrame(gym.Wrapper):
    def __init__(self, env_name, scoring_model_0, scoring_model_1, scoring_model_2, modify_reward=True, render_mode=None):
        # Create the original HalfCheetah environment
        env = gym.make(env_name, render_mode=render_mode)
        super().__init__(env)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scoring_model_0 = scoring_model_0.net
        self.scoring_model_0.eval()
        self.scoring_model_0.to(self.device)
        self.input_scaler_s_0 = scoring_model_0.input_scaler_s
        self.input_scaler_a_0 = scoring_model_0.input_scaler_a


        self.scoring_model_1 = scoring_model_1.net
        self.scoring_model_1.eval()
        self.scoring_model_1.to(self.device)
        self.input_scaler_s_1 = scoring_model_1.input_scaler_s
        self.input_scaler_a_1 = scoring_model_1.input_scaler_a


        self.scoring_model_2 = scoring_model_2.net
        self.scoring_model_2.eval()
        self.scoring_model_2.to(self.device)
        self.input_scaler_s_2 = scoring_model_2.input_scaler_s
        self.input_scaler_a_2 = scoring_model_2.input_scaler_a



        # scoring_model.net.prior_test = 0.5 # set the prior (ratio of pos samples in testing) to 0.5
        self.frame_num = 4
        self.old_obs = None
        self.obs_frames = []
        self.action_frames = []
        self.modify_reward = modify_reward

        self.traj_s = []
        self.traj_a = []
        self.traj_true_r = []
        self.traj_pred_r = []
        self.traj_opt_prob = []

        self.traj_true_acc_r = 0
        self.traj_pred_acc_r = [0,0,0]
        
    def set_modify_reward(self, value):
        self.modify_reward = value
    
    # Function to update the list with new data
    def update_list(self, my_list, new_data):
        # Remove the first element (index 0)
        my_list.pop(0)
        # Append the new data to the end of the list
        my_list.append(new_data)
        return my_list

    def step(self, action):
        # print(self.modify_reward)
        # Execute the action in the environment
        # print(self.env.step(action))

        obs, reward, terminated, truncated, info = self.env.step(action)
        # print("obs", obs.shape)
        # print("reward", reward)
        # print("action", action.shape)
        done = terminated or truncated
        # if done:
        #     print("Done ==================")
        
        if self.old_obs is None:
            modified_reward = 0 
            classifiers_pred = [0,0,0]
            self.traj_true_acc_r = 0
            self.traj_pred_acc_r = [0,0,0]
        else:
            # Modify the reward calculation here (customize as needed)
            if len(self.obs_frames) == self.frame_num:
                self.obs_frames = self.update_list(self.obs_frames, self.old_obs)
                self.action_frames = self.update_list(self.action_frames, action)

                modified_reward, reward_pred_orig, classifiers_pred = self._custom_reward(self.obs_frames, self.action_frames, reward)



            else:
                modified_reward = 0 # the four step has no reward
                classifiers_pred = [0,0,0]
                self.obs_frames.append(self.old_obs)
                self.action_frames.append(action)
                
        self.old_obs = obs
        # modified_reward = 0

        self.traj_s.append(self.old_obs)
        self.traj_a.append(action)
        self.traj_true_r.append(reward)
        self.traj_pred_r.append(modified_reward)
        self.traj_opt_prob.append(classifiers_pred)

        self.traj_true_acc_r += reward
        self.traj_pred_acc_r[0] += classifiers_pred[0]
        self.traj_pred_acc_r[1] += classifiers_pred[1]
        self.traj_pred_acc_r[2] += classifiers_pred[2]

        if done:
            self.old_obs = None
            
        if self.modify_reward:
            return obs, modified_reward, terminated, truncated, info
        else:
            # reward = 0
            return obs, reward, terminated, truncated, info

    def _custom_reward(self, obs, action, reward):
        
        obs_cp = cp(np.array(obs))
        action_cp = cp(np.array(action))
        # print("action_cp", actio
        def _get_reward(obs_cp, action_cp, scoring_model):
            # obs = self.input_scaler_s_0.transform(obs_cp.reshape(1, -1))
            # action = self.input_scaler_a_0.transform(action_cp.reshape(1, -1))
            obs = self.input_scaler_s_0.transform(obs_cp)
            action = self.input_scaler_a_0.transform(action_cp)
            # print("obs after transform ", obs.shape)
            obs = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
            # print("obs after torch ", obs.shape)
            action = torch.from_numpy(action).float().unsqueeze(0).to(self.device)
            # print("obs ", obs.shape)
            with torch.no_grad():
                opt_or_not, class_i = scoring_model.forward(obs, action)
            return opt_or_not, class_i
        
        opt_or_not_0, _ = _get_reward(obs_cp, action_cp, self.scoring_model_0)
        opt_or_not_1, _ = _get_reward(obs_cp, action_cp, self.scoring_model_1)
        opt_or_not_2, _ = _get_reward(obs_cp, action_cp, self.scoring_model_2)


        # obs = self.input_scaler_s_0.transform(obs_cp)
        # action = self.input_scaler_a_0.transform(action_cp)
        # obs = torch.from_numpy(obs[0]).float().unsqueeze(0).to(self.device)
        # action = torch.from_numpy(action[0]).float().unsqueeze(0).to(self.device)
        # with torch.no_grad():
        #     opt_or_not_1, class_i = self.scoring_model_1.forward(obs, action)


        # obs = self.input_scaler_s_0.transform(obs_cp)
        # action = self.input_scaler_a_0.transform(action_cp)
        # obs = torch.from_numpy(obs[0]).float().unsqueeze(0).to(self.device)
        # action = torch.from_numpy(action[0]).float().unsqueeze(0).to(self.device)
        # with torch.no_grad():
        #     opt_or_not_2, class_i = self.scoring_model_2.forward(obs, action)
        

        # opt_or_not = (opt_or_not_0 + opt_or_not_1 + opt_or_not_2) / 3.0
            
        # Vote
        opt_count = 0
        # print("opt_or_not_0", np.array(opt_or_not_0.cpu()[0][0]))
        # print("opt_or_not_1", opt_or_not_1)
        # print("opt_or_not_2", opt_or_not_2)
        if np.array(opt_or_not_0.cpu()[0][0]) > 0.5:
            opt_count += 1
        if np.array(opt_or_not_1.cpu()[0][0]) > 0.5:
            opt_count += 1
        if np.array(opt_or_not_2.cpu()[0][0]) > 0.5:
            opt_count += 1

        if opt_count >= 2:
            opt_or_not = 1
        else:
            opt_or_not = 0


        # print("opt_or_not", opt_or_not)
        # print("class_i", class_i)
            
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
            # reward_pred_orig = np.array(opt_or_not.cpu()[0][0])
            reward_pred_orig = opt_or_not
            reward_pred = reward_pred_orig
            reward_pred = np.max((1e-8, reward_pred_orig))
            reward_pred = np.min((1 - 1e-8, reward_pred))
        
        # Custom reward calculation based on the observation and original reward
        # Modify this function according to your specific reward logic
        # For example, you can use a different reward function based on the observation
        # modified_reward = reward * 0  # Modify the reward as an example
        
        return reward_pred, reward_pred_orig, [opt_or_not_0.cpu()[0][0], opt_or_not_1.cpu()[0][0], opt_or_not_2.cpu()[0][0]]