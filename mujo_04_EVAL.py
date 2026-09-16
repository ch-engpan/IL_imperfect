import gymnasium as gym
from gymnasium.wrappers.monitoring.video_recorder import VideoRecorder
# import gym
from sb3_contrib import TRPO, TQC
from stable_baselines3 import PPO, TD3, SAC, DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy

import os 
import re
import numpy as np
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import argparse
from utils_rl import eval_model, eval_model_multiEnvs
import json
from utils_plot import plot_reward_stepLen
import torch

"../results/alphaEst/logs_scoring_sin-train_sort_uu_loss_noReplacing/Hopper-v4_scoring_co_train_SAC_NetAgentSeed_3-3_epochs_100_RLseed_0__alpha_0613-un_gail_finetune_500_itrs_scoring_sort_05_stepnum0e5_both_uu_loss_test_uuloss_all/evaluations.npz'"
"  /results/alphaEst/logs_scoring_sin-train_sort_uu_loss_noReplacing/Hopper-v4_scoring_co_train_SAC_NetAgentSeed_3-3_epochs_100_RLseed_0__alpha_0613-un_gail_finetune_500_itrs_scoring_sort_05_stepnum0e5_both_uu_loss_test_uuloss_all"

def eval_best_checkpoints_models(env, env_name, RL_alg, seed_value, model_dir, file_suffix_name, n_eval_episodes=10, eval_best=True, reverse=False, save_trajs=True, 
                                 savename_reward=False, savename_steps=False, checkpoints_file = "checkpoints", save_env_init=True, plot_reward=True, multienv=False):

    np.random.seed(seed_value)
    print("env_name: ", env_name)

    env_init01 = env.reset()
    env_init02 = env.reset()
    
    if save_env_init:
        # Define the file name
        file_name = Path(model_dir +  'init_env_check_'+file_suffix_name+'.json')
        # Save the list to a JSON file
        with open(file_name, 'w') as file:
            json.dump([env_init01.tolist(), env_init02.tolist()], file)  
        

    def eval_save_model(model, env, n_eval_episodes, saving_path, save_trajs=True, savename_reward=False, savename_steps=False):
        if multienv:
            mean_reward, std_reward, traj_s_list, traj_a_list, traj_r_list, acc_reward_list, steps_list = eval_model_multiEnvs(model, env, 
                                                                                                                n_eval_episodes=n_eval_episodes,
                                                                                                                )
        else:
            mean_reward, std_reward, traj_s_list, traj_a_list, traj_r_list, acc_reward_list, steps_list = eval_model(model, env, 
                                                                                                                    n_eval_episodes=n_eval_episodes,
                                                                                                                    )
        data_to_save = {
            # 'env': env,
            # 'model': model.device('cpu'),
            'mean_reward': mean_reward,
            'std_reward': std_reward,
            'traj_s_list': traj_s_list,
            'traj_a_list': traj_a_list,
            'traj_r_list': traj_r_list,
            'acc_reward_list': acc_reward_list,
            'steps_list': steps_list
        }
        if save_trajs:
            if savename_reward:
                saving_path = saving_path.split(".pkl")[0] + "_meanReward_" + str(np.round(mean_reward, 2)) + "_.pkl"
            if savename_steps:
                saving_path = saving_path.split(".pkl")[0] + "_steps_" + str(int(np.sum(steps_list)/1000)) + "k_.pkl"

            with open(Path(saving_path), 'wb') as f:
                pickle.dump(data_to_save, f)
        print("acc_reward_list", acc_reward_list)
        print("mean_reward", mean_reward)
        print("std_reward", std_reward)
        print("steps_list", steps_list)
        print("steps_mean", np.mean(steps_list))
        print("steps_std", np.std(steps_list))
        print(" ")
        print("traj_s_list shape", len(traj_s_list), "traj_a_list shape", len(traj_a_list), "traj_r_list shape", len(traj_r_list))
        print("1st traj_s_list shape", np.array(traj_s_list[0]).shape, "1st traj_a_list shape", np.array(traj_a_list[0]).shape, "1st traj_r_list shape", np.array(traj_r_list[0]).shape)

        print("acc_reward_list shape", np.array(acc_reward_list).shape)
        print("mean_reward shape", np.array(mean_reward).shape)
        print("std_reward shape", np.array(std_reward).shape)
        print("steps_list shape", np.array(steps_list).shape)
        print("steps_mean shape", np.array(np.mean(steps_list)).shape)
        print("steps_std shape", np.array(np.std(steps_list)).shape)
        print(" ")

        """
        acc_reward_list [-1485.0628036260605, -1417.9066557884216, -1356.6000484228134, -1620.2361907958984, -1263.9853855371475, -1378.7834817171097, -1485.8406628370285, -1394.879420042038, -1166.3133057951927, -1459.5128991603851]
        mean_reward -1402.9120853722095
        std_reward 119.31272939174096
        steps_list [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]
        steps_mean 1000.0
        steps_std 0.0
        
        traj_s_list shape 10 
        traj_a_list shape 10 
        traj_r_list shape 10
        1st traj_s_list shape (1000, 27) 
        1st traj_a_list shape (1000, 8) 
        1st traj_r_list shape (1000,)

        acc_reward_list shape (10,)
        mean_reward shape ()
        std_reward shape ()
        steps_list shape (10,)
        steps_mean shape ()
        steps_std shape ()

        """
        return mean_reward, std_reward, np.mean(steps_list), np.std(steps_list)

    save_dir = model_dir

    os.makedirs(Path(save_dir+"trajs_"+file_suffix_name+"/"), exist_ok=True)



    def load_rl_model(model_path, RL_alg):
        print("=============================", "cuda" if torch.cuda.is_available() else "cpu", "=============================")
        if RL_alg == "TRPO":
            loaded_model = TRPO.load(model_path, device="cuda" if torch.cuda.is_available() else "cpu")
            loaded_model.set_parameters(model_path)
        elif RL_alg == "PPO":
            loaded_model = PPO.load(model_path, device="cuda" if torch.cuda.is_available() else "cpu")
            loaded_model.set_parameters(model_path)
        elif RL_alg == "TD3":
            loaded_model = TD3.load(model_path, device="cuda" if torch.cuda.is_available() else "cpu")
            loaded_model.set_parameters(model_path)
        elif RL_alg == "SAC":
            loaded_model = SAC.load(model_path, device="cuda" if torch.cuda.is_available() else "cpu")
            loaded_model.set_parameters(model_path)
        elif RL_alg == "TQC":
            loaded_model = TQC.load(model_path, device="cuda" if torch.cuda.is_available() else "cpu")
            loaded_model.set_parameters(model_path)
        elif RL_alg == "DQN":
            loaded_model = DQN.load(model_path, device="cuda" if torch.cuda.is_available() else "cpu")
            loaded_model.set_parameters(model_path)
        return loaded_model
    if eval_best:
        # EVAL best model --------------------------------------
        best_mode_path  = Path(save_dir + "best_model.zip")
        print(best_mode_path)
        loaded_model = load_rl_model(best_mode_path, RL_alg)
        r_mean_best, r_std_best, stepLen_mean_best, stepLenm_std_best = eval_save_model(loaded_model, env, n_eval_episodes=n_eval_episodes, saving_path=save_dir+"trajs_"+file_suffix_name+"/"+'best_model_trajs.pkl',
                                                                                        save_trajs=save_trajs, savename_reward=savename_reward, savename_steps=savename_steps)

    mean_reward_list = []
    std_reward_list = []
    mean_stepLen_list = []
    std_stepLen_list = []
    stepnum_list = []

    # EVAL checkpoints model  --------------------------------------
    checkpoint_num_list = []
    modelPaths = [] 
    for traj_file in os.listdir(Path(save_dir + checkpoints_file + "/")):
            if traj_file.endswith('.zip'):
                checkpoint = traj_file.split('_')[-2]
                checkpoint_num = int(re.findall('\d+', checkpoint)[0])
                checkpoint_num_list.append(checkpoint_num)
                # print("iter:", checkpoint_num)
                modelPaths.append(save_dir + "checkpoints/" + traj_file)
    checkpoint_num_keys = sorted(range(len(checkpoint_num_list)), key=lambda x: checkpoint_num_list[x], reverse=reverse)


    for i in (checkpoint_num_keys):
        # print(i)
        # traj_file = env_name + "_" + RL_alg + "_models_" + str(checkpoint_num_list[i]) + "_steps.zip"
        traj_file = modelPaths[i]
        model_path = Path(traj_file)
        print(model_path)
        # loaded_model = TRPO.load(model_path)
        loaded_model = load_rl_model(model_path, RL_alg)
        r_mean, r_std, stepLen_mean, stepLen_std = eval_save_model(loaded_model, env, n_eval_episodes=n_eval_episodes, saving_path=save_dir+"trajs_"+file_suffix_name+"/"+"model_checkpoints_" + str(checkpoint_num_list[i]) + "_steps_trajs.pkl",
                                                                   save_trajs=save_trajs, savename_reward=savename_reward, savename_steps=savename_steps)
        mean_reward_list.append(r_mean)
        std_reward_list.append(r_std)
        stepnum_list.append(checkpoint_num_list[i])
        mean_stepLen_list.append(stepLen_mean)
        std_stepLen_list.append(stepLen_std)

    if eval_best:
        # add the best model results
        mean_reward_list.append(r_mean_best)
        std_reward_list.append(r_std_best)
        stepnum_list.append(stepnum_list[-1] + 1*stepnum_list[0])
        mean_stepLen_list.append(stepLen_mean_best)
        std_stepLen_list.append(stepLenm_std_best)

    print("mean_reward_list: ", mean_reward_list)
    print("std_reward_list: ", std_reward_list)
    print("")
    print("mean_stepLen_list: ", mean_stepLen_list)
    print("std_stepLen_list: ", std_stepLen_list)
    print("")
    print("stepnum_list: ", stepnum_list)
    if plot_reward:
        plot_reward_stepLen(stepnum_list, mean_reward_list, std_reward_list, mean_stepLen_list, std_stepLen_list, env_name, save_dir, file_suffix_name)
    # save
    data_to_save = {
            'stepnum_list': stepnum_list,
            'mean_reward_list': mean_reward_list,
            'std_reward_list' : std_reward_list,
            'mean_stepLen_list': mean_stepLen_list,
            'std_stepLen_list' : std_stepLen_list
        }
    # save data
    saving_path = save_dir + "eval_data_reward_step_" + file_suffix_name + ".pkl"
    with open(Path(saving_path), 'wb') as f:
            pickle.dump(data_to_save, f)


if __name__ == '__main__':
    # python mujo_04_EVAL.py --env_idx 2 --test_idx 10  --rl_alg SAC  --seed 0 
    # python mujo_04_EVAL.py --env_idx 2 --test_idx 10  --rl_alg TQC  --seed 0 
    env_name_list = ["Ant-v4", "HalfCheetah-v4", "Hopper-v4", "Swimmer-v4", "Walker2d-v4"]
    
    folder_dir = "../results/logs_scoring/"
  
    file_suffix_name  = "extra_eval" # for names of file to be saved 

    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--env_idx', type=int, default='10', help='')
    parser.add_argument('--test_idx', type=str, default='10', help='')
    parser.add_argument('--seed', type=int, default='10', help='')
    parser.add_argument('--rl_alg', type=str, default='TRPO', help='TRPO or TD3 or PPO or TQC or SAC ')
    args = parser.parse_args()
    seed_value = args.seed

    RL_alg = args.rl_alg 

    if args.test_idx == "None":
        test_idx = ""
    else:
        test_idx = args.test_idx
    env_name = env_name_list[args.env_idx]
    model_dir = folder_dir + env_name + "_" + RL_alg + "_model"+"-un_using_"+str(test_idx)+"_RLseed_"+str(seed_value)+"/"

    # Create the HalfCheetah environment
    env = gym.make(env_name)

    # Wrap the environment
    env = DummyVecEnv([lambda: env])
    # env = VecNormalize(env, 
    #                 norm_obs=True,
    #                 norm_reward = False,
    #                 clip_obs = 10)
    env.seed(seed_value)

    eval_best_checkpoints_models(env, env_name, RL_alg, seed_value, model_dir, file_suffix_name)