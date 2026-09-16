import gymnasium as gym
import pickle
# import gym
from sb3_contrib import TRPO, TQC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO, TD3, SAC, DQN
import os
# from utils_scoring_uu_sort_paral import RL_Scoring

from utils_gail import GAIL, Demonstration_Buffer, Demonstration_Buffer_uu, Demonstration_Buffer_gail, GAIL_Discrim, Custom_Env
from mujo_04_EVAL import eval_best_checkpoints_models
from mujo_04_EVAL_results_npz import plot_results_npz
import argparse
import json
from pathlib import Path
# from utils_gail import GAIL_Discrim
from um_ssc_grid_mujuco import Scoring_model_net_multiFrame
import numpy as np
import torch
from mujo_10_EVAL_discriminator import plot_discriminator_acc_loss, plot_selflabel_num, plot_uuLearn_num
import multiprocessing
from utils_ileed import run_bc, run_mle
from utils_plot import plot_reward_stepLen, plot_evals_loglik_offlineMethods
import matplotlib.pyplot as plt
"""

Normal:


" python3 mujo_10_ileed_offline.py  --Expert_idx 0  --total_steps 3e6  --env 1  --noisy True --data_augm both   --rl_seed 2  --Pi_range_select 0 ",


"""


def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=============================", device, "=============================")
    print("=============================", device, "=============================")

    def true_false_convert(str):
        if str == "true" or str == "True":
            return True
        elif str == "false" or str == "False":
            return False
        else:
            raise ValueError("true or false")



    # -----------------------------------------------
    # opt_ratio_alpha is randomly generated from np.random using seed, and already labeled in the dataset, here just to correspond to the opt_ratio_alpha
    if args.Pi_range_select == 0:
        args.opt_ratio_alpha = "0.5"

    elif args.Pi_range_select == 1:
        args.opt_ratio_alpha = "0.1"
    
    elif args.Pi_range_select == 2:
        args.opt_ratio_alpha = "0.5"
    
    elif args.Pi_range_select == 4:
        args.opt_ratio_alpha = "0.15"

    clean_data = not true_false_convert(args.noisy)
    print("clean_data:", clean_data)



    env_name_list = ["Ant-v4", "HalfCheetah-v4", "Hopper-v4", "Swimmer-v4", "Walker2d-v4"]
    env_name = env_name_list[args.env]
    scoring_update_itrs = 500


    random_Pi_range_list = [(0.1, 0.9), 
                        (0.05, 0.15), (0.45, 0.55), (0.85, 0.95),  # 1 2 3
                        (0.05, 0.25), (0.4, 0.6),  (0.75, 0.95),   # 4 5 6

                        (0.05, 0.4), (0.05, 0.5), 
                        
                        (0.4, 0.6), (0.3, 0.7), 
                        
                        (0.8, 0.95), (0.7, 0.95), (0.6, 0.95), (0.5, 0.95),
                          
                        ]

    random_Pi_range = random_Pi_range_list[args.Pi_range_select]

    # root_folder = "../results/alphaEst/"
    root_folder = "../results/alphaEst_randomPi/"
    # random_Pi_range used for save_folder
    root_folder = root_folder[:-1] + "_PiRange"+str(random_Pi_range[0])[0]+ str(random_Pi_range[0])[2:] +"-"+str(random_Pi_range[1])[0]+ str(random_Pi_range[1])[2:] +"/"


    opt_ratio_alpha = args.opt_ratio_alpha


    seed_value = args.rl_seed
    expert_idxs = args.Expert_idx
    print("expert_idxs", expert_idxs)
    RL_alg = "SAC"

    # clean_data = True

    if clean_data:
        models_dir = root_folder + "/logs_ileed_clean/"
    else:
        models_dir = root_folder + "/logs_ileed_noisy/"

    os.makedirs(models_dir, exist_ok=True)

    suffix = "_stepnum" + str(int(args.total_steps/1e5)) + "e5_"  + args.data_augm


    if not clean_data:
        save_dir = models_dir+env_name+"_ILEED_"+RL_alg+"_NetAgentSeed_" + str(expert_idxs[0]) + "-" + str(expert_idxs[-1])+"_RLseed_"+str(seed_value)+"_noisy-un"+suffix+"/"

    else:
        save_dir = models_dir+env_name+"_ILEED_"+RL_alg+"_NetAgentSeed_" + str(expert_idxs[0]) + "-" + str(expert_idxs[-1])+"_RLseed_"+str(seed_value)+"-un"+suffix+"/"

    os.makedirs(save_dir, exist_ok=True)

    # suffix += "_noLossmix"
    # suffix += "_sigFine"
    # suffix += "_halfAgtFine"
    # suffix += "_optThrAgtFine05"
    # suffix += "_fixScor"
    


    uu_data_path = root_folder + "/noisy_data/"
    uu_data_path = uu_data_path + env_name+"_ExpertSeed_"+str(expert_idxs[0])+"_uu_data"+"_multi_frame_1_optStart200_alpha_"+opt_ratio_alpha[0]+opt_ratio_alpha[2:]+  "_newneg.npz"

    # load code
    print("Loading data from:", uu_data_path)
    loaded_data = np.load(uu_data_path, allow_pickle=True)
    keys = loaded_data.keys()
    
    print("priors_class:", loaded_data["priors_class"])
    print("Pi_s_train:", loaded_data["Pi_s_train"])
    print("priors_class_test:", loaded_data["priors_class_test"])
    print("Pi_test:", loaded_data["Pi_test"])
    print("input_dim_s:", loaded_data["input_dim_s"])
    print("input_dim_a:", loaded_data["input_dim_a"])
    print("input_scaler_s min :", loaded_data["input_scaler_s"].item().data_min_)
    print("input_scaler_s max :", loaded_data["input_scaler_s"].item().data_max_)
    print("input_scaler_a min :", loaded_data["input_scaler_a"].item().data_min_)
    print("input_scaler_a max :", loaded_data["input_scaler_a"].item().data_max_)
    print("frame_num:", loaded_data["frame_num"])
    print("")

    print("U_set_s_train shape:", loaded_data["U_set_s_train"].shape)
    print("U_set_a_train shape:", loaded_data["U_set_a_train"].shape)
    print("U_set_classLabels_train shape:", loaded_data["U_set_classLabels_train"].shape)
    print("U_sets_binLabels_train shape:", loaded_data["U_sets_binLabels_train"].shape)
    print("U_set_s_test shape:", loaded_data["U_set_s_test"].shape)
    print("U_set_a_test shape:", loaded_data["U_set_a_test"].shape)
    print("U_set_classLabels_test shape:", loaded_data["U_set_classLabels_test"].shape)
    print("U_sets_binLabels_test shape:", loaded_data["U_sets_binLabels_test"].shape)

    print("")
 
    print("seed:", loaded_data["seed"])
    print("env_idx:", loaded_data["env_idx"])
    print("env_name:", loaded_data["env_name"])
    print("opt_ratio_alpha:", loaded_data["opt_ratio_alpha"])
    print("opt_start_steps:", loaded_data["opt_start_steps"])


    binLabels=loaded_data["U_sets_binLabels_train"]
    true_labels = torch.from_numpy(binLabels[:, 0]).long()
    traj_s_noisy=loaded_data["U_set_s_train"]
    traj_a_noisy=loaded_data["U_set_a_train"]

    traj_s_opt = traj_s_noisy[true_labels == 1]
    traj_a_opt = traj_a_noisy[true_labels == 1]
    print("traj_s_opt: ", traj_s_opt.shape, " traj_a_opt: ", traj_a_opt.shape)
    print("traj_s_noisy: ", traj_s_noisy.shape, " traj_a_noisy: ", traj_a_noisy.shape)


    # U_set_s_train_opt = loaded_data["U_set_s_train"][loaded_data["U_sets_binLabels_train"][:, 0]>0.5]
    # U_set_a_train_opt = loaded_data["U_set_a_train"][loaded_data["U_sets_binLabels_train"][:, 0]>0.5]
    # U_set_s_train_nonopt = loaded_data["U_set_s_train"][loaded_data["U_sets_binLabels_train"][:, 0]<0.5]
    # U_set_a_train_nonopt = loaded_data["U_set_a_train"][loaded_data["U_sets_binLabels_train"][:, 0]<0.5]
    
    # print("U_set_s_train_opt: ", U_set_s_train_opt.shape, " U_set_a_train_opt: ", U_set_a_train_opt.shape)
    # print("U_set_s_train_opt samples: ", U_set_s_train_opt[559:561])
    # print("traj_s_opt samples: ", traj_s_opt[559:561])
    # print("U_set_s_train_nonopt: ", U_set_s_train_nonopt.shape, " U_set_a_train_nonopt: ", U_set_a_train_nonopt.shape)
    # print("")
    # print(np.array_equal(traj_s_opt, U_set_s_train_opt))
    # print(np.array_equal(traj_a_opt, U_set_a_train_opt))
    
    
    
    traj_s_noisy = torch.tensor(traj_s_noisy[:,0,:], dtype=torch.float32).to(device)
    traj_a_noisy = torch.tensor(traj_a_noisy[:,0,:], dtype=torch.float32).to(device)
   


    exp_num = loaded_data["U_set_classLabels_train"].shape[-1]

    demo_diff_exp_s = []
    demo_diff_exp_a = []
    true_labels_diff_exp = []
    for m in range(exp_num):
        idxs = np.where(loaded_data["U_set_classLabels_train"][:, m] == 1)[0]
        # print("Expert demo", m, " idxs:", idxs)
        demo_diff_exp_s.append( traj_s_noisy[idxs].cpu() )
        demo_diff_exp_a.append( traj_a_noisy[idxs].cpu() )
        true_labels_diff_exp.append( true_labels[idxs].cpu() )
    
    if clean_data:
        # use optimal data only
        for m in range(exp_num):
            print("Demo expert ", m, " num:", demo_diff_exp_s[m].shape[0])
            demo_diff_exp_s[m] = demo_diff_exp_s[m][true_labels_diff_exp[m]==1][:]
            demo_diff_exp_a[m] = demo_diff_exp_a[m][true_labels_diff_exp[m]==1][:]
            print("After clean, Demo expert ", m, " num:", demo_diff_exp_s[m].shape[0])


        # Find the maximum number of demos across experts
        max_count = max([demo_diff_exp_s[m].shape[0] for m in range(exp_num)])
        max_count = int(traj_a_noisy.shape[0] /exp_num )
        print("Target unified demo count:", max_count)

        for m in range(exp_num):
            cur_n = demo_diff_exp_s[m].shape[0]
            print(f"Expert {m}: {cur_n} → {max_count}")

            if cur_n == max_count:
                continue

            # number of additional demos needed
            needed = max_count - cur_n

            # sample WITH replacement from current expert's own demos
            idx = np.random.choice(cur_n, needed, replace=True)

            # expand dataset
            demo_diff_exp_s[m] = np.concatenate([demo_diff_exp_s[m], demo_diff_exp_s[m][idx]], axis=0)
            demo_diff_exp_a[m] = np.concatenate([demo_diff_exp_a[m], demo_diff_exp_a[m][idx]], axis=0)

            print(f"Expert {m} new count: {demo_diff_exp_s[m].shape[0]}")



        # pairs = [(5, 0), (4, 1), (3, 2)]
        # for donor, receiver in pairs:
        #     donor_s = demo_diff_exp_s[donor]
        #     donor_a = demo_diff_exp_a[donor]
        #     receiver_s = demo_diff_exp_s[receiver]
        #     receiver_a = demo_diff_exp_a[receiver]

        #     # how many demos to add so receiver matches donor
        #     needed = int((donor_s.shape[0] - receiver_s.shape[0])/2)
        #     if needed <= 0:
        #         continue

        #     # sample from donor
        #     idx = np.random.choice(donor_s.shape[0], needed, replace=False)

        #     # ---- ADD TO RECEIVER ----
        #     demo_diff_exp_s[receiver] = np.concatenate([receiver_s, donor_s[idx]], axis=0)
        #     demo_diff_exp_a[receiver] = np.concatenate([receiver_a, donor_a[idx]], axis=0)

        #     # ---- REMOVE FROM DONOR ----
        #     mask = np.ones(donor_s.shape[0], dtype=bool)
        #     mask[idx] = False

        #     demo_diff_exp_s[donor] = donor_s[mask]
        #     demo_diff_exp_a[donor] = donor_a[mask]

        #     print(f"After transfer: donor {donor} → {demo_diff_exp_s[donor].shape[0]}, "
        #         f"receiver {receiver} → {demo_diff_exp_s[receiver].shape[0]}")
            
        # input("Press Enter to continue...")


    # for m in range(exp_num):
    #     print("Demo expert ", m, " num:", demo_diff_exp_s[m].shape[0])
    demo_diff_exp_s = np.array(demo_diff_exp_s)
    demo_diff_exp_a = np.array(demo_diff_exp_a)
    # min max value for each expert demos
    print("min max values for each expert demos:", np.min(demo_diff_exp_s, axis=(1,2)), np.max(demo_diff_exp_s, axis=(1,2)))
    print("min max values for each expert demos:", np.min(demo_diff_exp_a, axis=(1,2)), np.max(demo_diff_exp_a, axis=(1,2)))
    # convert to tensor
    demo_diff_exp_s = torch.tensor(demo_diff_exp_s, dtype=torch.float32).to(device)
    demo_diff_exp_a = torch.tensor(demo_diff_exp_a, dtype=torch.float32).to(device)
    
    print("demo_diff_exp_s shape:", demo_diff_exp_s.shape)
    print("demo_diff_exp_a shape:", demo_diff_exp_a.shape)
    
    # torch.Size([15000, 27]) torch.Size([15000, 8])
    # next step, combine expert id with s, and a, and maybe s_next
    state_scaler = loaded_data["input_scaler_s"].item()
    action_scaler = loaded_data["input_scaler_a"].item()
    normalize = True
    from stable_baselines3.common.vec_env import SubprocVecEnv
    def make_env(env_name, modify_reward=True):
        def _init():
            env = Custom_Env(env_name, scoring_model=None, 
                 input_scaler_s=state_scaler,
                 input_scaler_a=action_scaler,
                  normalize=normalize,
                  modify_reward=modify_reward
                )   
            return env
        return _init

    # env = make_env(env_name,modify_reward=False)
    env = gym.make(env_name)
    data_tup = (demo_diff_exp_s, demo_diff_exp_a, None)

    num_envs = 10  # number of parallel environments
    env_eval = SubprocVecEnv([make_env(env_name, modify_reward=False) for i in range(num_envs)])
    env_eval.seed(seed_value)

    best_omega, networks, best_eval, best_log, all_logs, all_evals, all_log_likelihood, all_epochs, all_omegas  =  \
        run_mle(data_tup, env, env_eval, args.hidden_size, args.embed_dim, device=device, 
            seed = seed_value, scaler_s = state_scaler, scaler_a = action_scaler, num_iters = 100000)
    
    # best_omega, networks, best_eval, best_log, all_logs, all_evals, all_log_likelihood, all_omegas = \
    #     run_bc(data_tup, env, env_eval, args.hidden_size, args.embed_dim, device=device,
    #         seed = seed_value, scaler_s = state_scaler, scaler_a = action_scaler,)



    # save the trained scoring
    with open(save_dir+"ileed_scoring_model_evals.pkl", "wb") as f:
        pickle.dump({
            "best_omega": best_omega,
            "networks": networks,
            "state_scaler": state_scaler,
            "action_scaler": action_scaler,
            "best_eval": best_eval,
            "best_log": best_log,
            "all_logs": all_logs,
            "all_evals": all_evals,
            "all_log_likelihood": all_log_likelihood,
            "all_epochs": all_epochs,
            "all_omegas": all_omegas,
        }, f)
    

    true_best_eval = 0.0



    mean_reward_list = []
    std_reward_list = []
    mean_stepLen_list = []
    std_stepLen_list = []
    for i in range(len(all_evals)):
        mean_reward_list.append(all_evals[i][0])
        std_reward_list.append(all_evals[i][1])
        mean_stepLen_list.append(all_evals[i][2])
        std_stepLen_list.append(all_evals[i][3])
    
    best_idx = np.argmax(mean_reward_list)
    true_best_eval = mean_reward_list[best_idx]
    true_best_log = all_log_likelihood[best_idx]

    file_suffix_name = "evaluation_during_training" 
    plot_reward_stepLen(all_epochs, mean_reward_list, std_reward_list, mean_stepLen_list, std_stepLen_list, env_name, save_dir, file_suffix_name)


    import matplotlib.pyplot as plt
    from pathlib import Path



    plot_evals_loglik_offlineMethods(np.array(all_evals)[:,0], all_log_likelihood,
            best_eval[0], best_log,
            true_best_eval, true_best_log,
            save_dir, file_suffix_name="evals_loglik")

if __name__ == "__main__":
    multiprocessing.set_start_method("fork")

    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--rl_seed', type=int, default='10', help='')
    parser.add_argument('--Expert_idx', type=str, default='10', help='')
    parser.add_argument('--noisy', type=str, default='False', help='')
    parser.add_argument('--env', type=int, default='9', help='0-4 env list')
    parser.add_argument('--data_augm', type=str, default='both', help='  "mixup" or "normal" or "both"  ')
    parser.add_argument('--total_steps', type=float, default='None', help=' total steps of interaction with the environment')
    parser.add_argument('--Pi_range_select', type=int, default='0', help='0 - 9')
    # parser.add_argument('--opt_ratio_alpha', type=str, default='0.5', help=' alpha value for the optimal ratio - 0.25 or 0.5 or 0.75 or 1.0 or 0.0')
    # parser.add_argument('--opt_used_ratio_uuPi', type=float, default='1.0', help=' ratio of the optimal data used, comparison in uu_Pi estimation tests')

    parser.add_argument('--embed_dim', type=int, default=10, help="embedding dimension")
    parser.add_argument('--hidden_size', type=int, default=256, help="dimension of hidden layer in MLPs")
    args = parser.parse_args()
    main(args)



