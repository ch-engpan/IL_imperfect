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
from utils_scoring_uu_sort_paral import RL_Scoring, Custom_Env
from utils_scoring_selfLabel_uu_sort_paral import RL_Scoring_selfLabel
from utils_gail import Demonstration_Buffer, Demonstration_Buffer_uu
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


"""

Normal:


"python3 mujo_20_rl_use_scoring_single_train_sort_uu_alphaEst_paral.py  --new_scormodel false  --env 4 --Expert_idx 4 --total_steps 1.5e6  --data_augm both  --pre_label True  --rl_seed 0  --uuLoss_str uuloss_all",


"python3 mujo_20_rl_use_scoring_single_train_sort_uu_alphaEst_paral.py  --new_scormodel false  --env 0 --Expert_idx 1 --total_steps 3e6    --data_augm both   --pre_label True  --rl_seed 0 --uuLoss_str uuloss_Nagent  --agentReplay_fine True  --expertDemo_fine True   --expertDemo_label_method topk  --exp_opt_threshold 0.5   --topk_k 0.1  --Pi_range_select 1 ",

"python3 mujo_20_rl_use_scoring_single_train_sort_uu_alphaEst_paral.py  --new_scormodel false  --env 0 --Expert_idx 1 --total_steps 3e6    --data_augm both   --pre_label True  --rl_seed 0 --uuLoss_str uuloss_Nagent  --agentReplay_fine True  --expertDemo_fine True   --expertDemo_label_method topk  --exp_opt_threshold 0.5   --topk_k 0.1  --Pi_range_select 1 ",

" python3 mujo_20_rl_use_scoring_single_train_sort_uu_alphaEst_paral.py  --new_scormodel false  --env 0 --Expert_idx 1 --total_steps 3e6    --data_augm both   --pre_label True  --rl_seed 100 --uuLoss_str uuloss_Nagent  --agentReplay_fine True  --expertDemo_fine True   --expertDemo_label_method topk  --exp_opt_threshold 0.5   --topk_k 0.5  --Pi_range_select 0 --exp_loss PN_loss ",

# no mixup:
" python3 mujo_20_rl_use_scoring_single_train_sort_uu_alphaEst_paral.py  --new_scormodel false  --env 0 --Expert_idx 1 --total_steps 3e6    --data_augm normal   --pre_label True  --rl_seed 100 --uuLoss_str uuloss_Nagent  --agentReplay_fine True  --expertDemo_fine True   --expertDemo_label_method topk  --exp_opt_threshold 0.5   --topk_k 0.5  --Pi_range_select 0 ",


# oracle:
"python3 mujo_20_rl_use_scoring_single_train_sort_uu_alphaEst_paral.py  --new_scormodel false  --env 0 --Expert_idx 1 --total_steps 3e6    --data_augm both   --pre_label True  --rl_seed 100 --uuLoss_str uuloss_Nagent  --agentReplay_fine False  --expertDemo_fine False   --expertDemo_label_method    topk     --topk_k  0.5 --early_stop False   --Pi_range_select 1 --pi_given True  --exp_loss PN_loss --uu10_orcl True ",

"python3 mujo_20_rl_use_scoring_single_train_sort_uu_alphaEst_paral.py  --new_scormodel false  --env 0 --Expert_idx 1 --total_steps 3e6    --data_augm both   --pre_label True  --rl_seed 0 --uuLoss_str uuloss_Nagent  --agentReplay_fine False  --expertDemo_fine False   --expertDemo_label_method    topk     --topk_k  0.5 --early_stop False   --Pi_range_select 1 --pi_given True  --exp_loss PN_loss --uu10_orcl True ",



" python3 mujo_20_rl_use_scoring_single_train_sort_uu_alphaEst_paral.py  --new_scormodel false  --env 0 --Expert_idx 1 --total_steps 3e6    --data_augm both   --pre_label True  --rl_seed 100 --uuLoss_str uuloss_Nagent  --agentReplay_fine True  --expertDemo_fine True   --expertDemo_label_method topk  --exp_opt_threshold 0.5   --topk_k 0.5  --Pi_range_select 0  --exp_check_num 1 ",



"""

def true_false_convert(str):
    if str == "true" or str == "True":
        return True
    elif str == "false" or str == "False":
        return False
    else:
        raise ValueError("true or false")


def main(args):

    uu10_orcl = true_false_convert(args.uu10_orcl)
    expFine_true_labels = true_false_convert(args.expFine_true_labels)
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=============================", device, "=============================")
    print("=============================", device, "=============================")


    # -----------------------------------------------
    # opt_ratio_alpha is randomly generated from np.random using seed, and already labeled in the dataset, here just to correspond to the opt_ratio_alpha
    if args.Pi_range_select == 0:
        # if int(args.Expert_idx[0]) == 0:
        #     args.opt_ratio_alpha = "0.564"
        # elif int(args.Expert_idx[0]) == 1:
        #     args.opt_ratio_alpha = "0.324"
        # elif int(args.Expert_idx[0]) == 2:
        #     args.opt_ratio_alpha = "0.393"
        # elif int(args.Expert_idx[0]) == 3:
        #     args.opt_ratio_alpha = "0.613"
        # elif int(args.Expert_idx[0]) == 4:
        #     args.opt_ratio_alpha = "0.649"
        args.opt_ratio_alpha = "0.5"

    elif args.Pi_range_select == 1:
        args.opt_ratio_alpha = "0.1"
    
    elif args.Pi_range_select == 2:
        args.opt_ratio_alpha = "0.5"
    
    elif args.Pi_range_select == 4:
        args.opt_ratio_alpha = "0.15"
    
    if uu10_orcl: # for uu10 oracle, opt_ratio_alpha always 0.5, and topk_k also 0.5
        args.opt_ratio_alpha = "0.5"
        # args.topk_k = 0.5




    if args.env == 0 or args.env == 1 or args.env == 3 :
        scoring_pretrained_epochs = "epochs_"+str(50)
    elif args.env == 2 or args.env == 4:
        scoring_pretrained_epochs = "epochs_"+str(100)
    
    args.less_pretrained_epochs = true_false_convert(args.less_pretrained_epochs)
    if args.less_pretrained_epochs:
        if args.env == 0 or args.env == 1 or args.env == 3 :
            scoring_pretrained_epochs = "epochs_"+str(10)
        elif args.env == 2 or args.env == 4:
            scoring_pretrained_epochs = "epochs_"+str(15)



        
    pi_given = true_false_convert(args.pi_given)

    if not args.disc_loss_type == "uu_loss" or args.disc_loss_type == "binary_loss":
        raise ValueError("disc_loss_type should be 'uu_loss' or 'binary_loss' ")

    env_name_list = ["Ant-v4", "HalfCheetah-v4", "Hopper-v4", "Swimmer-v4", "Walker2d-v4"]
    env_name = env_name_list[args.env]
    scoring_update_itrs = 500

    # random_Pi_range_list = [(0.1, 0.9), 
    #                     (0.05, 0.2), (0.05, 0.3), 
                        
    #                     (0.4, 0.6), 
                        
    #                     (0.8, 0.95), (0.7, 0.95), 
                          
    #                     ]
    random_Pi_range_list = [(0.1, 0.9), 
                        (0.05, 0.15), (0.45, 0.55), (0.85, 0.95),  # 1 2 3
                        (0.05, 0.25), (0.4, 0.6),  (0.75, 0.95),   # 4 5 6
                        
                        
                        (0.05, 0.4), (0.05, 0.5), 
                        
                        (0.4, 0.6), (0.3, 0.7), 
                        
                        
                        (0.8, 0.95), (0.7, 0.95), (0.6, 0.95), (0.5, 0.95),
                        
                          
                        ]
    exp_check_num = args.exp_check_num
    pi_test = args.pi_test

    random_Pi_range = random_Pi_range_list[args.Pi_range_select]

    # root_folder = "../results/alphaEst/"
    root_folder = "../results/alphaEst_randomPi/"
    # random_Pi_range used for save_folder
    root_folder = root_folder[:-1] + "_PiRange"+str(random_Pi_range[0])[0]+ str(random_Pi_range[0])[2:] +"-"+str(random_Pi_range[1])[0]+ str(random_Pi_range[1])[2:] +"/"

    if args.bag_num != 6:
        root_folder = root_folder[:-1]  + "_bagNum"+str(args.bag_num)+"/"

    if pi_given:
        root_folder = root_folder[:-1] + "_piGiven/"
    
    if uu10_orcl:
        root_folder = root_folder[:-1] + "_uu10orcl/"

    if args.less_pretrained_epochs:
        root_folder = root_folder[:-1]  + "_lessEpochs/"
    
    root_folder = root_folder[:-1] + "_expCheckNum"+str(exp_check_num)+"/"
    root_folder = root_folder[:-1] + "_piTest"+str(args.pi_test)+"/"
    

    opt_ratio_alpha = args.opt_ratio_alpha




    seed_value = args.rl_seed
    expert_idxs = args.Expert_idx
    print("expert_idxs", expert_idxs)
    RL_alg = "SAC"

  


    new_scormodel = true_false_convert(args.new_scormodel)
    if new_scormodel:
        models_dir = root_folder + "/logs_scoring_sin-train" + "_new_scoring_noReplacing/"
    else:
        models_dir = root_folder + "/logs_scoring_sin-train" + "_noReplacing/"
    if pi_given:
        models_dir = models_dir[:-1] + "_piGiven/"
    if uu10_orcl:
        models_dir = models_dir[:-1] + "_uu10orcl/"
    models_dir = models_dir[:-1]  + "_expCheckNum"+str(exp_check_num)+"/"
    models_dir = models_dir[:-1] + "_piTest"+str(args.pi_test)+"/"


    os.makedirs(models_dir, exist_ok=True)


    if new_scormodel:
        suffix = "_gail_new_" + str(scoring_update_itrs) + "_itrs_scoring"
    else:
        suffix = "_gail_finetune_" + str(scoring_update_itrs) + "_itrs_scoring"

    suffix = suffix + "_stepnum" + str(int(args.total_steps/1e5)) + "e5_" + args.data_augm 

    # suffix = suffix +"_noMixupLoss"
    # suffix = suffix +"_originMethod"
    # suffix = suffix +"_test"

    # Add hyperparameters to suffix for tracking
    original_method = True


    expertDemo_N2P = False
    expertDemo_useN = False # True

    agentReplay_N2P = False ## True
    agentReplay_useP = False



    expertDemo_fine = true_false_convert(args.expertDemo_fine)  
    agentReplay_fine = true_false_convert(args.agentReplay_fine)
    early_stop = true_false_convert(args.early_stop)

    # expertDemo_label_method = "threshold" # "threshold" or "topk"
    expertDemo_label_method = args.expertDemo_label_method  # "threshold" or "topk"
    exp_opt_threshold = args.exp_opt_threshold
    # topk_k = 0.1
    # topk_k = args.topk_k



    if args.uuLoss_str == "uuloss_Nagent":
        uuLoss_Nagent = True
        uuLoss_all = False
    elif args.uuLoss_str == "uuloss_all":
        uuLoss_Nagent = False
        uuLoss_all = True
    else:
        uuLoss_Nagent = False
        uuLoss_all = False


    uuLoss_str = args.uuLoss_str




    # Load two scoring models for co-training --------------------------------------

    scoring_model_dir = root_folder + "/scoring_model_alphaEst/"
    name_suffix = "multi_frame_1_optStart200"

    scoring_dir = scoring_model_dir +env_name+"_ExpertSeed_"+str(expert_idxs[0])+"_scoring_model-un_"+name_suffix+"_alpha_"+opt_ratio_alpha[0]+opt_ratio_alpha[2:]+"_newneg/"

    scoring_model_path = scoring_dir + "scroring_model_"+env_name+"_"+scoring_pretrained_epochs + "_NetSeed_"
    net_idx = expert_idxs[-1]
    scoring_model_A_path = scoring_model_path +str(net_idx) + ".pkl"

    # ----------------- scoring model B ----------------- for co-training ----------------- but not used in this script
    scoring_model_B_path = scoring_model_A_path

    with open(scoring_model_A_path, 'rb') as f:
        scoring_model_A = pickle.load(f)

    with open(scoring_model_B_path, 'rb') as f:
        scoring_model_B = pickle.load(f)


    print("scoring_model_A s min", scoring_model_A.input_scaler_s.data_min_)
    print("scoring_model_A s max", scoring_model_A.input_scaler_s.data_max_)
    print("scoring_model_A a min", scoring_model_A.input_scaler_a.data_min_)
    print("scoring_model_A a max", scoring_model_A.input_scaler_a.data_max_)

    print("")


    print("scoring_model_A priors_corr", scoring_model_A.net.priors_corr)
    print("scoring_model_A Pi", scoring_model_A.net.Pi)
    print("")
    print("")

    Pis_mean = np.mean(scoring_model_A.net.Pi)
    # print("scoring_model_A Pi mean", Pis_mean)
    # round to 2 decimal places floor
    Pis_mean = np.floor(Pis_mean * 100) / 100
    # print("scoring_model_A Pi mean (floored)", Pis_mean)

    if args.topk_k <= 0:
        topk_k = Pis_mean
        print("Using Pi mean for topk_k:", topk_k)
    else:
        topk_k = args.topk_k
        print("Using provided topk_k:", topk_k)

    print(" ")

    # topk_k = Pis_mean

    # input("enter")

    

    if uuLoss_Nagent and uuLoss_all:
        raise ValueError("uuLoss_Nagent and uuLoss_all cannot be both True")

    if not (uuLoss_Nagent or uuLoss_all):
        suffix += f"_origMethod_{original_method}_expFine_{expertDemo_fine}_expN2P_{expertDemo_N2P}_expUseN_{expertDemo_useN}_agtN2P_{agentReplay_N2P}_agtUseP_{agentReplay_useP}"
    else:
    # uuLoss_str
        suffix += f"_{uuLoss_str}"
    
        if (not expertDemo_fine) and (not agentReplay_fine):
            suffix += "_noRelabel"
        else:
            suffix += "_expFine_"+str(expertDemo_fine)[0]+"_agtFine_"+str(agentReplay_fine)[0]
            if expertDemo_fine:
                suffix += f"_{expertDemo_label_method}"
                if expertDemo_label_method == "threshold":
                    exp_opt_threshold_str = str(exp_opt_threshold)
                    suffix += "_expOptThr_" + exp_opt_threshold_str[0] + exp_opt_threshold_str[2:]  
                elif expertDemo_label_method == "topk":
                    topk_k_str = str(topk_k)
                    suffix += "_expTopk_" + topk_k_str[0] + topk_k_str[2:]  
                else:
                    raise ValueError("expertDemo_label_method should be 'threshold' or 'topk' ")
            if early_stop:
                suffix += "_earlyStop30num"
            
        if args.exp_loss == "PN_loss":
            suffix += "_expLossPN"
        elif args.exp_loss == "uu_loss":
            suffix += ""
        else:
            raise ValueError("exp_loss should be 'PN_loss' or 'uu_loss' ")
        
        if expFine_true_labels:
            suffix = suffix + "_expFineTrueLabels/"



    # suffix += "_noLossmix"
    # suffix += "_sigFine"
    # suffix += "_halfAgtFine"
    # suffix += "_optThrAgtFine05"
    # suffix += "_fixScor"
    num_envs = args.num_envs
    if num_envs != 32:
        suffix += "_env"+str(num_envs)


    save_dir = models_dir + env_name + "_scoring_" + RL_alg + "_NetAgentSeed_" + str(expert_idxs[0]) + "-" + str(expert_idxs[-1]) + "_" + scoring_pretrained_epochs + "_RLseed_" + str(seed_value) + "_" + "alpha_" + opt_ratio_alpha[0] + opt_ratio_alpha[2:] + "-un" + suffix + "/"




    uu_data_path = root_folder + "/noisy_data/"
    uu_data_path = uu_data_path + env_name+"_ExpertSeed_"+str(expert_idxs[0])+"_uu_data"+"_multi_frame_1_optStart200_alpha_"+opt_ratio_alpha[0]+opt_ratio_alpha[2:]+  "_newneg.npz"

    # load code
    loaded_data = dict(np.load(uu_data_path, allow_pickle=True))
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

    # input("Press Enter to continue...")

    # reduce the duplicated data:
    if uu10_orcl and args.Pi_range_select == 1:

        loaded_data["U_set_s_train"] = np.concatenate((loaded_data["U_set_s_train"][:3000], loaded_data["U_set_s_train"][27000:]), axis=0)
        loaded_data["U_set_a_train"] = np.concatenate((loaded_data["U_set_a_train"][:3000], loaded_data["U_set_a_train"][27000:]), axis=0)
        loaded_data["U_set_classLabels_train"] = np.concatenate((loaded_data["U_set_classLabels_train"][:3000], loaded_data["U_set_classLabels_train"][27000:]), axis=0)
        loaded_data["U_sets_binLabels_train"] = np.concatenate((loaded_data["U_sets_binLabels_train"][:3000], loaded_data["U_sets_binLabels_train"][27000:]), axis=0)
        print("Reduced duplicated data for uu10_orcl and Pi_range_select 1")
        print("U_set_s_train shape:", loaded_data["U_set_s_train"].shape)
        print("U_set_a_train shape:", loaded_data["U_set_a_train"].shape)
        print("U_set_classLabels_train shape:", loaded_data["U_set_classLabels_train"].shape)
        print("U_sets_binLabels_train shape:", loaded_data["U_sets_binLabels_train"].shape)
        
        print("U_set_classLabels_train example: ", loaded_data["U_set_classLabels_train"][3000-5:3000+5])
        print("U_sets_binLabels_train example: ", loaded_data["U_sets_binLabels_train"][3000-5:3000+5])


    # input("Press Enter to continue...")


    with torch.no_grad():
        scoring_model_A.net.eval()
        # convert to tensor
        uu_data_s = torch.tensor(loaded_data["U_set_s_train"], dtype=torch.float32).to(device)
        uu_data_a = torch.tensor(loaded_data["U_set_a_train"], dtype=torch.float32).to(device)
        opt_or_not, class_i, _ = scoring_model_A.net.forward(uu_data_s, uu_data_a)
        opt_or_not = opt_or_not.cpu()

    traj_s_labeled_opt = np.array(loaded_data["U_set_s_train"][opt_or_not[:, 0]>0.5])
    traj_a_labeled_opt = np.array(loaded_data["U_set_a_train"][opt_or_not[:, 0]>0.5])
    traj_s_labeled_nonopt = np.array(loaded_data["U_set_s_train"][opt_or_not[:, 0]<0.5])
    traj_a_labeled_nonopt = np.array(loaded_data["U_set_a_train"][opt_or_not[:, 0]<0.5])

    #count
    print("Number of labeled optimal trajectories:", traj_s_labeled_opt.shape[0])
    print("Number of labeled non-optimal trajectories:", traj_s_labeled_nonopt.shape[0])



    # TEST if the data from saved npz same as here --------------------------------------

    # U_set_s_train_opt = loaded_data["U_set_s_train"][loaded_data["U_sets_binLabels_train"][:, 0]>0.5]
    # U_set_a_train_opt = loaded_data["U_set_a_train"][loaded_data["U_sets_binLabels_train"][:, 0]>0.5]
    # U_set_s_train_nonopt = loaded_data["U_set_s_train"][loaded_data["U_sets_binLabels_train"][:, 0]<0.5]
    # U_set_a_train_nonopt = loaded_data["U_set_a_train"][loaded_data["U_sets_binLabels_train"][:, 0]<0.5]
    # traj_s_labeled_opt = []
    # traj_a_labeled_opt = []
    # traj_s_labeled_nonopt = []
    # traj_a_labeled_nonopt = []

    # with torch.no_grad():
    #     uu_data_s = torch.tensor(U_set_s_train_opt, dtype=torch.float32).to(device)
    #     uu_data_a = torch.tensor(U_set_a_train_opt, dtype=torch.float32).to(device)
    #     opt_or_not, class_i, _ = scoring_model_A.net.forward(uu_data_s, uu_data_a)
    #     opt_or_not = opt_or_not.cpu()
    
    
    # traj_s_labeled_opt = np.array(U_set_s_train_opt[opt_or_not[:, 0]>0.5])
    # traj_a_labeled_opt = np.array(U_set_a_train_opt[opt_or_not[:, 0]>0.5])

    # traj_s_labeled_nonopt = np.array(U_set_s_train_opt[opt_or_not[:, 0]<0.5])
    # traj_a_labeled_nonopt = np.array(U_set_a_train_opt[opt_or_not[:, 0]<0.5])

    # with torch.no_grad():
    #     uu_data_s = torch.tensor(U_set_s_train_nonopt, dtype=torch.float32).to(device)
    #     uu_data_a = torch.tensor(U_set_a_train_nonopt, dtype=torch.float32).to(device)
    #     opt_or_not, class_i, _ = scoring_model_A.net.forward(uu_data_s, uu_data_a)
    #     opt_or_not = opt_or_not.cpu()


    # traj_s_labeled_opt = np.concatenate((traj_s_labeled_opt, U_set_s_train_nonopt[opt_or_not[:, 0]>0.5]), axis=0)
    # traj_a_labeled_opt = np.concatenate((traj_a_labeled_opt, U_set_a_train_nonopt[opt_or_not[:, 0]>0.5]), axis=0)

    # traj_a_labeled_nonopt = np.concatenate((traj_a_labeled_nonopt, U_set_a_train_nonopt[opt_or_not[:, 0]<0.5]), axis=0)
    # traj_s_labeled_nonopt = np.concatenate((traj_s_labeled_nonopt, U_set_s_train_nonopt[opt_or_not[:, 0]<0.5]), axis=0)

    # #count
    # print("Number of labeled optimal trajectories:", traj_s_labeled_opt.shape[0])
    # print("Number of labeled non-optimal trajectories:", traj_s_labeled_nonopt.shape[0])

    # # convert to dtype=torch.float32

    # traj_s_labeled_opt = torch.tensor(traj_s_labeled_opt, dtype=torch.float32).to(device)
    # traj_a_labeled_opt = torch.tensor(traj_a_labeled_opt, dtype=torch.float32).to(device)
    # traj_s_labeled_nonopt = torch.tensor(traj_s_labeled_nonopt, dtype=torch.float32).to(device)
    # traj_a_labeled_nonopt = torch.tensor(traj_a_labeled_nonopt, dtype=torch.float32).to(device)

    # print("traj_s_labeled_opt", traj_s_labeled_opt[228:233])
    # print("traj_a_labeled_opt", traj_a_labeled_opt[228:233])
    # print("traj_s_labeled_nonopt", traj_s_labeled_nonopt[228:233])
    # print("traj_a_labeled_nonopt", traj_a_labeled_nonopt[228:233])
    
    #  --------------------------------------

    # scoring_model_A.net
    # print("scoring_model_A net Pi:", scoring_model_A.net.Pi)
    # print("scoring_model_A net priors_class:", scoring_model_A.net.priors_corr)
    # print("scoring_model_A net Pi_test:", scoring_model_A.net.Pi_test)
    # input("check point")



    # Create discriminator model --------------------------------------
    env_test = gym.make(env_name)
    state_shape = env_test.observation_space.shape[0]
    action_shape = env_test.action_space.shape[0]
    # scoring_model = GAIL_Discrim(state_shape, action_shape,) # discrmdiscrmdiscrm


    if (new_scormodel or opt_ratio_alpha == "0.0" or opt_ratio_alpha == "1.0") and not uu10_orcl:
        priors_class = np.random.rand(20)
        Pi = np.random.rand(20)
        Pi_test = 0.5
        scoring_net_new = Scoring_model_net_multiFrame(priors_class, Pi, Pi_test, state_shape, action_shape, frame_num = 1) # scorescorescore
    # there is no need to learn / pretrain a scoring model, if the alpha is 0.0 or 1.0
        scoring_model_A.net = scoring_net_new
        scoring_model_B.net = scoring_net_new
        print("")
        print("NEW scoring model is used")
        print("")

    # from pprint import pprint
    # pprint(vars(env.env))

    normalize = True


    from stable_baselines3.common.vec_env import SubprocVecEnv
    def make_env(env_name, modify_reward=True):
        def _init():
            env = Custom_Env(env_name, scoring_model_A, 
                    input_scaler_s=scoring_model_A.input_scaler_s, 
                    input_scaler_a=scoring_model_A.input_scaler_a,
                    normalize=normalize, 
                    modify_reward=modify_reward
    )
            return env
        return _init
    
    num_envs = num_envs # number of parallel environments
    vec_env = SubprocVecEnv([make_env(env_name, modify_reward=True) for i in range(num_envs)])
    vec_env.seed(seed_value)




    # Create RL agent --------------------------------------
    RL_params = json.load(open(Path("./RL_parameters/mujoco_params.json")))

    if RL_alg == "SAC":
        rl_model = SAC('MlpPolicy', vec_env, verbose=0,
                    use_sde = False,
                    learning_starts=100,
                    batch_size=1024,
                    learning_rate=2e-3,
                    seed=seed_value, device=device)
        



    eval_freq = 15000 / num_envs
    check_points_freq = 1 * eval_freq

    checkpoint_callback = CheckpointCallback(
                                            save_freq=int(check_points_freq/eval_freq),
                                            save_path=save_dir + "checkpoints/",
                                            name_prefix=env_name+"_"+RL_alg+"_models",
                                            save_replay_buffer=False,
                                            save_vecnormalize=True,
                                            verbose=1,
                                            )

    eval_callback = EvalCallback(vec_env, 
                                callback_after_eval = checkpoint_callback,
                                best_model_save_path=save_dir,
                                log_path=save_dir, 
                                eval_freq=eval_freq,
                                deterministic=True, 
                                render=False,
                                verbose = 1)

    # Load demonstration data --------------------------------------
    trajs_dir = root_folder + "/opt_nonopt_trajs_alphaEst/"
    expert_seed = args.Expert_idx[0]

    saved_trajs_path = trajs_dir + "opt_nonopt_trajs_"+env_name+"_ExpertSeed_"+str(expert_seed)+"_NetSeed"+str(expert_seed)+"_"+scoring_pretrained_epochs+"_pseudo_labeled_alpha_"+opt_ratio_alpha[0]+opt_ratio_alpha[2:]+".pkl"

    demon_normalized = True

    # only if opt_ratio_alpha != 0.0 or 1.0, pre_label can maybe be used
    if args.pre_label == "False" or args.pre_label == "false":
        # ablation test - not using pre-labeled data
        saved_trajs_path = trajs_dir + "opt_nonopt_trajs_"+env_name+"_sacExpertSeed_"+str(0)+"_noisy_opt"+"_newneg"+".pkl"
        print("\n")
        print("pre_label ************************************** ", args.pre_label)
        print("\n")
        demon_normalized = False
    elif args.pre_label == "True" or args.pre_label == "true":
        demon_normalized = True
    else:
        raise ValueError("pre_label should be True or False")

    print("\n")
    print("demo_normalized ", demon_normalized)
    print("saved_trajs_path ", saved_trajs_path)



    if uuLoss_all or uuLoss_Nagent:
        demonstrations_exp = Demonstration_Buffer_uu( traj_s=loaded_data["U_set_s_train"], traj_a=loaded_data["U_set_a_train"], bagLabels=loaded_data["U_set_classLabels_train"], binLabels=loaded_data["U_sets_binLabels_train"],
                                                     priors_class=loaded_data["priors_class"], Pi_s=loaded_data["Pi_s_train"],
                                                     device=device,
                                                    scaler_s=loaded_data["input_scaler_s"].item(), scaler_a=loaded_data["input_scaler_a"].item(),
                                                    uuLoss_all = uuLoss_all,
                                                    opt_threshold=exp_opt_threshold,
                                                    top_k_percent=topk_k,
                                                    expertDemo_label_method = expertDemo_label_method,
                                                    )
        demonstrations_exp.update_opt_demo(scoring_model_A.net, first_label=True)
        if uu10_orcl or expFine_true_labels:
            demonstrations_exp.update_opt_demo_trueLabels()
            # input("Using true labels for expertDemo_fine, press Enter to continue...")

        if new_scormodel and uu10_orcl:
            scoring_model_A.net = Scoring_model_net_multiFrame(scoring_model_A.net.priors_corr, scoring_model_A.net.Pi, scoring_model_A.net.Pi_test, 
                                                            loaded_data["input_dim_s"], loaded_data["input_dim_a"], frame_num = loaded_data["frame_num"].item()).to(device)



    else:
        demonstrations_exp = Demonstration_Buffer(saved_trajs_path, device=device,  single_frame=False)

    print("loaded dataset")



    """
    Stage 1

    scoring_lr = 1e-4
    scoring_batch_size = 1000
    scoring_optimizer = Adam
    var_low_threshold for of est. expertise levels = 0.0005

    per epochs = 5 update expertise levels

    epochs = 10 for Ant HalfCheetah  Swimmer
    epochs = 20 for Walker2d

    Stage 2
    
    rl_model = SAC - stable_baselines3
    use_sde = False,
    learning_starts=100,
    rl_batch_size=1024,
    rl_learning_rate=2e-3,
    rl_learning_starts=100,
    rl_buffer_size=1e6,
    rl_train_timesteps_each_itr=1.5e4,

    scoring_lr = 1e-4
    scoring_batch_size = 1024
    scoring_optimizer = Adam
    scoring_update_each_itr = 500

    delta_earlystop = 30 for continuous 5 steps

    Total iterations = 200 

    """



    # Create RL_Scoring_Co-Train agent --------------------------------------
    
    lr_scoring = 1e-4
    
    print("")
    print("scoring_model_A_path ", scoring_model_A_path)
    print("saved_trajs_path ", saved_trajs_path)
    print("")
    # # get all attributes of scoring_model_A.net
    # print("scoring_model_A.net", scoring_model_A.net.__dict__)
    # print("")
    # print("scoring_model_A.net", vars(scoring_model_A.net))

    train_method = 1

    if train_method == 0:
        RL_scoring_train = RL_Scoring
    elif train_method == 1:
        RL_scoring_train = RL_Scoring_selfLabel



    rl_scoring_train = RL_scoring_train(demonstrations_exp, batch_size=1024, device=device,
                                            rl_algo=rl_model,  rl_callback = eval_callback,
                                            rl_train_timesteps=int(1.5e4),
                                            
                                            scoring_model_A=scoring_model_A, 
                                            scoring_model_B=scoring_model_B,
                                            n_scoring_updates_per_round=scoring_update_itrs, 
                                            lr_scoring=lr_scoring,

                                            scaler_state=scoring_model_A.input_scaler_s,
                                            scaler_action=scoring_model_A.input_scaler_a, 


                                            data_augm = args.data_augm,

                                            demon_normalized = demon_normalized,
                                            original_method = original_method,
                                            expertDemo_fine = expertDemo_fine,
                                            expertDemo_N2P = expertDemo_N2P,
                                            expertDemo_useN = expertDemo_useN,
                                            agentReplay_N2P = agentReplay_N2P,
                                            agentReplay_useP = agentReplay_useP,

                                            agentReplay_fine= agentReplay_fine,

                                            uuLoss_Nagent = uuLoss_Nagent,
                                            uuLoss_all = uuLoss_all,
                                            early_stop = early_stop,

                                            exp_loss = args.exp_loss,
                                            )

    # Save original expert demonstration state-action pairs before training
    if not (uuLoss_all or uuLoss_Nagent):
        original_expert_demo_opt_states = demonstrations_exp.states_opt.clone().cpu().numpy()
        original_expert_demo_opt_actions = demonstrations_exp.actions_opt.clone().cpu().numpy()
        original_expert_demo_nonopt_states = demonstrations_exp.states_nonopt.clone().cpu().numpy()
        original_expert_demo_nonopt_actions = demonstrations_exp.actions_nonopt.clone().cpu().numpy()

    rl_scoring_train.train(int(args.total_steps), )


    vec_env.close()

    # EVAL best & checkpoints models --------------------------------------
    model_dir = save_dir
    file_suffix_name = "after_training"
    # env.envs[0].set_modify_reward(False)

    # env = Custom_Env(env_name, scoring_model_A, 
    #                 input_scaler_s=scoring_model_A.input_scaler_s, 
    #                 input_scaler_a=scoring_model_A.input_scaler_a,
    #                 normalize=normalize)
    # env = DummyVecEnv([lambda: env])
    # print("reset env", env.reset())
    # env.seed(seed_value)
    # env.envs[0].set_modify_reward(False)
    
    num_envs = 10  # number of parallel environments
    env = SubprocVecEnv([make_env(env_name, modify_reward=False) for i in range(num_envs)])
    env.seed(seed_value)


    eval_best_checkpoints_models(env, env_name, RL_alg, seed_value, model_dir, file_suffix_name, eval_best=False, multienv=True)


    # Save the trained rl_scoring_train model as pkl file
    with open(save_dir+"rl_scoring_co_train_model.pkl", 'wb') as f:
        pickle.dump({   'disc_acc_agent': rl_scoring_train.acc_agent_list,
                        'disc_acc_expert': rl_scoring_train.acc_expert_list,
                        'disc_loss': rl_scoring_train.loss_disc_list,

                        'disc_model_A': rl_scoring_train.scoring_model_A,
                        'disc_model_B': rl_scoring_train.scoring_model_B,
                        's_max': rl_scoring_train.s_max,
                        's_min': rl_scoring_train.s_min,
                        'a_max': rl_scoring_train.a_max,
                        'a_min': rl_scoring_train.a_min,
                    }, f)

    if train_method == 1 and not (uuLoss_all or uuLoss_Nagent):
        # save the trained rl_scoring_train model as pkl file
        with open(save_dir+"rl_scoring_finegrained_data.pkl", 'wb') as f:

            pickle.dump({ 
                            'exp_N2P_num': rl_scoring_train.exp_N2P_num,
                            'exp_P2N_num': rl_scoring_train.exp_P2N_num,
                            'expDemo_opt_num': rl_scoring_train.expDemo_opt_num,
                            'expDemo_nonopt_num': rl_scoring_train.expDemo_nonopt_num,
                            'agt_N2P_num': rl_scoring_train.agt_N2P_num,
                            
                            # Track replay buffer sizes during training
                            'agent_replay_buffer_opt_size': rl_scoring_train.agent_replay_buffer_opt_size,
                            'agent_replay_buffer_nonopt_size': rl_scoring_train.agent_replay_buffer_nonopt_size,

                            # Save state-action pairs from agent replay buffers
                            'agent_replay_buffer_opt_states': rl_scoring_train.agent_replay_buffer_opt.state_tensor.cpu().numpy(),
                            'agent_replay_buffer_opt_actions': rl_scoring_train.agent_replay_buffer_opt.action_tensor.cpu().numpy(),
                            'agent_replay_buffer_nonopt_states': rl_scoring_train.agent_replay_buffer_nonopt.state_tensor.cpu().numpy(),
                            'agent_replay_buffer_nonopt_actions': rl_scoring_train.agent_replay_buffer_nonopt.action_tensor.cpu().numpy(),
                            
                            # Save original expert demonstration state-action pairs (before training)
                            'original_expert_demo_opt_states': original_expert_demo_opt_states,
                            'original_expert_demo_opt_actions': original_expert_demo_opt_actions,
                            'original_expert_demo_nonopt_states': original_expert_demo_nonopt_states,
                            'original_expert_demo_nonopt_actions': original_expert_demo_nonopt_actions,
                            
                            # Save final expert demonstration state-action pairs (after training)
                            'expert_demo_opt_states': rl_scoring_train.demonstrations.states_opt.cpu().numpy(),
                            'expert_demo_opt_actions': rl_scoring_train.demonstrations.actions_opt.cpu().numpy(),
                            'expert_demo_nonopt_states': rl_scoring_train.demonstrations.states_nonopt.cpu().numpy(),
                            'expert_demo_nonopt_actions': rl_scoring_train.demonstrations.actions_nonopt.cpu().numpy(),
                        }, f)
            
    elif train_method == 1 and (uuLoss_all or uuLoss_Nagent):
# relabel_earlyStop_list
        with open(save_dir+"rl_scoring_uu_learning_data.pkl", 'wb') as f:
            pickle.dump({
                            'expDemo_opt_num': rl_scoring_train.expDemo_opt_num,
                            'TP_exp': rl_scoring_train.TP_exp,
                            'TN_exp': rl_scoring_train.TN_exp,
                            'FP_exp': rl_scoring_train.FP_exp,
                            'FN_exp': rl_scoring_train.FN_exp,
                            'recall_exp': rl_scoring_train.recall_exp,
                            'unprecision_exp': rl_scoring_train.unprecision_exp,
                            'true_exp_P_num' : rl_scoring_train.true_exp_P_num,
                            'true_exp_N_num' : rl_scoring_train.true_exp_N_num,

                            'agt_N2P_num': rl_scoring_train.agt_N2P_num,
                            'exp_N2P_num': rl_scoring_train.exp_N2P_num,
                            'exp_P2N_num': rl_scoring_train.exp_P2N_num,

                            'relabel_earlyStop': rl_scoring_train.relabel_earlyStop_list,
                        }, f)

    file_suffix_name = "evaluation_during_training" 
    plot_results_npz(env_name, save_dir, file_suffix_name)

    plot_discriminator_acc_loss(save_dir+"rl_scoring_co_train_model.pkl", env_name, save_dir, "after_training")

    if train_method == 1 and not (uuLoss_all or uuLoss_Nagent): 
        plot_selflabel_num(save_dir+"rl_scoring_finegrained_data.pkl", env_name, save_dir, "after_training")

    elif train_method == 1 and (uuLoss_all or uuLoss_Nagent):
        plot_uuLearn_num(save_dir+"rl_scoring_uu_learning_data.pkl", env_name, save_dir, "after_training")



if __name__ == "__main__":
    multiprocessing.set_start_method("fork")

    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--rl_seed', type=int, default='10', help='')
    parser.add_argument('--Expert_idx', type=list, default='10', help='')
    parser.add_argument('--data_augm', type=str, default='False', help='  "mixup" or "normal" or "both"  ')
    parser.add_argument('--new_scormodel', type=str, default='false', help='true, false')
    parser.add_argument('--env', type=int, default='9', help='0-4 env list')

    parser.add_argument('--sort_loss', type=str, default='True', help=' True or False')
    parser.add_argument('--sort_ratio', type=float, default='0.5', help=' ratio for the sorting - 0.25 or 0.5 or 0.75')
    parser.add_argument('--disc_loss_type', type=str, default='uu_loss', help=' "uu_loss" or "binary_loss" ') # "binary_loss" or "uu_loss"



    parser.add_argument('--total_steps', type=float, default='None', help=' total steps of interaction with the environment') 
    parser.add_argument('--alpha_noise_level', type=float, default='0.0', help=' noise level for the alpha value - 0 or 0.2 or 0.4 or 0.6')
    
    parser.add_argument('--pre_label', type=str, default='True', help=' True or False')
    parser.add_argument('--uuLoss_str', type=str, default='None', help=' uuloss_Nagent or uuloss_all')

    """   
    neither:
        uses pseudo-labeled opt/nonopt pickle
        expert = positive, agent = negative
        normal P/N imitation-style scoring update

    uuLoss_Nagent:
        uses raw U-data NPZ
        expert uses U-bag loss or pseudo-label loss
        agent is forced negative by binary loss

    uuLoss_all:
        uses raw U-data NPZ
        expert uses U-bag loss or pseudo-label loss
        agent becomes an extra Pi=0 bag and uses bag CE loss
    """


    parser.add_argument('--expertDemo_fine', type=str, default='False', help=' expertDemo_fine should be True or False ')
    parser.add_argument('--agentReplay_fine', type=str, default='False', help=' agentReplay_fine should be True or False ')
    parser.add_argument('--expertDemo_label_method', type=str, default='threshold', help=' expertDemo_label_method should be "threshold" or "topk" ')
    parser.add_argument('--exp_opt_threshold', type=float, default='0.5', help=' exp_opt_threshold for demo opt from exp uu data')
    parser.add_argument('--topk_k', type=float, default='0.2', help=' top-k value for demo opt from exp uu data')


    parser.add_argument('--Pi_range_select', type=int, default='0', help='0 - 9')
    parser.add_argument('--early_stop', type=str, default='True', help=' early_stop should be True or False ')
    parser.add_argument('--pi_given', type=str, default='False', help=' pi_given should be True or False ')
    parser.add_argument('--num_envs', type=int, default='32', help='number of parallel envs, 1, 2, 4, 8, 16')
    parser.add_argument('--exp_loss', type=str, default='uu_loss', help=' exp_loss should be "uu_loss" or "PN_loss" ')
    # args.bag_num
    parser.add_argument('--bag_num', type=int, default='6', help=' bag_num should be 2, 4, 6, 8 ')
    parser.add_argument('--uu10_orcl', type=str, default='False', help=' uu10_orcl should be True or False ')
    # args.expFine_true_labels
    parser.add_argument('--expFine_true_labels', type=str, default='False', help=' expFine_true_labels should be True or False ')
    # less_pretrained_epochs
    parser.add_argument('--less_pretrained_epochs', type=str, default='False', help=' less_pretrained_epochs should be True or False ')
    parser.add_argument('--exp_check_num', type=int, default='5', help='number of samples to check by experts')
    parser.add_argument('--pi_test', type=float, default='0.5', help='the Pi value for test data')

    args = parser.parse_args()
    main(args)