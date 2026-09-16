import gymnasium as gym
import pickle
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3 import PPO, TD3, SAC, DQN
import os
from utils_gail import Demonstration_Buffer_gail, Custom_Env
from mujo_04_EVAL import eval_best_checkpoints_models
from mujo_04_EVAL_results_npz import plot_results_npz
import argparse
import numpy as np
import torch
from mujo_10_EVAL_discriminator import plot_discriminator_acc_loss, plot_selflabel_num, plot_uuLearn_num
import multiprocessing

from utils_wgail_paral import WGAIL_Discrim, WGAIL


"""

Normal:


" python3 mujo_20_wgail_paral.py  --Expert_idx 0 --data_augm both  --total_steps 3e6  --env 4  --rl_seed 0 --Pi_range_select 0 ",

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

    clean_data = False
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
        models_dir = root_folder + "/logs_wgail_clean/"
    else:
        models_dir = root_folder + "/logs_wgail_noisy/"

    os.makedirs(models_dir, exist_ok=True)

    suffix = "_stepnum" + str(int(args.total_steps/1e5)) + "e5_"  + args.data_augm


    if not clean_data:
        save_dir = models_dir+env_name+"_WGAIL_"+RL_alg+"_NetAgentSeed_" + str(expert_idxs[0]) + "-" + str(expert_idxs[-1])+"_RLseed_"+str(seed_value)+"_wgail_discrm_"+str(scoring_update_itrs)+"_iters"+"_noisy-un"+suffix+"/"

    else:
        save_dir = models_dir+env_name+"_WGAIL_"+RL_alg+"_NetAgentSeed_" + str(expert_idxs[0]) + "-" + str(expert_idxs[-1])+"_RLseed_"+str(seed_value)+"_wgail_discrm_"+str(scoring_update_itrs)+"_iters"+"-un"+suffix+"/"


    # suffix += "_noLossmix"
    # suffix += "_sigFine"
    # suffix += "_halfAgtFine"
    # suffix += "_optThrAgtFine05"
    # suffix += "_fixScor"
    


    uu_data_path = root_folder + "/noisy_data/"
    uu_data_path = uu_data_path + env_name+"_ExpertSeed_"+str(expert_idxs[0])+"_uu_data"+"_multi_frame_1_optStart200_alpha_"+opt_ratio_alpha[0]+opt_ratio_alpha[2:]+  "_newneg.npz"

    # load code
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
    

 
    # input("check")

    if clean_data:
        demonstrations_exp = Demonstration_Buffer_gail( traj_s=traj_s_opt, traj_a=traj_a_opt, 
                                                    scaler_s=loaded_data["input_scaler_s"].item(), scaler_a=loaded_data["input_scaler_a"].item(),
                                                   device=device,
                                                    )
    else:
        demonstrations_exp = Demonstration_Buffer_gail( traj_s=traj_s_noisy, traj_a=traj_a_noisy, 
                                                    scaler_s=loaded_data["input_scaler_s"].item(), scaler_a=loaded_data["input_scaler_a"].item(),
                                                   device=device,
                                                    )
        
    

    # Create discriminator model --------------------------------------
    env_test = gym.make(env_name)
    state_shape = env_test.observation_space.shape[0]
    action_shape = env_test.action_space.shape[0]

    scoring_model = WGAIL_Discrim(state_shape, action_shape,)


    normalize = True
    from stable_baselines3.common.vec_env import SubprocVecEnv
    def make_env(env_name, modify_reward=True):
        def _init():
            env = Custom_Env(env_name, scoring_model, 
                 input_scaler_s=demonstrations_exp.scaler_s, 
                 input_scaler_a=demonstrations_exp.scaler_a,
                  normalize=normalize,
                  modify_reward=modify_reward
)
            return env
        return _init
    
    num_envs = 32 # number of parallel environments
    vec_env = SubprocVecEnv([make_env(env_name, modify_reward=True) for i in range(num_envs)])
    vec_env.seed(seed_value)


    # Create RL agent --------------------------------------
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

    # Create Ril agent --------------------------------------

    wGail = WGAIL(demonstrations_exp, demo_batch_size=1024, device=device,
                
            discrim=scoring_model, 
            n_disc_updates_per_round=scoring_update_itrs, lr_disc=1e-4,

            gen_algo=rl_model, 
            gen_train_timesteps=int(1.5e4), 
            gen_callback=eval_callback,
            
            scaler_state=demonstrations_exp.scaler_s, 
            scaler_action=demonstrations_exp.scaler_a,

            data_augm = args.data_augm,
            )

    wGail.train(int(args.total_steps))
    vec_env.close()


    # EVAL best & checkpoints models --------------------------------------
    model_dir = save_dir
    file_suffix_name = "after_training"

    num_envs = 10  # number of parallel environments
    env = SubprocVecEnv([make_env(env_name, modify_reward=False) for i in range(num_envs)])
    env.seed(seed_value)

    eval_best_checkpoints_models(env, env_name, RL_alg, seed_value, model_dir, file_suffix_name, eval_best=False, multienv=True)



    # save the trained model as pkl file
    with open(save_dir+"wgail_model.pkl", 'wb') as f:
        
        pickle.dump({   'disc_acc_agent': wGail.acc_agent_list,
                        'disc_acc_expert': wGail.acc_expert_list,
                        'disc_loss': wGail.loss_disc_list,

                        'discrim': wGail.discrim,
            
                        's_max': wGail.s_max,
                        's_min': wGail.s_min,
                        'a_max': wGail.a_max,
                        'a_min': wGail.a_min,

                        'confi_list': wGail.confi_list,
                    }, f)
    print("wGail.confi_list", wGail.confi_list)
    print("wGail.confi_list shape", np.array(wGail.confi_list).shape)


    # plot the results
    file_suffix_name = "evaluation_during_training" 
    plot_results_npz(env_name, save_dir, file_suffix_name)

    plot_discriminator_acc_loss(save_dir+"wgail_model.pkl", env_name, save_dir, "after_training")


if __name__ == "__main__":
    multiprocessing.set_start_method("fork")

    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--rl_seed', type=int, default='10', help='')
    parser.add_argument('--Expert_idx', type=str, default='10', help='')
    parser.add_argument('--env', type=int, default='9', help='0-4 env list')
    parser.add_argument('--data_augm', type=str, default='both', help='  "mixup" or "normal" or "both"  ')
    parser.add_argument('--total_steps', type=float, default='None', help=' total steps of interaction with the environment')
    parser.add_argument('--Pi_range_select', type=int, default='0', help='0 - 9')
    # parser.add_argument('--opt_ratio_alpha', type=str, default='0.5', help=' alpha value for the optimal ratio - 0.25 or 0.5 or 0.75 or 1.0 or 0.0')
    # parser.add_argument('--opt_used_ratio_uuPi', type=float, default='1.0', help=' ratio of the optimal data used, comparison in uu_Pi estimation tests')

    args = parser.parse_args()
    main(args)



