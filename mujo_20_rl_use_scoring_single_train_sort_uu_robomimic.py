import gymnasium as gym
import pickle
# import gym
from sb3_contrib import TRPO, TQC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO, TD3, SAC, DQN
import os
from old_scripts.utils_scoring_uu_sort import RL_Scoring, Custom_Env
from utils_gail import Demonstration_Buffer
from mujo_04_EVAL import eval_best_checkpoints_models
from mujo_04_EVAL_results_npz import plot_results_npz
import argparse
import json
from pathlib import Path
# from utils_gail import GAIL_Discrim
from um_ssc_grid_mujuco import Scoring_model_net_multiFrame
import numpy as np
import torch
from mujo_10_EVAL_discriminator import plot_discriminator_acc_loss

import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import h5py


"""
"python3 mujo_20_rl_use_scoring_single_train_sort.py  --new_scormodel false  --env 1  --Expert_idx 0  --sort_loss True  --data_augm mixup  --rl_seed 0  ",
"python3 mujo_20_rl_use_scoring_single_train_sort.py  --new_scormodel false  --env 1  --Expert_idx 0  --sort_loss True  --data_augm both  --rl_seed 0  ",

"python3 mujo_20_rl_use_scoring_single_train_sort.py  --new_scormodel false  --env 1  --Expert_idx 0  --sort_loss False  --data_augm mixup  --rl_seed 0  ",
"python3 mujo_20_rl_use_scoring_single_train_sort.py  --new_scormodel false  --env 1  --Expert_idx 0  --sort_loss False  --data_augm both  --rl_seed 0  ",

"python3 mujo_20_rl_use_scoring_single_train_sort_uu.py  --new_scormodel false  --env 0  --Expert_idx 0 --total_steps 1.5e6  --sort_loss False --disc_loss_type uu_loss  --data_augm normal  --rl_seed 99  ",



python3 mujo_20_rl_use_scoring_single_train_sort_uu.py  --new_scormodel false  --env 1 --Expert_idx 0 --total_steps 2.5e6  --sort_loss True   --disc_loss_type uu_loss  --data_augm both  --opt_ratio_alpha 0.25  --rl_seed 99


python3 mujo_20_rl_use_scoring_single_train_sort_uu.py  --new_scormodel false  --env 1 --Expert_idx 0 --total_steps 2.5e6  --sort_loss True   --disc_loss_type uu_loss  --data_augm both  --opt_ratio_alpha 1.0  --rl_seed 99



Normal:
python3 mujo_20_rl_use_scoring_single_train_sort_uu_robomimic.py  --new_scormodel false  --env 0 --Expert_idx 0 --total_steps 2.5e6  --sort_loss False  --sort_ratio 0.5   --disc_loss_type uu_loss  --data_augm both  --opt_ratio_alpha 0.5  --alpha_noise_level 0.0  --pre_label True  --rl_seed 99


python3 mujo_20_rl_use_scoring_single_train_sort_uu_robomimic.py  --new_scormodel false  --env 0 --Expert_idx 0 --total_steps 2.5e6  --sort_loss True  --sort_ratio 0.5   --disc_loss_type uu_loss  --data_augm both  --opt_ratio_alpha 0.5  --alpha_noise_level 0.0  --pre_label True  --rl_seed 0
python3 mujo_20_rl_use_scoring_single_train_sort_uu_robomimic.py  --new_scormodel false  --env 1 --Expert_idx 0 --total_steps 2.5e6  --sort_loss True  --sort_ratio 0.5   --disc_loss_type uu_loss  --data_augm both  --opt_ratio_alpha 0.5  --alpha_noise_level 0.0  --pre_label True  --rl_seed 0
python3 mujo_20_rl_use_scoring_single_train_sort_uu_robomimic.py  --new_scormodel false  --env 2 --Expert_idx 0 --total_steps 2.5e6  --sort_loss True  --sort_ratio 0.5   --disc_loss_type uu_loss  --data_augm both  --opt_ratio_alpha 0.5  --alpha_noise_level 0.0  --pre_label True  --rl_seed 0
python3 mujo_20_rl_use_scoring_single_train_sort_uu_robomimic.py  --new_scormodel false  --env 3 --Expert_idx 0 --total_steps 2.5e6  --sort_loss True  --sort_ratio 0.5   --disc_loss_type uu_loss  --data_augm both  --opt_ratio_alpha 0.5  --alpha_noise_level 0.0  --pre_label True  --rl_seed 0


Different alpha:
python3 mujo_20_rl_use_scoring_single_train_sort_uu.py  --new_scormodel false  --env 1 --Expert_idx 0 --total_steps 2.5e6  --sort_loss True  --sort_ratio 0.5   --disc_loss_type uu_loss  --data_augm both  --opt_ratio_alpha 0.25  --alpha_noise_level 0.0  --pre_label True  --rl_seed 99
python3 mujo_20_rl_use_scoring_single_train_sort_uu.py  --new_scormodel false  --env 1 --Expert_idx 0 --total_steps 2.5e6  --sort_loss True  --sort_ratio 0.5   --disc_loss_type uu_loss  --data_augm both  --opt_ratio_alpha 0.5   --alpha_noise_level 0.0  --pre_label True  --rl_seed 99
python3 mujo_20_rl_use_scoring_single_train_sort_uu.py  --new_scormodel false  --env 1 --Expert_idx 0 --total_steps 2.5e6  --sort_loss True  --sort_ratio 0.5   --disc_loss_type uu_loss  --data_augm both  --opt_ratio_alpha 0.75  --alpha_noise_level 0.0  --pre_label True  --rl_seed 99


Noisy alpha:
python3 mujo_20_rl_use_scoring_single_train_sort_uu.py  --new_scormodel false  --env 1 --Expert_idx 0 --total_steps 2.5e6  --sort_loss True  --sort_ratio 0.5   --disc_loss_type uu_loss  --data_augm both  --opt_ratio_alpha 0.5   --alpha_noise_level 0.2  --pre_label True  --rl_seed 99
python3 mujo_20_rl_use_scoring_single_train_sort_uu.py  --new_scormodel false  --env 1 --Expert_idx 0 --total_steps 2.5e6  --sort_loss True  --sort_ratio 0.5   --disc_loss_type uu_loss  --data_augm both  --opt_ratio_alpha 0.5   --alpha_noise_level 0.4  --pre_label True  --rl_seed 99
python3 mujo_20_rl_use_scoring_single_train_sort_uu.py  --new_scormodel false  --env 1 --Expert_idx 0 --total_steps 2.5e6  --sort_loss True  --sort_ratio 0.5   --disc_loss_type uu_loss  --data_augm both  --opt_ratio_alpha 0.5   --alpha_noise_level 0.6  --pre_label True  --rl_seed 99



Different sort_loss ratio:
python3 mujo_20_rl_use_scoring_single_train_sort_uu.py  --new_scormodel false  --env 1 --Expert_idx 0 --total_steps 2.5e6  --sort_loss True  --sort_ratio 0.25   --disc_loss_type uu_loss  --data_augm both  --opt_ratio_alpha 0.5  --alpha_noise_level 0.0  --pre_label True  --rl_seed 99
python3 mujo_20_rl_use_scoring_single_train_sort_uu.py  --new_scormodel false  --env 1 --Expert_idx 0 --total_steps 2.5e6  --sort_loss True  --sort_ratio 0.75   --disc_loss_type uu_loss  --data_augm both  --opt_ratio_alpha 0.5  --alpha_noise_level 0.0  --pre_label True  --rl_seed 99


python3 mujo_20_rl_use_scoring_single_train_sort_uu.py  --new_scormodel false  --env 1 --Expert_idx 2 --total_steps 2.5e6  --sort_loss True  --sort_ratio 0.25   --disc_loss_type uu_loss  --data_augm both  --opt_ratio_alpha 0.5  --alpha_noise_level 0.0  --pre_label True  --rl_seed 0
python3 mujo_20_rl_use_scoring_single_train_sort_uu.py  --new_scormodel false  --env 1 --Expert_idx 2 --total_steps 2.5e6  --sort_loss True  --sort_ratio 0.75   --disc_loss_type uu_loss  --data_augm both  --opt_ratio_alpha 0.5  --alpha_noise_level 0.0  --pre_label True  --rl_seed 0


without sort_loss:
python3 mujo_20_rl_use_scoring_single_train_sort_uu.py  --new_scormodel false  --env 1 --Expert_idx 0 --total_steps 2.5e6  --sort_loss False  --sort_ratio 0.5   --disc_loss_type uu_loss  --data_augm both  --opt_ratio_alpha 0.5  --alpha_noise_level 0.0  --pre_label True  --rl_seed 99


Not Prelabeled: 
python3 mujo_20_rl_use_scoring_single_train_sort_uu.py  --new_scormodel false  --env 1 --Expert_idx 0 --total_steps 2.5e6  --sort_loss True  --sort_ratio 0.5   --disc_loss_type uu_loss  --data_augm both  --opt_ratio_alpha 0.5  --alpha_noise_level 0.0  --pre_label False  --rl_seed 99


"""


parser = argparse.ArgumentParser(description=None)
parser.add_argument('--rl_seed', type=int, default='10', help='')
parser.add_argument('--Expert_idx', type=list, default='10', help='')
parser.add_argument('--data_augm', type=str, default='False', help='  "mixup" or "normal" or "both"  ')
parser.add_argument('--new_scormodel', type=str, default='false', help='true, false')
parser.add_argument('--env', type=int, default='9', help='0-4 env list')
parser.add_argument('--sort_loss', type=str, default='None', help=' True or False')
parser.add_argument('--total_steps', type=float, default='None', help=' total steps of interaction with the environment') 
parser.add_argument('--disc_loss_type', type=str, default='None', help=' "uu_loss" or "binary_loss" ') # "binary_loss" or "uu_loss"
parser.add_argument('--opt_ratio_alpha', type=str, default='0.5', help=' alpha value for the optimal ratio - 0.25 or 0.5 or 0.75')
parser.add_argument('--alpha_noise_level', type=float, default='0.0', help=' noise level for the alpha value - 0 or 0.2 or 0.4 or 0.6')
parser.add_argument('--sort_ratio', type=float, default='0.5', help=' ratio for the sorting - 0.25 or 0.5 or 0.75')
parser.add_argument('--pre_label', type=str, default='True', help=' True or False')



args = parser.parse_args()

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

if not args.disc_loss_type == "uu_loss" or args.disc_loss_type == "binary_loss":
    raise ValueError("disc_loss_type should be 'uu_loss' or 'binary_loss' ")

env_name_list = [ "Lift",  "PickPlaceCan",  "NutAssemblySquare",  "TwoArmTransport", ]
normalize = False 
env_name = env_name_list[args.env]
scoring_update_itrs = 500

IL_params = json.load(open(Path("./RL_parameters/IL_parameters.json")))
scoring_pretrained_epochs = "epochs_"+str(IL_params[env_name[:]]["scoring_model_pre_iters"]) 

opt_ratio_alpha = args.opt_ratio_alpha
# check if opt_ratio_alpha = 0.25 or 0.5 or 0.75 or 1.0 or 0.0, ohterwise ValueError
if opt_ratio_alpha != "0.25" and opt_ratio_alpha != "0.5" and opt_ratio_alpha != "0.75" and opt_ratio_alpha != "1.0" and opt_ratio_alpha != "0.0":
  raise ValueError("opt_ratio_alpha should be 0.25 or 0.5 or 0.75 or 1.0 or 0.0")

sort_ratio = args.sort_ratio
sort_loss = true_false_convert(args.sort_loss)


seed_value = args.rl_seed
expert_idxs = args.Expert_idx
print("expert_idxs", expert_idxs)
RL_alg = "SAC"

if sort_loss:
    sort_loss_str = "_sort"
else:
    sort_loss_str = ""


new_scormodel = true_false_convert(args.new_scormodel)
if new_scormodel:
    models_dir = "../results/logs_scoring_sin-train"+sort_loss_str + "_" + args.disc_loss_type + "_new_scoring_noReplacing/"
    if opt_ratio_alpha != "0.5":
        models_dir = models_dir + "alpha_"+opt_ratio_alpha[0]+opt_ratio_alpha[2:]+"/"
    if opt_ratio_alpha == "0.5" and args.alpha_noise_level != 0:
        models_dir = models_dir + "alpha_"+opt_ratio_alpha[0]+opt_ratio_alpha[2:]+"_noisy_"+str(args.alpha_noise_level)[0]+ str(args.alpha_noise_level)[2:]+"/"
    if args.pre_label == "False" or args.pre_label == "false":
        models_dir = models_dir + "noPreLabel/"
else:
    models_dir = "../results/logs_scoring_sin-train"+sort_loss_str + "_" + args.disc_loss_type + "_noReplacing/"
    if opt_ratio_alpha != "0.5":
        models_dir = models_dir + "alpha_"+opt_ratio_alpha[0]+opt_ratio_alpha[2:]+"/"
    if opt_ratio_alpha == "0.5" and args.alpha_noise_level != 0:
        models_dir = models_dir + "alpha_"+opt_ratio_alpha[0]+opt_ratio_alpha[2:]+"_noisy_"+str(args.alpha_noise_level)[0]+ str(args.alpha_noise_level)[2:]+"/"  
    if args.pre_label == "False" or args.pre_label == "false":
        models_dir = models_dir + "noPreLabel/"
os.makedirs(models_dir, exist_ok=True)


if new_scormodel:
    suffix = "_gail_new_" + str(scoring_update_itrs) + "_itrs_scoring"
else:
    suffix = "_gail_finetune_" + str(scoring_update_itrs) + "_itrs_scoring"
    if sort_loss:
        suffix = suffix + sort_loss_str + "_" + str(sort_ratio)[0] + str(sort_ratio)[2:]
suffix = suffix + "_stepnum" + str(int(args.total_steps/1e5)) + "e5_" + args.data_augm + "_" + args.disc_loss_type

scoring_model_dir = "../results/scoring_model_noReplacing/"
name_suffix = "multi_frame_1_optStart0"


save_dir = models_dir+env_name+"_scoring_co_train_"+RL_alg+"_NetAgentSeed_"+str(expert_idxs[0])+"-"+str(expert_idxs[-1])+"_"+scoring_pretrained_epochs+"_RLseed_"+str(seed_value)+ "_" +  "-un"+suffix+"/"

# Load two scoring models for co-training --------------------------------------
scoring_dir = scoring_model_dir +env_name+"_ExpertSeed_"+str(expert_idxs[0])+"-"+str(expert_idxs[0])+"_scoring_model-un_"+name_suffix+ "_alpha_"+opt_ratio_alpha[0]+opt_ratio_alpha[2:]+ "_newneg/"

if opt_ratio_alpha == "0.5" and args.alpha_noise_level != 0:
    # ablation test - scoring model trained using noisy opt_ratio (alpha)
    scoring_dir = scoring_model_dir +env_name+"_ExpertSeed_"+str(expert_idxs[0])+"-"+str(expert_idxs[0])+"_scoring_model-un_"+name_suffix+"_alpha_"+opt_ratio_alpha[0]+opt_ratio_alpha[2:]+"_noisy_"+str(args.alpha_noise_level)[0]+ str(args.alpha_noise_level)[2:]+"_newneg/"

if opt_ratio_alpha == "0.75" or opt_ratio_alpha == "0.25":
    scoring_dir = scoring_model_dir +env_name+"_ExpertSeed_"+str(expert_idxs[0])+"-"+str(expert_idxs[0])+"_scoring_model-un_"+name_suffix+"_alpha_"+opt_ratio_alpha[0]+opt_ratio_alpha[2:]+"_newneg/"
scoring_model_path = scoring_dir + "scroring_model_"+env_name+"_"+scoring_pretrained_epochs + "_NetSeed_"
net_idx = expert_idxs[-1]
scoring_model_A_path = scoring_model_path +str(net_idx) + ".pkl"


scoring_dir = scoring_model_dir +env_name+"_ExpertSeed_"+str(expert_idxs[-1])+"-"+str(expert_idxs[-1])+"_scoring_model-un_"+name_suffix + "_alpha_"+opt_ratio_alpha[0]+opt_ratio_alpha[2:]+ "_newneg/"

if opt_ratio_alpha == "0.5" and args.alpha_noise_level != 0:
    # ablation test - scoring model trained using noisy opt_ratio (alpha)
    scoring_dir = scoring_model_dir +env_name+"_ExpertSeed_"+str(expert_idxs[-1])+"-"+str(expert_idxs[-1])+"_scoring_model-un_"+name_suffix+"_alpha_"+opt_ratio_alpha[0]+opt_ratio_alpha[2:]+"_noisy_"+str(args.alpha_noise_level)[0]+ str(args.alpha_noise_level)[2:]+"_newneg/"

if opt_ratio_alpha == "0.75" or opt_ratio_alpha == "0.25":
    scoring_dir = scoring_model_dir +env_name+"_ExpertSeed_"+str(expert_idxs[-1])+"-"+str(expert_idxs[-1])+"_scoring_model-un_"+name_suffix+"_alpha_"+opt_ratio_alpha[0]+opt_ratio_alpha[2:]+"_newneg/"
net_idx = expert_idxs[-1]
scoring_model_path = scoring_dir + "scroring_model_"+env_name+"_"+scoring_pretrained_epochs+"_NetSeed_"
scoring_model_B_path = scoring_model_path +str(net_idx) + ".pkl"


with open(scoring_model_A_path, 'rb') as f:
    scoring_model_A = pickle.load(f)

print("scoring_model_A_path", scoring_model_A_path)
print("scoring_model_B_path", scoring_model_B_path)

with open(scoring_model_B_path, 'rb') as f:
    scoring_model_B = pickle.load(f)

print("scoring_model_A s max", scoring_model_A.input_scaler_s.data_max_)
print("scoring_model_A s min", scoring_model_A.input_scaler_s.data_min_)
print("scoring_model_A a max", scoring_model_A.input_scaler_a.data_max_)
print("scoring_model_A a min", scoring_model_A.input_scaler_a.data_min_)
print("")

print("scoring_model_B s max", scoring_model_B.input_scaler_s.data_max_)
print("scoring_model_B s min", scoring_model_B.input_scaler_s.data_min_)
print("scoring_model_B a max", scoring_model_B.input_scaler_a.data_max_)
print("scoring_model_B a min", scoring_model_B.input_scaler_a.data_min_)


# Create discriminator model --------------------------------------
dataset_path  = "../robomimic_datasets/"+ env_name +"/mh/low_dim_v141.hdf5"
dataset = h5py.File(dataset_path, 'r')

state_shape = scoring_model_A.input_scaler_s.data_max_.shape[0]
action_shape = scoring_model_A.input_scaler_a.data_max_.shape[0]



priors_class = np.random.rand(20)
Pi = np.random.rand(20)
Pi_test = 0.5

scoring_net_new = Scoring_model_net_multiFrame(priors_class, Pi, Pi_test, state_shape, action_shape, frame_num = 1) # scorescorescore

if new_scormodel or opt_ratio_alpha == "0.0" or opt_ratio_alpha == "1.0":
# there is no need to learn / pretrain a scoring model, if the alpha is 0.0 or 1.0
    scoring_model_A.net = scoring_net_new
    scoring_model_B.net = scoring_net_new
    print("")
    print("NEW scoring model is used")
    print("")




# Create customized environment --------------------------------------
from utils_robomimic_gym_wrapper import RobomimicGymWrapper
import robosuite as suite



if env_name == "Lift" or env_name == "PickPlaceCan" or env_name == "NutAssemblySquare":
    robots = ["Panda"]
    horizon = 1000

elif env_name == "TwoArmTransport":
    robots = ["Panda", "Panda"]
    horizon = 2000

env_make = suite.make(env_name,
                 robots=robots,
                 use_camera_obs = False,
                 has_renderer = True,
                 )

env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)
env = EnvUtils.create_env_from_metadata(
    env_meta=env_meta,
    env_name=env_name, 
    render=False, 
    render_offscreen=False,
    use_image_obs=False, 
)

# add some attributes to the env to make it compatible with the customized GymWrapper
env.robots = env_make.robots
env.reward_scale = 1.0
env.use_object_obs = True
env.use_camera_obs = False
env.action_spec = (np.ones(env.action_dimension) * -1, np.ones(env.action_dimension))

env = RobomimicGymWrapper(env, horizon = horizon)

env = Custom_Env(env_name, scoring_model_A, 
                 input_scaler_s=scoring_model_A.input_scaler_s, 
                 input_scaler_a=scoring_model_A.input_scaler_a,
                 normalize = normalize, env=env
)


env = DummyVecEnv([lambda: env])
env.seed(seed_value)


# Create RL agent --------------------------------------
RL_params = json.load(open(Path("./RL_parameters/mujoco_params.json")))

if RL_alg == "SAC":
    rl_model = SAC('MlpPolicy', env, verbose=1,
                use_sde = False,
                learning_starts=100,
                batch_size=1024,
                learning_rate=2e-3,
                seed=seed_value, device=device,
                )

eval_freq = 15000
check_points_freq = 1 * eval_freq

checkpoint_callback = CheckpointCallback(
  save_freq=int(check_points_freq/eval_freq),
  save_path=save_dir + "checkpoints/",
  name_prefix=env_name+"_"+RL_alg+"_models",
  save_replay_buffer=False,
  save_vecnormalize=True,
  verbose=1,
)

eval_callback = EvalCallback(env, 
                             callback_after_eval = checkpoint_callback,
                             best_model_save_path=save_dir,
                             log_path=save_dir, 
                             eval_freq=eval_freq,
                             deterministic=True, 
                             render=False,
                             verbose = 1)

# Load demonstration data --------------------------------------
trajs_dir = "../results/opt_nonopt_trajs_noReplacing/"
expert_seed = args.Expert_idx[0]
saved_trajs_path = trajs_dir + "opt_nonopt_trajs_"+env_name+"_ExpertSeed_"+str(expert_seed)+"_NetSeed"+str(expert_seed)+"_"+scoring_pretrained_epochs+"_pseudo_labeled"+"_alpha_"+opt_ratio_alpha[0]+opt_ratio_alpha[2:]+".pkl"
if opt_ratio_alpha == "0.5" and args.alpha_noise_level != 0:
    saved_trajs_path = trajs_dir + "opt_nonopt_trajs_"+env_name+"_ExpertSeed_"+str(expert_seed)+"_NetSeed"+str(expert_seed)+"_"+scoring_pretrained_epochs+"_pseudo_labeled_alpha_"+opt_ratio_alpha[0]+opt_ratio_alpha[2:]+"_noisy_"+str(args.alpha_noise_level)[0]+ str(args.alpha_noise_level)[2:]+".pkl"

if opt_ratio_alpha == "0.75" or opt_ratio_alpha == "0.25":
    # ablation test - using demons with different average opt_ratio 
    saved_trajs_path = trajs_dir + "opt_nonopt_trajs_"+env_name+"_ExpertSeed_"+str(expert_seed)+"_NetSeed"+str(expert_seed)+"_"+scoring_pretrained_epochs+"_pseudo_labeled_alpha_"+opt_ratio_alpha[0]+opt_ratio_alpha[2:]+".pkl"

if opt_ratio_alpha == "0.0" or opt_ratio_alpha == "1.0":
    # ablation test - using demons with different average opt_ratio 
    saved_trajs_path = trajs_dir + "opt_nonopt_trajs_"+env_name+"_sacExpertSeed_"+str(expert_seed)+"_noisy_opt"+"_newneg_alpha_"+opt_ratio_alpha[0]+opt_ratio_alpha[2:]+ ".pkl"
    demon_normalized = False
else:
    demon_normalized = True


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


demonstrations_exp = Demonstration_Buffer(saved_trajs_path, device=device,  single_frame=False)


# Create RL_Scoring_Co-Train agent --------------------------------------
if args.disc_loss_type == "uu_loss":
    lr_scoring = 1e-4
elif args.disc_loss_type == "binary_loss":
    lr_scoring = 1e-4

rl_scoring_train = RL_Scoring(demonstrations_exp, batch_size=1024, device=device,
                                        rl_algo=rl_model,  rl_callback = eval_callback,
                                        rl_train_timesteps=int(1.5e4),
                                         
                                        scoring_model_A=scoring_model_A, 
                                        scoring_model_B=scoring_model_B,
                                        n_scoring_updates_per_round=scoring_update_itrs, 
                                        lr_scoring=lr_scoring,

                                        scaler_state=scoring_model_A.input_scaler_s,
                                        scaler_action=scoring_model_A.input_scaler_a, 

                                        sort_ratio = sort_ratio,
                                        sort_loss = sort_loss,

                                        data_augm = args.data_augm,

                                        loss_type = args.disc_loss_type,

                                        demon_normalized = demon_normalized,

                                        normalize = normalize,
                                         )


rl_scoring_train.train(int(args.total_steps), )


# EVAL best & checkpoints models --------------------------------------
model_dir = save_dir
file_suffix_name = "after_training"
env.envs[0].set_modify_reward(False)
eval_best_checkpoints_models(env, env_name, RL_alg, seed_value, model_dir, file_suffix_name, eval_best=False)


# save the trained rl_scoring_train model as pkl file
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


file_suffix_name = "evaluation_during_training" 
plot_results_npz(env_name, save_dir, file_suffix_name)

plot_discriminator_acc_loss(save_dir+"rl_scoring_co_train_model.pkl", env_name, save_dir, "after_training")
