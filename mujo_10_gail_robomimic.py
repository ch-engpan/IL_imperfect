from utils_gail import *
import gymnasium as gym
from sb3_contrib import TRPO, TQC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO, TD3, SAC, DQN
import numpy as np
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import argparse
import torch
import json
from utils_plot import plot_reward_stepLen
from utils_rl import get_TRPO_RL_params, get_TD3_RL_params, get_PPO_RL_params
from mujo_04_EVAL_results_npz import plot_results_npz
from mujo_04_EVAL import eval_best_checkpoints_models
import os
from mujo_10_EVAL_discriminator import plot_discriminator_acc_loss

"""

# 2024.7.2 run - test if work with GAIL
python3 mujo_10_gail_robomimic.py  --agent_idx 0  --total_steps 2.5e6  --env 0  --noisy False --data_augm mixup   --rl_seed 0 

python3 mujo_10_gail_robomimic.py  --agent_idx 0  --total_steps 2.5e6  --env 1  --noisy False --data_augm mixup   --rl_seed 0 


"""
parser = argparse.ArgumentParser(description=None)
parser.add_argument('--rl_seed', type=int, default='10', help='')
parser.add_argument('--agent_idx', type=str, default='10', help='')
parser.add_argument('--noisy', type=str, default='False', help='')
parser.add_argument('--env', type=int, default='9', help='0-4 env list')
parser.add_argument('--data_augm', type=str, default='False', help='  "mixup" or "normal" or "both"  ')
parser.add_argument('--total_steps', type=float, default='None', help=' total steps of interaction with the environment') 
parser.add_argument('--opt_ratio_alpha', type=str, default='0.5', help=' alpha value for the optimal ratio - 0.25 or 0.5 or 0.75 or 1.0 or 0.0')


args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("=============================", device, "=============================")
print("=============================", device, "=============================")

scoring_update_itrs = 500


env_name_list = ["Ant-v4", "HalfCheetah-v4", "Hopper-v4", "Swimmer-v4", "Walker2d-v4"]
env_name_list = [ "Lift",  "PickPlaceCan",  "NutAssemblySquare",  "TwoArmTransport", ]

env_name = env_name_list[args.env]

seed_value = args.rl_seed
expert_seed = args.agent_idx
RL_alg = "SAC"

opt_ratio_alpha = args.opt_ratio_alpha
# check if opt_ratio_alpha = 0.25 or 0.5 or 0.75 or 1.0 or 0.0, ohterwise ValueError
if opt_ratio_alpha != "0.25" and opt_ratio_alpha != "0.5" and opt_ratio_alpha != "0.75" and opt_ratio_alpha != "1.0" and opt_ratio_alpha != "0.0" and opt_ratio_alpha != "0.9" and opt_ratio_alpha != "0.8":
  raise ValueError("opt_ratio_alpha should be 0.25 or 0.5 or 0.75 or 1.0 or 0.0 or 0.9 or 0.8")

models_dir = "../results/logs_gail_noReplacing/"
if opt_ratio_alpha != "0.5":
  models_dir = models_dir + "alpha_"+opt_ratio_alpha[0]+opt_ratio_alpha[2:] + "/"

os.makedirs(models_dir, exist_ok=True)
trajs_dir = "../results/opt_nonopt_trajs_noReplacing/"
suffix = "_newneg" + "_stepnum" + str(int(args.total_steps/1e5)) + "e5_" + args.data_augm


if args.noisy == "True":
  save_dir = models_dir+env_name+"_GAIL_"+RL_alg+"_AgentSeed_"+str(expert_seed)+"_RLseed_"+str(seed_value)+"_"+str(scoring_update_itrs)+"_iters"+"_noisy-un"+suffix+"/"
  saved_trajs_path = trajs_dir + "opt_nonopt_trajs_"+env_name+"_robomimicSeed_"+str(expert_seed)+"_noisy_opt"+"_newneg"+".pkl"

  if opt_ratio_alpha != "0.5":
    # ablation test - using demons with different average opt_ratio 
    saved_trajs_path = trajs_dir + "opt_nonopt_trajs_"+env_name+"_robomimicSeed_"+str(expert_seed)+"_noisy_opt"+"_newneg_alpha_"+opt_ratio_alpha[0]+opt_ratio_alpha[2:]+".pkl"


elif args.noisy == "False":
  save_dir = models_dir+env_name+"_GAIL_"+RL_alg+"_AgentSeed_"+str(expert_seed)+"_RLseed_"+str(seed_value)+"_"+str(scoring_update_itrs)+"_iters"+"-un"+suffix+"/"
  saved_trajs_path = trajs_dir + "opt_nonopt_trajs_"+env_name+"_robomimicSeed_"+str(expert_seed)+"_opt_newneg"+".pkl"

  if opt_ratio_alpha != "0.5":
    # ablation test - using demons with different average opt_ratio 
    saved_trajs_path = trajs_dir + "opt_nonopt_trajs_"+env_name+"_robomimicSeed_"+str(expert_seed)+"_opt_newneg_alpha_"+opt_ratio_alpha[0]+opt_ratio_alpha[2:]+".pkl"


# Load demonstration data --------------------------------------
demonstrations_exp = Demonstration_Buffer(saved_trajs_path, device=device)

states, actions,_ = demonstrations_exp.sample(10)
print(states.requires_grad)
print(actions.shape)

# Create discriminator model --------------------------------------
if env_name == "Lift":
    state_shape = 19
    action_shape = 7
elif env_name == "PickPlaceCan":
    state_shape = 23
    action_shape = 7
elif env_name == "NutAssemblySquare":
    state_shape = 23
    action_shape = 7
elif env_name == "TwoArmTransport":
    state_shape = 59
    action_shape = 14


scoring_model = GAIL_Discrim(state_shape, action_shape,
)

# Create customized environment --------------------------------------

from utils_robomimic_gym_wrapper import RobomimicGymWrapper
import robosuite as suite
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import h5py

dataset_path  = "../robomimic_datasets/"+ env_name +"/mh/low_dim_v141.hdf5"
dataset = h5py.File(dataset_path, 'r')



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
normalize = False 
env = Custom_Env(env_name, scoring_model, 
                 input_scaler_s=demonstrations_exp.scaler_s,
                 input_scaler_a=demonstrations_exp.scaler_a,
                 normalize = normalize, env=env
)


env = DummyVecEnv([lambda: env])
env.seed(seed_value)


# Create RL agent --------------------------------------
RL_params = json.load(open(Path("./RL_parameters/mujoco_params.json")))

if RL_alg == "SAC":
    rl_model = SAC('MlpPolicy', env, verbose=0,
                use_sde = False,
                learning_starts=100,
                batch_size=1024,
                learning_rate=2e-3,
                seed=seed_value, device=device)

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

# Create GAIL agent --------------------------------------
Gail = GAIL(demonstrations_exp, demo_batch_size=1024, device=device,
            
            discrim=scoring_model, 
            n_disc_updates_per_round=scoring_update_itrs, lr_disc=1e-4,

            gen_algo=rl_model, 
            gen_train_timesteps=int(1.5e4), gen_callback=eval_callback,
            
            scaler_state=demonstrations_exp.scaler_s, 
            scaler_action=demonstrations_exp.scaler_a,

            data_augm = args.data_augm,
            )

Gail.train(int(args.total_steps))


# EVAL best & checkpoints models --------------------------------------
model_dir = save_dir
file_suffix_name = "after_training"
env.envs[0].set_modify_reward(False)
eval_best_checkpoints_models(env, env_name, RL_alg, seed_value, model_dir, file_suffix_name, eval_best=False)


# save the trained model as pkl file
with open(save_dir+"gail_model.pkl", 'wb') as f:
    
    pickle.dump({   'disc_acc_agent': Gail.acc_agent_list,
                    'disc_acc_expert': Gail.acc_expert_list,
                    'disc_loss': Gail.loss_disc_list,

                    'discrim': Gail.discrim,
        
                    's_max': Gail.s_max,
                    's_min': Gail.s_min,
                    'a_max': Gail.a_max,
                    'a_min': Gail.a_min,
                 }, f)


# plot the results
file_suffix_name = "evaluation_during_training" 
plot_results_npz(env_name, save_dir, file_suffix_name)

plot_discriminator_acc_loss(save_dir+"gail_model.pkl", env_name, save_dir, "after_training")
