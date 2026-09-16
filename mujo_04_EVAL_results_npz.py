import numpy as np
from pathlib import Path
import argparse
from utils_plot import plot_reward_stepLen


def plot_results_npz(env_name, save_dir, file_suffix_name):
    
    npz_file_path = save_dir + "evaluations.npz"
    print("npz file_path: ", npz_file_path)
    # Load the file
    data = np.load(npz_file_path)
    ep_lengths = data['ep_lengths']
    results = data['results']
    timesteps = data['timesteps']

    # print("ep_lengths: ", ep_lengths)
    # print("results: ", results)
    # print("timesteps: ", timesteps)

    mean_reward_list = []
    std_reward_list = []
    mean_stepLen_list = []
    std_stepLen_list = []

    for i in range(len(results)):
        mean_reward_list.append(np.mean(results[i]))
        std_reward_list.append(np.std(results[i]))
        mean_stepLen_list.append(np.mean(ep_lengths[i]))
        std_stepLen_list.append(np.std(ep_lengths[i]))

    stepnum_list = timesteps
    # print("stepnum_list: ", len(stepnum_list))
    # print("mean_reward_list: ", len(results))

    plot_reward_stepLen(stepnum_list, mean_reward_list, std_reward_list, mean_stepLen_list, std_stepLen_list, env_name, save_dir, file_suffix_name)


if __name__ == '__main__':

    # python mujo_04_EVAL.py --test_idx 0   --env_idx 0
    env_name_list = ["Ant-v4", "HalfCheetah-v4", "Hopper-v4", "Swimmer-v4", "Walker2d-v4"]
    model_dir = "../results/logs_scoring/"
    # seed_value = 1
    RL_alg = "TRPO"
    file_suffix_name = "evaluation_during_training" # for names of file to be saved 

    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--env_idx', type=int, default='10', help='')
    parser.add_argument('--test_idx', type=str, default='10', help='')
    parser.add_argument('--seed', type=int, default='10', help='')
    args = parser.parse_args()
    seed_value = args.seed

    if args.test_idx == "None":
        test_idx = ""
    else:
        test_idx = args.test_idx

    env_name = env_name_list[args.env_idx]
    print("env_name: ", env_name)

    # save_dir = model_dir + env_name + "_" + RL_alg + "_model_seed_"+str(seed_value)+"-un"+str(test_idx)+"/"
    save_dir = model_dir + env_name + "_" + RL_alg + "_model"+"-un_using_"+str(test_idx)+"_RLseed_"+str(seed_value)+"/"

    plot_results_npz(env_name, save_dir, file_suffix_name)