import numpy as np
from pathlib import Path
import argparse
from utils_plot import plot_reward_stepLen
import matplotlib.pyplot as plt
import pickle

def plot_acc_loss(stepnum_list, disc_acc_agent, disc_acc_expert, disc_loss, env_name, save_dir, file_suffix_name):
    fig, axs = plt.subplots(4, 1, figsize=(8, 8))

    # First subplot: 
    axs[0].set_title('Discriminator acc - Agent vs. Expert ---' + env_name)
    axs[0].plot(stepnum_list, disc_acc_agent, label='Agent')
    axs[0].plot(stepnum_list, disc_acc_expert, label='Expert')
    axs[0].legend()
    axs[0].set_xlabel('Timesteps')
    axs[0].set_ylabel('Accuracy')
    axs[0].grid(True)

    # Second subplot (enlarged): 
    start_steps = int(0.95*len(stepnum_list))
    axs[1].set_title('Discriminator acc - Agent vs. Expert ---' + env_name)
    axs[1].plot(stepnum_list[start_steps:], disc_acc_agent[start_steps:], label='Agent')
    axs[1].plot(stepnum_list[start_steps:], disc_acc_expert[start_steps:], label='Expert')
    axs[1].legend()
    axs[1].set_xlabel('Timesteps')
    axs[1].set_ylabel('Accuracy')
    axs[1].grid(True)

    # Third subplot: 
    axs[2].set_title('Discriminator loss ---' + env_name)
    disc_loss = np.array(disc_loss)
    axs[2].plot(stepnum_list, disc_loss[:,0], label='Loss')
    axs[2].plot(stepnum_list, disc_loss[:,1], label='Loss agent')
    axs[2].plot(stepnum_list, disc_loss[:,2], label='Loss expert')
    if disc_loss.shape[1] > 3:
        axs[2].plot(stepnum_list, disc_loss[:,3], label='Loss penalty')
    axs[2].legend()
    axs[2].set_xlabel('Timesteps')
    axs[2].set_ylabel('Loss')
    axs[2].grid(True)

    # Forth subplot (enlarged):
    axs[3].set_title('Discriminator loss ---' + env_name)
    
    axs[3].plot(stepnum_list[start_steps:], disc_loss[:,0][start_steps:], label='Loss')
    axs[3].plot(stepnum_list[start_steps:], disc_loss[:,1][start_steps:], label='Loss agent')
    axs[3].plot(stepnum_list[start_steps:], disc_loss[:,2][start_steps:], label='Loss expert')
    if disc_loss.shape[1] > 3:
        axs[3].plot(stepnum_list[start_steps:], disc_loss[:,3][start_steps:], label='Loss penalty')
    axs[3].legend()
    axs[3].set_xlabel('Timesteps')
    axs[3].set_ylabel('Loss')
    axs[3].grid(True)

    plt.tight_layout()
    save_dir = Path(save_dir + "/Discriminator_acc_loss_plots_" + file_suffix_name + ".pdf")
    print("save_dir: ", save_dir)
    plt.savefig(save_dir)
    # plt.show()
    plt.close()

def plot_discriminator_acc_loss(model_path, env_name, save_dir, file_suffix_name):
    # load the trained model as pkl file
    with open (model_path, 'rb') as f:
        data = pickle.load(f)
        disc_acc_agent = data['disc_acc_agent']
        disc_acc_expert = data['disc_acc_expert']
        disc_loss = data['disc_loss']
  
    stepnum_list = np.arange(len(disc_acc_agent))
    
    # pick up data every 5 steps
    step_gap = 5
    disc_acc_agent = disc_acc_agent[::step_gap]
    disc_acc_expert = disc_acc_expert[::step_gap]
    disc_loss = disc_loss[::step_gap]
    stepnum_list = stepnum_list[::step_gap]

    print("disc_acc_agent: ", np.array(disc_acc_agent).shape)
    print("disc_acc_expert: ", np.array(disc_acc_expert).shape)
    print("disc_loss: ", np.array(disc_loss).shape)

    plot_acc_loss(stepnum_list, disc_acc_agent, disc_acc_expert, disc_loss, env_name, save_dir, file_suffix_name)



def plot_uuLearn_num(model_path, env_name, save_dir, file_suffix_name):
    # load the trained model as pkl file

    """
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
    """


    with open (model_path, 'rb') as f:
        data = pickle.load(f)
        expDemo_opt_num = data['expDemo_opt_num']
        TP_exp = data['TP_exp']
        TN_exp = data['TN_exp']
        FP_exp = data['FP_exp']
        FN_exp = data['FN_exp']
        recall_exp = data['recall_exp']
        unprecision_exp = data['unprecision_exp']
        true_exp_P_num = data['true_exp_P_num']
        true_exp_N_num = data['true_exp_N_num']
        agt_N2P_num = data['agt_N2P_num']
        exp_N2P_num = data['exp_N2P_num']
        exp_P2N_num = data['exp_P2N_num']
        relabel_earlyStop = data['relabel_earlyStop']

    stepnum_list = np.arange(len(expDemo_opt_num))


    # Determine number of subplots based on available data
    num_plots = 11 + 2
    fig, axs = plt.subplots(num_plots, 1, figsize=(8, 2*num_plots))

    axs[0].set_title('Expert Demo opt num ---' + env_name)
    axs[0].plot(stepnum_list, expDemo_opt_num, label='Expert Demo opt num')
    axs[0].set_xlabel('Timesteps')
    axs[0].set_ylabel('Num')
    axs[0].grid(True)
    axs[0].legend()

    axs[1].set_title('Expert True Positive num ---' + env_name)
    axs[1].plot(stepnum_list, TP_exp, label='Expert True Positive num')
    axs[1].set_xlabel('Timesteps')
    axs[1].set_ylabel('Num')
    axs[1].grid(True)

    axs[1].plot(stepnum_list, true_exp_P_num, label='Expert True Positive num (ground truth)')
    axs[1].set_xlabel('Timesteps')
    axs[1].set_ylabel('Num')
    axs[1].grid(True)
    axs[1].legend()

    axs[2].set_title('Expert True Negative num ---' + env_name)
    axs[2].plot(stepnum_list, TN_exp, label='Expert True Negative num')
    axs[2].set_xlabel('Timesteps')
    axs[2].set_ylabel('Num')
    axs[2].grid(True)

    axs[2].plot(stepnum_list, true_exp_N_num, label='Expert True Negative num (ground truth)')
    axs[2].set_xlabel('Timesteps')
    axs[2].set_ylabel('Num')
    axs[2].grid(True)
    axs[2].legend()

    axs[3].set_title('Expert False Positive num ---' + env_name)
    axs[3].plot(stepnum_list, FP_exp, label='Expert False Positive num')
    axs[3].set_xlabel('Timesteps')
    axs[3].set_ylabel('Num')
    axs[3].grid(True)
    axs[3].legend()

    axs[4].set_title('Expert False Negative num ---' + env_name)
    axs[4].plot(stepnum_list, FN_exp, label='Expert False Negative num')
    axs[4].set_xlabel('Timesteps')
    axs[4].set_ylabel('Num')
    axs[4].grid(True)
    axs[4].legend()

    axs[5].set_title('Expert recall (all real P, how much labeled as P) ---' + env_name)
    axs[5].plot(stepnum_list, recall_exp, label='Expert recall')
    axs[5].set_xlabel('Timesteps')
    axs[5].set_ylabel('Num')
    axs[5].grid(True)
    axs[5].legend()

    axs[6].set_title('Expert unprecision (all labeled P, how much actually N) ---' + env_name)
    axs[6].plot(stepnum_list, unprecision_exp, label='Expert unprecision')
    axs[6].set_xlabel('Timesteps')
    axs[6].set_ylabel('Num')
    axs[6].grid(True)
    axs[6].legend()

    axs[7].set_title('Agent N2P num ---' + env_name)
    axs[7].plot(stepnum_list, agt_N2P_num, label='Agent N2P num')
    axs[7].set_xlabel('Timesteps')
    axs[7].set_ylabel('Num')
    axs[7].grid(True)
    axs[7].legend()

    axs[8].set_title('Expert N2P num ---' + env_name)
    axs[8].plot(stepnum_list, exp_N2P_num, label='Expert N2P num')
    axs[8].set_xlabel('Timesteps')
    axs[8].set_ylabel('Num')
    axs[8].grid(True)
    axs[8].legend()

    axs[9].set_title('Expert N2P num ---' + env_name)
    axs[9].plot(stepnum_list, exp_N2P_num, label='Expert N2P num')
    axs[9].set_xlabel('Timesteps')
    axs[9].set_ylabel('Num')
    axs[9].grid(True)
    axs[9].legend()
    # fix y axis range from 0 to 50
    axs[9].set_ylim(0, 50)

    axs[10].set_title('Expert P2N num ---' + env_name)
    axs[10].plot(stepnum_list, exp_P2N_num, label='Expert P2N num')
    axs[10].set_xlabel('Timesteps')
    axs[10].set_ylabel('Num')
    axs[10].grid(True)
    axs[10].legend()

    axs[11].set_title('Expert P2N num ---' + env_name)
    axs[11].plot(stepnum_list, exp_P2N_num, label='Expert P2N num')
    axs[11].set_xlabel('Timesteps')
    axs[11].set_ylabel('Num')
    axs[11].grid(True)
    axs[11].legend()
    # fix y axis range from 0 to 50
    axs[11].set_ylim(0, 50)

    axs[12].set_title('Relabel Early Stop ---' + env_name)
    axs[12].plot(stepnum_list, relabel_earlyStop, label='Relabel Early Stop')
    axs[12].set_xlabel('Timesteps')
    axs[12].set_ylabel('Num')
    axs[12].grid(True)
    axs[12].legend()


    # # Add replay buffer size plots if data is available
    # if has_replay_buffer_data:
    #     replay_stepnum_list = np.arange(len(agent_replay_buffer_opt_size))
        
    #     axs[5].set_title('Agent Replay Buffer Opt Size ---' + env_name)
    #     axs[5].plot(replay_stepnum_list, agent_replay_buffer_opt_size, label='Opt Buffer Size', color='green')
    #     axs[5].set_xlabel('Timesteps')
    #     axs[5].set_ylabel('Buffer Size')
    #     axs[5].grid(True)

    #     axs[5].set_title('Agent Replay Buffer NonOpt Size ---' + env_name)
    #     axs[5].plot(replay_stepnum_list, agent_replay_buffer_nonopt_size, label='NonOpt Buffer Size', color='red')
    #     axs[5].set_xlabel('Timesteps')
    #     axs[5].set_ylabel('Buffer Size')
    #     axs[5].grid(True)
    #     axs[5].legend()

    plt.tight_layout()
    save_dir = Path(save_dir + "/uu_learn_plots_" + file_suffix_name + ".pdf")
    print("save_dir: ", save_dir)
    plt.savefig(save_dir)
    plt.close()



def plot_selflabel_num(model_path, env_name, save_dir, file_suffix_name):
    # load the trained model as pkl file
    with open (model_path, 'rb') as f:
        data = pickle.load(f)
        exp_N2P_num = data['exp_N2P_num']
        exp_P2N_num = data['exp_P2N_num']
        expDemo_opt_num = data['expDemo_opt_num']
        expDemo_nonopt_num = data['expDemo_nonopt_num']
        agt_N2P_num = data['agt_N2P_num']

    stepnum_list = np.arange(len(exp_N2P_num))

    # Check if replay buffer size data exists (only available when train_method == 1)
    try:
        # Try to load finegrained data file which has replay buffer sizes
        with open(model_path, 'rb') as f:
            finegrained_data = pickle.load(f)
            agent_replay_buffer_opt_size = finegrained_data.get('agent_replay_buffer_opt_size', [])
            agent_replay_buffer_nonopt_size = finegrained_data.get('agent_replay_buffer_nonopt_size', [])
        has_replay_buffer_data = len(agent_replay_buffer_opt_size) > 0
    except:
        has_replay_buffer_data = False
        agent_replay_buffer_opt_size = []
        agent_replay_buffer_nonopt_size = []

    # Determine number of subplots based on available data
    num_plots = 6 if has_replay_buffer_data else 5
    fig, axs = plt.subplots(num_plots, 1, figsize=(8, 2*num_plots))

    axs[0].set_title('Expert N2P num ---' + env_name)
    axs[0].plot(stepnum_list, exp_N2P_num, label='Expert N2P num')
    axs[0].set_xlabel('Timesteps')
    axs[0].set_ylabel('Num')
    axs[0].grid(True)

    axs[1].set_title('Expert P2N num ---' + env_name)
    axs[1].plot(stepnum_list, exp_P2N_num, label='Expert P2N num')
    axs[1].set_xlabel('Timesteps')
    axs[1].set_ylabel('Num')
    axs[1].grid(True)

    axs[2].set_title('Expert Demo opt num ---' + env_name)
    axs[2].plot(stepnum_list, expDemo_opt_num, label='Expert Demo opt num')
    axs[2].set_xlabel('Timesteps')
    axs[2].set_ylabel('Num')
    axs[2].grid(True)

    axs[3].set_title('Expert Demo nonopt num ---' + env_name)
    axs[3].plot(stepnum_list, expDemo_nonopt_num, label='Expert Demo nonopt num')
    axs[3].set_xlabel('Timesteps')
    axs[3].set_ylabel('Num')
    axs[3].grid(True)

    axs[4].set_title('Agent N2P num ---' + env_name)
    axs[4].plot(stepnum_list, agt_N2P_num, label='Agent N2P num')
    axs[4].set_xlabel('Timesteps')
    axs[4].set_ylabel('Num')
    axs[4].grid(True)

    # Add replay buffer size plots if data is available
    if has_replay_buffer_data:
        replay_stepnum_list = np.arange(len(agent_replay_buffer_opt_size))
        
        axs[5].set_title('Agent Replay Buffer Opt Size ---' + env_name)
        axs[5].plot(replay_stepnum_list, agent_replay_buffer_opt_size, label='Opt Buffer Size', color='green')
        axs[5].set_xlabel('Timesteps')
        axs[5].set_ylabel('Buffer Size')
        axs[5].grid(True)

        axs[5].set_title('Agent Replay Buffer NonOpt Size ---' + env_name)
        axs[5].plot(replay_stepnum_list, agent_replay_buffer_nonopt_size, label='NonOpt Buffer Size', color='red')
        axs[5].set_xlabel('Timesteps')
        axs[5].set_ylabel('Buffer Size')
        axs[5].grid(True)
        axs[5].legend()

    plt.tight_layout()
    save_dir = Path(save_dir + "/Selflabel_num_plots_" + file_suffix_name + ".pdf")
    print("save_dir: ", save_dir)
    plt.savefig(save_dir)
    plt.close()


if __name__ == '__main__':

    """
    "python3 mujo_10_EVAL_discriminator.py  ",

    """
    sort_ratio = 0.5

    env_name_list = ["Ant-v4", "HalfCheetah-v4", "Hopper-v4", "Swimmer-v4", "Walker2d-v4"]
    env_name = env_name_list[0]

    models_dir = "../results/logs_scoring_sin-train_sort/"

    save_dir = models_dir+"Ant-v4_scoring_co_train_SAC_NetAgentSeed_0-0_epochs_50_RLseed_0_-un_gail_finetune_scoring_sort_05/"
    plot_discriminator_acc_loss(save_dir+"rl_scoring_co_train_model.pkl", env_name, save_dir, "after_training_new")
