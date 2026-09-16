import argparse
from utils_gail import *
import torch
from torch.utils.data import DataLoader, Dataset
from um_ssc_grid_mujuco import Scoring_model
from utils_ssc import load_data, Split_train_test, gen_data
from sklearn.preprocessing import MinMaxScaler
from utils_alpha_est_ssc_self2 import SSC_alpha_self
import time
from copy import deepcopy as cp

 
#  python mujo_02_trajs-Udata_alpha_est_entropy_main_uu10orcl.py   --env 4 --uu_seed 1 --method 5 --Pi_range_select 0
# python mujo_02_trajs-Udata_alpha_est_entropy_main_uu10orcl.py   --env 4  --method 5 --Pi_range_select 0 --uu_seed 0
"""
"python mujo_02_trajs-Udata_alpha_est_entropy_main_uu10orcl.py   --env 0 --uu_seed 1 --method 5 --Pi_range_select 1",
"python mujo_02_trajs-Udata_alpha_est_entropy_main_uu10orcl.py   --env 1 --uu_seed 1 --method 5 --Pi_range_select 1",
"python mujo_02_trajs-Udata_alpha_est_entropy_main_uu10orcl.py   --env 2 --uu_seed 1 --method 5 --Pi_range_select 1",
"python mujo_02_trajs-Udata_alpha_est_entropy_main_uu10orcl.py   --env 3 --uu_seed 1 --method 5 --Pi_range_select 1",
"python mujo_02_trajs-Udata_alpha_est_entropy_main_uu10orcl.py   --env 4 --uu_seed 1 --method 5 --Pi_range_select 1",


"python mujo_02_trajs-Udata_alpha_est_entropy_main_uu10orcl.py   --env 0 --uu_seed 1 --method 5 --Pi_range_select 2",
"python mujo_02_trajs-Udata_alpha_est_entropy_main_uu10orcl.py   --env 1 --uu_seed 1 --method 5 --Pi_range_select 2",
"python mujo_02_trajs-Udata_alpha_est_entropy_main_uu10orcl.py   --env 2 --uu_seed 1 --method 5 --Pi_range_select 2",
"python mujo_02_trajs-Udata_alpha_est_entropy_main_uu10orcl.py   --env 3 --uu_seed 1 --method 5 --Pi_range_select 2",
"python mujo_02_trajs-Udata_alpha_est_entropy_main_uu10orcl.py   --env 4 --uu_seed 1 --method 5 --Pi_range_select 2",


"python mujo_02_trajs-Udata_alpha_est_entropy_main_uu10orcl.py   --env 0 --uu_seed 1 --method 5 --Pi_range_select 3",
"python mujo_02_trajs-Udata_alpha_est_entropy_main_uu10orcl.py   --env 1 --uu_seed 1 --method 5 --Pi_range_select 3",
"python mujo_02_trajs-Udata_alpha_est_entropy_main_uu10orcl.py   --env 2 --uu_seed 1 --method 5 --Pi_range_select 3",
"python mujo_02_trajs-Udata_alpha_est_entropy_main_uu10orcl.py   --env 3 --uu_seed 1 --method 5 --Pi_range_select 3",
"python mujo_02_trajs-Udata_alpha_est_entropy_main_uu10orcl.py   --env 4 --uu_seed 1 --method 5 --Pi_range_select 3",






"python mujo_02_trajs-Udata_alpha_est_entropy_main_uu10orcl.py   --env 0 --uu_seed 0 --method 5 --Pi_range_select 0  ",
"python mujo_02_trajs-Udata_alpha_est_entropy_main_uu10orcl.py   --env 0 --uu_seed 1 --method 5 --Pi_range_select 0  ",
"python mujo_02_trajs-Udata_alpha_est_entropy_main_uu10orcl.py   --env 0 --uu_seed 2 --method 5 --Pi_range_select 0  ",
"python mujo_02_trajs-Udata_alpha_est_entropy_main_uu10orcl.py   --env 0 --uu_seed 3 --method 5 --Pi_range_select 0  ",
"python mujo_02_trajs-Udata_alpha_est_entropy_main_uu10orcl.py   --env 0 --uu_seed 4 --method 5 --Pi_range_select 0  ",



"""

parser = argparse.ArgumentParser(description=None)
parser.add_argument('--env', type=int, default='9', help='0-4 env list')
parser.add_argument('--uu_seed', type=int, default='0', help='')
parser.add_argument('--method', type=int, default='0', help='0: D_pi_adam, 1: D_pi_bo, 2: pi_self')
parser.add_argument('--Pi_range_select', type=int, default='0', help='0.1-0.9')
parser.add_argument('--pi_given', type=str, default='False', help=' pi_given should be True or False ')
parser.add_argument('--bag_num', type=int, default='6', help='2 or 6')  

args = parser.parse_args()

def true_false_convert(str):
    if str == "true" or str == "True":
        return True
    elif str == "false" or str == "False":
        return False
    else:
        raise ValueError("true or false")




env_name_list = ["Ant-v4", "HalfCheetah-v4", "Hopper-v4", "Swimmer-v4", "Walker2d-v4"]
env_name = env_name_list[args.env]

pi_given = true_false_convert(args.pi_given)

###########################
uu10_active = True
less_epochs = True
pi_given = True # fix pi_given to True when uu10_active is True
###########################

# random_Pi_range_list = [(0.1, 0.9), 
#                         (0.05, 0.2), (0.05, 0.3), (0.05, 0.4), (0.05, 0.5), 
                        
#                         (0.4, 0.6), (0.3, 0.7), 
                        
                        
#                         (0.8, 0.95), (0.7, 0.95), (0.6, 0.95), (0.5, 0.95),
                          
#                         ]

random_Pi_range_list = [(0.1, 0.9),  # 0
                        
                        (0.05, 0.15), (0.45, 0.55), (0.85, 0.95),  # 1 2 3
                        (0.05, 0.25), (0.4, 0.6),  (0.75, 0.95), # 4 5 6

                        (0.2, 0.8), # 7

                        (0.05, 0.4), (0.05, 0.5), 
                        
                        (0.4, 0.6), (0.3, 0.7), 
                        
                        
                        (0.8, 0.95), (0.7, 0.95), (0.6, 0.95), (0.5, 0.95),
                        
                          
                        ]


random_Pi_range = random_Pi_range_list[args.Pi_range_select]

# seed_value = args.rl_seed

# RL_alg = "SAC"
# models_dir = "../results/logs_gail_noReplacing/"
trajs_dir = "../results/opt_nonopt_trajs_noReplacing/"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("=============================", device, "=============================")
print("=============================", device, "=============================")


# Load demonstration data - noisy --------------------------------------
save_trajs_path = trajs_dir + "opt_nonopt_trajs_"+env_name+"_sacExpertSeed_"+"0"+"_newneg_fullDataset.pkl"
opt_traj_s_set, opt_traj_a_set, nonopt_traj_s_set, nonopt_traj_a_set, s_maxmin, a_maxmin, opt_start_steps = load_data(save_trajs_path)



scaler_s = MinMaxScaler()
scaler_s.fit(s_maxmin)

scaler_a = MinMaxScaler()
scaler_a.fit(a_maxmin)

s_dim = s_maxmin[0].shape[0]
a_dim = a_maxmin[0].shape[0]


train_ratio = 0.9
seed_value = args.uu_seed
bag_num_train = args.bag_num
# bag_num_train = 6
opt_ratio_alpha = "0.5"
bag_sizes_train = np.ones(bag_num_train) * 5000
opt_traj_s_set_train, opt_traj_a_set_train, opt_traj_s_set_test, opt_traj_a_set_test = Split_train_test(opt_traj_s_set, opt_traj_a_set, train_ratio, seed_value)
nonopt_traj_s_set_train, nonopt_traj_a_set_train, nonopt_traj_s_set_test, nonopt_traj_a_set_test = Split_train_test(nonopt_traj_s_set, nonopt_traj_a_set, train_ratio, seed_value)

print("opt_traj_s_set_train: ", len(opt_traj_s_set_train), " opt_traj_s_set_test: ", len(opt_traj_s_set_test))
print("nonopt_traj_s_set_train: ", len(nonopt_traj_s_set_train), " nonopt_traj_s_set_test: ", len(nonopt_traj_s_set_test))
# input("check")

U_set_s_train, U_set_a_train, U_set_classLabels_train, U_sets_binLabels_train, Pi_s_train, priors_class_train = gen_data(
    bag_num_train, bag_sizes_train, [opt_traj_s_set_train, opt_traj_a_set_train],  [nonopt_traj_s_set_train, nonopt_traj_a_set_train] , opt_ratio_alpha, seed=seed_value, testing_pi=None, 
    random_Pi=True, random_Pi_range=random_Pi_range)


print("Pi_s_train: ", Pi_s_train)
print("var of Pi_s_train: ", np.var(Pi_s_train))
print("priors_class_train: ", priors_class_train)
print("U_set_s_train samples: ", U_set_s_train.shape)
# print("U_set_s_train example: ", U_set_s_train[:5])
# print("U_set_classLabels_train example: ", U_set_classLabels_train[:5])
# print("U_sets_binLabels_train example: ", U_sets_binLabels_train[:5])
print("U_sets_binLabels_train shape: ", U_sets_binLabels_train.shape)
print("U_set_classLabels_train shape: ", U_set_classLabels_train.shape)





U_set_s_train = torch.tensor((scaler_s.transform(U_set_s_train[:,0,:])) [:, np.newaxis, :]).float().to(device)
U_set_a_train = torch.tensor((scaler_a.transform(U_set_a_train[:,0,:])) [:, np.newaxis, :]).float().to(device)

true_labels = torch.from_numpy(U_sets_binLabels_train[:, 0]).long()
U_set_s_train_opt = U_set_s_train[true_labels==1]
U_set_a_train_opt = U_set_a_train[true_labels==1]
U_set_s_train_nonopt = U_set_s_train[true_labels==0]
U_set_a_train_nonopt = U_set_a_train[true_labels==0]
print("traj_s_opt: ", U_set_s_train_opt.shape, " traj_a_opt: ", U_set_a_train_opt.shape)
print("traj_s_nonopt: ", U_set_s_train_nonopt.shape, " traj_a_nonopt: ", U_set_a_train_nonopt.shape)

if uu10_active:
    if U_set_s_train_opt.shape[0] != U_set_s_train_nonopt.shape[0]:
        bag_size_ratio = int(U_set_s_train_nonopt.shape[0] / U_set_s_train_opt.shape[0])
        # duplicate the optimal data to make the number of optimal and non-optimal data equal
        U_set_s_train_opt = U_set_s_train_opt.repeat(bag_size_ratio, 1, 1)
        U_set_a_train_opt = U_set_a_train_opt.repeat(bag_size_ratio, 1, 1)

        print("After uu10 active - duplicated optimal data")
        print("traj_s_opt: ", U_set_s_train_opt.shape, " traj_a_opt: ", U_set_a_train_opt.shape)
        print("traj_s_nonopt: ", U_set_s_train_nonopt.shape, " traj_a_nonopt: ", U_set_a_train_nonopt.shape)
        # input("check uu10 active - duplicated optimal data")

    U_set_s_train = torch.cat((U_set_s_train_opt, U_set_s_train_nonopt), dim=0)
    U_set_a_train = torch.cat((U_set_a_train_opt, U_set_a_train_nonopt), dim=0)


    if U_set_s_train_opt.shape[0] == U_set_s_train_nonopt.shape[0]:
        # only 2 bags

        U_set_classLabels_train = torch.zeros((U_set_s_train.shape[0], 2 )).long().to(device)
        U_set_classLabels_train[:U_set_s_train_opt.shape[0], 0] = 1
        U_set_classLabels_train[U_set_s_train_opt.shape[0]:, 1] = 1
        # copy U_set_classLabels_train to numpy
        U_set_classLabels_train = U_set_classLabels_train.cpu().numpy()
        U_sets_binLabels_train = cp(U_set_classLabels_train)

        Pi_s_train = np.array([1, 0])
        priors_class_train = [U_set_s_train_opt.shape[0]/U_set_s_train.shape[0], U_set_s_train_nonopt.shape[0]/U_set_s_train.shape[0]]


    else:
        # bag_num_train = int(U_set_s_train_nonopt.shape[0] / U_set_s_train_opt.shape[0]) + 1
        # print("After uu10 active - bag_num_train: ", bag_num_train)

        # U_set_classLabels_train = torch.zeros((U_set_s_train.shape[0], bag_num_train )).long().to(device)
        # for i in range(bag_num_train):
        #     if i < bag_num_train -1:
        #         U_set_classLabels_train[i*U_set_s_train_opt.shape[0]:(i+1)*U_set_s_train_opt.shape[0], i] = 1
        #     else:
        #         U_set_classLabels_train[i*U_set_s_train_opt.shape[0]:, i] = 1
        # # copy U_set_classLabels_train to numpy
        # U_set_classLabels_train = U_set_classLabels_train.cpu().numpy()
        # U_sets_binLabels_train = [[1,0] for _ in range(U_set_s_train_opt.shape[0])] + [[0,1] for _ in range(U_set_s_train_nonopt.shape[0])]
        # U_sets_binLabels_train = np.array(U_sets_binLabels_train)
        # Pi_s_train = np.array([1] * 1 + [0] * (bag_num_train -1))
        # priors_class_train = np.array([1/bag_num_train] * bag_num_train)
        raise ValueError("for uu10_active, the number of optimal and non-optimal samples should be equal")






    print("After uu10 active - ")
    print("U_set_s_train shape: ", U_set_s_train.shape, " U_set_a_train shape: ", U_set_a_train.shape)
    print("U_set_classLabels_train example: ", U_set_classLabels_train[15000-5:15000+5])
    print("U_sets_binLabels_train example: ", U_sets_binLabels_train[15000-5:15000+5])
    print("U_set_classLabels_train example: ", U_set_classLabels_train[3000-5:3000+5])
    print("U_sets_binLabels_train example: ", U_sets_binLabels_train[3000-5:3000+5])
    print("U_sets_binLabels_train shape: ", U_sets_binLabels_train.shape)
    print("U_set_classLabels_train shape: ", U_set_classLabels_train.shape)
    print("Pi_s_train: ", Pi_s_train)
    print("priors_class_train: ", priors_class_train)
    # input("check uu10 active")



bag_num_test = 1
bag_sizes_test = np.ones(bag_num_test) * 5000
testing_pi = 0.5
U_set_s_test, U_set_a_test, U_set_classLabels_test, U_sets_binLabels_test, Pi_s_test, priors_class_test = gen_data(
    bag_num_test, bag_sizes_test, [opt_traj_s_set_test, opt_traj_a_set_test],  [nonopt_traj_s_set_test, nonopt_traj_a_set_test] , opt_ratio_alpha, seed=seed_value, testing_pi=testing_pi)
U_set_a_test = torch.tensor((scaler_a.transform(U_set_a_test[:,0,:])) [:, np.newaxis, :]).float().to(device)
U_set_s_test = torch.tensor((scaler_s.transform(U_set_s_test[:,0,:])) [:, np.newaxis, :]).float().to(device)

# get the randomly generated Pi_s to calculate the optimal ratio alpha
opt_ratio_alpha = np.mean(Pi_s_train)
opt_ratio_alpha = str(np.round(opt_ratio_alpha, 3))

batch_size = int(U_set_s_train.shape[0] / 10)
print("batch_size: ", batch_size)
batch_size = int(30000 / 10)
print("batch_size: ", batch_size)
# input("check batch size")



class StateActionDataset(Dataset):
    def __init__(self, states, actions):
        self.states = states
        self.actions = actions

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        state = self.states[idx]
        action = self.actions[idx]
        return state, action

noisy_data_loader = DataLoader(StateActionDataset(U_set_s_train, U_set_a_train), batch_size=batch_size, shuffle=True)
priors_class_train = np.array(priors_class_train)
# Pi_s_train = np.array([0.5 for _ in range(6)]) # initialize Pi_s
# Pi_s_train = np.array([0.8, 0.2, 0.6, 0.4, 0.7, 0.3])
# Pi_s_train = np.array([0.6, 0.1, 0.7, 0.1, 0.65, 0.2])
Pi_s_test = np.array([0.5]) # initialize Pi_s

frame_num = 1

scoring_model = Scoring_model(priors_class = priors_class_train, Pi = Pi_s_train, Pi_test = Pi_s_test[0], input_dim_s=s_dim, input_dim_a=a_dim, input_scaler_s=scaler_s, input_scaler_a=scaler_a,
                            frame_num = frame_num)


save_folder = "../results/alphaEst_randomPi/"
# random_Pi_range used for save_folder
save_folder = save_folder[:-1] + "_PiRange"+str(random_Pi_range[0])[0]+ str(random_Pi_range[0])[2:] +"-"+str(random_Pi_range[1])[0]+ str(random_Pi_range[1])[2:] +"/"
# add time date -hours min- second
# save_folder = save_folder[:-1] + "_"+ time.strftime("%Y%m%d-%H%M%S") +"/"

if bag_num_train != 6:
    save_folder = save_folder[:-1]  + "_bagNum"+str(bag_num_train)+"/"

if pi_given:
    save_folder = save_folder[:-1]  + "_piGiven/"

if uu10_active:
    save_folder = save_folder[:-1]  + "_uu10orcl/"

if less_epochs:
    save_folder = save_folder[:-1]  + "_lessEpochs/"


saving_path = save_folder + "scoring_model_alphaEst/"+ env_name+"_ExpertSeed_"+str(args.uu_seed)+"_scoring_model-un"+"_multi_frame_"+str(frame_num)+"_optStart"+str(opt_start_steps)+"_alpha_"+opt_ratio_alpha[0]+opt_ratio_alpha[2:]
# if args.noisy_alpha == "True":
#     saving_path = saving_path +"_"+opt_ratio_alpha_noise
saving_path = saving_path + "_newneg/"

train_epoch_num = 25

lr_uu_classifier = 1e-4/10*10
uu_classifier_data = {"priors_class": priors_class_train,
                        "Pi_test": Pi_s_test,
                        "input_dim_s": s_dim,
                        "input_dim_a": a_dim,
                        "input_scaler_s": scaler_s,
                        "input_scaler_a": scaler_a,
                        "frame_num": frame_num,
                      
    
                    "U_set_s_train": U_set_s_train, 
                      "U_set_a_train": U_set_a_train, 
                      "U_set_classLabels_train": U_set_classLabels_train, 
                      "U_sets_binLabels_train": U_sets_binLabels_train, 
                        "U_set_s_test": U_set_s_test,
                        "U_set_a_test": U_set_a_test,
                        "U_set_classLabels_test": U_set_classLabels_test,
                        "U_sets_binLabels_test": U_sets_binLabels_test,
                        "train_epoch_num": train_epoch_num,
                        "lr": lr_uu_classifier,
                        "batch_size": 1000,
                        "saving_path": saving_path,
                        "seed": seed_value,
                        
                        "env_idx":args.env,
                        "env_name":env_name,
                        "opt_ratio_alpha":opt_ratio_alpha,
                        "opt_start_steps":opt_start_steps,
                        }

print("priors_class_train: ", priors_class_train)
# input("check")



# _, _, _ = scoring_model.net_training(x_input_train = [U_set_s_train, U_set_a_train], y_output_train = U_set_classLabels_train, y_output_bin_train = U_sets_binLabels_train,
#                             x_input_test = [U_set_s_test, U_set_a_test], y_output_test = U_set_classLabels_test,  y_output_bin_test = U_sets_binLabels_test, 

#                             epoch_num=train_epoch_num, lr=1e-4/10, batch_size=1000, saving_path=saving_path, 
#                             seed = seed_value)


if args.method == 5:
    ssc_gan_class = SSC_alpha_self


print("demo noisy data shape ", U_set_s_train.shape)
print("demo noisy data shape ", U_set_a_train.shape)

print("demo noisy data max ", torch.max(U_set_s_train))
print("demo noisy data min ", torch.min(U_set_s_train))
print("demo noisy data max ", torch.max(U_set_a_train))
print("demo noisy data min ", torch.min(U_set_a_train))



ssc_gan = ssc_gan_class(scoring_model, uu_classifier_data, noisy_data_loader, 
                device=device,seed_value=seed_value, init_alpha_mode = "random", save_folder=save_folder,
                random_Pi_range = random_Pi_range,
                 pi_given= pi_given,)

if args.env == 0 or args.env == 1 or args.env == 3:
    epochs = 10
elif args.env == 2 or args.env == 4:
    epochs = 20

if less_epochs:
    if args.env == 0 or args.env == 1 or args.env == 3:
        epochs = 2
    elif args.env == 2 or args.env == 4:
        epochs = 3

ssc_gan.train(epochs=epochs, lr=1e-4,
               saving_path=saving_path)


# print("Pi_s_train ", Pi_s_train)
# print("priors_class_train ", priors_class_train)
# print("U_set_s_train ", U_set_s_train[-5:])
# print("U_set_a_train ", U_set_a_train[-5:])
# print("U_set_classLabels_train ", U_set_classLabels_train[-5:])
# print("U_sets_binLabels_train ", U_sets_binLabels_train[-5:])

# print("U_set_s_train shape", (U_set_s_train.shape,))

# print("")
# print("U_set_s_train min ", U_set_s_train[:,0,:].min(dim=0).values, )
# print("U_set_s_train max ", U_set_s_train[:,0,:].max(dim=0).values, )
# print("")
# print("U_set_a_train min ", U_set_a_train[:,0,:].min(dim=0).values, )
# print("U_set_a_train max ", U_set_a_train[:,0,:].max(dim=0).values, )
# print("")

# print("scaler_s min max ", scaler_s.data_min_, scaler_s.data_max_)
# print("scaler_a min max ", scaler_a.data_min_, scaler_a.data_max_)

# input("Press Enter to continue...")

# Save all noisy data for future use -
uu_classifier_data = {"priors_class": priors_class_train,
                      "Pi_s_train": Pi_s_train,
                    "priors_class_test": priors_class_test,
                        "Pi_test": Pi_s_test,

                        "input_dim_s": s_dim,
                        "input_dim_a": a_dim,
                        "input_scaler_s": scaler_s,
                        "input_scaler_a": scaler_a,
                        "frame_num": frame_num,
                      
    
                    "U_set_s_train": U_set_s_train.cpu(), 
                      "U_set_a_train": U_set_a_train.cpu(), 
                      "U_set_classLabels_train": U_set_classLabels_train, 
                      "U_sets_binLabels_train": U_sets_binLabels_train, 
                        "U_set_s_test": U_set_s_test.cpu(),
                        "U_set_a_test": U_set_a_test.cpu(),
                        "U_set_classLabels_test": U_set_classLabels_test,
                        "U_sets_binLabels_test": U_sets_binLabels_test,


                        "train_epoch_num": train_epoch_num,
                        "lr": lr_uu_classifier,
                        "batch_size": 1000,
        
                        "seed": seed_value,
                        
                        "env_idx":args.env,
                        "env_name":env_name,
                        "opt_ratio_alpha":opt_ratio_alpha,
                        "opt_start_steps":opt_start_steps,
                        }

import os
saving_path = save_folder + "/noisy_data/"
os.makedirs(saving_path, exist_ok=True) 
saving_path = saving_path + env_name+"_ExpertSeed_"+str(args.uu_seed)+"_uu_data"+"_multi_frame_"+str(frame_num)+"_optStart"+str(opt_start_steps)+"_alpha_"+opt_ratio_alpha[0]+opt_ratio_alpha[2:]+  "_newneg_.npz"

# save uu_classifier_data as npz
np.savez(saving_path, **uu_classifier_data, allow_pickle=True)

print("demo noisy data shape ", U_set_s_train.shape)
print("demo noisy data shape ", U_set_a_train.shape)

print("demo noisy data max ", torch.max(U_set_s_train))
print("demo noisy data min ", torch.min(U_set_s_train))
print("demo noisy data max ", torch.max(U_set_a_train))
print("demo noisy data min ", torch.min(U_set_a_train))


"""

env 0: epochs = 10
env 1: epochs = 10
env 2: epochs = 20
env 3: epochs = 10
env 4: epochs = 20



"""
# TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.
# a = torch.tensor([1.1434 ,2.5456 ,3.235235]).to("cuda:0")
# a.numpy()
# a.cpu().numpy()

