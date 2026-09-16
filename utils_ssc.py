import numpy as np
from copy import deepcopy as cp
import json
from pathlib import Path
import pickle
import random


# export PATH="/home/robo/anaconda3/bin:$PATH"
# source /home/robo/anaconda3/bin/activate

        # "alpha_0.5": {

        #     "alpha_1": 0.8,
        #     "alpha_2": 0.2,

        #     "alpha_3": 0.6,
        #     "alpha_4": 0.4,

        #     "alpha_5": 0.7,
        #     "alpha_6": 0.3,

        #     "alpha_7": 0.9,
        #     "alpha_8": 0.1

        # },

def gen_data(bag_num, bag_sizes, opt_set, non_opt_set, opt_ratio_alpha, seed, testing_pi=None, random_Pi=False, random_Pi_range=(0.1, 0.9)):
    # randomly generate Pi_s (proportion of positive samples in each bag)
    if testing_pi is not None:
        Pi_s = [testing_pi]
    else:
        if not random_Pi:
            Pi_s = get_Pi_determ(sets=bag_num, alpha = opt_ratio_alpha, seed_value=seed)
        else:
            Pi_s = get_Pi_uniform(sets=bag_num, pi_max=random_Pi_range[1], pi_min=random_Pi_range[0])
            # Pi_s = get_Pi(sets=bag_num, pi_max=random_Pi_range[1], pi_min=random_Pi_range[0], seed_value=seed)
    print("True Pi_s ", Pi_s)

    # unlabled sets (bag_num, bag_size)
    replace = False if testing_pi is None else True

    U_sets_idx = get_U_sets_idx(bag_num, bag_sizes, Pi_s, opt_num = len(opt_set[0]), non_opt_num = len(non_opt_set[0]), seed_value=seed, replace=replace)

    # prior about proportion, how many unlabeled samples in each bag / bag total size
    priors_class = [bag_size / sum(bag_sizes) for bag_size in bag_sizes]

    print("priors_class: ", priors_class)
    print("--- len(U_sets_idx): ", len(U_sets_idx))
    print("--- U_sets_idx[0].shape: ", U_sets_idx[0].shape)
    # print("")

    U_set_s, U_set_classLabels, U_sets_binLabels_s = get_U_set(U_sets_idx, opt_set[0], non_opt_set[0])
    U_set_a, _, _ = get_U_set(U_sets_idx, opt_set[1], non_opt_set[1])
    print("U_set_s: ", U_set_s.shape)
    print("U_set_a :", U_set_a.shape)
    print("U_set_classLabels: ", U_set_classLabels.shape)
    print("U_sets_binLabels_s: ", U_sets_binLabels_s.shape)
    print("\n")
    return U_set_s, U_set_a, U_set_classLabels, U_sets_binLabels_s, Pi_s, priors_class




def Split_train_test(traj_s_set, traj_a_set, train_ratio, seed):
    num_train = int(train_ratio * len(traj_s_set))
    traj_idx_list = list(range(len(traj_s_set)))
    random.seed(seed)
    random.shuffle(traj_idx_list)
    
    traj_s_set_train = traj_s_set[traj_idx_list[:num_train]]
    traj_a_set_train = traj_a_set[traj_idx_list[:num_train]]
    traj_s_set_test = traj_s_set[traj_idx_list[num_train:]]
    traj_a_set_test = traj_a_set[traj_idx_list[num_train:]]
    # print(traj_idx_list)
    return traj_s_set_train, traj_a_set_train, traj_s_set_test, traj_a_set_test


def load_data(save_trajs_path):
    with open(save_trajs_path, 'rb') as f:
        loaded_data = pickle.load(f)
        
    opt_traj_s_set = loaded_data['opt_traj_s_set']
    opt_traj_a_set = loaded_data['opt_traj_a_set']
    nonopt_traj_s_set = loaded_data['nonopt_traj_s_set']
    nonopt_traj_a_set = loaded_data['nonopt_traj_a_set']
    scaler_s = loaded_data['scaler_s']
    scaler_a = loaded_data['scaler_a']
    opt_start_steps = loaded_data['opt_start_steps']

    s_max = scaler_s.data_max_
    s_min = scaler_s.data_min_
    a_max = scaler_a.data_max_
    a_min = scaler_a.data_min_
    return opt_traj_s_set, opt_traj_a_set, nonopt_traj_s_set, nonopt_traj_a_set, np.vstack((s_max, s_min)), np.vstack((a_max, a_min)), opt_start_steps



def get_Pi_determ(sets, alpha, seed_value=0):
    # load the pi from the json file
    np.random.seed(seed_value)

    Pi_json = json.load(open(Path("./RL_parameters/ssc_alpha.json")))
    Pi_json = Pi_json["alpha_"+str(alpha)]
    Pi_list = []
    for i in range(sets):
        Pi_list.append(Pi_json["alpha_"+str(i+1)])
    Pi = np.array(Pi_list)

    return Pi

def get_Pi(sets, pi_max=0.9, pi_min=0.1, seed_value=0):
    # uniform random priors frow [0.1, 0.9]
    np.random.seed(seed_value)
    Pi = np.random.rand(sets) * (pi_max - pi_min) + pi_min
    return Pi

def get_Pi_uniform(sets, pi_max=0.9, pi_min=0.1,):
    # uniform rsplit from pi_max and pi_min
    Pi = np.linspace(pi_min, pi_max, sets)
    return Pi

def get_U_sets_idx(bag_num, bag_sizes, Pi_s, opt_num, non_opt_num, seed_value=0, replace=False):
    U_sets_idx = []
    np.random.seed(seed_value)
    for i in range(bag_num):
        pos_num = int(np.round(bag_sizes[i] * Pi_s[i]))
        neg_num = int(bag_sizes[i] - pos_num)

        print("opt_num: ", opt_num)
        
        print("pos_num: ", pos_num)

        print("non_opt_num: ", non_opt_num)
        print("neg_num: ", neg_num)
        print("")


        pos_idx = 1 * np.random.choice(opt_num, pos_num, replace=replace)  + 1  # randomly choose number for pos_num times from range(opt_set)
        neg_idx = -1 * np.random.choice(non_opt_num, neg_num, replace=replace)  - 1

        U_set_idx = np.concatenate((pos_idx, neg_idx))     
        np.random.shuffle(U_set_idx)
        U_sets_idx.append(U_set_idx)

        # print("==", i, "==")
        # print("Pi: ", Pi_s[i])
        # print("bag_sizes ", bag_sizes[i])
        # print("pos_num: ", pos_num)
        # print("neg_num: ", neg_num)

        # print("pos_idx: ", pos_idx.shape)
        # print(pos_idx)
        # print("neg_idx: ", neg_idx.shape)
        # print(neg_idx)
        # print("U_set_idx: ", U_set_idx.shape)
        # print(U_set_idx)

    return U_sets_idx

def get_U_set(U_sets_idx, opt_set, non_opt_set):
    U_set = []
    U_set_class = []
    U_sets_binLabels = []
    for i in range(len(U_sets_idx)):
        # print("i ", i)
        for j in U_sets_idx[i]:
            if j > 0:
                U_set.append(opt_set[ j - 1])
                # U_sets_pos.append([True])
                U_sets_binLabels.append(np.eye(2)[0]) # True
            elif j < 0:
                U_set.append(non_opt_set[ -1 * (j + 1) ])
                # U_sets_pos.append([False])
                U_sets_binLabels.append(np.eye(2)[1]) # False
            else:
                raise ValueError("U_sets_idx is wrong")
            U_set_class.append(np.eye(len(U_sets_idx))[i])

            # print(np.eye(len(U_sets_idx))[i])
    
    return np.array(U_set), np.array(U_set_class), np.array(U_sets_binLabels)

def encode_state(state, grid_size):
    encoded_state = np.zeros(grid_size**2)
    encoded_state[state[1] * grid_size + state[0]] = 1
    return encoded_state.tolist()


# def encode_action(action):
#     if action == 0:
#         return [1, 0, 0, 0]
#     elif action == 1:
#         return [0, 1, 0, 0]
#     elif action == 2:
#         return [0, 0, 1, 0]
#     elif action == 3:
#         return [0, 0, 0, 1]
#     else:
#         raise ValueError("Invalid action.")


# def decode_action(encoded_action):
#     if encoded_action == [1, 0, 0, 0]:
#         return 0
#     elif encoded_action == [0, 1, 0, 0]:
#         return 1
#     elif encoded_action == [0, 0, 1, 0]:
#         return 2
#     elif encoded_action == [0, 0, 0, 1]:
#         return 3
#     else:
#         raise ValueError("Invalid action.")
    

def encode_action(a_labels, num_classes):
    """
    Convert class labels to one-hot encoded vectors.

    Args:
    - labels: List or numpy array of class labels (e.g., [1, 2, 3, ...])
    - num_classes: Total number of classes

    Returns:
    - one_hot: Numpy array of one-hot encoded vectors
    """

    # Initialize an array of zeros to store the one-hot encoded vectors
    one_hot = np.zeros((len(a_labels), num_classes))

    # Set the appropriate element in each row to 1
    for i, label in enumerate(a_labels):
        one_hot[i, label] = 1  

    return one_hot



def decode_action(one_hot_vectors):
    """
    Decode one-hot encoded vectors to class labels.

    Args:
    - one_hot_vectors: Numpy array of one-hot encoded vectors

    Returns:
    - labels: List of decoded class labels
    """

    # Extract the index of the maximum value along the columns
    labels = np.argmax(one_hot_vectors, axis=1) 

    return labels.tolist()


import sys
def check_storage_mb(parameter):
    size_bytes = sys.getsizeof(parameter)
    size_mb = size_bytes / (1024 * 1024)
    print(f"The size of the parameter is approximately {size_mb:.2f} MB.")

# # Example usage
# check_storage_mb(None)  # Check the size of None
# check_storage_mb(42)    # Check the size of an integer
# check_storage_mb("Hello, World!")  # Check the size of a string
# check_storage_mb([1, 2, 3, 4, 5])  # Check the size of a list
