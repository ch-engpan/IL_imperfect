import torch
import numpy as np
from um_ssc_grid_mujuco import Scoring_model_net_multiFrame
# import wandb
import datetime
import time
import csv
import itertools
from copy import deepcopy as cp
import os
import matplotlib.pyplot as plt
from um_ssc_grid_mujuco import plot_learning_loss
from pathlib import Path
import pickle
from itertools import combinations

class SSC_alpha_self():
    def __init__(self, uu_classifier, uu_classifier_data, noisy_data_loader,
                  device = None, seed_value=None, init_alpha_mode = "half_rank", save_folder = "./",
                  random_Pi_range = (0.1, 0.9), pi_given = False):
        
        np.random.seed(seed_value)
        self.device = device
        self.noisy_data_loader = noisy_data_loader
        self.save_folder = save_folder

        self.true_Pi = uu_classifier.net.Pi
        self.pi_given = pi_given
        print("check self.true_Pi: ", self.true_Pi  )
        if pi_given:
            print("pi_given is True, so use the given pi")
        else:
            print("pi_given is False, so estimate pi")
        # input("Press Enter to continue...")
        # self.random_Pi_range = random_Pi_range

        # numbers = [0, 1, 2, 3, 4, 5]
        # Get all combinations of selecting 3 numbers from the list
        # combinations = list(itertools.combinations(numbers, 3))
        # self.params_init_list = []   
        # # Print the combinations
        # for combination in combinations:
        #     params_init = np.array([0.9 for _ in range(6)]) # Pi_s for uu_classifier
        #     params = cp(params_init)
        #     for idx in combination:
        #         params[idx] = 0.1
        #     self.params_init_list.append(params)
        
        
        self.params_init_list = []
        # pick up the index of three max values from self.true_Pi
        idx = np.argsort(self.true_Pi, )
        idx_max = idx[int(0.5*len(idx)):]
        idx_min = idx[:int(0.5*len(idx))]

        # def get_ramdom_pi(idx_max, idx_min):
        def get_ramdom_pi(idx_min, idx_max):
            init_pi = np.array([0.1 for _ in range(self.true_Pi.shape[0])]) 
            for i_th, i in enumerate(idx_max):
                random_num = np.random.uniform(0, 1)
                while random_num < 0.5: # try to get a value larger than 0.5
                    random_num = np.random.uniform(0, 1)
                init_pi[i] = random_num
                init_pi[idx_min[i_th]] = 1 - random_num
            return init_pi
        
        # def get_ramdom_pi(idx_max, idx_min):
        #     init_pi = np.array([0.1 for _ in range(6)])
        #     half_value = (random_Pi_range[0] + random_Pi_range[1]) / 2
        #     for i_th, i in enumerate(idx_max):
        #         random_num = np.random.uniform(random_Pi_range[0], random_Pi_range[1])
        #         while random_num < half_value: # try to get a value larger than 0.5
        #             random_num = np.random.uniform(random_Pi_range[0], random_Pi_range[1])
        #         init_pi[i] = random_num
        #         init_pi[idx_min[i_th]] = 1 - random_num
        #     return init_pi
            
        if not pi_given:
            if init_alpha_mode == "half_rank":
                init_pi1 = get_ramdom_pi(idx_max, idx_min)
                self.params_init_list.append(init_pi1) # set first three max values as 0.9, else 0.1


            elif init_alpha_mode == "random":
        
                combos = list(combinations([i for i in range(len(self.true_Pi))], int(0.5*len(self.true_Pi))))
                print(f"Total combinations: {len(combos)}")

                if len(combos) < 20:
                    # duplicate to have at least 20 combinations
                    multiplier = 20 // len(combos) + 1
                    combos = combos * multiplier
                print(f"Total combinations after duplication: {len(combos)}")
                combos_rest = []
                for combo in combos:
                    print(combo)
                    rest = tuple(x for x in [i for i in range(len(self.true_Pi))] if x not in combo)
                    combos_rest.append(rest)
                print(f"Total combinations: {(combos_rest)}")
                
                for i, combo in enumerate(combos):
                    init_pi = get_ramdom_pi(combo, combos_rest[i])
                    init_pi = np.clip(init_pi, 0.1, 0.9)
                    self.params_init_list.append(init_pi)
        else:
            self.params_init_list.append(self.true_Pi)
        print("self.params_init_list ", self.params_init_list)
        print("len(self.params_init_list) ", len(self.params_init_list))

        print("self.true_Pi: ", self.true_Pi)
        # input("Press Enter to continue...")

        # init_pi2 = np.array([0.1 for _ in range(6)])
        # pi_candidates = np.array([1/7, 2/7, 3/7, 4/7, 5/7, 6/7])
        # for index, i in enumerate(idx):
        #     print(i)
        #     init_pi2[i] = pi_candidates[index]

        
        # cahnge to tensor
        self.uu_classifier = uu_classifier
        self.uu_data = uu_classifier_data

        self.first_row_csv = True


    def train(self, epochs, lr=1e-4,
              saving_path = "", exp_check_num=3):

        def get_index_of_data_from_class(data_s, data_a, class_label, true_bin_label):

            # Convert one-hot encoded labels to class indices
            class_indices = np.argmax(class_label, axis=1)

            # Get indices of training data for each class
            class_to_indices = {}
            for idx, class_idx in enumerate(class_indices):
                if class_idx not in class_to_indices:
                    class_to_indices[class_idx] = []
                class_to_indices[class_idx].append(idx)
            # Print the result
            data_s_all =[]
            data_a_all =[]
            bin_labels_all = []
            for class_label, indices in class_to_indices.items():

                data_s_all.append(data_s[indices])

                data_a_all.append(data_a[indices])

                bin_labels_all.append(true_bin_label[indices])
                # print(f"Class {class_label}: Indices {indices[:10]}")
            return data_s_all, data_a_all, bin_labels_all

        def pred_opt(model, data_s, data_a, y_true_bin, final_iteration=False, check_num=3):
            y_predicted_bin, _, _ = model.forward(data_s, data_a)
            y_predicted_bin = y_predicted_bin[:, 0:2] 
            labels_pred = torch.argmax(y_predicted_bin.data, dim=1)
            
            num_pos = (labels_pred == 0).sum().item()
            num_neg = (labels_pred == 1).sum().item()
            Pi = num_pos / (num_pos + num_neg)
            if num_pos + num_neg != len(data_s):
                print("num_pos + num_neg != len(data_s)")
                raise ValueError
            
            # random pick up 3 samples and check if the labels are correct
            if final_iteration:
                y_true_bin = torch.tensor(y_true_bin, dtype=torch.float32).to(self.device)
                labels_true = torch.argmax(y_true_bin, dim=1)
                check_num = check_num
                random_indices = np.random.choice(len(data_s), check_num, replace=False)

                correct_count = 0
                pi_reverse_signal = []
                for i in random_indices:
                    # print(f"data_s[{i}]: {data_s[i]}, data_a[{i}]: {data_a[i]}, true label: {labels_true[i]}, predicted label: {labels_pred[i]}")
                    if labels_true[i] == labels_pred[i]:
                        correct_count += 1
                        pi_reverse_signal.append(True)
                    else:
                        pi_reverse_signal.append(False)
                print(f"Correctly classified {correct_count} out of {check_num} samples.")
                # pi_reverse_signal = correct_count == check_num
            else:
                pi_reverse_signal = None


            return Pi, pi_reverse_signal

        other_saved_data_allinit = []

        for param_init_idx, param_init in enumerate(self.params_init_list):
            self.uu_classifier.net = Scoring_model_net_multiFrame(self.uu_data["priors_class"], 
                                                param_init, 
                                                self.uu_data["Pi_test"][0], 
                                                self.uu_data["input_dim_s"], self.uu_data["input_dim_a"], 
                                                self.uu_data["frame_num"])
            self.uu_classifier.net.to(self.device)
            U_set_s_class, U_set_a_class, U_set_bin_labels = get_index_of_data_from_class(self.uu_data["U_set_s_train"], self.uu_data["U_set_a_train"], self.uu_data["U_set_classLabels_train"], self.uu_data["U_sets_binLabels_train"])
            uu_bag_loss_train_list = []
            uu_bin_loss_train_list = []
            uu_bin_loss_val_list = []
            acc_uu_train_list = []
            acc_uu_val_list = []
            acc_uu_train_th_list = []
            acc_uu_val_th_list = []
            epoch_plt = []

            alpha_loss_list = []
            for epoch in range(epochs):
   
                acc_uu_train = 0
                acc_uu_val = 0
                acc_uu_train_th = 0
                acc_uu_val_th = 0

                uu_bag_loss_train = 0
                acc_uu_bag_train = 0

                epoch_num_uu = 5
                batch_size = self.uu_data["batch_size"]
                seed=self.uu_data["seed"]
                _, uu_loss, _, acc_uu = self.uu_classifier.net_training(x_input_train = [
                                    self.uu_data["U_set_s_train"], self.uu_data["U_set_a_train"]],

                                    y_output_train = self.uu_data["U_set_classLabels_train"],
                                    y_output_bin_train = self.uu_data["U_sets_binLabels_train"],
                                    x_input_test = [self.uu_data["U_set_s_test"], self.uu_data["U_set_a_test"]],
                                    y_output_test = self.uu_data["U_set_classLabels_test"],
                                    y_output_bin_test = self.uu_data["U_sets_binLabels_test"],
                                    # epoch_num=self.uu_data["train_epoch_num"],
                                    epoch_num = epoch_num_uu,
                                    # lr=self.uu_data["lr"],
                                    lr=lr,
                                    batch_size=self.uu_data["batch_size"],
                                    saving_path=self.uu_data["saving_path"],
                                    seed=self.uu_data["seed"],
                                    plot_loss = False,)
                
                Pi_cal = []
                final_iteration = (epoch == epochs - 1)
                
                for j in range(len(U_set_s_class)):
                    
                    pi_cal_class, _ = pred_opt(self.uu_classifier.net, U_set_s_class[j], U_set_a_class[j], U_set_bin_labels[j], final_iteration = final_iteration,
                                                               check_num = exp_check_num)
                    Pi_cal.append(pi_cal_class)

                if final_iteration:
                    pi_reverse_signals = []
       
                    print("U_set_s_class_all shape ", self.uu_data["U_set_s_train"].shape , U_set_s_class[0].shape)
                    print("U_set_a_class_all shape ", self.uu_data["U_set_a_train"].shape , U_set_a_class[0].shape)
                    print("U_set_bin_labels_all shape ", self.uu_data["U_sets_binLabels_train"].shape , U_set_bin_labels[0].shape)
                        

                    _, pi_reverse_signal = pred_opt(self.uu_classifier.net, self.uu_data["U_set_s_train"], self.uu_data["U_set_a_train"], self.uu_data["U_sets_binLabels_train"],
                                               final_iteration = final_iteration,
                                                               check_num = exp_check_num)
                    pi_reverse_signals.append(pi_reverse_signal)
                    if final_iteration:
                        print("pi_reverse_signal ", pi_reverse_signal)
                
                if self.pi_given:
                    print("Pi cal     : ", Pi_cal)

                    Pi_cal = self.true_Pi # use the given pi
                    print("Pi cal_use : ", Pi_cal)
                    # input("pi_given is True, press Enter to continue...")

                
                if final_iteration:
                    reverse_flag = False
                    # var_low_threshold = 0.002: # (0.1-0.9)
                    var_low_threshold = 0.0005
                    # minmax_low_threshold = 0.04

                    # var_low_threshold = 0.0001 ? # (0.05-0.2 seed 0)
                    var_Pi_cal = np.var(Pi_cal)
                    print("")
                    print("Pi_cal ", Pi_cal)
                    print("var_Pi_cal ", var_Pi_cal)
                    if var_Pi_cal < var_low_threshold:
                        print("Pi_cal has low variance, learning failed, ")
                        indicator_str = "failed_lowVar"
                    else:
                        print("pi_reverse_signals ", pi_reverse_signals)
                        # check if the Pi_cal is all True or all False
                        if all(pi_reverse_signals):
                            print("All pi_reverse_signals are True")
                            indicator_str = "allTrue_success"
                        elif not any(pi_reverse_signals):
                            print("All pi_reverse_signals are False")
                            # reverse Pi_cal
                            Pi_cal = 1 - np.array(Pi_cal)
                            indicator_str = "allFalse_reverse_success"
                            reverse_flag = True
                        else:
                            print("Pi_cal has both True and False values, keeping the original values")
                            # if more than half of the signals are False, reverse Pi_cal
                            if pi_reverse_signals.count(False) > len(pi_reverse_signals) / 2:
                                print("More than half of the pi_reverse_signals are False, reversing Pi_cal")
                                Pi_cal = 1 - np.array(Pi_cal)
                                indicator_str = "moreFalse_reverse_success"
                                reverse_flag = True
                            else:
                                print("More than half of the pi_reverse_signals are True, keeping Pi_cal unchanged")
                                indicator_str = "moreTrue_success"
                                # aa = bb
                    self.uu_classifier.net.reverse_flag = reverse_flag
                    print("indicator_str ------------------->1", indicator_str)
                # add gaussian noise to Pi_cal
                Pi_cal = np.array(Pi_cal)
                Pi_cal = np.clip(Pi_cal, 0, 1)

                self.uu_classifier.net.Pi = np.array(Pi_cal)
                

                pi_loss = np.sum(np.abs(Pi_cal - self.true_Pi))
                alpha_loss_list.append(pi_loss)


                acc_uu_train += acc_uu[0]
                acc_uu_val += acc_uu[1]
                acc_uu_train_th += acc_uu[2]
                acc_uu_val_th += acc_uu[3]
                acc_uu_bag_train += acc_uu[4]
                
                uu_bag_loss_train_list.append(uu_loss[0])
                uu_bin_loss_train_list.append(uu_loss[1])
                uu_bin_loss_val_list.append(uu_loss[2])

                acc_uu_train_list.append(acc_uu[0])
                acc_uu_val_list.append(acc_uu[1])
                acc_uu_train_th_list.append(acc_uu[2])
                acc_uu_val_th_list.append(acc_uu[3])

                epoch_plt.append(epoch_num_uu*(epoch+1))

                # uu_bag_loss_train += np.min(uu_bag_loss_train_list)
                uu_bag_loss_train += uu_loss[0]
                uu_bag_loss_train_min = np.min(uu_bag_loss_train_list)
                print("--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- uu_bag_loss_train", uu_bag_loss_train)
            

            epoch += 1 # start from 0
            print("")
            print("Epoch ====", epoch)
            print("param_init ===========================>>>>>>>>>", np.round(param_init, 4),  "  param_init_idx ", param_init_idx)
            print("Pi_true ==============================>>>>>>>>>", np.round(self.true_Pi, 4))
            print("Pi_cal ===============================>>>>>>>>>", np.round(Pi_cal, 4))

            print("")
            print(f"Epoch [{epoch}/{epochs}] | uu_bag_loss_train {uu_bag_loss_train} | uu_bag_loss_train_min: {uu_bag_loss_train_min} | acc_uu_bag_train: {acc_uu_bag_train}")
            print(f" acc_uu_train [{acc_uu_train}] | acc_uu_val: {acc_uu_val} | acc_uu_train_th: {acc_uu_train_th} | acc_uu_val_th: {acc_uu_val_th}")
            print("=====================================================================================================")
            print("\n")

            
            save_folder = self.save_folder + "Pi_estimation_csv/"
            # loss_Pi = mean of abs(Pi_cal - self.true_Pi)
            loss_pi = np.mean(np.abs(Pi_cal - self.true_Pi))

            if (acc_uu_val_list[-1] > 0.9 and acc_uu_train_list[-1] > 0.9 and loss_pi < 0.05 and (indicator_str =="allTrue_success" or indicator_str =="moreTrue_success")) or (acc_uu_val_list[-1] < 0.1 and acc_uu_train_list[-1] < 0.1 and loss_pi < 0.05 and (indicator_str == "allFalse_reverse_success" or indicator_str == "moreFalse_reverse_success")):
                decision_str = "success"
            elif (acc_uu_val_list[-1] > 0.8 and acc_uu_train_list[-1] > 0.8 and loss_pi < 0.05 and (indicator_str =="allTrue_success" or indicator_str =="moreTrue_success")) or (acc_uu_val_list[-1] < 0.2 and acc_uu_train_list[-1] < 0.2 and loss_pi < 0.05 and (indicator_str == "allFalse_reverse_success" or indicator_str == "moreFalse_reverse_success")):
                decision_str = "success 80%"
            elif (acc_uu_val_list[-1] <= 0.9 and acc_uu_train_list[-1] > 0.9 and loss_pi < 0.05 and (indicator_str =="allTrue_success" or indicator_str =="moreTrue_success")) or (acc_uu_val_list[-1] >= 0.1 and acc_uu_train_list[-1] < 0.1 and loss_pi < 0.05 and (indicator_str == "allFalse_reverse_success" or indicator_str == "moreFalse_reverse_success")):
                decision_str = "train acc 90%, val acc low"
            elif (acc_uu_val_list[-1] <= 0.8 and acc_uu_train_list[-1] > 0.8 and loss_pi < 0.05 and (indicator_str =="allTrue_success" or indicator_str =="moreTrue_success")) or (acc_uu_val_list[-1] >= 0.2 and acc_uu_train_list[-1] < 0.2 and loss_pi < 0.05 and (indicator_str == "allFalse_reverse_success" or indicator_str == "moreFalse_reverse_success")):
                decision_str = "train acc 80%, val acc low"
            elif loss_pi < 0.05 and indicator_str == "failed_lowVar":
                decision_str = "alpha_OK, but low var fail"
            elif loss_pi < 0.05:
                decision_str = "alpha_OK, why acc low?"
            else:
                decision_str = ""

            if indicator_str != "failed_lowVar" and loss_pi > 0.05:
                warn_str = f"wrong alpha but not low var > {var_low_threshold}"
            elif  loss_pi > 0.05 and var_Pi_cal >= 0.0001:
                warn_str = "wrong alpha but not low var > 0.0001"
            else:
                warn_str = " "
            
            if decision_str == "success" and indicator_str != "failed_lowVar":
                align_str = " indicator good"
            elif decision_str == "success" and indicator_str == "failed_lowVar":
                align_str = "XXXXX"
            elif indicator_str != "failed_lowVar":
                align_str = "False Positive Indicator"
            else:
                align_str = ""

            if indicator_str == "allTrue_success" or indicator_str == "moreTrue_success":
                use_str = " use "
            else:
                use_str = " "

            os.makedirs(save_folder, exist_ok=True)



            # with open(save_folder + 'uu_self_Pi_cal_env_'+str(self.uu_data["env_idx"])+"_seed_"+ str(self.uu_data["seed"])+'_self_uu_lr1e-4_ranking_.csv',  mode='a', newline='') as file:
            #     writer = csv.writer(file)
            #     if self.first_row_csv:
            #         writer.writerow(['param_init_idx', 'epoch', 'uu_bag_loss_train',  'uu_bag_loss_train min', 'acc_uu_bag_train', 'acc_uu_train', 'acc_uu_val', 'acc_uu_train_th', 'acc_uu_val_th', 
            #                             'true Pi[0]', 'est. Pi[0]',  'true Pi[1]', 'est. Pi[1]', 'true Pi[2]', 'est. Pi[2]',
            #                             'true Pi[3]', 'est. Pi[3]',  'true Pi[4]', 'est. Pi[4]', 'true Pi[5]',   'est. Pi[5]',
            #                             'param_init[0]', 'param_init[1]', 'param_init[2]', 'param_init[3]', 'param_init[4]', 'param_init[5]', 'error Pi','Indicator status', 'var_Pi', 'True status', "Align", 'warning', 'Usage'
            #             ])
            #         self.first_row_csv = False
                
            #     writer.writerow([param_init_idx, epoch*epoch_num_uu, uu_bag_loss_train, uu_bag_loss_train_min, acc_uu_bag_train, acc_uu_train, acc_uu_val, acc_uu_train_th, acc_uu_val_th, 
            #                      self.true_Pi[0], Pi_cal[0], self.true_Pi[1], Pi_cal[1], self.true_Pi[2], Pi_cal[2],
            #                         self.true_Pi[3], Pi_cal[3], self.true_Pi[4], Pi_cal[4], self.true_Pi[5], Pi_cal[5],
            #                         param_init[0], param_init[1], param_init[2], param_init[3], param_init[4], param_init[5], loss_pi, indicator_str, var_Pi_cal, decision_str, align_str, warn_str, use_str
                                  
            #     ])

            # other_saved_data_allinit.append([param_init_idx, epoch*epoch_num_uu, uu_bag_loss_train,   uu_bag_loss_train_min,    acc_uu_bag_train, acc_uu_train, acc_uu_val, acc_uu_train_th, acc_uu_val_th,
            #                     self.true_Pi[0], Pi_cal[0], self.true_Pi[1], Pi_cal[1], self.true_Pi[2], Pi_cal[2],
            #                     self.true_Pi[3], Pi_cal[3], self.true_Pi[4], Pi_cal[4], self.true_Pi[5], Pi_cal[5],
            #                     param_init[0], param_init[1], param_init[2], param_init[3], param_init[4], param_init[5],  loss_pi, indicator_str, var_Pi_cal, decision_str, align_str, warn_str, use_str
            # ])



            with open(save_folder + 'uu_self_Pi_cal_env_' + str(self.uu_data["env_idx"]) + "_seed_" + str(self.uu_data["seed"]) + '_self_uu_lr1e-4_ranking_.csv', mode='a', newline='') as file:
                writer = csv.writer(file)
                if self.first_row_csv:
                    header = [
                        'param_init_idx', 'epoch', 'uu_bag_loss_train', 'uu_bag_loss_train min',
                        'acc_uu_bag_train', 'acc_uu_train', 'acc_uu_val',
                        'acc_uu_train_th', 'acc_uu_val_th'
                    ]
                    # Add true/est Pi headers dynamically
                    for i in range(len(Pi_cal)):
                        header.append(f'true Pi[{i}]')
                        header.append(f'est. Pi[{i}]')
                    # Add param_init headers dynamically
                    for i in range(len(param_init)):
                        header.append(f'param_init[{i}]')
                    header += ['error Pi', 'Indicator status', 'var_Pi', 'True status', "Align", 'warning', 'Usage']
                    writer.writerow(header)
                    self.first_row_csv = False

                row = [
                    param_init_idx, epoch * epoch_num_uu, uu_bag_loss_train, uu_bag_loss_train_min,
                    acc_uu_bag_train, acc_uu_train, acc_uu_val,
                    acc_uu_train_th, acc_uu_val_th
                ]
                # Add true/est Pi dynamically
                for t, e in zip(self.true_Pi, Pi_cal):
                    row.extend([t, e])
                # Add param_init dynamically
                row.extend(param_init)
                row += [loss_pi, indicator_str, var_Pi_cal, decision_str, align_str, warn_str, use_str]
                writer.writerow(row)

            other_saved_data_allinit.append(row)




            file_name = "scoring_net_itr_%.d_lr_%.d_e-6_batch_%.d_" % (epochs*epoch_num_uu, lr*1e6, batch_size)

            # plot loss
            save_path = saving_path + "lossFig_" + file_name + "class_" + "seed_" + str(seed) + "_.pdf"
            plot_learning_loss(uu_bag_loss_train_list, None, epoch_plt, None, None, save_path,  label="(class loss)")
            save_path = saving_path + "lossFig_" + file_name + "bin_" + "seed_" + str(seed) + "_.pdf"
            plot_learning_loss(uu_bin_loss_train_list, uu_bin_loss_val_list, epoch_plt, None, None, save_path,  label="(bin loss)")
            save_path = saving_path + "lossFig_" + file_name + "acc_" +"seed_"+ str(seed) + "_.pdf"
            plot_learning_loss(acc_uu_train_list, acc_uu_val_list, epoch_plt, acc_uu_train_th_list, acc_uu_val_th_list, save_path,  label="(acc)")

            save_path = saving_path + "lossFig_" + file_name + "alphaLoss_" +"seed_"+ str(seed) + "_.pdf"
            plot_learning_loss(alpha_loss_list, None, epoch_plt, None, None, save_path,  label="(alpha loss)")

            # save the net
            save_path_net = Path(saving_path + "scroring_model_"+self.uu_data["env_name"] +"_epochs_"+str(epochs*epoch_num_uu) +"_NetSeed_"+str(seed)+"_.pkl")
            with open(save_path_net, "wb") as file:
                pickle.dump(self.uu_classifier, file)


            # Label the data using the trained scoring model --------------------------------------
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            U_set_s_train = self.uu_data["U_set_s_train"]
            U_set_a_train = self.uu_data["U_set_a_train"]
            U_sets_binLabels_train = self.uu_data["U_sets_binLabels_train"]

            # pick up optimal and nonoptimal data
            U_set_s_train_opt = U_set_s_train[U_sets_binLabels_train[:, 0]>0.5]
            U_set_a_train_opt = U_set_a_train[U_sets_binLabels_train[:, 0]>0.5]
            U_set_s_train_nonopt = U_set_s_train[U_sets_binLabels_train[:, 0]<0.5]
            U_set_a_train_nonopt = U_set_a_train[U_sets_binLabels_train[:, 0]<0.5]

            # label the optimal data and cauculate the accuracy =====

            U_set_s_train_opt = U_set_s_train_opt.to(device)
            U_set_a_train_opt  = U_set_a_train_opt.to(device)

            with torch.no_grad():
                self.uu_classifier.net.eval()
                opt_or_not, class_i, _ = self.uu_classifier.net.forward(U_set_s_train_opt, U_set_a_train_opt)
                opt_sigmoid = opt_or_not[:, 0]
                if indicator_str == "allFalse_reverse_success" or indicator_str == "moreFalse_reverse_success":
                    opt_sigmoid = 1 - opt_sigmoid # reverse the labels 

            classification_info = {}
    
            print("true opt num ", U_set_s_train_opt.shape)
            print("labeled opt num ", opt_or_not[opt_sigmoid>0.5].shape)
            print("labeled nonopt num ", opt_or_not[opt_sigmoid<0.5].shape)
            print("")

            classification_info["true_opt_num"] = U_set_s_train_opt.shape
            classification_info["label_opt_num"] = opt_or_not[opt_sigmoid>0.5].shape
            classification_info["label_nonopt_num"] = opt_or_not[opt_sigmoid<0.5].shape

            traj_s_labeled_opt = np.array(U_set_s_train_opt[opt_sigmoid>0.5].cpu())
            traj_a_labeled_opt = np.array(U_set_a_train_opt[opt_sigmoid>0.5].cpu())
            traj_a_labeled_nonopt = np.array(U_set_a_train_opt[opt_sigmoid<0.5].cpu())
            traj_s_labeled_nonopt = np.array(U_set_s_train_opt[opt_sigmoid<0.5].cpu())

            # label the nonoptimal data and cauculate the accuracy =====
            U_set_s_train_nonopt = U_set_s_train_nonopt.to(device)
            U_set_a_train_nonopt = U_set_a_train_nonopt.to(device)
            with torch.no_grad():
                opt_or_not, class_i, _ = self.uu_classifier.net.forward(U_set_s_train_nonopt, U_set_a_train_nonopt)
                opt_sigmoid = opt_or_not[:, 0]
                if indicator_str == "allFalse_reverse_success" or indicator_str == "moreFalse_reverse_success":
                    opt_sigmoid = 1 - opt_sigmoid # reverse the labels 

            print("true nonopt num ", U_set_s_train_nonopt.shape)
            print("labeled opt num - ", opt_or_not[opt_sigmoid>0.5].shape)
            false_opt_num = opt_or_not[opt_sigmoid>0.5].shape[0] 
            print("labeled nonopt num - ", opt_or_not[opt_sigmoid<0.5].shape)
            print("")

            classification_info["true_nonopt_num"] = U_set_s_train_nonopt.shape
            classification_info["label_opt_num_"] = opt_or_not[opt_sigmoid>0.5].shape
            classification_info["label_nonopt_num_"] = opt_or_not[opt_sigmoid<0.5].shape

            traj_s_labeled_opt = np.concatenate((traj_s_labeled_opt, U_set_s_train_nonopt[opt_sigmoid>0.5].cpu()), axis=0)
            traj_a_labeled_opt = np.concatenate((traj_a_labeled_opt, U_set_a_train_nonopt[opt_sigmoid>0.5].cpu()), axis=0)
            traj_a_labeled_nonopt = np.concatenate((traj_a_labeled_nonopt, U_set_a_train_nonopt[opt_sigmoid<0.5].cpu()), axis=0)
            traj_s_labeled_nonopt = np.concatenate((traj_s_labeled_nonopt, U_set_s_train_nonopt[opt_sigmoid<0.5].cpu()), axis=0)

            print("traj_s_labeled_opt ", traj_s_labeled_opt.shape)
            print("traj_a_labeled_opt ", traj_a_labeled_opt.shape)
            print("traj_s_labeled_nonopt ", traj_s_labeled_nonopt.shape)
            print("traj_a_labeled_nonopt ", traj_a_labeled_nonopt.shape)

            classification_info["traj_s_labeled_opt"] = traj_s_labeled_opt.shape
            classification_info["traj_a_labeled_opt"] = traj_a_labeled_opt.shape
            classification_info["traj_s_labeled_nonopt"] = traj_s_labeled_nonopt.shape
            classification_info["traj_a_labeled_nonopt"] = traj_a_labeled_nonopt.shape
            classification_info["est Pi"] = Pi_cal
            classification_info["true Pi"] = self.true_Pi


            # save the labeled data ---------------------------------------------------
            # # Save opt and nonopt data ===============================================
            # # normalized data
            save_data = True
            if save_data:
                save_trajs_dir =  self.save_folder + "opt_nonopt_trajs_alphaEst/"
                os.makedirs(save_trajs_dir, exist_ok=True)
                expert_seed_str = str(seed)
                save_trajs_path = save_trajs_dir + "opt_nonopt_trajs_"+ self.uu_data["env_name"] +"_ExpertSeed_"+expert_seed_str+"_NetSeed"+str(seed) +"_epochs_"+str(epochs*epoch_num_uu) +"_pseudo_labeled_alpha_"+self.uu_data["opt_ratio_alpha"][0]+self.uu_data["opt_ratio_alpha"][2:]
    
                save_trajs_path = Path(save_trajs_path + "_.pkl")

                # data is normalized
                with open(save_trajs_path, 'wb') as f:
                    pickle.dump({'opt_traj_s_set': traj_s_labeled_opt, 
                                'opt_traj_a_set': traj_a_labeled_opt, 
                                'nonopt_traj_s_set': traj_s_labeled_nonopt, 
                                'nonopt_traj_a_set': traj_a_labeled_nonopt,
                                    'scaler_s': self.uu_data["input_scaler_s"],
                                    'scaler_a': self.uu_data["input_scaler_a"],
                                    'opt_start_steps': self.uu_data["opt_start_steps"],
                                    }, f)
                    
                # save classification_info as txt file
                classification_info_path = save_trajs_dir + "opt_nonopt_trajs_"+self.uu_data["env_name"]+"_ExpertSeed_"+expert_seed_str+"_NetSeed"+str(seed) +"_epochs_"+str(epochs*epoch_num_uu) +"_classification_info_alpha_"+self.uu_data["opt_ratio_alpha"][0]+self.uu_data["opt_ratio_alpha"][2:]

                classification_info_path = Path(classification_info_path + "_.txt")

                with open(classification_info_path, 'w') as f:
                    for key in classification_info.keys():
                        f.write("%s: %s\n" % (key, classification_info[key]))
                    f.write("\n" )

            if indicator_str == "allTrue_success" or indicator_str == "moreTrue_success":
                print("Good indicator, stop here.")
                break

            