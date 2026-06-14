import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import random_split, DataLoader, TensorDataset
from pathlib import Path
import copy
import matplotlib.pyplot as plt
import time
import random
import os




class Scoring_model_net(nn.Module):
    def __init__(self, priors_class, Pi, Pi_test, input_dim_s, input_dim_a,):
        super(Scoring_model_net, self).__init__()
        self.priors_corr = priors_class
        self.Pi = Pi
        self.Pi_test = Pi_test
        print("priors_class ",  priors_class)
        print("Pi ", Pi)    
        print("Pi_test ", Pi_test)

        self.relu = nn.LeakyReLU()
        bias_control = True # why better without bias?

        self.state_layer = nn.Sequential(
            nn.Linear(input_dim_s, 64, bias=bias_control),
            nn.LeakyReLU(),
             nn.Linear(64, 64, bias=bias_control),
            nn.LeakyReLU()
        )

        self.action_layer = nn.Sequential(
            nn.Linear(input_dim_a, 64, bias=bias_control),
            nn.LeakyReLU(),
            nn.Linear(64, 64, bias=bias_control),
            nn.LeakyReLU()
        )

        self.dropout1 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(32, 32, bias=bias_control)
        self.bn1 = nn.BatchNorm1d(32)

        self.dropout2 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(32, 32, bias=bias_control)
        self.bn2 = nn.BatchNorm1d(32)

        self.dropout3 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(128, 128, bias=bias_control)
        self.bn3 = nn.BatchNorm1d(32)

        self.dropout4 = nn.Dropout(0.2)
        self.fc4 = nn.Linear(128, 128, bias=bias_control)
        self.bn4 = nn.BatchNorm1d(32)

        self.fc5 = nn.Linear(128, 1, bias=bias_control)
    
    def T_func(self, g):
        c = 0
        d = 0
        output = []
        sets = len(self.Pi)

        for i in range(sets):
            c += self.priors_corr[i] * (self.Pi[i] - self.Pi_test)
            d += (1 - self.Pi[i]) * self.priors_corr[i] * self.Pi_test

        for i in range(sets):
            a = self.priors_corr[i] * (self.Pi[i] - self.Pi_test)
            b = (1 - self.Pi[i]) * self.Pi_test * self.priors_corr[i]
            output.append((a * g + b) / (c * g + d))

        res = torch.cat(output, dim=1)
        return res

    def forward(self, x_s, x_a):
        # x_s = self.input_scaler_s.transform(x_s)
        # x_a = self.input_scaler_a.transform(x_a)
        # start_time = time.time()
        # x = self.dropout1(x)
        if True:
            x_state = self.state_layer(x_s)
            x_action = self.action_layer(x_a)
            x = torch.cat((x_state, x_action), dim=1)

        else:
            x = self.fc1(x)
            # # x = self.bn1(x)
            x = self.relu(x)

            # # x = self.dropout2(x)
            x = self.fc2(x)
            # # x = self.bn2(x)
            x = self.relu(x)

        # x = self.dropout3(x)
        x = self.fc3(x)
        # x = self.bn3(x)
        x = self.relu(x)

        # x = self.dropout4(x)
        x = self.fc4(x)
        # x = self.bn4(x)
        x = self.relu(x)

        x = self.fc5(x)
        binary_y = torch.sigmoid(x)
        # t1 = time.time()
        

        class_y = self.T_func(binary_y)
        # class_y = torch.rand(20, 20, requires_grad=True)

        # t2 = time.time()

        # print("forward time: ", t1 - start_time)
        # print("T_func time: ", t2 - t1)
        
        return torch.cat((binary_y, 1-binary_y), dim=1), class_y
    

class Scoring_model_net_multiFrame(nn.Module):
    def __init__(self, priors_class, Pi, Pi_test, input_dim_s, input_dim_a, frame_num = 4, ):
        super(Scoring_model_net_multiFrame, self).__init__()
        self.priors_corr = priors_class
        self.Pi = Pi
        self.Pi_test = Pi_test
        self.frame_num = frame_num
        print("priors_class ",  priors_class)
        print("Pi ", Pi)    
        print("Pi_test ", Pi_test)
        print("input_dim_s ", input_dim_s)
        print("input_dim_a ", input_dim_a)

        self.relu = nn.LeakyReLU()
        bias_control = True # why better without bias?
        
        self.state_layers = nn.ModuleList()
        self.action_layers = nn.ModuleList()
        for i in range(frame_num):
            layer = nn.Sequential(
                nn.Linear(input_dim_s, 64, bias=bias_control),
                nn.LeakyReLU(),
                nn.Linear(64, 64, bias=bias_control),
                nn.LeakyReLU()
            )
            self.state_layers.append(layer)

            layer = nn.Sequential(
                nn.Linear(input_dim_a, 64, bias=bias_control),
                nn.LeakyReLU(),
                nn.Linear(64, 64, bias=bias_control),
                nn.LeakyReLU()
            )
            self.action_layers.append(layer)

        # self.state_layer = nn.Sequential(
        #     nn.Linear(input_dim_s, 64, bias=bias_control),
        #     nn.LeakyReLU(),
        #      nn.Linear(64, 64, bias=bias_control),
        #     nn.LeakyReLU()
        # )

        # self.action_layer = nn.Sequential(
        #     nn.Linear(input_dim_a, 64, bias=bias_control),
        #     nn.LeakyReLU(),
        #     nn.Linear(64, 64, bias=bias_control),
        #     nn.LeakyReLU()
        # )

        # self.dropout1 = nn.Dropout(0.2)
        # self.fc1 = nn.Linear(32, 32, bias=bias_control)
        # self.bn1 = nn.BatchNorm1d(32)

        # self.dropout2 = nn.Dropout(0.2)
        # self.fc2 = nn.Linear(32, 32, bias=bias_control)
        # self.bn2 = nn.BatchNorm1d(32)

        # self.dropout3 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(128*frame_num, 128*frame_num, bias=bias_control)
        # self.bn3 = nn.BatchNorm1d(32)

        # self.dropout4 = nn.Dropout(0.2)
        self.fc4 = nn.Linear(128*frame_num, 128*frame_num, bias=bias_control)
        # self.bn4 = nn.BatchNorm1d(32)

        self.fc5 = nn.Linear(128*frame_num, 1, bias=bias_control)
    
    def T_func(self, g,):

        Pi = self.Pi
 
        c = 0
        d = 0
        output = []
        sets = len(Pi)

        for i in range(sets):
            c += self.priors_corr[i] * (Pi[i] - self.Pi_test)
            d += (1 - Pi[i]) * self.priors_corr[i] * self.Pi_test

        for i in range(sets):
            a = self.priors_corr[i] * (Pi[i] - self.Pi_test)
            b = (1 - Pi[i]) * self.Pi_test * self.priors_corr[i]
            output.append((a * g + b) / (c * g + d))

        res = torch.cat(output, dim=1)
        return res

    def forward(self, x_s, x_a,):
        # print("x_s shape: ", x_s.shape)
        # print("x_a shape: ", x_a.shape)
        # x_s = self.input_scaler_s.transform(x_s)
        # x_a = self.input_scaler_a.transform(x_a)
        # start_time = time.time()
        # x = self.dropout1(x)
        if True:
            for i in range(self.frame_num):
                x_state = self.state_layers[i](x_s[:,i,:])
                x_action = self.action_layers[i](x_a[:,i,:])

                # x_state = self.state_layer(x_s[:,i,:])
                # x_action = self.action_layer(x_a[:,i,:])

                if i == 0:
                    x_state_all = x_state
                    x_action_all = x_action
                else:
                    x_state_all = torch.cat((x_state_all, x_state), dim=1)
                    x_action_all = torch.cat((x_action_all, x_action), dim=1)

            x = torch.cat((x_state_all, x_action_all), dim=1)
            # print("x shape: ", x.shape)

        else:
            x = self.fc1(x)
            # # x = self.bn1(x)
            x = self.relu(x)

            # # x = self.dropout2(x)
            x = self.fc2(x)
            # # x = self.bn2(x)
            x = self.relu(x)

        # x = self.dropout3(x)
        x = self.fc3(x)
        x_fc3 = x
        # x = self.bn3(x)
        x = self.relu(x)

        # x = self.dropout4(x)
        x = self.fc4(x)
        x_fc4 = x
        # x = self.bn4(x)
        x = self.relu(x)

        x = self.fc5(x)
        x_fc5 = x
        logits = x
        binary_y = torch.sigmoid(x)
        # t1 = time.time()
        
        

        class_y = self.T_func(binary_y,)
        # class_y = torch.rand(20, 20, requires_grad=True)

        # t2 = time.time()

        # print("forward time: ", t1 - start_time)
        # print("T_func time: ", t2 - t1)
        infos = {"x_state_all": x_state_all, "x_action_all": x_action_all, "x_fc3": x_fc3, "x_fc4": x_fc4, "x_fc5": x_fc5, 
                 "binary_y": binary_y, "class_y": class_y, "x_s": x_s, "x_a": x_a}
        return torch.cat((binary_y, 1-binary_y, logits), dim=1), class_y, infos


class Scoring_model_net_multiFrame_v(nn.Module):
    def __init__(self, priors_class, Pi, Pi_test, input_dim_s, input_dim_a, frame_num = 4, ):
        super(Scoring_model_net_multiFrame_v, self).__init__()
        self.priors_corr = priors_class
        self.Pi = Pi
        self.Pi_test = Pi_test
        self.frame_num = frame_num
        print("priors_class ",  priors_class)
        print("Pi ", Pi)    
        print("Pi_test ", Pi_test)

        self.relu = nn.LeakyReLU()
        bias_control = True # why better without bias?
        
        self.state_layers = nn.ModuleList()
        self.state_v_layers = nn.ModuleList()
        self.action_layers = nn.ModuleList()
        for i in range(frame_num):
            layer = nn.Sequential(
                nn.Linear(input_dim_s, 64, bias=bias_control),
                nn.LeakyReLU(),
                nn.Linear(64, 32, bias=bias_control),
                nn.LeakyReLU()
            )
            self.state_layers.append(layer)

            layer = nn.Sequential(
                nn.Linear(3, 64, bias=bias_control),
                nn.LeakyReLU(),
                nn.Linear(64, 32, bias=bias_control),
                nn.LeakyReLU()
            )
            self.state_v_layers.append(layer)

            layer = nn.Sequential(
                nn.Linear(input_dim_a, 64, bias=bias_control),
                nn.LeakyReLU(),
                nn.Linear(64, 64, bias=bias_control),
                nn.LeakyReLU()
            )
            self.action_layers.append(layer)



        # self.state_layer = nn.Sequential(
        #     nn.Linear(input_dim_s, 64, bias=bias_control),
        #     nn.LeakyReLU(),
        #      nn.Linear(64, 64, bias=bias_control),
        #     nn.LeakyReLU()
        # )

        # self.action_layer = nn.Sequential(
        #     nn.Linear(input_dim_a, 64, bias=bias_control),
        #     nn.LeakyReLU(),
        #     nn.Linear(64, 64, bias=bias_control),
        #     nn.LeakyReLU()
        # )

        # self.dropout1 = nn.Dropout(0.2)
        # self.fc1 = nn.Linear(32, 32, bias=bias_control)
        # self.bn1 = nn.BatchNorm1d(32)

        # self.dropout2 = nn.Dropout(0.2)
        # self.fc2 = nn.Linear(32, 32, bias=bias_control)
        # self.bn2 = nn.BatchNorm1d(32)

        self.dropout3 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(128*frame_num, 128*frame_num, bias=bias_control)
        self.bn3 = nn.BatchNorm1d(32)

        self.dropout4 = nn.Dropout(0.2)
        self.fc4 = nn.Linear(128*frame_num, 32*frame_num, bias=bias_control)
        self.bn4 = nn.BatchNorm1d(32)

        self.fc5 = nn.Linear(32*frame_num, 1, bias=bias_control)
    
    def T_func(self, g):
        c = 0
        d = 0
        output = []
        sets = len(self.Pi)

        for i in range(sets):
            c += self.priors_corr[i] * (self.Pi[i] - self.Pi_test)
            d += (1 - self.Pi[i]) * self.priors_corr[i] * self.Pi_test

        for i in range(sets):
            a = self.priors_corr[i] * (self.Pi[i] - self.Pi_test)
            b = (1 - self.Pi[i]) * self.Pi_test * self.priors_corr[i]
            output.append((a * g + b) / (c * g + d))

        res = torch.cat(output, dim=1)
        return res

    def forward(self, x_s, x_a):
        # print("x_s shape: ", x_s.shape)
        # print("x_a shape: ", x_a.shape)
        # x_s = self.input_scaler_s.transform(x_s)
        # x_a = self.input_scaler_a.transform(x_a)
        # start_time = time.time()
        # x = self.dropout1(x)
        if True:
            for i in range(self.frame_num):
                x_state = self.state_layers[i](x_s[:,i,:])
                x_state_v = self.state_v_layers[i](x_s[:, i, 13:16])
                x_action = self.action_layers[i](x_a[:,i,:])

                # x_state = self.state_layer(x_s[:,i,:])
                # x_action = self.action_layer(x_a[:,i,:])

                if i == 0:
                    x_state_all = x_state
                    x_state_v_all = x_state_v
                    x_action_all = x_action
                else:
                    x_state_all = torch.cat((x_state_all, x_state), dim=1)
                    x_state_v_all = torch.cat((x_state_v_all, x_state_v), dim=1)
                    x_action_all = torch.cat((x_action_all, x_action), dim=1)

            x = torch.cat((x_state_all, x_state_v_all, x_action_all), dim=1)
            # print("x shape: ", x.shape)

        else:
            x = self.fc1(x)
            # # x = self.bn1(x)
            x = self.relu(x)

            # # x = self.dropout2(x)
            x = self.fc2(x)
            # # x = self.bn2(x)
            x = self.relu(x)

        # x = self.dropout3(x)
        x = self.fc3(x)
        # x = self.bn3(x)
        x = self.relu(x)

        # x = self.dropout4(x)
        x = self.fc4(x)
        # x = self.bn4(x)
        x = self.relu(x)

        x = self.fc5(x)
        binary_y = torch.sigmoid(x)
        # t1 = time.time()
        

        class_y = self.T_func(binary_y)
        # class_y = torch.rand(20, 20, requires_grad=True)

        # t2 = time.time()

        # print("forward time: ", t1 - start_time)
        # print("T_func time: ", t2 - t1)
        
        return torch.cat((binary_y, 1-binary_y), dim=1), class_y



# def split_train_val(x_input, y_output, data_num, ram_split_data):
#     def _split_in_out(_in_output):
#         _in_output = np.array(_in_output)
#         _input = np.expand_dims(_in_output[:, 0], axis=1)
#         _output = np.expand_dims(_in_output[:, 1], axis=1)
#         return _input, _output

#     if ram_split_data:
#         data = list(zip(x_input, y_output))
#         train_dataset, val_dataset = random_split(data, [data_num[0], data_num[1]])
#         train_input, train_output = _split_in_out(train_dataset)
#         val_input, val_output = _split_in_out(val_dataset)
#     else:
#         train_input, train_output = x_input[0:data_num[0]], y_output[0:data_num[0]]
#         val_input, val_output = x_input[data_num[0]:], y_output[data_num[0]:]

#     return train_input, train_output, val_input, val_output


def train_val_loader(train_input_s, train_input_a, train_output, train_output_bin,
                     val_input_s, val_input_a, val_output, val_output_bin, batch_size, device):

    def _torch_batch(_train_input_s, _train_input_a, _train_output, _train_output_bin,
                      batch_size, shuffle, device):
        # if _train_input_s is not tensor
        if not torch.is_tensor(_train_input_s):
            _train_input_s = torch.from_numpy(_train_input_s).float().to(device)
            _train_input_a = torch.from_numpy(_train_input_a).float().to(device)
        else:
            _train_input_s = _train_input_s.float().to(device)
            _train_input_a = _train_input_a.float().to(device)
        
        _train_output = torch.from_numpy(_train_output).float().to(device)
        _train_output_bin = torch.from_numpy(_train_output_bin).float().to(device)

        _dataset = TensorDataset(_train_input_s, _train_input_a, _train_output, _train_output_bin)
        _train_loader = DataLoader(_dataset, batch_size=batch_size, shuffle=shuffle)
        return _train_loader

    train_loader = _torch_batch(train_input_s, train_input_a, train_output, train_output_bin, 
                                batch_size=batch_size, shuffle=True, device=device)
    val_loader = _torch_batch(val_input_s, val_input_a, val_output, val_output_bin,
                              batch_size=batch_size, shuffle=False, device=device)
    return train_loader, val_loader



def plot_learning_loss(training_loss, validation_loss, epoch_set, train_acc_th, val_acc_th, save_path, label="", th_pos=None):
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))

    axs[0].plot(epoch_set, training_loss, label='Training Loss' + label)
    if validation_loss is not None:
        axs[0].plot(epoch_set, validation_loss, label='Validation Loss' + label)
        axs[0].set_title('Training and Validation '+ label+', tr: %.4f val: %.4f' % (np.mean(training_loss[-10:]), np.mean(validation_loss[-10:])) + label)
        if train_acc_th is not None:
            axs[0].plot(epoch_set, train_acc_th, label='Training Acc th ' + label)
            axs[0].plot(epoch_set, val_acc_th, label='Val Acc th ' + label)
            axs[0].set_title('Training and Validation '+ label+', tr: %.4f tr_th: %.4f val: %.4f val_th: %.4f' % (np.mean(training_loss[-10:]), np.mean(train_acc_th[-10:]), np.mean(validation_loss[-10:]), np.mean(val_acc_th[-10:])))
    else:
        axs[0].set_title('Training and Validation Loss, tr: %.4f ' % (np.mean(training_loss[-10:]),))
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel(label)

    starting_epoch = int(len(epoch_set)/2)
    axs[1].plot(epoch_set[starting_epoch:], training_loss[starting_epoch:], label='Training Loss ' + label)
    if validation_loss is not None:
        axs[1].plot(epoch_set[starting_epoch:], validation_loss[starting_epoch:], label='Validation Loss ' + label)
        if train_acc_th is not None:
            axs[1].plot(epoch_set[starting_epoch:], train_acc_th[starting_epoch:], label='Training Acc th' + label)
            axs[1].plot(epoch_set[starting_epoch:], val_acc_th[starting_epoch:], label='Val Acc th' + label)
    axs[1].set_title('Training and Validation Loss ' + label)
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel(label)

    plt.legend()
    plt.savefig(save_path)
    # plt.savefig(save_path, format="svg")
    # plt.show()
    plt.close()

def cal_accuracy(y_predicted_bin, y_true_bin, bag_cal = False):
    labels_pred = torch.argmax(y_predicted_bin.data, dim=1)
    labels_true = torch.argmax(y_true_bin.data, dim=1)
    correct_preds = torch.sum(labels_pred == labels_true).item()
    acc = correct_preds / len(labels_true)

    if bag_cal:
        print("y_predicted_bin: ", y_predicted_bin[:5])
        print("y_true_bin: ", y_true_bin[:5])
        print("labels_pred: ", labels_pred[:5])
        print("labels_true: ", labels_true[:5])
        print("correct_preds: ", correct_preds)
        print("len(labels_true): ", len(labels_true))
        print("acc: ", acc)
        print("\n")
    # exit()
    
    # print(f"cal_accuracy   : {acc * 100:.6f}%")
    return acc


def cal_accuracy_th(model_predictions, actual_labels, device, threshold_positive = 0.85):
    # 设置阈值
    threshold_negative = 1-threshold_positive

    # 使用阈值将模型的预测转化为二元标签
    predicted_labels = torch.zeros(model_predictions.size(), dtype=torch.int)  # 先将所有标签初始化为错误类
    predicted_labels[model_predictions[:, 0] >= threshold_positive, 0] = 1  # 预测为类别 0
    predicted_labels[model_predictions[:, 1] >= threshold_positive, 1] = 1  # 预测为类别 1
    predicted_labels[(model_predictions[:, 0] < threshold_positive) & (model_predictions[:, 0] > threshold_negative)] = -1  # 预测为错误类
    # 计算正确分类的数量
    correct_predictions = torch.sum(torch.all(predicted_labels.to(device) == actual_labels, dim=1))
    # 计算正确率
    accuracy = correct_predictions.item() / len(actual_labels)        
    # print(f"cal_accuracy_th: {accuracy * 100:.6f}%")
    # print("\n")
    return accuracy


class Scoring_model():
    def __init__(self, priors_class, Pi, Pi_test, input_dim_s, input_dim_a, input_scaler_s, input_scaler_a, frame_num = 4,):
        self.net = Scoring_model_net_multiFrame(priors_class, Pi, Pi_test, input_dim_s, input_dim_a, frame_num = frame_num)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("device: ", self.device)
        # self.loss_func = torch.nn.MSELoss()
        self.loss_func = torch.nn.CrossEntropyLoss()
        self.loss_func.to(self.device)
        self.input_scaler_s = input_scaler_s
        self.input_scaler_a = input_scaler_a
        self.y_output_scaler = None
        self.th_pos = 0.85
        self.frame_num = frame_num
    
    def net_training(self, x_input_train, y_output_train, y_output_bin_train,
                      x_input_test, y_output_test, y_output_bin_test, 
                      
                      epoch_num=int(1e4), lr=1e-4, batch_size=200,
                      device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), saving_path="",
                     seed=0, plot_loss= True):
        

        test_idx = seed

        if False:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # If using CUDA
            np.random.seed(seed)
            random.seed(seed)


        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
        
        

        # x_input[0] is the state, x_input[1] is the action (action array needs no normalization)
        # print("train data shape (state): ", x_input_train[0].shape)
        # print("test data shape (state): ", x_input_test[0].shape)
        # print("train data shape (action): ", x_input_train[1].shape)
        # print("test data shape (action): ", x_input_test[1].shape)

        # print("max train data (state): ", np.max(x_input_train[0]))
        # print("min train data (state): ", np.min(x_input_train[0]))
        # print("max test data (state): ", np.max(x_input_test[0]))
        # print("min test data (state): ", np.min(x_input_test[0]))
        # print("max train data (action): ", np.max(x_input_train[1]))
        # print("min train data (action): ", np.min(x_input_train[1]))
        # print("max test data (action): ", np.max(x_input_test[1]))
        # print("min test data (action): ", np.min(x_input_test[1]))

        
        # train_input_s, train_output, train_output_bin = x_input_train[0], y_output_train, y_output_bin_train
        # val_input, val_output, val_output_bin = x_input_test[0], y_output_test, y_output_bin_test

        # x_input_scaler, train_input_scaled_s = data_normalizing(x_input_train[0])
        # X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
        # X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

        # print(train_input_scaled)
        # y_output_scaler, train_output_scaled = data_normalizing(y_output_train)

        # val_input_scaled_s = x_input_scaler.transform(x_input_test[0])
        # val_output_scaled = y_output_scaler.transform(val_output)
        val_output_scaled = y_output_test


        train_loader, val_loader = train_val_loader(train_input_s = x_input_train[0],   train_input_a = x_input_train[1], 
                                                    train_output = y_output_train,     train_output_bin = y_output_bin_train, 

                                                     val_input_s = x_input_test[0],  val_input_a = x_input_test[1], 
                                                     val_output = y_output_test,    val_output_bin = y_output_bin_test, 

                                                     batch_size = batch_size, device = self.device)
  

        self.net.to(self.device)
        Pi_parameters = [{'params': self.net.Pi}]
        # print("self.net.Pi: ", self.net.Pi)
        nn_parameters = [{'params': self.net.parameters()}]
        optimizer_nn = torch.optim.Adam(nn_parameters, lr=lr, weight_decay=0e-05)

 
        
        # optimizer_nn_pi = torch.optim.Adam(Pi_parameters + nn_parameters, lr=lr, weight_decay=0e-05)
        en_EM_pi = False

        en_opt_pi = False
        if en_opt_pi:
            optimizer_pi = torch.optim.Adam(Pi_parameters, lr=lr*10, weight_decay=0e-05)

        # print("nn_parameters: ", nn_parameters)
        # # Print the shape of each parameter in the network
        # for param_group in nn_parameters:
        #     for param in param_group['params']:
        #         print(f"Parameter Shape: {param.shape}")

        # for name, param in self.net.named_parameters():
        #     print(f"Parameter Name: {name}, Shape: {param.shape}")

        # optimizer = torch.optim.Adam(nn_parameters, lr=lr, weight_decay=0e-05)
        

        train_loss_set = []
        train_loss_set_bin = []
        train_acc_bag_set = []
        train_acc_set = []
        train_acc_set_th = []
        val_loss_set_bin = []
        val_acc_set = []
        val_acc_set_th = []

        epoch_set = []

        for epoch in range(epoch_num):
            epoch_counter = 0
            self.net.train()
            # start_time = time.time()
            train_loss_sum = 0
            acc_bag_train_sum = 0
            for batch_i, (x_batch_s, x_batch_a, y_batch, y_batch_bin) in enumerate(train_loader):
                # print("x_batch_s: ", x_batch_s)
                # print("x_batch_a: ", x_batch_a)
                
                start_time_in = time.time()
                # scaling data here because int8 data needs less storage (8 times) space compared to float64
                # x_batch_s = x_batch_s / self.x_input_scaler


                y_predicted_bin, y_predicted, _ = self.net.forward(x_batch_s, x_batch_a)
                # y_predicted_bin
                y_predicted_bin = y_predicted_bin[:, 0:2] # only use the first two columns, get rid of the logits
                # print("y_predicted_bin", y_predicted_bin.shape)
                # print("y_batch_bin", y_batch_bin.shape)
                # print("y_predicted", y_predicted.shape)
                # print("y_batch", y_batch.shape)
                # t1 = time.time()
                train_loss = self.loss_func(y_predicted, y_batch)
                train_loss_sum += train_loss.item()

                acc_bag_train = cal_accuracy(y_predicted, y_batch, bag_cal = False)
                acc_bag_train_sum += acc_bag_train

                # t2 = time.time()

                optimizer_nn.zero_grad()
                if en_opt_pi:
                    optimizer_pi.zero_grad()
                train_loss.backward(retain_graph=True)
                optimizer_nn.step()
                # print("self.net.Pi ++++++++: ", self.net.Pi)

                if en_opt_pi:
                    if epoch%10 == 0:
                        optimizer_pi.step()


                # t3 = time.time()
                # print("forward time: ", t3 - start_time_in)

            # t4 = time.time()
            # print("epoch time: ", t4 - start_time)


            # if (epoch + 1) % 1 == 0:
            #     print("epoch: ", epoch)

            if (epoch + 1) % 1 == 0:
                val_loss_sum = 0
                accuracy_sum = 0
                accuracy_sum_th = 0
                counter = 0
                self.net.eval()
                with torch.no_grad():
                    train_loss_bin = self.loss_func(y_predicted_bin, y_batch_bin)
                    # print("Train ===================" )
                    train_acc = cal_accuracy(y_predicted_bin, y_batch_bin)
                    train_acc_th = cal_accuracy_th(y_predicted_bin, y_batch_bin, device = self.device, threshold_positive=self.th_pos)
                    if train_acc_th > train_acc:
                        print("Train Wrong!")
                
                    for _, (x_batch_s, x_batch_a, _, y_batch_bin) in enumerate(val_loader):
                        # scaling data here because int8 data needs less storage (8 times) space compared to float64
                        # x_batch_s = x_batch_s / self.x_input_scaler

                        y_predicted_bin, y_predicted, _ = self.net.forward(x_batch_s, x_batch_a)
                        y_predicted_bin = y_predicted_bin[:, 0:2] # only use the first two columns, get rid of the logits
                        val_loss_sum += self.loss_func(y_predicted_bin, y_batch_bin).item()

                        # Calculate the number of correctly classified
                        # print("Val ===================" )
                        acc = cal_accuracy(y_predicted_bin, y_batch_bin)
                        accuracy_sum += acc
                        acc_th= cal_accuracy_th(y_predicted_bin, y_batch_bin, device = self.device, threshold_positive=self.th_pos)
                        accuracy_sum_th += acc_th

                        if acc_th > acc:
                            print("Val Wrong!")
                        counter += 1

                    val_loss = val_loss_sum / counter
                    accuracy = accuracy_sum / counter
                    accuracy_th = accuracy_sum_th / counter
            
                train_loss_set.append(train_loss_sum/len(train_loader))
                train_acc_bag_set.append(acc_bag_train_sum/len(train_loader))
                train_loss_set_bin.append(train_loss_bin.item())
                train_acc_set.append(train_acc)
                train_acc_set_th.append(train_acc_th)

                val_loss_set_bin.append(val_loss)
                val_acc_set.append(accuracy)
                val_acc_set_th.append(accuracy_th)
                epoch_set.append(epoch)

            if (epoch + 1) % 5 == 0:
                print("")
                print(
                    f" Epoch {epoch + 1} train loss: {train_loss_set[-1]:.8f}, train (bin) loss: {train_loss_bin.item():.5f}, train bag acc: {train_acc_bag_set[-1]:.5f}    train accuracy: {train_acc:.5f}, train accuracy th: {train_acc_th:.5f}, val (bin) loss: {val_loss:.5f}, val accuracy: {accuracy:.5f}, val accuracy th: {accuracy_th:.5f} ")
                print("self.net.Pi ++++++++: ", self.net.Pi)
                
        # save trained net
        file_name = "scoring_net_itr_%.d_lr_%.d_e-6_batch_%.d_" % (epoch_num, lr*1e6, batch_size)
        save_path_net = Path(saving_path + file_name +"test_"+ str(test_idx) + ".pickle")

        os.makedirs(os.path.dirname(saving_path), exist_ok=True)

        # plot loss
        if plot_loss:
            save_path = saving_path + "lossFig_" + file_name + "class_" + "test_" + str(test_idx) + "_.pdf"
            plot_learning_loss(train_loss_set, None, epoch_set, None, None, save_path,  label="(class loss)")
            save_path = saving_path + "lossFig_" + file_name + "bin_" + "test_" + str(test_idx) + "_.pdf"
            plot_learning_loss(train_loss_set_bin, val_loss_set_bin, epoch_set,  None, None,save_path, label="(bin loss)")
            save_path = saving_path + "lossFig_" + file_name + "acc_" +"test_"+ str(test_idx) + "_.pdf"
            plot_learning_loss(train_acc_set, val_acc_set, epoch_set, train_acc_set_th, val_acc_set_th, save_path, label="(accuaracy)", th_pos = self.th_pos)


        loss_set = [train_loss_set[-1], np.mean(train_loss_set_bin[-10:]), np.mean(val_loss_set_bin[-10:]),]
        acc_set = [np.mean(train_acc_set[-5:]), np.mean(val_acc_set[-5:]),np.mean(train_acc_set_th[-5:]), np.mean(val_acc_set_th[-5:]), np.mean(train_acc_bag_set[-5:])]
        print("train loss: ", loss_set[0])

        return self.net, loss_set, save_path_net, acc_set
    
    def net_val(self, x_input_train, y_output_train, y_output_bin_train,
                      x_input_test, y_output_test, y_output_bin_test, 
                      
                      epoch_num=int(1e4), lr=1e-4, batch_size=200,
                      device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), saving_path="",
                     seed=0):
        
        train_loader, val_loader = train_val_loader(train_input_s = x_input_train[0],   train_input_a = x_input_train[1], 
                                            train_output = y_output_train,     train_output_bin = y_output_bin_train, 

                                                val_input_s = x_input_test[0],  val_input_a = x_input_test[1], 
                                                val_output = y_output_test,    val_output_bin = y_output_bin_test, 

                                                batch_size = batch_size, device = self.device)
        
        self.net.to(self.device)
        Pi_parameters = [{'params': self.net.Pi}]
        print("self.net.Pi: ", self.net.Pi)

        train_loss_set = []
        train_loss_set_bin = []
        train_acc_bag_set = []
        train_acc_set = []
        train_acc_set_th = []
        val_loss_set_bin = []
        val_acc_set = []
        val_acc_set_th = []


        self.net.eval()
        with torch.no_grad():
            train_loss_sum = 0
            acc_bag_train_sum = 0
            for step, (x_batch_s, x_batch_a, y_batch, y_batch_bin) in enumerate(train_loader):    
                
                y_predicted_bin, y_predicted, _ = self.net.forward(x_batch_s, x_batch_a)
                # y_predicted_bin
                y_predicted_bin = y_predicted_bin[:, 0:2] 
                train_loss = self.loss_func(y_predicted, y_batch)
                train_loss_sum += train_loss.item()

                acc_bag_train = cal_accuracy(y_predicted, y_batch)
                acc_bag_train_sum += acc_bag_train

        
        val_loss_sum = 0
        accuracy_sum = 0
        accuracy_sum_th = 0
        counter = 0
        self.net.eval()

        with torch.no_grad():
            train_loss_bin = self.loss_func(y_predicted_bin, y_batch_bin)
            train_acc = cal_accuracy(y_predicted_bin, y_batch_bin)
            train_acc_th = cal_accuracy_th(y_predicted_bin, y_batch_bin, device = self.device, threshold_positive=self.th_pos)

            for _, (x_batch_s, x_batch_a, _, y_batch_bin) in enumerate(val_loader):
                y_predicted_bin, y_predicted, _ = self.net.forward(x_batch_s, x_batch_a)
                y_predicted_bin = y_predicted_bin[:, 0:2] # only use the first two columns, get rid of the logits
                val_loss_sum += self.loss_func(y_predicted_bin, y_batch_bin).item()

                acc = cal_accuracy(y_predicted_bin, y_batch_bin)
                accuracy_sum += acc
                acc_th= cal_accuracy_th(y_predicted_bin, y_batch_bin, device = self.device, threshold_positive=self.th_pos)
                accuracy_sum_th += acc_th

                counter += 1

            val_loss = val_loss_sum / counter
            accuracy = accuracy_sum / counter
            accuracy_th = accuracy_sum_th / counter
    

        train_loss_set.append(train_loss_sum/len(train_loader))
        train_acc_bag_set.append(acc_bag_train_sum/len(train_loader))
        train_loss_set_bin.append(train_loss_bin.item())
        train_acc_set.append(train_acc)
        train_acc_set_th.append(train_acc_th)

        val_loss_set_bin.append(val_loss)
        val_acc_set.append(accuracy)
        val_acc_set_th.append(accuracy_th)

        loss_set = [np.mean(train_loss_set[-10:]), np.mean(train_loss_set_bin[-10:]), np.mean(val_loss_set_bin[-10:]),]
        acc_set = [np.mean(train_acc_set[-5:]), np.mean(val_acc_set[-5:]),np.mean(train_acc_set_th[-5:]), np.mean(val_acc_set_th[-5:]), np.mean(train_acc_bag_set[-5:])]

        return self.net, loss_set, None, acc_set
