# This file contains all algorithm functions we are running in a way that they return
# a numpy array of [mean_reward, std_reward]

# IMPORTS
import numpy as np
import sys
import os
import pdb
from utils_rl import eval_model_multiEnvs
# expertrank
# from ileed.utils.helpers import evaluate_model
# from ileed.utils.helpers import state_featurizer_loss
from tqdm import trange
import tqdm
from copy import deepcopy as cp
# rllab

#  data_tup (s, a, s_next)
# (num_expert, traj_size, s_dim), (num_expert, traj_size, a_dim), (num_expert, traj_size, s_dim)

def run_bc(data_tup, env, env_eval, hidden_size, embed_dim, device, seed, scaler_s=None, scaler_a=None, num_iters = 200000):
    '''
    Runs the Behavioural Cloning Algorithm using Cross Entropy Loss
    Inputs:
        env         - Gym environment class
        hidden_size - integer for dimension of the one hidden layer
        embed_dim   - dimension to embed the skill vector omega
    Outputs:
        numpy.array containing mean and std of reward evaluated over 1000 episodes
    '''
    # load env params
    obs_space = env.observation_space
    act_space = env.action_space
    indim = np.prod(obs_space.shape)
    outdim = 1 # dummy for continuous
    print("indim:", indim , "outdim:", outdim)

    # setup exp
    network_args = {'indim':indim,
                    'outdim':outdim,
                    'hidden_size':hidden_size,
                    'embed_dim':embed_dim,
                    'device':device,
                    'seed':seed,
                    'use_latent':False
                    }
    # run
    best_omega, networks, best_eval, best_log, all_logs, all_evals, all_log_likelihood, all_epochs, all_omegas = learn(
                data_tup, 
                env,
                network_args, 
                no_sigmoid=False, 
                # loss_type='bc',
                loss_type='bc_continuous',
                num_iters = num_iters,
                n_eval = 10,
                num_restarts = 20,
                env_eval=env_eval,
                scaler_s=scaler_s,
                scaler_a=scaler_a,
    )
    # eval
    # mean_reward, std_reward = evaluate_model(networks['action_network'], env, n_eval_episodes=1000)
    # print("learned reward avg: %.2f std: %.2f" % (mean_reward, std_reward))
    # return np.array([mean_reward,std_reward])

    return best_omega, networks, best_eval, best_log, all_logs, all_evals, all_log_likelihood, all_epochs, all_omegas

def run_mle(data_tup, env, env_eval, hidden_size, embed_dim, device, seed, scaler_s=None, scaler_a=None, num_iters = 200000):
    '''
    Runs our Annotation Algorithm
    Inputs:
        env         - Gym environment class
        env_eval    - Gym environment class for evaluation
        hidden_size - integer for dimension of the one hidden layer
        embed_dim   - dimension to embed the skill vector omega
    Outputs:
        numpy.array containing mean and std of reward evaluated over 1000 episodes
    '''
    # load env params
    obs_space = env.observation_space
    act_space = env.action_space
    indim = np.prod(obs_space.shape)
    outdim = 1 # dummy for continuous
    print("obs_space:", obs_space , "act_space:", act_space)
    print("indim:", indim , "outdim:", outdim)

    # setup exp
    network_args = {'indim':indim,
                    'outdim':outdim,
                    'hidden_size':hidden_size,
                    'embed_dim':embed_dim,
                    'device':device,
                    'seed':seed,
                    'use_latent':False
                    }
    # run
    
    best_omega, networks, best_eval, best_log, all_logs, all_evals, all_log_likelihood, all_epochs, all_omegas = learn(
                data_tup, 
                env,
                network_args, 
                no_sigmoid=False, 
                # loss_type='irt',
                loss_type='irt_continuous',
                num_iters = num_iters,
                n_eval = 10,
                num_restarts = 20,
                env_eval=env_eval,
                scaler_s=scaler_s,
                scaler_a=scaler_a,
                )
    # eval
    # mean_reward, std_reward = evaluate_model(networks['action_network'], env, n_eval_episodes=1000)
    # print("learned reward avg: %.2f std: %.2f" % (mean_reward, std_reward))
    # return np.array([mean_reward,std_reward])
    return best_omega, networks, best_eval, best_log, all_logs, all_evals, all_log_likelihood, all_epochs, all_omegas

def run_mle_state(data_tup, env, hidden_size, embed_dim, device, seed):
    '''
    Runs our Annotation Algorithm WITH the transient network
    Inputs:
        env         - Gym environment class
        hidden_size - integer for dimension of the one hidden layer
        embed_dim   - dimension to embed the skill vector omega
    Outputs:
        numpy.array containing mean and std of reward evaluated over 1000 episodes
    '''
    # load env params
    obs_space = env.observation_space
    act_space = env.action_space
    indim = np.prod(obs_space.shape)
    outdim = act_space.n
    # setup exp
    network_args = {'indim':indim,
                    'outdim':outdim,
                    'hidden_size':hidden_size,
                    'embed_dim':embed_dim,
                    'device':device,
                    'seed':seed,
                    'use_latent':True
                    }
    # run
    _, networks, _, _, _ = learn(
                data_tup, 
                env,
                network_args, 
                no_sigmoid=False, 
                loss_type='irt',
                num_iters = 2000,
                n_eval = 10,
                num_restarts = 20)
    # eval
    mean_reward, std_reward, traj_s_list, traj_a_list, traj_r_list, acc_reward_list, steps_list = \
        eval_model_multiEnvs(networks['action_network'], env, n_eval_episodes=1000)
    return np.array([mean_reward,std_reward])



from torch.optim import Adam
from torch import inf, randn, no_grad
#------------------------------------------------------------------------------------#
# Main Learning Function
#------------------------------------------------------------------------------------#
def learn(data_tup, env, network_args, no_sigmoid, loss_type, num_iters, n_eval, num_restarts, env_eval=None, scaler_s=None, scaler_a=None):
    '''
    Main loss function for training our networks used in learn.py
    Inputs:
        data_tup             
        networks 
        embedding_dim           
        no_sigmoid        
        loss_type   loss_type='irt', loss_type='bc', or irt_continuous, bc_continuous
    
    Outputs:
        log likelihood of loss 

    TODO: Some of these variables should be generalized (learning rates and number of iterations)
    '''
    # init constants needed
    device = network_args['device']
    M = len(data_tup[0])
    # init where we store data
    best_log = inf
    best_eval = None
    best_omega = None
    all_logs = []
    all_evals = []
    all_log_likelihood = []
    all_epochs = []
    all_omegas = []
    # iterate searching for smallest NLL

    num_restarts = 1
    for n in trange(num_restarts):
        # init the networks
        state_featurizer_network = MLP(input_size=network_args['indim'], 
                                    hidden_size=network_args['hidden_size'], 
                                    output_size=network_args['embed_dim'],
                                    device=device,
                                    seed=network_args['seed']+n).to(device)
        
        # define action network for continuous or discrete action space
        if loss_type in ['irt_continuous', 'bc_continuous']:
            action_network = ContinuousActionNetwork(
                input_size=network_args['indim'],
                hidden_size=network_args['hidden_size'],
                action_dim=env.action_space.shape[0],  # continuous dimension
                num_mixtures=1,
                seed=network_args['seed']+n,
                device=device,
                scaler_s=scaler_s, 
                scaler_a=scaler_a,
            ).to(device)
        else:
            action_network = MLP(
                input_size=network_args['indim'],
                hidden_size=network_args['hidden_size'],
                output_size=network_args['outdim'],
                device=device,
                seed=network_args['seed']+n).to(device)
            
        # this network maps from latent embedding to next state (in embedded space)
        latent_transition_network = MLP(input_size=network_args['embed_dim'] + 1, 
                                        hidden_size=network_args['hidden_size'], 
                                        output_size=network_args['embed_dim'],
                                        device=device,
                                        seed=network_args['seed']+n).to(device) if network_args['use_latent'] else None 
       
        
        # the omegas to be learned for the M experts
        omega = randn(M, network_args['embed_dim'], requires_grad=True, device=device)
        # one optim for the networks, one for the omegas
        optim1 = Adam(list(state_featurizer_network.parameters()) + list(action_network.parameters()), lr=2e-3)
        optim2 = Adam([omega], lr=1e-3)
        # also learn the latent_transition_network if provided
        if latent_transition_network is not None: optim3 = Adam(list(state_featurizer_network.parameters()) + list(latent_transition_network.parameters()), lr=1e-3)
        # go over iterations taking derivative of the NLL
        # for i in range(num_iters):
        for i in tqdm.tqdm(range(num_iters), desc="iter", leave=True):
            log_likelihood = loss(data_tup, state_featurizer_network, action_network, omega, no_sigmoid, loss_type)
            optim1.zero_grad()
            optim2.zero_grad()
            (log_likelihood).backward()
            optim1.step()
            optim2.step()
            if latent_transition_network is not None:
                aux_loss = state_featurizer_loss(data_tup, state_featurizer_network, latent_transition_network)
                optim3.zero_grad()
                aux_loss.backward()
                optim3.step()

            if (i+1) % 1000 == 0 or i == 5:
                print(f"Restart {n+1}/{num_restarts}, Iteration {i+1}/{num_iters}, Loss: {log_likelihood.item():.4f}")
                # evaluate the model
                with no_grad():
                    action_network.eval()
                    mean_reward, std_reward, traj_s_list, traj_a_list, traj_r_list, acc_reward_list, steps_list = \
                        eval_model_multiEnvs(action_network, env_eval, n_eval_episodes=n_eval)
 
                    print(f"                                                                           Evaluation over {n_eval} episodes: Mean Reward = {mean_reward:.2f}, Std Reward = {std_reward:.2f}")
                    all_evals.append((mean_reward, std_reward, np.mean(steps_list), np.std(steps_list)))
                    all_log_likelihood.append(log_likelihood.item())
                    all_epochs.append(i+1)
                    # print omega gradients
                    # print("omega gradients:", omega.grad)
                    # print("omega:", omega)
                    all_omegas.append(cp(omega).detach().cpu().numpy())
                    action_network.train()

                # once you are done store the log loss if it is better
                with no_grad():
                    # compute log
                    # final_log = loss(data_tup, state_featurizer_network, action_network, omega, no_sigmoid, loss_type)
                    all_logs.append(log_likelihood.item())
                    # eval net
                    if log_likelihood.item() < best_log:
                        best_log = log_likelihood.item()
                        networks = {
                            'state_featurizer_network': state_featurizer_network,
                            'action_network': action_network,
                            'latent_transition_network': latent_transition_network,
                        }
                        best_eval = (mean_reward, std_reward)
                        best_omega = omega
                    # print("best_log so far:", best_log, "best_eval so far:", best_eval)
    return best_omega, networks, best_eval, best_log, all_logs, all_evals, all_log_likelihood, all_epochs, all_omegas


from torch.nn.functional import softmax, cross_entropy
from torch import sigmoid, sum, log

def loss(data_tup, state_featurizer_network, action_network, omega, no_sigmoid, loss_type):
    '''
    Main loss function for training our networks used in learn.py
    Inputs:
        data_tup             
        state_featurizer_network 
        action_network    
        omega       
        no_sigmoid        
        loss_type  loss_type='irt', loss_type='bc', or irt_continuous, bc_continuous
    
    Outputs:
        log likelihood of loss 
    '''
    # breakpoint()
    states, actions, _ = data_tup
    
    if loss_type == 'irt':
        prob_a_vec = action_network.prob_forward(states) 
        out = state_featurizer_network(states) # difficulty of state
        # (num_expert, traj_size, s_embed_dim)
        sigma = sigmoid(sum(out * omega.unsqueeze(1), dim=-1))
        # (num_expert, traj_size) <- (num_expert, traj_size, s_embed_dim) * (num_expert, 1, s_embed_dim)  


        # prob_a_vec[i][j][k] = prob of action k being optimal at state[i][j] 
        log_likelihood = IRT_logloss(sigma=sigma, actions=actions, prob_a_vec=prob_a_vec)

    elif loss_type == 'bc':
        prob_a_vec = action_network.prob_forward(states) 
        log_likelihood = cross_entropy(log(prob_a_vec.reshape(actions.shape[0]*actions.shape[1],-1)),actions.reshape(-1))
    
    elif loss_type == 'irt_continuous':
        # Continuous IRT
        out = state_featurizer_network(states)
        sigma = torch.sigmoid(torch.sum(out * omega.unsqueeze(1), dim=-1))
        alpha, mu, sigma_gmm = action_network(states)
        log_likelihood = IRT_logloss_continuous(
            sigma=sigma, actions=actions, alpha=alpha, mu=mu, sigma_gmm=sigma_gmm)
        # num_expert = N
        # traj_size = T
        # num_mixtures = K

        # sigma: (N, T)

        # alpha: (N, T, K)
        # mu: (N, T, K, a_dim)
        # sigma_gmm: (N, T, K, a_dim)


    elif loss_type == 'bc_continuous':
        # Continuous Behavioral Cloning
        # gmm = action_network.get_mixture_distribution(states)
        # log_likelihood = -gmm.log_prob(actions).sum()

        alpha, mu, sigma_gmm = action_network(states)
        log_likelihood = BC_logloss_continuous(
            actions=actions, alpha=alpha, mu=mu, sigma_gmm=sigma_gmm)
        # actions: (num_expert, traj_size, a_dim)
    
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")



    return log_likelihood


#------------------------------------------------------------------------------------#
# PART II: losses used in loss.py (for main algorithm in learn.py)
#------------------------------------------------------------------------------------#
from torch import gather
def IRT_logloss(sigma, actions, prob_a_vec):
    '''
    This loss is based on the IRT model, as of now assuming worst case scenario 
    is random actions.

    Inputs:
        sigma
        actions: [num_expert, traj_size, num_actions] expert actions taken, type is long 
        prob_a_vec: [num_expert, traj_size, num_actions] probability of each action being optimal
    
    Outputs:
        log likelihood of loss 
    '''
    # sigma[i][j] = probability of expert i making the correct action at states[i][j]
    # print(actions)
    # breakpoint()
    
    prob_a = gather(input=prob_a_vec, dim=2, index=actions) 
    # prob_a[i][j] = prob of action taken by expert i being optimal at states[i][j]

    prob_a = prob_a.squeeze(-1) # shape (num_expert, traj_size)

    log_likelihood = log(sigma*prob_a + (1-sigma)*(1-prob_a)/(prob_a_vec.size(-1)-1))
    # shape (num_expert, traj_size)   <- shape (num_expert, traj_size) * (num_expert, traj_size) + (num_expert, traj_size) * (num_expert, traj_size)
    
    # log_likelihood = prob_a*log(sigma) + (1-prob_a)/(prob_a_vec.size(-1)-1)*log((1-sigma))


    return -1*log_likelihood.sum()


# def IRT_logloss_continuous(sigma, actions, alpha, mu, sigma_gmm):
#     """
#     Continuous version of IRT log loss using GMM policy.
    
#     Inputs:
#         sigma: [batch] expertise level ρ_φ(s, ω_i)
#         actions: [batch, action_dim] continuous expert actions
#         alpha: [batch, num_mixtures] mixture weights
#         mu: [batch, num_mixtures, action_dim] means
#         sigma_gmm: [batch, num_mixtures, action_dim] std devs (before scaling)

#         # num_expert = N
#         # traj_size = T
#         # num_mixtures = K

#         # sigma: (N, T)

#         # action: (N, T, a_dim)

#         # alpha: (N, T, K)
#         # mu: (N, T, K, a_dim)
#         # sigma_gmm: (N, T, K, a_dim)

#     """
#     batch_size, num_mixtures, action_dim = mu.shape

#     # Scale variance according to equation: σ_j(s)/ρ_φ(s, ω_i)
#     sigma_scaled = sigma_gmm / sigma.unsqueeze(-1).unsqueeze(-1)

#     # Compute log-prob of actions under each Gaussian component
#     normal_log_probs = -0.5 * (((actions.unsqueeze(1) - mu) / sigma_scaled) ** 2).sum(-1) \
#                        - action_dim * 0.5 * torch.log(2 * torch.pi) \
#                        - sigma_scaled.log().sum(-1)

#     # Weight by mixture coefficients (α_j)
#     log_prob = torch.logsumexp(torch.log(alpha) + normal_log_probs, dim=-1)

#     # Negative log-likelihood
#     nll = -log_prob.sum()
#     return nll


def IRT_logloss_continuous(sigma, actions, alpha, mu, sigma_gmm):
    """
    Continuous-action IRT log-loss based on Eq. (3) from the paper.

    Inputs:
        sigma:      (N, T)                # expertise level ρ_φ(s, ω_i)
        actions:    (N, T, a_dim)         # expert actions
        alpha:      (N, T, K)             # GMM mixture weights
        mu:         (N, T, K, a_dim)      # GMM means μ*_j(s)
        sigma_gmm:  (N, T, K, a_dim)      # GMM stds σ*_j(s)
    Returns:
        scalar negative log-likelihood loss
    """
    N, T, K, a_dim = mu.shape

    # === 1. Scale GMM variance by 1 / ρ_φ(s, ω_i)
    sigma_scaled = sigma_gmm / sigma.unsqueeze(-1).unsqueeze(-1)  # (N, T, K, a_dim)

    # === 2. Compute log probability of expert action under each Gaussian component
    # Normal distribution log-prob:
    # log N(a; μ, σ) = -0.5 * ((a - μ)/σ)^2 - log(σ) - 0.5*log(2π)
    diff = (actions.unsqueeze(2) - mu) / sigma_scaled  # (N, T, K, a_dim)
    log_component = (
        -0.5 * (diff ** 2).sum(-1)
        - sigma_scaled.log().sum(-1)
        - 0.5 * a_dim * torch.log(torch.tensor(2 * torch.pi))
    )  # (N, T, K)

    # === 3. Mixture: sum over components weighted by α_j
    # log π(a|s, ω_i, φ, πθ*) = log(Σ_j α_j N(a; μ_j*, σ_j*/ρ))
    log_prob = torch.logsumexp(torch.log(alpha + 1e-8) + log_component, dim=-1)  # (N, T)

    # === 4. Negative log-likelihood
    nll = -log_prob.sum()  # or .mean(), depending on objective normalization

    return nll



def BC_logloss_continuous(actions, alpha, mu, sigma_gmm):
    """
    Continuous-action IRT log-loss based on Eq. (3) from the paper.

    Inputs:
        sigma:      (N, T)                # expertise level ρ_φ(s, ω_i)
        actions:    (N, T, a_dim)         # expert actions
        alpha:      (N, T, K)             # GMM mixture weights
        mu:         (N, T, K, a_dim)      # GMM means μ*_j(s)
        sigma_gmm:  (N, T, K, a_dim)      # GMM stds σ*_j(s)
    Returns:
        scalar negative log-likelihood loss
    """
    N, T, K, a_dim = mu.shape

    # === 2. Compute log probability of expert action under each Gaussian component
    # Normal distribution log-prob:
    # log N(a; μ, σ) = -0.5 * ((a - μ)/σ)^2 - log(σ) - 0.5*log(2π)
    diff = (actions.unsqueeze(2) - mu) / sigma_gmm  # (N, T, K, a_dim)
    log_component = (
        -0.5 * (diff ** 2).sum(-1)
        - sigma_gmm.log().sum(-1)
        - 0.5 * a_dim * torch.log(torch.tensor(2 * torch.pi))
    )  # (N, T, K)

    # === 3. Mixture: sum over components weighted by α_j
    # log π(a|s, ω_i, φ, πθ*) = log(Σ_j α_j N(a; μ_j*, σ_j*/ρ))
    log_prob = torch.logsumexp(torch.log(alpha + 1e-8) + log_component, dim=-1)  # (N, T)

    # === 4. Negative log-likelihood
    nll = -log_prob.sum()  # or .mean(), depending on objective normalization

    return nll




from torch.nn import Module
from torch.nn import Linear, ReLU, Sequential
from torch import manual_seed
class MLP(Module):
    '''
    MLP class with three linear layers with two ReLU non-linearities between,
    uses softmax at the output if called via prob_forward() method.
    '''
    def __init__(self, input_size, hidden_size, output_size, device, seed=123):
        super(MLP, self).__init__()
        self.seed = manual_seed(seed)
        self.device = device
        self.net = Sequential(
            Linear(input_size, hidden_size),
            ReLU(inplace=True),
            Linear(hidden_size, hidden_size),
            ReLU(inplace=True),
            Linear(hidden_size, output_size),
        )

    def forward(self, x):
        return self.net(x)
    
    def prob_forward(self, x):
        return softmax(self.net(x), dim=-1)
    

import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical, MixtureSameFamily

class ContinuousActionNetwork(nn.Module):
    """
    Gaussian Mixture Model policy network for continuous action spaces.
    Outputs mixture weights, means, and standard deviations for each mixture component.

    num_expert = N

    traj_size = T

    input_size = s_dim

    action_dim = a_dim

    num_mixtures = K

    """
    def __init__(self, input_size, hidden_size, action_dim, num_mixtures=5, seed=123, device='cpu',
                    scaler_s=None, scaler_a=None):
        super().__init__()
        torch.manual_seed(seed)
        self.num_mixtures = num_mixtures
        self.action_dim = action_dim

        self.shared = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.device = device

        self.fc_alpha = nn.Linear(hidden_size, num_mixtures)  # mixture weights
        self.fc_mu = nn.Linear(hidden_size, num_mixtures * action_dim) # means
        self.fc_logsigma = nn.Linear(hidden_size, num_mixtures * action_dim) # log std deviations

        scaler_state, scaler_action = scaler_s, scaler_a
        self.s_max = torch.tensor(scaler_state.data_max_, dtype=torch.float32).to(self.device)
        self.s_min = torch.tensor(scaler_state.data_min_, dtype=torch.float32).to(self.device)
        self.a_max = torch.tensor(scaler_action.data_max_, dtype=torch.float32).to(self.device)
        self.a_min = torch.tensor(scaler_action.data_min_, dtype=torch.float32).to(self.device)

    def forward(self, x):
        # x: (N, T, s_dim)
        h = self.shared(x)
        # h: (N, T, hidden_size)
        alpha = F.softmax(self.fc_alpha(h), dim=-1)
        # before view: (N, T, K)
        # after view: (N, T, K)
        mu = self.fc_mu(h).view(x.size(0), x.size(1), self.num_mixtures, self.action_dim)
        # before view: (N, T, K * a_dim)
        # after view: (N, T, K, a_dim)
        logsigma = self.fc_logsigma(h).view(x.size(0), x.size(1), self.num_mixtures, self.action_dim)
        # before view: (N, T, K * a_dim)
        # after view: (N, T, K, a_dim)
        sigma = torch.exp(logsigma)
        # sigma: (N, T, K, a_dim)
        return alpha, mu, sigma
    
        # alpha: (N, T, K)
        # mu: (N, T, K, a_dim)
        # sigma: (N, T, K, a_dim)

    def get_mixture_distribution(self, x):
        alpha, mu, sigma = self.forward(x)
        
        mix = Categorical(alpha)
        # Shape: (N, T) batch shape, each element is a categorical over K

        comp = Normal(mu, sigma)
        # comp: batch shape (N, T, K), event shape (a_dim)
        gmm = MixtureSameFamily(mix, comp)
        # batch (N, T), event (a_dim)
        return gmm
    
    def norm_state(self, x_s):
        x_s = (x_s - self.s_min) / (self.s_max - self.s_min)
        return x_s.to(torch.float32), 

    def denorm_action(self, x_a):
        x_a = x_a * (self.a_max - self.a_min) + self.a_min
        return x_a
    
    def predict(self, x, deterministic=False):
        # print("state in predict:", x.shape)
        
        """
        Sample an action from the GMM policy given state x.
        If deterministic=True, return the expected mean action under the mixture.
        """
        # check if x is tensor, if not convert
        x = torch.tensor(x, dtype=torch.float32).to(self.device) if not isinstance(x, torch.Tensor) else x.to(self.device)
        x = self.norm_state(x)[0]
        # print("state in predict:", x.shape)
        # input("check state shape")
        
        x = x.unsqueeze(0) # Add batch dimension
        alpha, mu, sigma = self.forward(x)
        # alpha: (N, T, K)
        # mu, sigma: (N, T, K, a_dim)
        deterministic = True
        if deterministic:
            # Weighted mean over mixture components
            # Expand alpha to match (N, T, K, a_dim)
            weighted_mean = torch.sum(alpha.unsqueeze(-1) * mu, dim=2)
            # weighted_mean: (N, T, a_dim)
            weighted_mean = weighted_mean.squeeze(0)  # Remove batch dimension if N=1
            # print("deterministic action:", weighted_mean.shape)
            weighted_mean = self.denorm_action(weighted_mean)
            weighted_mean = weighted_mean.cpu().numpy() # convert to numpy
            # print("denormed deterministic action:", weighted_mean.shape)
            # input("check action shape")
  
            return weighted_mean, None
        else:
            # Sample mixture indices for each (N, T)
            mix_idx = torch.multinomial(alpha.view(-1, self.num_mixtures), 1).view(alpha.shape[:-1]) # shape (N, T)
            # Gather the corresponding component parameters
            mu_selected = torch.gather(mu, 2, mix_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, self.action_dim)).squeeze(2) # shape (N, T, a_dim)    
            sigma_selected = torch.gather(sigma, 2, mix_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, self.action_dim)).squeeze(2) # shape (N, T, a_dim)
            # Sample from the selected Normal distributions
            action = mu_selected + sigma_selected * torch.randn_like(sigma_selected)
            action = action.squeeze(0)  # Remove batch dimension if N=1
            action = self.denorm_action(action)
            action = action.cpu().numpy()  # convert to numpy

            return action, None
        
        """
        Sample an action from the GMM policy given state x.
        """
        gmm = self.get_mixture_distribution(x)
        if deterministic:
            # Choose the component with the highest weight
            _, max_indices = gmm.mixture_distribution.probs.max(dim=-1)
            mu = gmm.component_distribution.mean
            actions = mu[torch.arange(mu.size(0)).unsqueeze(1), torch.arange(mu.size(1)).unsqueeze(0), max_indices]
        else:
            actions = gmm.sample()
        return actions
        
