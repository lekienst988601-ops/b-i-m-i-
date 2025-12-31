import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import Adam
import torch.nn.functional as F
from robot_neural_network import QNetwork, GaussianPolicy

def update(target, source, tau):
    """Update Method for SAC Only

    Args:
        target: Target Parameters to change
        source: Origin Parameters
        tau (float): Soft Update Policy Value
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

class SAC(object):
    """
    Soft Actor-Critic
    """
    def __init__(self, num_inputs, num_actions, update_interval, hidden_layer_size, learning_rate, device, epsilon, gamma, tau, alpha = 0.2):
        self.epsilon = epsilon
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.update_interval = update_interval

        self.device = device
        self.hidden_layer_size = hidden_layer_size
        self.learning_rate = learning_rate
        action_space = num_actions

        self.critic = QNetwork(num_inputs = num_inputs, num_actions = action_space, hidden_layer_size = self.hidden_layer_size).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.learning_rate)

        self.critic_target = QNetwork(num_inputs = num_inputs, num_actions = action_space, hidden_layer_size = self.hidden_layer_size).to(device=self.device)
        update(target = self.critic_target, source = self.critic, tau = 1)

        self.policy = GaussianPolicy(num_inputs = num_inputs, num_actions = action_space, hidden_layer_size = self.hidden_layer_size, action_space = action_space, epsilon = self.epsilon).to(self.device)
        self.policy_optim = Adam(self.policy.parameters(), lr = self.learning_rate)

    def select_action(self, state, epsilon, evaluate = False):
        """
        Selects action based on epsilon greedy algorithm

        Args:
            state (np.array): array of observations
            epsilon (float): epsilon-greedy value 
            evaluate (bool, optional): False for training, True for testing. Defaults to False.

        Returns:
            action (int): action
        """
        import random
        explore = True if random.random() < epsilon else False

        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)

        if explore: #random action
            action = random.uniform(-6.15, 6.15) # max velocity of wheels
            action = torch.tensor([[action]], device=self.device, dtype=torch.float)

        else: # greedy
            if evaluate == False:
                action, _, _ = self.policy.sample(state)
            else:
                _, _, action = self.policy.sample(state)
        
        return action.detach().cpu().numpy()[0]
    
    def update_weights(self, memory, batch_size, updates):
        """update model weights
        """
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)
        
        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
        qf1, qf2 = self.critic(state_batch, action_batch)
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        pi, log_pi, _ = self.policy.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()
        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if updates % self.update_interval == 0:
            update(self.critic_target, self.critic, self.tau)

        return policy_loss.detach()
