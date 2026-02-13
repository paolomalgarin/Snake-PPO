# Modello PPO
import torch
from torch import nn
from torch.distributions import Categorical
from torch.optim import Adam
import numpy as np
from tools.beautyful_progress_bar import PBar

class FeedForwardNN(nn.Module):

    def __init__(self, obs_dim, action_dim):
        super().__init__()

        self.input_layer = nn.Linear(obs_dim, 128*3)
        self.hidden_layer1 = nn.Linear(128*3, 128)
        self.hidden_layer2 = nn.Linear(128, 128)
        self.hidden_layer3 = nn.Linear(128, 32)
        self.relu = nn.ReLU()
        
        self.policy_head = nn.Linear(32, action_dim)
        self.value_head = nn.Linear(32, 1)

    def forward(self, obs):
        # Convert observation to tensor if it's a numpy array
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)

        x = self.input_layer(obs)
        x = self.relu(x)
        x = self.hidden_layer1(x)
        x = self.relu(x)
        x = self.hidden_layer2(x)
        x = self.relu(x)
        x = self.hidden_layer3(x)
        x = self.relu(x)

        logits = self.policy_head(x)
        value = self.value_head(x)
        return logits, value


# CREDITS:
# https://medium.com/@eyyu/coding-ppo-from-scratch-with-pytorch-part-2-4-f9d8b8aa938a
# PPO ALGO (pseudo code):
# https://spinningup.openai.com/en/latest/_images/math/e62a8971472597f4b014c2da064f636ffe365ba3.svg

class PPOAgent:

    def __init__(self, env):
        # Extract environment information
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.n # Now ONLY handles Discrete actions

        self._init_hyperparameters()

        # PPO ALG STEP 1
        # Initialize actor and critic networks
        self.actor = FeedForwardNN(self.obs_dim, self.act_dim)
        self.critic = FeedForwardNN(self.obs_dim, 1)

        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

        # (useless for discrete action spaces)
        # # Create the covariance matrix
        # # Note that I chose 0.5 for stdev arbitrarily.
        # self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
        # self.cov_mat = torch.diag(self.cov_var)
    
    def learn(self, total_timesteps):
        t_so_far = 0 # Timesteps simulated so far
        pbar = PBar(total_timesteps, preset="training")
        
        # PPO ALG STEP 2
        while(t_so_far < total_timesteps):
            # PPO ALG STEP 3
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()

            # Calculate how many timesteps we collected this batch
            t_so_far += np.sum(batch_lens)
            pbar.update(np.sum(batch_lens))

            # Calculate V_{phi, k}
            V, _ = self.evaluate(batch_obs, batch_acts)

            # PPO ALG STEP 5 (step 4 is in rollout function)
            # Calculate advantage
            A_k = batch_rtgs - V.detach()

            # Normalize advantages
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            for _ in range(self.n_updates_per_iteration):
                # Epoch code (where we perform multiple updates on the actor and critic networks)

                # Calculate pi_theta(a_t | s_t)
                V, curr_log_probs = self.evaluate(batch_obs, batch_acts)
                
                # Calculate ratios
                ratios = torch.exp(curr_log_probs - batch_log_probs)

                # Calculate surrogate losses
                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

                actor_loss = (-torch.min(surr1, surr2)).mean()
                critic_loss = nn.MSELoss()(V, batch_rtgs)
                
                # Calculate gradients and perform backward propagation for actor network
                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor_optim.step()

                # Calculate gradients and perform backward propagation for critic network
                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()
        
        pbar.close()


    def _init_hyperparameters(self):
        # Default values for hyperparameters
        self.timestamps_per_batch = 4800        # timesteps per batch (a batch is a number of timesteps before updating PPO's policy)
        self.max_timestamps_per_episode = 1600  # timesteps per episode (an episode is a game inside the env)
        self.gamma = 0.95
        self.n_updates_per_iteration = 5        # Number of epoch, used to perform multiple updates on the actor and critic networks
        self.clip = 0.2
        self.lr = 3e-4

    def rollout(self):
        # Data of a batch
        batch_obs = []             # batch observations
        batch_acts = []            # batch actions
        batch_log_probs = []       # log probs of each action
        batch_rews = []            # batch rewards
        batch_rtgs = []            # batch rewards-to-go
        batch_lens = []            # episodic lengths in batch


        # Number of timesteps run so far this batch
        t = 0

        # => BATCH <=
        while t < self.timestamps_per_batch:
            
            # Rewards of this episode
            ep_rewards = []

            obs, _ = self.env.reset()
            done = False

            # => EPISODE <=
            for ep_t in range(self.max_timestamps_per_episode):

                t += 1

                # Collect current observation
                batch_obs.append(obs)

                action, log_prob = self.get_action(obs)
                obs, rew, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                # Collect current reward, action, and log prob
                ep_rewards.append(rew)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)

                if(done):
                    break
            
            # Collect episode length and rewards
            batch_lens.append(ep_t + 1)  # +1 because timestep starts at 0
            batch_rews.append(ep_rewards)
        
        # Reshape data as tensors before returning
        batch_obs = torch.tensor(np.array(batch_obs), dtype=torch.float)
        batch_acts = torch.tensor(np.array(batch_acts), dtype=torch.long)
        batch_log_probs = torch.tensor(np.array(batch_log_probs), dtype=torch.float)

        # PPO ALG STEP 4
        batch_rtgs = self.compute_rtgs(batch_rews)

        # Return the data of the batch 
        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens

    def get_action(self, obs):
        # Query the actor network for a mean action
        mean, _ = self.actor.forward(obs)

        # Create a Categorical Distribution
        dist = Categorical(logits=mean)
        
        # Sample an action from the distribution and get its log prob
        action = dist.sample()
        log_prob = dist.log_prob(action)

        # Return the sampled action and the log prob of that action
        # Note that I'm calling detach() since the action and log_prob  
        # are tensors with computation graphs, so I want to get rid
        # of the graph and just convert the action to numpy array.
        # log prob as tensor is fine. Our computation graph will
        # start later down the line.
        return action.detach().numpy(), log_prob.detach()

    def compute_rtgs(self, batch_rewards):
        # Calculate the rewards-to-go (rtg) per episode per batch to return
        batch_rtgs = []

        # Iterate through each episode backwards to maintain the same order in batch_rtgs
        for ep_rews in reversed(batch_rewards):

            discounted_reward = 0  # The discounted reward so far

            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)
            
        # Convert the rewards-to-go into a tensor
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)

        return batch_rtgs
    
    def evaluate(self, batch_obs, batch_acts):
        # Asking critic network for a value V for each obs in batch_obs
        V, _ = self.critic.forward(batch_obs)
        V = V.squeeze()

        # Calculate the log probabilities of batch actions using most recent actor network
        mean, _ = self.actor.forward(batch_obs)
        dist = Categorical(logits=mean)
        log_probs = dist.log_prob(batch_acts.squeeze())

        # Return predicted values V and log probs log_probs
        return V, log_probs