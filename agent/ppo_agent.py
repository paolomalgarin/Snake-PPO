# Modello PPO
import torch
from torch import nn
import numpy as np
import json, os

class ActorCritic(nn.Module):

    def __init__(self, obs_dim, action_dim, hidden_dim=128):
        super().__init__()

        self.input_layer = nn.Linear(obs_dim, hidden_dim)
        self.hidden_layer = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs):
        x = self.input_layer(obs)
        x = self.relu(x)
        x = self.hidden_layer(x)
        x = self.relu(x)

        logits = self.policy_head(x)
        value = self.value_head(x)
        return logits, value



class RolloutBuffer:

    def __init__(self):
        self.clear()

    def store(self, obs, action, log_prob, value, reward, done):
        self.obs.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)
    
    def clear(self):
        self.obs = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []



class PPOAgent:
    
    def __init__(self, obs_dim, action_dim, config=None):
        self.config = config or {
            'gamma': 0.99,      # discount factor
            'lambda': 0.95,     # GAE lambda
            'epsilon': 0.2,     # clip parameter
            'lr': 3e-4,         # learning rate
            'epochs': 10,       # training epochs per update
            'batch_size': 64,
            'value_coef': 0.5,  # value loss coefficient
            'entropy_coef': 0.01
        }
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = ActorCritic(obs_dim, action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['lr'])
        self.buffer = RolloutBuffer()

    def select_actions(self, obs):
        # usa la policy corrente per selezionare un azione date le osservazioni
        obs = self._preprocess_obs(obs)
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits, value = self.model(obs_tensor)
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        
        return action.item(), log_prob.item(), value.item()

    def collect_rollout(self, env, num_steps=2048):
        # Raccoglie esperienze interagendo con l'ambiente
        obs, _ = env.reset()
        for _ in range(num_steps):
            action, log_prob, value = self.select_actions(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Memo: l'obs di SnakeEnv è un dict, devi trasformarlo in vettore
            flat_obs = self._preprocess_obs(obs)
            self.buffer.store(flat_obs, action, log_prob, value, reward, done)
            
            obs = next_obs
            if done:
                obs, _ = env.reset()
        
        return self.buffer

    def compute_gae(self):
        #Calcola i GAE (Generalized Advantage Estimation)
        rewards = torch.tensor(self.buffer.rewards)
        values = torch.tensor(self.buffer.values)
        dones = torch.tensor(self.buffer.dones)
        
        advantages = torch.zeros_like(rewards)
        last_advantage = 0
        last_value = 0  # oppure l'ultimo value predetto
        
        # Calcola GAE all'indietro
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = last_value
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.config['gamma'] * next_value * (1 - dones[t]) - values[t]
            advantages[t] = last_advantage = delta + self.config['gamma'] * self.config['lambda'] * (1 - dones[t]) * last_advantage
        
        returns = advantages + values
        return advantages, returns

    def ppo_update(self):
        # Updates the policy using the ppo algorithm
        # Prepara dati
        obs = torch.tensor(self.buffer.obs)
        actions = torch.tensor(self.buffer.actions)
        old_log_probs = torch.tensor(self.buffer.log_probs)
        old_values = torch.tensor(self.buffer.values)
        
        # Calcola GAE
        advantages, returns = self.compute_gae()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Multiple epoch update
        for epoch in range(self.config['epochs']):
            # Mini-batch update
            indices = torch.randperm(len(obs))
            
            for start in range(0, len(obs), self.config['batch_size']):
                end = start + self.config['batch_size']
                batch_idx = indices[start:end]
                
                batch_obs = obs[batch_idx].to(self.device)
                batch_actions = actions[batch_idx].to(self.device)
                batch_old_log_probs = old_log_probs[batch_idx].to(self.device)
                batch_advantages = advantages[batch_idx].to(self.device)
                batch_returns = returns[batch_idx].to(self.device)
                
                # Forward pass
                logits, values = self.model(batch_obs)
                probs = torch.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                
                # Calcola nuove probabilità
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                # Ratio per PPO
                ratio = (new_log_probs - batch_old_log_probs).exp()
                
                # PPO Clipped Loss
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.config['epsilon'], 1 + self.config['epsilon']) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value Loss
                value_loss = ((values.squeeze() - batch_returns) ** 2).mean()
                
                # Total Loss
                loss = policy_loss + self.config['value_coef'] * value_loss - self.config['entropy_coef'] * entropy
                
                # Update
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
        
        self.buffer.clear()

    def save(self, path='agent/resoults', model_name='ppo_agent'):
        # Saves the model
        # Creates the directory if does not exist
        os.makedirs(path, exist_ok=True)
        
        # Creates the file
        model_path = os.path.join(path, f"{model_name}.pth")
        
        # Saves the model
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'obs_dim': self.model.input_layer.in_features,
            'action_dim': self.model.policy_head.out_features,
        }, model_path)        
        print(f"Model saved in: {model_path}")

    def load(self, path):
        # Loads the model (returns true if loads successfully)
        try:            
            # Checks file existance
            if not os.path.exists(path):
                print(f"File not found ({path})")
                return False
                
            checkpoint = torch.load(path, map_location=self.device)
            
            # Checks model dimensions
            current_obs_dim = self.model.input_layer.in_features
            current_action_dim = self.model.policy_head.out_features
            
            if current_obs_dim != checkpoint.get('obs_dim', current_obs_dim):
                print(f"WARNING: obs_dim not matching! Saved model: {checkpoint.get('obs_dim')}, Current: {current_obs_dim}")
                
            if current_action_dim != checkpoint.get('action_dim', current_action_dim):
                print(f"WARNING: action_dim not matching! Saved model: {checkpoint.get('action_dim')}, Current: {current_action_dim}")
            
            # Loads model wheights
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print("Model load: SUCCESS")
            return True
            
        except Exception as e:
            print(f"Model load: ERROR \n{e}")
            import traceback
            traceback.print_exc()
            return False


    def _preprocess_obs(self, obs_dict):
        """Converti dict observation in vettore"""
        grid = obs_dict["grid"].flatten() / 3.0  # Normalizza 0-1
        direction_onehot = np.zeros(4)
        direction_onehot[obs_dict["direction"]] = 1
        return np.concatenate([grid, direction_onehot])
