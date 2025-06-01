import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque

class MA_PS_DQN(nn.Module):
    def __init__(self, args):
        super(MA_PS_DQN, self).__init__()
        self.hidden_dim = getattr(args, 'hidden_dim', 32)
        self.hidden_dim2 = getattr(args, 'hidden_dim2', 32)
        self.args = args
        self.num_agents = args.num_agents
        # Shared feature extraction layers
        self.feature_extraction = nn.Sequential(
            nn.Linear(args.num_link * args.num_link_fea, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim , self.hidden_dim2),
            nn.ReLU()
        )

        self.q_values = nn.ModuleList([
            nn.Linear(self.hidden_dim2 , args.action_dim) for _ in range(args.num_agents)
        ])

    def forward(self, states):
        features = self.feature_extraction(states)
        
        q_values_list = []
        for i in range(self.num_agents):
            q_values_list.append(self.q_values[i](features))
        q_values = torch.stack(q_values_list, dim=1)
        
        return q_values


class ps_dqn_agent:
    def __init__(self, args):
        self.num_agents = args.num_agents
        self.action_size = args.action_dim
        self.lr = getattr(args, 'lr', 0.001)
        self.gamma = getattr(args, 'gamma', 0.9)
        self.batch_size = getattr(args, 'batch_size', 32)
        self.buffer_size = getattr(args, 'buffer_size', 2000)
        self.tau = getattr(args, 'tau', 0.005)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.dqn_model = MA_PS_DQN(args).to(self.device)
        self.dqn_target = MA_PS_DQN(args).to(self.device)

        self.dqn_opt = optim.Adam(params=self._get_parameters(self.dqn_model),lr=self.lr)

        self.memory = deque(maxlen=self.buffer_size)  

    def _get_parameters(self, networks):
        params = []
        params += list(networks.parameters())
        return params

    
    def update_target(self):
        self.dqn_target.load_state_dict(self.dqn_model.state_dict())

    def get_action(self, state, epsilon, **info):
        state_tensor = torch.tensor(state, dtype=torch.float32).detach().to(self.device)
        
        with torch.no_grad():
            q_value = self.dqn_model(state_tensor)
        q_value =  q_value.squeeze(0)
        if np.random.rand() <= epsilon:
            action = torch.randint(0, self.action_size, (self.num_agents,))
        else:
            action = torch.argmax(q_value, dim=-1)
        action = action.cpu().data.numpy()
        return action, {}

    def append_sample(self, info, next_state, reward):
        state = info[1].get("input_state")
        actions = info[0].get("action")
        self.memory.append((state, actions, next_state, reward))

    def save_model(self, path):
        torch.save(self.dqn_model, path+'/ps_dqn_model')

    def load_model(self, path):
        self.dqn_model = torch.load(path+'/ps_dqn_model', map_location=self.device)

    def update(self):
        mini_batch = random.sample(self.memory, self.batch_size)
        states = torch.tensor([i[0] for i in mini_batch], dtype=torch.float32).to(self.device)
        actions = torch.tensor([i[1] for i in mini_batch], dtype=torch.float32).to(self.device)
        next_states = torch.tensor([i[2] for i in mini_batch], dtype=torch.float32).to(self.device)
        rewards = torch.tensor([i[3] for i in mini_batch], dtype=torch.float32).to(self.device)
        
        q_value = self.dqn_model(states)
        main_q_value = q_value.gather(dim=-1, index=actions.unsqueeze(-1).long()).squeeze()
            
        with torch.no_grad():
            next_q_value = self.dqn_target(next_states)
            max_q_value, _ = torch.max(next_q_value, dim=-1)
            target_q_value = rewards + self.gamma * max_q_value
        
        dqn_loss = F.mse_loss(main_q_value, target_q_value)
        print("dqn_loss:", dqn_loss.item())
        self.dqn_opt.zero_grad()
        dqn_loss.backward()
        self.dqn_opt.step()

        return {"dqn_loss": dqn_loss.item()}
