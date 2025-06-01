import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import copy


class MF_dqn(nn.Module):
    def __init__(self, args):
        super(MF_dqn, self).__init__()
        self.hidden_dim = getattr(args, 'hidden_dim', 32)
        self.hidden_dim2 = getattr(args, 'hidden_dim2', 32)
        self.args = args
        self.num_agents = args.num_agents
        # Shared feature extraction layers
        self.feature_extraction = nn.Sequential(
            nn.Linear(args.num_link * args.num_link_fea + args.action_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim , self.hidden_dim2),
            nn.ReLU()
        )

        self.q_values = nn.ModuleList([
            nn.Linear(self.hidden_dim2 , args.action_dim) for _ in range(args.num_agents)
        ])

    def forward(self, states, mean_field):
        agent_input = torch.cat((states, mean_field), dim=-1)
        features = self.feature_extraction(agent_input)
        q_values_list = []
        for i in range(self.num_agents):
            q_values_list.append(self.q_values[i](features))
        q_values = torch.stack(q_values_list, dim=1)
        
        return q_values


class meanfield_agent:
    def __init__(self, args):
        self.num_agents = args.num_agents
        self.action_size = args.action_dim
        self.lr = getattr(args, 'lr', 0.001)
        self.gamma = getattr(args, 'gamma', 0.9)
        self.batch_size = getattr(args, 'batch_size', 32)
        self.buffer_size = getattr(args, 'buffer_size', 500)
        self.tau = getattr(args, 'tau', 0.005)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.MF_dqn = MF_dqn(args).to(self.device)
        self.target_MF_dqn = copy.deepcopy(self.MF_dqn)
        self.params = list(self.MF_dqn.parameters())
        
        self.opt = optim.Adam(params=self.params,lr=self.lr)

        self.memory = deque(maxlen=self.buffer_size)  

    def _get_parameters(self, networks):
        params = []
        params += list(networks.parameters())
        return params

    def update_target(self):
        self.target_MF_dqn.load_state_dict(self.MF_dqn.state_dict())

    def prepare_step(self, state, epsilon):
        info = {}
        current_mean = np.ones(self.action_size, dtype=np.float32) / self.action_size
        info["mean_field"] = [current_mean]
        _, output_info = self.get_action([state], epsilon, **info)
                        
        return output_info.get("mf")
    
    def get_action(self, state, epsilon, **info):
        first_mean_field = info.get("mean_field")
        state_tensor = torch.tensor(state, dtype=torch.float32).detach().to(self.device)
        mean_field_tensor = torch.tensor(first_mean_field, dtype=torch.float32).detach().to(self.device)
        
        q_value = self.MF_dqn(state_tensor, mean_field_tensor)
        q_value =  q_value.squeeze(0)
        if np.random.rand() <= epsilon:
            action = torch.randint(0, self.action_size, (self.num_agents,))
        else:
            action = torch.argmax(q_value, dim=-1)
        action = action.cpu().data.numpy()
        
        one_hot_action = np.zeros(
            (self.num_agents, self.action_size),
            dtype=np.float32
        )
        one_hot_action[action] = 1.0
        mean_field = np.mean(one_hot_action, axis=0)
        
        return action, {"mf": mean_field}

    def append_sample(self, info, next_state, reward):
        state = info[1].get("input_state")
        actions = info[0].get("action")
        mean_field = info[0].get("mf")
        self.memory.append((state, actions, mean_field, next_state, reward))

    def save_model(self, path):
        torch.save(self.MF_dqn, path+'/meanfield_model')

    def load_model(self, path):
        self.MF_dqn = torch.load(path+'/meanfield_model', map_location=self.device)

    def update(self):
        mini_batch = random.sample(self.memory, self.batch_size)
        states = torch.tensor([i[0] for i in mini_batch], dtype=torch.float32).to(self.device)
        actions = torch.tensor([i[1] for i in mini_batch], dtype=torch.float32).to(self.device)
        mean_fields = torch.tensor([i[2] for i in mini_batch], dtype=torch.float32).to(self.device)
        next_states = torch.tensor([i[3] for i in mini_batch], dtype=torch.float32).to(self.device)
        rewards = torch.tensor([i[4] for i in mini_batch], dtype=torch.float32).to(self.device)

        current_mean = torch.ones((self.batch_size, self.action_size), dtype=torch.float32).to(self.device) / self.action_size

        q_value = self.MF_dqn(states, mean_fields)
        main_q_value = q_value.gather(dim=1, index=actions.unsqueeze(-1).long()).squeeze()
            
        with torch.no_grad():
            next_q_value = self.target_MF_dqn(next_states, current_mean)
            max_q_value, action = torch.max(next_q_value, dim=-1)
            one_hot_actions = F.one_hot(action, num_classes=self.action_size).float()
                
        next_mean_field = torch.mean(one_hot_actions, dim=1)
                
        with torch.no_grad():
            next_q_value = self.target_MF_dqn(next_states, next_mean_field)
            max_q_value, _ = torch.max(next_q_value, dim=-1)
            target_q_value = rewards + self.gamma * max_q_value
                
        dqn_loss = F.mse_loss(main_q_value, target_q_value)
        self.opt.zero_grad()
        dqn_loss.backward()
        self.opt.step()

        return {"dqn_loss": dqn_loss.item()}
