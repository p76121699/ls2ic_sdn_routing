import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import time
from collections import deque


class MA_PS_DQN_A(nn.Module):
    def __init__(self, args):
        super(MA_PS_DQN_A, self).__init__()
        
        self.num_agents = args.num_agents
        self.hidden_dim = getattr(args, 'hidden_dim', 32)
        self.hidden_dim2 = getattr(args, 'hidden_dim2', 32)
        self.encoder_h = getattr(args, 'encoder_h', 16)
        self.attend_heads= getattr(args, 'attend_heads', 1)
        self.attend_dim= getattr(args, 'attend_dim', 16)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = nn.Sequential(nn.Linear(args.num_link, self.encoder_h),nn.ReLU())
        self.multihead_attn = nn.MultiheadAttention(self.attend_dim, self.attend_heads,batch_first=True)

        self.feature_extraction = nn.Sequential(
            nn.Linear(args.num_link * args.num_link_fea + self.attend_dim , self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim , self.hidden_dim2),
            nn.ReLU()
        )

        # Independent output layers for each agent
        self.q_values = nn.ModuleList([
            nn.Linear(self.hidden_dim2, args.action_dim) for _ in range(self.num_agents)
        ])

        for layer in self.feature_extraction:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')
        for layer in self.encoder:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')
        
    def cal_attention_v(self,path_vector,training=False):
        if not training:
            path_vector = [path_vector]
            path_vector_tensor = torch.tensor(path_vector, dtype=torch.float32).to(self.device)
        else:
            path_vector_tensor = torch.tensor(path_vector, dtype=torch.float32).to(self.device)

        e_path_vector_tensor = self.encoder(path_vector_tensor)
        attention_vector,_ = self.multihead_attn(e_path_vector_tensor,e_path_vector_tensor,e_path_vector_tensor)
        
        return attention_vector

    def forward(self, states, attention_vector):
        features_ = torch.cat((states.unsqueeze(1).expand(-1, self.num_agents, -1), attention_vector), dim=-1)
        features = self.feature_extraction(features_)
        
        q_values_list = []
        for i in range(self.num_agents):
            q_values_list.append(self.q_values[i](features[:, i]))
        q_values = torch.stack(q_values_list, dim=1)
        
        return q_values

class ps_dqn_a_agent:
    def __init__(self, args):
        self.num_agents = args.num_agents
        self.action_size = args.action_dim
        self.lr = getattr(args, 'lr', 0.001)
        self.gamma = getattr(args, 'gamma', 0.9)
        self.batch_size = getattr(args, 'batch_size', 32)
        self.buffer_size = getattr(args, 'buffer_size', 2000)
        self.tau = getattr(args, 'tau', 0.005)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.dqn_model = MA_PS_DQN_A(args).to(self.device)
        self.dqn_target = MA_PS_DQN_A(args).to(self.device)
        
        self.params = list(self.dqn_model.parameters())
        self.dqn_opt = optim.Adam(params=self._get_parameters(self.dqn_model),lr=self.lr)

        self.memory = deque(maxlen=self.buffer_size)
        
    def _get_parameters(self, networks):
        params = []
        params += list(networks.parameters())
        return params

    def update_target(self):
        self.dqn_target.load_state_dict(self.dqn_model.state_dict())

    def get_action(self, state, epsilon, **info):
        attention_vector = info.get("att_vector")
        state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device)
        e_attention_vector = attention_vector.clone().unsqueeze(0)
        q_value = self.dqn_model(state_tensor, e_attention_vector).detach()
        
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
        path_vector = info[1].get("path_vector")
        self.memory.append((state, actions, next_state, reward, path_vector))

    def save_model(self, path):
        torch.save(self.dqn_model, path+'/ps_dqn_a_model')

    def load_model(self, path):
        self.dqn_model = torch.load(path+'/ps_dqn_a_model', map_location=self.device)
    
    def update(self):
        mini_batch = random.sample(self.memory, self.batch_size)
        states = torch.tensor([i[0] for i in mini_batch], dtype=torch.float32).to(self.device)
        actions = torch.tensor([i[1] for i in mini_batch], dtype=torch.float32).to(self.device)
        next_states = torch.tensor([i[2] for i in mini_batch], dtype=torch.float32).to(self.device)
        rewards = torch.tensor([i[3] for i in mini_batch], dtype=torch.float32).to(self.device)
        path_vector = [i[4] for i in mini_batch]
        
        attention_vector= self.dqn_model.cal_attention_v(path_vector,training=True)
        target_attention_vector= self.dqn_target.cal_attention_v(path_vector, training=True)

        q_value = self.dqn_model(states, attention_vector)
        main_q_value = q_value.gather(dim=-1, index=actions.unsqueeze(-1).long()).squeeze()

        with torch.no_grad():
            next_q_value = self.dqn_target(next_states, target_attention_vector).detach()
            max_q_value, _ = torch.max(next_q_value, dim=-1)
            target_q_value = rewards + self.gamma * max_q_value

        dqn_loss = F.mse_loss(main_q_value, target_q_value)
        self.dqn_opt.zero_grad()
        dqn_loss.backward()
        self.dqn_opt.step()

        return {"dqn_loss": dqn_loss.item()}
