import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.jit
import numpy as np
import random
import copy
import os
import json
from collections import deque

class MultiheadAttention(nn.Module):
    def __init__(self, query_dim, kv_dim, output_dim, embed_dim, num_heads):
        super(MultiheadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.query_proj = nn.Linear(query_dim, embed_dim)
        self.keyvalue_proj = nn.Linear(kv_dim, embed_dim*2)
        self.out_proj = nn.Linear(embed_dim, output_dim)
    
    def forward(self, query, keyvalue):
        bs, na, sl, fd = query.size()

        query = query.reshape(bs * na, sl, fd)
        keyvalue   = keyvalue.reshape(bs * na, sl, fd)

        Q = self.query_proj(query)
        KV = self.keyvalue_proj(keyvalue)
        K, V = torch.chunk(KV, 2, dim=-1)
        #    [bs*na, seq_len, num_heads, head_dim] -> [bs*na, num_heads, seq_len, head_dim]
        Q = Q.reshape(bs * na, sl, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.reshape(bs * na, sl, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.reshape(bs * na, sl, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1) 

        weighted_values = torch.matmul(attn_weights, V)

        weighted_values = weighted_values.transpose(1, 2).contiguous()
        weighted_values = weighted_values.reshape(bs * na, sl, -1)

        output = self.out_proj(weighted_values)

        output = output.reshape(bs, na, sl, -1)

        return output

class ls2ic(nn.Module):
    def __init__(self, args):
        super(ls2ic, self).__init__()
        self.num_agents = args.num_agents
        self.action_dim = args.action_dim
        self.hidden_dim = getattr(args, 'hidden_dim', 32)
        self.message_dim = getattr(args, 'message_dim', 32)
        
        self.input_attention = MultiheadAttention(
            args.num_link_fea, args.num_link_fea, args.num_link_fea, self.hidden_dim, 4
        )
        self.layer_norm = nn.LayerNorm(args.num_link_fea)
        
        self.input_linear = nn.Linear(args.num_link * args.num_link_fea, self.hidden_dim)
        self.actor_rnn = nn.GRUCell(input_size=self.hidden_dim, hidden_size=self.hidden_dim)
        
        feat = self.hidden_dim
        
        self.message_shared_net = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU()
        )
        
        self.message_generator = nn.ModuleList([
            nn.Linear(self.hidden_dim, self.message_dim) for _ in range(args.num_agents)
        ])
        
        self.attention_message = MultiheadAttention(
            self.hidden_dim, self.message_dim, self.hidden_dim, self.hidden_dim, 2
        )
        self.message_layer_norm = nn.LayerNorm(self.hidden_dim)
        
        self.q_shared_net = nn.Sequential(
            nn.Linear(feat + self.message_dim, self.hidden_dim),
            nn.ReLU()
        )
        self.agent_heads = nn.ModuleList([
            nn.Linear(self.hidden_dim, self.action_dim) for _ in range(args.num_agents)
        ])
        
    def forward(self, states, hidden_state):
        bs, num_agents, num_seq, seq_dim = states.size()
        
        aft_att = self.input_attention(states, states)
        aft_att = self.layer_norm(states + aft_att)
        aft_att = aft_att.reshape(bs, num_agents, -1)
        encoded_states = self.input_linear(aft_att.reshape((bs * num_agents, -1)))
        hidden_state = hidden_state.reshape((bs * num_agents, -1))
        h = self.actor_rnn(encoded_states, hidden_state)
        h = h.reshape((bs, num_agents, -1))
        
        messages_emb = self.message_shared_net(h)
        messages_list = []
        for i in range(self.num_agents):
            messages_list.append(self.message_generator[i](messages_emb[:, i, :]))
        messages = torch.stack(messages_list, dim=1)
        
        total_detached = messages.detach().sum(dim=1, keepdim=True)
        mean_message = (total_detached - messages.detach() + messages) / messages.size(1)
        local_feat = h.unsqueeze(2)
        global_msg = mean_message.unsqueeze(2)
        after_comm = self.attention_message(local_feat, global_msg).squeeze(2)
        after_comm = self.message_layer_norm(h + after_comm.squeeze(2))
        
        feat = torch.cat((h, after_comm), dim=-1)
        
        q_values_emb = self.q_shared_net(feat)
        q_values_list = []
        for i in range(self.num_agents):
            q_values_list.append(self.agent_heads[i](q_values_emb[:, i]))
        q_values = torch.stack(q_values_list, dim=1)
        
        return q_values, h, mean_message
        
    def cal_inc_reward(self, states, hidden_state):
        with torch.no_grad():
            bs, num_agents, num_seq, seq_dim = states.size()
            
            aft_att = self.input_attention(states, states)
            aft_att = self.layer_norm(states + aft_att)
            aft_att = aft_att.reshape(bs, num_agents, -1)
            encoded_states = self.input_linear(aft_att.reshape((bs * num_agents, -1)))
            hidden_state = hidden_state.reshape((bs * num_agents, -1))
            h = self.actor_rnn(encoded_states, hidden_state)
            h = h.reshape((bs, num_agents, -1))
            
            messages_emb = self.message_shared_net(h)
            messages_list = []
            for i in range(self.num_agents):
                messages_list.append(self.message_generator[i](messages_emb[:, i, :]))
            messages = torch.stack(messages_list, dim=1)
            
            total_detached = messages.detach().sum(dim=1, keepdim=True)
            mean_message = (total_detached - messages.detach() + messages) / messages.size(1)
            local_feat = h.unsqueeze(2)
            global_msg = mean_message.unsqueeze(2)
            after_comm = self.attention_message(local_feat, global_msg).squeeze(2)
            after_comm = self.message_layer_norm(h + after_comm.squeeze(2))
            
            #----------------------------------------------------------------------------
            feat_com = torch.cat((h, after_comm), dim=-1)
            
            feat_com = self.q_shared_net(feat_com)
            q_values_com_list = []
            for i in range(self.num_agents):
                q_values_com_list.append(self.agent_heads[i](feat_com[:, i]))
            q_values_com = torch.stack(q_values_com_list, dim=1)
            #----------------------------------------------------------------------------
            
            #----------------------------------------------------------------------------
            after_comm = after_comm * 0
            feat_nocom = torch.cat((h, after_comm), dim=-1)
        
            feat_nocom = self.q_shared_net(feat_nocom)
            q_values_nocom_list = []
            for i in range(self.num_agents):
                q_values_nocom_list.append(self.agent_heads[i](feat_nocom[:, i]))
            q_values_nocom = torch.stack(q_values_nocom_list, dim=1)
            #----------------------------------------------------------------------------
            
            q_values_com = q_values_com.max(dim=-1)[0]
            q_values_nocom = q_values_nocom.max(dim=-1)[0]
            
            inc_reward = q_values_com - q_values_nocom

        return inc_reward
    
    def init_hidden(self):
        return self.input_linear.weight.new(1, self.hidden_dim).zero_()

class VAE(nn.Module):
    def __init__(self, args):
        super(VAE, self).__init__()
        self.latent_dim = getattr(args, 'latent_dim', 32)
        self.hidden_dim = getattr(args, 'hidden_dim2', 64)
        
        input_dim = args.num_link * args.num_link_fea
        
        self.fc1 = nn.Linear(input_dim, self.hidden_dim)
        self.fc_mu = nn.Linear(self.hidden_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(self.hidden_dim, self.latent_dim)

        self.fc2 = nn.Linear(self.latent_dim, self.hidden_dim)
        self.fc3 = nn.Linear(self.hidden_dim, input_dim)
    
    def encode(self, x):
        h = F.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std) 
        return mu + eps * std
    
    def decode(self, z):
        h = F.relu(self.fc2(z))
        return self.fc3(h)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

class influence_model(nn.Module):
    def __init__(self, args):
        super(influence_model, self).__init__()
        self.num_agents = args.num_agents
        self.action_dim = args.action_dim
        
        self.message_dim = getattr(args, 'message_dim', 32)
        self.hidden_dim = getattr(args, 'hidden_dim', 32)
        
        self.message_encoder = nn.Linear(self.message_dim, self.hidden_dim)
        self.action_encoder = nn.ModuleList([
            nn.Linear(self.action_dim, self.hidden_dim) for _ in range(self.num_agents)
        ])
        
        self.inference = nn.Sequential(
            nn.Linear(self.hidden_dim*3, self.hidden_dim*2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim*2, self.hidden_dim*2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim*2, self.hidden_dim)
        )

    def forward(self, z, z1, messages, actions):
        bs, time_seq, num_agents, _ = messages.size()
        
        de_messages = self.message_encoder(messages)
        de_actions = []
        for i in range(num_agents):
            de_actions.append(self.action_encoder[i](actions[:, :, i]))
        de_actions = torch.stack(de_actions, dim=2)
        
        delta_state = z1 - z
        
        feat = torch.cat((z, de_messages, de_actions), dim=-1)
        pred = self.inference(feat)
        
        loss = ((delta_state - pred)**2).sum() / self.hidden_dim
        return loss

def initialize_weights_kaiming(layer):
    if isinstance(layer, nn.Linear):
        nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')
        nn.init.uniform_(layer.weight, -1e-1, 1e-1)
        nn.init.constant_(layer.bias, 0.1)
    
class ls2ic_agent():
    def __init__(self, args):
        self.num_agents = args.num_agents
        self.action_dim = args.action_dim
        self.seq_dim = args.num_link
        self.num_link_fea = args.num_link_fea
        
        self.message_dim = getattr(args, 'message_dim', 32)
        self.lr = getattr(args, 'lr', 0.0005)
        self.vae_lr = getattr(args, 'vae_lr', 0.005)
        self.gamma = getattr(args, 'gamma', 0.9)
        self.tau = getattr(args, 'tau', 0.0005)
        self.vae_tau = getattr(args, 'vae_tau', 0.001)
        self.time_seq = getattr(args, 'time_seq', 8)
        self.mini_batch = getattr(args, 'mini_batch', 4)
        self.batch_size = self.time_seq * self.mini_batch
        self.buffer_size = getattr(args, 'buffer_size', 500)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.actor = ls2ic(args).to(self.device)
        self.actor.apply(initialize_weights_kaiming)
        self.target_actor = copy.deepcopy(self.actor)
        
        self.state_vae = VAE(args).to(self.device)
        self.state_vae.apply(initialize_weights_kaiming)
        self.target_s_vae = copy.deepcopy(self.state_vae)
        
        self.aux_model = influence_model(args).to(self.device)
        self.aux_model.apply(initialize_weights_kaiming)
        
        self.hidden_states = None
        self.tar_hidden_states = None
        
        self.params = list(self.actor.parameters()) + list(self.aux_model.parameters())
        self.opt = optim.Adam(self.params, lr=self.lr)
        self.vae_params = list(self.state_vae.parameters())
        self.vae_opt = optim.Adam(self.vae_params, lr=self.vae_lr)
        
        self.memory = deque(maxlen=self.buffer_size)
    
    def init_hidden(self, network, batch_size):
        return network.init_hidden().unsqueeze(0).expand(batch_size, self.num_agents, -1)
    
    def _get_parameters(self, networks):
        params = []
        params += list(networks.parameters())
        return params

    def update_target(self):
        for target_param, local_param in zip(self.target_s_vae.parameters(), self.state_vae.parameters()):
            target_param.data.copy_(self.vae_tau * local_param.data + (1.0 - self.vae_tau) * target_param.data)
            
        for target_param, local_param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
    
    def get_action(self, state, epsilon, **info):
        state_tensor = torch.tensor(state, dtype=torch.float32).detach().to(self.device)
        
        with torch.no_grad():
            q_value, self.hidden_states, _ = self.actor(state_tensor, self.hidden_states)
        
        q_value =  q_value.squeeze(0)
        if np.random.rand() <= epsilon:
            action = torch.randint(0, self.action_dim, (self.num_agents,))
        else:
            action = torch.argmax(q_value, dim=-1)
        action = action.cpu().data.numpy()
        q_value = q_value.cpu().data.numpy()
        return action, {"logits": q_value}

    def append_sample(self, info, next_state, reward):
        state = info[1].get("input_state")
        actions = info[0].get("action")
        logits = info[0].get("logits")
        self.memory.append((state, actions, logits, next_state, reward))
    
    def sample_window(self, window_size):
        '''
        states: (batch_size, window_size, num_agent, state_dim)
        actions: (batch_size, window_size, num_agent) 
        next_states: (batch_size, window_size, num_agent, state_dim)
        rewards: (batch_size, window_size, num_agent)
        '''
        memory_len = len(self.memory)
        valid_range = memory_len - window_size + 1

        start_indices = np.random.randint(0, valid_range, size=self.mini_batch)
        
        batch_states = []
        batch_actions = []
        batch_logits = []
        batch_next_states = []
        batch_rewards = []
        
        for start in start_indices:
            window = [self.memory[i] for i in range(start, start + window_size)]
            states, actions, logits, next_states, rewards = zip(*window)
            batch_states.append(np.array(states))
            batch_actions.append(np.array(actions))
            batch_logits.append(np.array(logits))
            batch_next_states.append(np.array(next_states))
            batch_rewards.append(np.array(rewards))
        
        batch_states = torch.tensor(batch_states, dtype=torch.float32, device=self.device)
        batch_actions = torch.tensor(batch_actions, dtype=torch.long, device=self.device)
        batch_logits = torch.tensor(batch_logits, dtype=torch.float32, device=self.device)
        batch_next_states = torch.tensor(batch_next_states, dtype=torch.float32, device=self.device)
        batch_rewards = torch.tensor(batch_rewards, dtype=torch.float32, device=self.device)
        
        return batch_states, batch_actions, batch_logits, batch_next_states, batch_rewards
    
    def save_model(self, path):
        torch.save(self.actor, path+'/ls2ic_model')
        torch.save(self.state_vae, path+'/vae_model')
        torch.save(self.aux_model, path+'/aux_model')
        
    def load_model(self, path):
        self.actor = torch.load(path+'/ls2ic_model', map_location=self.device)
        
    def update(self):
        sample_result = self.sample_window(self.time_seq)
    
        states, actions, logits, next_states, rewards = sample_result
        
        tar_next_hidden = self.init_hidden(self.target_actor, self.mini_batch)
        next_hidden = self.init_hidden(self.actor, self.mini_batch)
        tar_next_out = []
        next_out = []
        with torch.no_grad():
            for t in range(self.time_seq):
                tar_next_q, tar_next_hidden, _ = self.target_actor(next_states[:, t], tar_next_hidden)
                next_q, next_hidden, _ = self.actor(next_states[:, t, :, :], next_hidden)
                tar_next_out.append(tar_next_q)
                next_out.append(next_q)
        tar_next_out = torch.stack(tar_next_out, dim=1) 
        best_actions = torch.stack(next_out, dim=1).argmax(dim=-1, keepdim=True)
        tar_next_qvals = tar_next_out.gather(3, best_actions).squeeze(-1)

        train_hidden = self.init_hidden(self.actor, self.mini_batch)
        online_out = []
        inc_rewards = []
        mess_out = []
        for t in range(self.time_seq):
            inc_reward = self.actor.cal_inc_reward(states[:, t], train_hidden.detach())
            q_values, train_hidden, mess = self.actor(states[:, t], train_hidden)
            inc_rewards.append(inc_reward)
            online_out.append(q_values)
            mess_out.append(mess)
        online_out = torch.stack(online_out, dim=1)
        inc_rewards = torch.stack(inc_rewards, dim=1)
        mess_out = torch.stack(mess_out, dim=1)
        
        inc_rewards = inc_rewards * F.softmax(rewards, dim=-1)
        inc_rewards = inc_rewards.sum(dim=-1, keepdim=True) - inc_rewards
        inc_rewards = inc_rewards / (1 + torch.abs(inc_rewards))
        
        targets = rewards + 0.1 * inc_rewards + self.gamma * tar_next_qvals
        
        main_q_value = online_out.gather(3, actions.unsqueeze(-1)).squeeze(-1)

        td_error = (targets.detach() - main_q_value)
        
        td_loss = (td_error**2).sum() / self.batch_size
        
        bs, time_seq, num_agents, _ = mess_out.size()
        s = states.view(bs, time_seq, num_agents, -1)
        s1 = next_states.view(bs, time_seq, num_agents, -1)
        with torch.no_grad():
            mu, logvar = self.target_s_vae.encode(s)
            mu1, logvar1 = self.target_s_vae.encode(s1)
            z = self.target_s_vae.reparameterize(mu, logvar)
            z1 = self.target_s_vae.reparameterize(mu1, logvar1)
            
        aux_loss = self.aux_model(z, z1, mess_out, logits) / self.batch_size
        loss = td_loss + 5e2 * aux_loss
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        
        x_recon, mu, logvar = self.state_vae(s)
        vae_loss = F.mse_loss(x_recon, s, reduction='sum') / self.batch_size
        vae_loss += -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / self.batch_size
        self.vae_opt.zero_grad()
        vae_loss.backward()
        self.vae_opt.step()
        
        return {"dqn_loss": td_loss.item(), "aux_loss": aux_loss.item(), "vae_loss": vae_loss.item()}


