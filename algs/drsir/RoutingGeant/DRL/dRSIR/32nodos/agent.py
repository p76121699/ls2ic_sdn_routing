import numpy as np
import collections
import torch
import torch.nn as nn
import torch.optim as optim
import time

# ---------------------------
# Network (Q-function approximator)
# ---------------------------
class Network(nn.Module):
    def __init__(self, input_size, output_size, lr, hidden_size=[50]):
        """
        :param input_size: 輸入尺寸（狀態向量大小）
        :param output_size: 輸出尺寸（動作空間大小）
        :param lr: 學習率
        :param hidden_size: 隱藏層大小（此處只採用1個隱藏層）
        """
        super(Network, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        # 定義網路結構：1個隱藏層，採用 ReLU 激活函數
        self.fc1 = nn.Linear(input_size, hidden_size[0])
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size[0], output_size)

        # 定義 optimizer 與 loss function
        self.optimizer = optim.Adam(self.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08)
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        """前向傳播，給定狀態向量返回每個動作的 Q 值。"""
        h1 = self.relu(self.fc1(x))
        out = self.fc2(h1)
        return out

    def train_step(self, inputs, targets, actions_one_hot):
        """
        :param inputs: Tensor, shape [batch_size, input_size]
        :param targets: Tensor, shape [batch_size]，對應 target Q 值（由 reward 與下一狀態最小 Q 值計算）
        :param actions_one_hot: Tensor, shape [batch_size, output_size]
        :return: loss 值 (float)
        """
        self.optimizer.zero_grad()
        qvalues = self.forward(inputs)  # shape: [batch_size, output_size]
        # 根據 one-hot 取出採用的動作對應的 Q 值
        preds = torch.sum(qvalues * actions_one_hot, dim=1)
        loss = self.loss_fn(preds, targets)
        loss.backward()
        self.optimizer.step()
        return loss.item()

# ---------------------------
# Memory (Experience Replay)
# ---------------------------
class Memory(object):
    """Experience Replay Memory"""
    def __init__(self, max_size):
        self.buffer = collections.deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        buffer_size = len(self.buffer)
        indices = np.random.choice(np.arange(buffer_size), size=batch_size, replace=False)
        return [self.buffer[i] for i in indices]

    def __len__(self):
        return len(self.buffer)

# ---------------------------
# Agent (Deep Q-learning Agent)
# ---------------------------
class Agent(object):
    def __init__(self,
                 state_space_size,
                 action_space_size,
                 target_update_freq=900,  # 每 target_update_freq 步更新目標網路
                 discount=0.7,
                 batch_size=15,
                 max_explore=1,
                 min_explore=0.05,
                 anneal_rate=(1/1500),
                 replay_memory_size=100000,
                 replay_start_size=200,
                 lr=0.001):
        """
        :param state_space_size: 狀態向量大小
        :param action_space_size: 動作空間大小
        """
        self.lr = lr
        self.action_space_size = action_space_size

        # 建立線上網路與目標網路
        self.online_network = Network(state_space_size, action_space_size, lr)
        self.target_network = Network(state_space_size, action_space_size, lr)
        self.update_target_network()

        self.target_update_freq = target_update_freq
        self.discount = discount
        self.batch_size = batch_size

        # 探索參數，採用 ε-貪婪策略
        self.max_explore = max_explore + (anneal_rate * replay_start_size)
        self.min_explore = min_explore
        self.anneal_rate = anneal_rate
        self.steps = 0

        # 經驗回放記憶庫
        self.memory = Memory(replay_memory_size)
        self.replay_start_size = replay_start_size

    def handle_episode_start(self):
        self.last_state = None
        self.last_action = None

    def step(self, state, reward, training=True):
        """
        根據當前狀態與上一個動作的獎勵來選擇新動作，同時存儲經驗。
        :param state: 當前狀態（numpy array），shape 為 [state_space_size] 或其他符合模型輸入的尺寸
        :param reward: 上一動作的獎勵
        :param training: 是否處於訓練模式
        :return: 選擇的動作 (整數)
        """
        action = self.policy(state, training)

        if training:
            self.steps += 1
            if self.last_state is not None:
                experience = {
                    "state": self.last_state,
                    "action": self.last_action,
                    "reward": reward,
                    "next_state": state
                }
                self.memory.add(experience)

            if self.steps > self.replay_start_size:
                self.train_network()
                if self.steps % self.target_update_freq == 0:
                    self.update_target_network()

        self.last_state = state
        self.last_action = action
        return action
    
    def save_model(self):
        torch.save(self.online_network, "model")


    def load_model(self):
        self.online_network = torch.load("model", map_location=self.device)
    
    def policy(self, state, training):
        """
        ε-貪婪策略：
          - 在訓練時，以概率探索隨機動作，否則採用貪婪策略選擇使 Q 值最小（成本最小）的動作
        :param state: 當前狀態（numpy array）
        :param training: 是否為訓練模式
        :return: 選擇的動作（整數）
        """
        explore_prob = self.max_explore - (self.steps * self.anneal_rate)
        if training and max(explore_prob, self.min_explore) > np.random.rand():
            action = np.random.randint(self.action_space_size)
        else:
            # 將狀態轉換為 tensor 並增加 batch 維度
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                qvalues = self.online_network(state_tensor)  # shape: [1, action_space_size]
            # 由於成本越低越好，因此取最小的 Q 值所對應的動作
            action = int(torch.argmin(qvalues, dim=1).item())
        return action

    def update_target_network(self):
        """將線上網路權重複製給目標網路。"""
        self.target_network.load_state_dict(self.online_network.state_dict())

    def train_network(self):
        """從記憶庫中取樣並更新線上網路。"""
        batch = self.memory.sample(self.batch_size)
        # 提取 batch 中的狀態、動作、獎勵與下一狀態
        states = np.array([b["state"] for b in batch])
        actions = np.array([b["action"] for b in batch])
        rewards = np.array([b["reward"] for b in batch])
        next_states = np.array([b["next_state"] for b in batch])

        # 轉換成 torch tensor
        states_tensor = torch.tensor(states, dtype=torch.float32)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        next_states_tensor = torch.tensor(next_states, dtype=torch.float32)
        # 對動作進行 one-hot 編碼
        actions_one_hot = np.eye(self.action_space_size)[actions]
        actions_one_hot_tensor = torch.tensor(actions_one_hot, dtype=torch.float32)

        # 從目標網路獲得下一狀態的 Q 值，並取最小值（因為我們要最小化成本）
        with torch.no_grad():
            next_qvalues = self.target_network(next_states_tensor)
            next_q_min, _ = torch.min(next_qvalues, dim=1)

        targets = rewards_tensor + self.discount * next_q_min

        loss = self.online_network.train_step(states_tensor, targets, actions_one_hot_tensor)
        return loss

