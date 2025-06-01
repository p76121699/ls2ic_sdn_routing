import numpy as np

class ospf_agent:
    def __init__(self, args):
        self.args = args
        self.num_agents = args.num_agents
        self.action_size = args.action_dim

    def get_action(self, state, epsilon, **info):

        action = np.zeros(self.num_agents, dtype=int)

        return action, {}

    def append_sample(self, info, next_state, reward):
        pass

    def update_target(self):
        pass

    def save_model(self, path):
        pass

    def load_model(self, path):
        pass

    def update(self):
        return {}