config = {
    "algs_name": "ls2ic",
    "action_dim": 20,
    "message_dim": 32,
    "lr": 0.0005,
    "vae_lr": 0.005,
    "gamma": 0.9,
    "tau": 0.0005,
    "vae_tau": 0.001,
    "time_seq": 8,
    "batch_size": 4,
    "buffer_size": 500,
    "hidden_dim": 32,
    "hidden_dim2": 64,
    "latent_dim": 32,
    "softupdate_freq": 30,
    "rnn": True,
    "epsilon_ini": 0.9,
    "epsilon_final": 0.1,
    "epsilon": 0.9,
    "use_global_state": False,
    "epsilon_first_phase": 1500,
    "epsilon_second_phase": 0,
    "use_delta_reward": False,
}
