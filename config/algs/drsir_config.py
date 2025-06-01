config = {
    "algs_name": "drsir",
    "episodes": 50,
    "target_update_freq": 100, #1000, #cada n steps se actualiza la target network
    "discount": 0.1,
    "batch_size": 15,
    "max_explore":1,
    "min_explore": 0.05,
    "anneal_rate": (1/400), #1/100000),
    "replay_memory_size": 100000,  #100000,
    "replay_start_size": 400,
    "lr": 0.01
}