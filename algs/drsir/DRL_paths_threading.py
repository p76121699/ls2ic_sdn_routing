import time
import json,ast
import csv
import pandas as pd
from algs.drsir import setting
import numpy as np
import importlib
import pathlib
import sys
import os

def _prepare_env(topology: str):
    """
    topology: '23' / '32' (字串)
    回傳 (env_module, agent_module, base_path)
    """
    base_path = pathlib.Path(__file__).parent / "RoutingGeant/DRL/dRSIR" / f"{topology}nodos"
    sys.path.insert(0, str(base_path))

    env_mod = importlib.import_module(f"environment_test_{topology}nodes")
    agent_mod = importlib.import_module("agent")  # 同資料夾的 agent.py
    return env_mod, agent_mod, base_path

def append_multiple_lines(file_name, lines_to_append):
    # Open the file in append & read mode ('a+')
    with open(file_name, "a+") as file_object:
        appendEOL = False
        # Move read cursor to the start of file.
        file_object.seek(0)
        # Check if file is not empty
        data = file_object.read(100)
        if len(data) > 0:
            appendEOL = True
        # Iterate over each string in the list
        for line in lines_to_append:
            # If file is not empty then append '\n' before first line for
            # other lines always append '\n' before appending line
            if appendEOL == True:
                file_object.write("\n")
            else:
                appendEOL = True
            # Append element at the end of file
            file_object.write(line)

def get_paths_base(num_nodes):
    file_base = './algs/drsir/RoutingGeant/DRL/dRSIR/'+str(num_nodes)+'nodos/paths_weight.json'
    with open(file_base,'r') as json_file:
        paths_dict = json.load(json_file)
        paths_base = ast.literal_eval(json.dumps(paths_dict))
        return paths_base

def get_paths_DRL():
    file_DRL = './results/drsir/drl_paths.json'
    with open(file_DRL,'r') as json_file:
        paths_dict = json.load(json_file)
        paths_DRL = ast.literal_eval(json.dumps(paths_dict))
        return paths_DRL

def stretch(paths, paths_base, src, dst):
   
    if isinstance(paths.get(str(src)).get(str(dst))[0],list):
        # print (paths.get(str(src)).get(str(dst))[0],'----', paths_base.get(str(src)).get(str(dst)))
        add_stretch = float(len(paths.get(str(src)).get(str(dst))[0])) - float(len(paths_base.get(str(src)).get(str(dst))))
        mul_stretch = float(len(paths.get(str(src)).get(str(dst))[0])) / float(len(paths_base.get(str(src)).get(str(dst))))
        return add_stretch, mul_stretch
    else:
        # print (paths.get(str(src)).get(str(dst)),'----', paths_base.get(str(src)).get(str(dst)))
        add_stretch = float(len(paths.get(str(src)).get(str(dst)))) - float(len(paths_base.get(str(src)).get(str(dst))))
        mul_stretch = float(len(paths.get(str(src)).get(str(dst)))) / float(len(paths_base.get(str(src)).get(str(dst))))
        return add_stretch, mul_stretch

def calc_all_stretch(cont, num_nodes):
    paths_base = get_paths_base(num_nodes)
    paths_DRL = get_paths_DRL()
    cont_DRL = 0
    total_paths = 0
    switches = [i for i in range(1,num_nodes+1)]
    a = time.time()
    with open('./results/drsir/stretch/'+str(cont)+'_stretch.csv','w') as csvfile:
        header = ['src','dst','add_st','mul_st']
        file = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
        file.writerow(header)
        for src in switches:
            for dst in switches:
                if src != dst:
                    total_paths += 1
                    add_stretch_DRL, mul_stretch_DRL = stretch(paths_DRL, paths_base, src, dst)
                    if add_stretch_DRL != 0:
                        cont_DRL += 1
                    # print('Additive stretch RL: ', add_stretch_DRL)
                    # print('Multi stretch RL: ', mul_stretch_DRL)
                    file.writerow([src,dst,add_stretch_DRL,mul_stretch_DRL])
    total_time = time.time() - a
    return total_time

def get_all_paths(config, env_mod, agent_mod, episode_rewards, episode_states_all, episode_duration_all, episodes, metrics_csv):
    t = time.time()
    # env = environment_test_23nodes.Environment()
    env = env_mod.Environment()
    #env = environment_test_48nodes.Environment()
    # env = environment_test_64nodes.Environment()
    state_space_size = env.obs_pm_size
    action_space_size = env.act_space_size

    target_update_freq = config["target_update_freq"]
    discount = config["discount"]
    batch_size = config["batch_size"]
    max_explore = config["max_explore"]
    min_explore = config["min_explore"]
    anneal_rate = config["anneal_rate"]
    replay_memory_size = config["replay_memory_size"]
    replay_start_size = config["replay_start_size"]
    lr = config["lr"]

    agente = agent_mod.Agent(state_space_size, action_space_size,target_update_freq, #1000, #cada n steps se actualiza la target network
                         discount, batch_size, max_explore, min_explore,
                         anneal_rate, replay_memory_size, replay_start_size,lr)
            
    for episode in range(episodes):
        ini_ep = time.time()
        ep = time.time()
        agente.handle_episode_start()
        s = [np.float32(env.reset())] #state inicial
        episode_reward = 0
        episode_state_list = []
        r = 0
        d = False
        while True:   
            a = agente.step(s,r)
            s_, r, d, _ = env.step(a)
            episode_reward += r
            episode_state_list.append(s)

            if d:
                end_ep = time.time()
                episode_rewards[episode].append(episode_reward)
                episode_states_all[episode].append(episode_state_list)
                episode_duration_all[episode].append(end_ep-ini_ep)
                break
            s = [np.float32(s_)]
    # print('\nEpisode rewards',episode_rewards)

    #Recover paths corresponding to each action for states
    file = './algs/drsir/RoutingGeant/DRL/dRSIR/'+str(len(env.topo_nodes))+'nodos/k_paths.json'
    with open(file,'r') as json_file:
        k_paths = json.load(json_file)
        k_paths_dict = ast.literal_eval(json.dumps(k_paths))

    #Use trained model to find choosen actions
    drl_paths = {src: {dst: [] for dst in range(1,len(env.topo_nodes)+1) if src != dst} for src in range(1,len(env.topo_nodes)+1)}
    for src in range(1,len(env.topo_nodes)+1):
                for dst in range(1,len(env.topo_nodes)+1):
                    if src != dst:
                        state = [np.float32(env.obs_space.index((src,dst)))]
                        action = agente.step(state,0,False)
                        path = k_paths_dict[str(src)][str(dst)][int(action)]
                        drl_paths[src][dst].append(path)
    
    avg_delay, avg_packet_loss, avg_throughput, max_link_utilization = compute_network_metrics(config)
    timestamp = time.time()
    with open(metrics_csv, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([timestamp, avg_delay, avg_packet_loss, avg_throughput, max_link_utilization])
        
    print("Eval metrics saved at {:.3f}: delay {:.3f}, loss {:.3f}, throughput {:.3f}, max util {:.3f}%".format(
            timestamp, avg_delay, avg_packet_loss, avg_throughput, max_link_utilization))
            
    time_DRL = time.time() - t
    return drl_paths, time_DRL, episode_rewards, episode_states_all, episode_duration_all

def compute_network_metrics(config):
    try:
        net_info = pd.read_csv('./results/drsir/net_info.csv')
    except Exception as e:
        print("Error reading net_info.csv:", e)
        return 0, 0, 0, 0

    capacity_dict = {}
    try:
        bw_r_file = f'dataset/{config["topology"]}_traffic/bw_r.txt'
        with open(bw_r_file, 'r') as file:
            for line in file:
                data = line.strip().split(',')
                if len(data) < 4:
                    continue
                src, dst, _, bw = map(int, data)
                link = (src, dst)
                reverse_link = (dst, src)
                capacity_dict[link] = bw
                capacity_dict[reverse_link] = bw
    except Exception as e:
        print("Error reading bw_r.txt:", e)
        capacity_dict = {}

    delays = []
    packet_losses = []
    throughputs = []
    utilizations = []
    for _, row in net_info.iterrows():
        try:
            node1 = int(row['node1'])
            node2 = int(row['node2'])
        except Exception as e:
            continue

        delay = row['delay']
        pkloss = row['pkloss']
        free_bw = row['bwd'] / 1000.0

        cap = capacity_dict.get((node1, node2), 200)
        
        throughput = (2 * cap) - free_bw
        utilization = (2 * cap - free_bw) / (2 * cap)
        
        delays.append(delay)
        packet_losses.append(pkloss)
        throughputs.append(throughput)
        utilizations.append(utilization)
    
    if len(delays) == 0:
        return 0, 0, 0, 0

    avg_delay = np.mean(delays)
    avg_packet_loss = np.mean(packet_losses)
    avg_link_throughput = np.mean(throughputs)
    max_link_utilization = max(utilizations) * 100.0

    return avg_delay, avg_packet_loss, avg_link_throughput, max_link_utilization

def DRL_thread(config): #cambiar para que lo llame a drl

    waiting_time = 30
    print("waiting ",waiting_time," second, then start testing")
    time.sleep(waiting_time)

    print('Enter thread')
    cont = 0
    episodes = config["episodes"]
    # ---------TRAINNING AND RECOVERING OF PATHS----------
    # For running after deciding 

    episode_rewards = [[] for _ in range(episodes)]
    episode_states_all = [[] for _ in range(episodes)]
    episode_duration_all = [[] for _ in range(episodes)]
    iteration_times = []
    
    metrics_csv = f'./results/drsir/{config["tm_id"]}_eval_metrics.csv'
    if not os.path.exists(metrics_csv):
        with open(metrics_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["timestamp", "avg_delay", "avg_packet_loss", "avg_throughput", "max_link_utilization"])

    topo = config["num_node"]
    env_mod, agent_mod, _ = _prepare_env(topo)
    while cont < 30:  
    
        a = time.time()
        cont = cont + 1
        drl_paths, time_DRL, episode_rewards, episode_states_all, episode_duration_all = get_all_paths(config, env_mod, agent_mod, episode_rewards, episode_states_all, episode_duration_all, episodes, metrics_csv)
        
        #write choosen paths
        with open('./results/drsir/drl_paths.json','w') as json_file:
            json.dump(drl_paths, json_file, indent=2)

        # print('time_DRL',time_DRL)
        time_stretch = calc_all_stretch(cont, topo)
        iteration_times.append(time_DRL) # print('time_stretch' , time_stretch)
        sleep = setting.MONITOR_PERIOD - time_DRL - time_stretch
        
        if sleep > 0:
            print("**"+str(cont)+"**time remaining drl and stretch",sleep)
            time.sleep(sleep)
        else:
            print("**"+str(cont)+"**time remaining drl and stretch",sleep)        
            time.sleep(0.2)
        # print(time.time()-a)
    
    file_info_eps = "./results/drsir/episode_info.txt"
    list_of_lines = ["Episodes: "+str(episodes),"Iterations: "+str(cont),"Time iteration: "+str(iteration_times)+"\n",str(episode_rewards)+"\n",str(episode_states_all)+"\n", str(episode_duration_all)+"\n"]
    append_multiple_lines(file_info_eps, list_of_lines)
    # print("Episode rewards: ", episode_rewards)
    
