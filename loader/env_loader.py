from mininet.net import Mininet
from mininet.node import RemoteController
from mininet.link import TCLink
from mininet.log import info, setLogLevel
import time
import os

def build_topo(env_config):
    net = Mininet(controller=RemoteController, link=TCLink)
    setLogLevel("info")
    info("*** Add Controller ***\n")
    net.addController("c0", controller=RemoteController, ip='127.0.0.1')

    info("*** Add Switches and host ***\n")
    for i in range(1, env_config["num_node"] + 1):
        net.addSwitch(f"s{i}")
        net.addHost(f"h{i}", mac=f"00:00:00:00:00:{i:02d}")
        net.addLink(f"s{i}", f"h{i}")

    info("*** Add Inter-Switch Links from BW file ***\n")
    added_links = set()
    with open(env_config["bw_file"], 'r') as f:
        for line in f:
            if not line.strip():
                continue
            src, dst, _, bw = line.strip().split(",")
            src, dst = int(src), int(dst)
            bw = float(bw)
            link_key = tuple(sorted((src, dst)))
            if link_key not in added_links:
                net.addLink(f"s{src}", f"s{dst}", bw=bw)
                added_links.add(link_key)

    info("*** Network Start ***\n")
    net.start()
    return net

def start_traffic(net, env_config, mode="train"):
    """
    Start the traffic generation process, mode supports 'train' / 'test'
    """
    tm_ids = env_config["tm_list_train"] if mode == "train" else env_config["tm_list_test"]
    tm_duration = env_config["tm_duration_training"] if mode == "train" else env_config["tm_duration_test"]
    tm_prefix = env_config["tm_prefix"]  # e.g. 32nodos_24tm/TM-{tm_id}/
    num_hosts = env_config["num_node"]

    for tm_id in tm_ids:
        print(f"--- TM {tm_id} ---")

        for i in range(1, num_hosts + 1):
            hname = f"h{i}"
            suffix = f"0{i}" if i < 10 else str(i)
            server_sh = os.path.join(tm_prefix.format(tm_id=tm_id), f"Servers/server_{suffix}.sh")
            net.get(hname).popen(f"sh {server_sh}")

        time.sleep(10)

        for i in range(1, num_hosts + 1):
            hname = f"h{i}"
            suffix = f"0{i}" if i < 10 else str(i)
            client_sh = os.path.join(tm_prefix.format(tm_id=tm_id), f"Clients/client_{suffix}.sh")
            net.get(hname).popen(f"sh {client_sh}")

        time.sleep(tm_duration)

        os.system("sudo killall -p iperf3")
        print("next TM\n")

def start_single_traffic(net, env_config, input_):
    
    tm_prefix = env_config["tm_prefix"]
    num_hosts = env_config["num_node"]
    
    print("################################################")
    for i in range(1, num_hosts + 1):
        hname = f"h{i}"
        suffix = f"0{i}" if i < 10 else str(i)
        server_sh = os.path.join(tm_prefix.format(tm_id=input_), f"Servers/server_{suffix}.sh")
        net.get(hname).popen(f"sh {server_sh}")
    time.sleep(10)
    for i in range(1, num_hosts + 1):
        hname = f"h{i}"
        suffix = f"0{i}" if i < 10 else str(i)
        client_sh = os.path.join(tm_prefix.format(tm_id=input_), f"Clients/client_{suffix}.sh")
        net.get(hname).popen(f"sh {client_sh}")

    time.sleep(360)
    os.popen("sudo killall -p iperf3")
    time.sleep(2)