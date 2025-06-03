# SDN + DRL Setup Guide
This guide outlines the complete installation and usage steps for setting up an SDN environment using Mininet, Ryu controller, and our custom DRL-based routing solution. The recommended Python version is 3.8.10.

## Table of Contents
1. Installing Mininet
2. Creating a Conda Virtual Environment
3. Installing and Modifying Ryu
4. Installing ls2ic_sdn_routing
5. training and testing
6. Analysis

### 1. Installing Mininet
Mininet is a lightweight network emulator commonly used for SDN research.

```
cd ~
git clone https://github.com/mininet/mininet 
```

After cloning, a mininet directory will be created. Proceed to install:

```
cd mininet
git tag   # list available versions
git checkout -b 2.3.1b1
cd util 
./install.sh -a
```

Verify the installation:

```
sudo mn
```

Exit Mininet:

```
quit
```

### 2. Creating a Conda Virtual Environment (Recommended)

We recommend using Miniconda to create a clean Python environment and avoid dependency conflicts:

```
conda create -n sdn1 python=3.8 -y
conda activate sdn1
```

### 3. Installing and Modifying Ryu
Clone the Ryu SDN controller:
```
cd ~
git clone https://github.com/faucetsdn/ryu
```

Modify `switches.py` (Located at `ryu/ryu/topology/switches.py`)

Update the `PortData` class (around line 246):

Before:
```
class PortData(object):
    def __init__(self, is_down, lldp_data):
        super(PortData, self).__init__()
        self.is_down = is_down
        self.lldp_data = lldp_data
        self.timestamp = None
        self.sent = 0
```

After (add self.delay = 0):
```
class PortData(object):
    def __init__(self, is_down, lldp_data):
        super(PortData, self).__init__()
        self.is_down = is_down
        self.lldp_data = lldp_data
        self.timestamp = None
        self.sent = 0
        self.delay = 0
```

Modify `lldp_packet_in_handler` (around line 711):

Original:
```
@set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
def lldp_packet_in_handler(self, ev):
    if not self.link_discovery:
        return

    msg = ev.msg
    try:
        src_dpid, src_port_no = LLDPPacket.lldp_parse(msg.data)
    except LLDPPacket.LLDPUnknownFormat:
        # This handler can receive all the packets which can be
        # not-LLDP packet. Ignore it silently
        return

    dst_dpid = msg.datapath.id
    if msg.datapath.ofproto.OFP_VERSION == ofproto_v1_0.OFP_VERSION:
        dst_port_no = msg.in_port
    elif msg.datapath.ofproto.OFP_VERSION >= ofproto_v1_2.OFP_VERSION:
        dst_port_no = msg.match['in_port']
    else:
        LOG.error('cannot accept LLDP. unsupported version. %x',
                  msg.datapath.ofproto.OFP_VERSION)

    src = self._get_port(src_dpid, src_port_no)
    if not src or src.dpid == dst_dpid:
        return
```

Modified:
```
@set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
def lldp_packet_in_handler(self, ev):
    recv_timestamp = time.time()   #加這一行
    if not self.link_discovery:
        return

    msg = ev.msg
    try:
        src_dpid, src_port_no = LLDPPacket.lldp_parse(msg.data)
    except LLDPPacket.LLDPUnknownFormat:
        # This handler can receive all the packets which can be
        # not-LLDP packet. Ignore it silently
        return

    dst_dpid = msg.datapath.id
    if msg.datapath.ofproto.OFP_VERSION == ofproto_v1_0.OFP_VERSION:
        dst_port_no = msg.in_port
    elif msg.datapath.ofproto.OFP_VERSION >= ofproto_v1_2.OFP_VERSION:
        dst_port_no = msg.match['in_port']
    else:
        LOG.error('cannot accept LLDP. unsupported version. %x',
                  msg.datapath.ofproto.OFP_VERSION)

    #加下面這段
    for port in self.ports.keys():
        if src_dpid == port.dpid and src_port_no == port.port_no:
            send_timestamp = self.ports[port].timestamp
            if send_timestamp:
                self.ports[port].delay = recv_timestamp - send_timestamp
    src = self._get_port(src_dpid, src_port_no)
    if not src or src.dpid == dst_dpid:
        return
```

#### Install Ryu
```
cd ~/ryu
pip uninstall setuptools
pip install setuptools==58.0.0
python3 setup.py install
pip3 install .
```

Test the installation:
```
ryu-manager
```
Press `Ctrl + C` to exit.

### 4. Installing ls2ic_sdn_routing
Install iperf3:
```
sudo apt install iperf3
```

Clone the project and install dependencies:
```
cd ~
git clone https://github.com/p76121699/ls2ic_sdn_routing.git
cd ls2ic_sdn_routing
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## 5. training and testing

### 5.1 Preparing Datasets

Use the following commands to generate topologies and traffic matrices:

```
python3 prepare_dataset.py --topology 32node --tms 144tm
python3 prepare_dataset.py --topology 32node --tms 24tm
python3 prepare_dataset.py --topology geant --tms 24tm
```

### 5.2 Configuring Environment Parameters

Edit the following file:
`config/controller/simple_monitor_config.py`
Example configuration:

```
config = {
    "conda_sh": "/home/username/miniconda3/etc/profile.d/conda.sh",
    "conda_env": "sdn1", # The name of the previously created conda environment
    "controller_dir": "/home/username/ls2ic_sdn_routing/utils",
    "controller_entry": "simple_monitor.py"
}
```

To find the full path of `conda.sh`, run:
```
find $HOME -name "conda.sh"
```

Typically, updating the username field in the path is sufficient.


### 5.3 Training

There is no need to manually activate the conda environment — the program will do this automatically.
```
conda deactivate
```

Run the training script:
```
sudo python3 main.py --env 32node_144tm --alg ls2ic train
sudo python3 main.py --env geant --alg ps_dqn_a train
```
#### NOTE: 
* Algorithms like drisr and ospf do not require training.
* Training/testing splits are defined in the config files under `config/env`.
* If your controller subprocess takes longer to start due to a slower machine, the DRL agent subprocess might encounter an error.
In this case, close all subprocesses and the main program, then re-run main.py.
Follow the shutdown procedure below to ensure a clean restart.
* To safely terminate training:
Step-by-step shutdown:

1. Press `Ctrl + C` in both the controller and DRL agent windows.
2. Press `Ctrl + C` in the main window (Mininet), then run:
```
sudo killall -p iperf3
sudo mn -c
```

This ensures all services are cleanly stopped.

### 5.4 Testing

After training is complete, run test mode to evaluate performance on the test set:
```
sudo python3 main.py --env 32node_144tm --alg ls2ic test
```

To test a single traffic matrix manually:
```
sudo python3 test_single_tm.py --env 32node_144tm --alg ls2ic
```

Then enter the traffic matrix ID (e.g., for 144 TMs):
```
06
41
```

### 5. Monitoring and Performance Analysis

You can analyze training and test results in the stats directory using the provided .ipynb or .py scripts.
We recommend using Visual Studio Code, selecting the sdn1 environment for best compatibility.


### Final Notes
For additional details, please refer to the README.md files in each subdirectory.
This guide assumes a Linux-based system and may require minor adjustments on other platforms.