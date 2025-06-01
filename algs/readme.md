# Custom Algorithm Integration Guide

If you would like to integrate your own algorithm (referred to as `XXX`), please follow the structure described below to ensure smooth execution within our framework.

1. Use `ospf` as the Minimal Reference
For a minimal working example, refer to the `ospf.py` file in this directory. This file represents the most basic implementation template.

2. File Naming Convention
The name of your algorithm file `XXX.py` must correspond to the `algs_name` defined in `config/algs/XXX_config.py`.

The train_loader module will instantiate your DRL agent based on the algs_name field provided in the config.

3. Register Your Algorithm
After creating `XXX.py`, remember to register your algorithm in `algs/__init__.py` to make it discoverable by the training framework.