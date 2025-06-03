# Results Directory
This directory stores all training logs, test results, and runtime communication files between the DRL agent and the SDN controller.

### File Generation and Responsibilities
1. DRL Agent Output
The training and testing process, managed by `loader/train_loader.py`, automatically saves output under:
```
results/<algs_name>/
```
This includes:
* `drl_paths.json`: The action file used by the controller.
* `dqn_loss.txt`, `output.txt`, and other training logs.
* Evaluation metrics (e.g., '_eval_metrics.csv', 'net_metrics.csv') during testing.

2. Controller Output
During operation, `utils/simple_monitor.py` (the Ryu controller) writes environment-related files here for the agent to read:
* `net_info.csv`: Encode the original network state into the agent state.
* `paths_metrics.json`: Contains performance metrics such as link delay, throughput, and loss (used as reward signals).

### Analysis Tools
To visualize and analyze training/test results, refer to the scripts under the `stats/` directory.
These tools directly access files in the `results/<algs_name>/` folders.
