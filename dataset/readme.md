# Dataset Preparation

To generate traffic scripts, please refer to the README located in the `traffic_generator/` directory.

If you would like to use a custom network topology, please follow these steps:

## Step-by-Step Instructions

1. Create a new folder under this directory to store traffic data.
Name it in the format `XXX_Traffic`, e.g., `48node_traffic`.
2. Inside the folder, create a subdirectory to store traffic matrices.
Use the format `XXX_YYtm`, e.g., `48node_72tm`.
This name will later be used as the `--env` argument when running `main.py`.
3. Create a topology file named `bw_r.txt` inside the `XXX_Traffic` folder.
This file defines the network topology and must follow the format:

```
node1, node2, <unused>, link_capacity
```
This file will be used by Mininet to build the network topology, and is required.

Note: The third column is ignored and can contain any placeholder value.

4. Generate top-k shortest paths using the script `get_k_paths.py`.
This defines the action space for the DRL agent.

```
python dataset/get_k_paths.py
```

Before running the script, open `get_k_paths.py` and update the following lines to match your folder name:

```
if __name__ == "__main__":
    dataset_folder = "48node_traffic"  # ← Change this to your actual folder
    bw_r_file = os.path.join(dataset_folder, "bw_r.txt")
    output_file = os.path.join(dataset_folder, "k_paths.json")
    generate_k_paths(bw_r_file, output_file, k=20)
```