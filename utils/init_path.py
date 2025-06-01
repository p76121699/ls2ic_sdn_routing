import os
import json

def init_paths(env_config, algs_config):
    # Step 1: Read k_paths.json
    k_paths_file = env_config["k_paths_file"]
    with open(k_paths_file, "r") as f:
        k_paths = json.load(f)

    # Step 2: Assemble a new drl_paths structure, taking only the first path of each pair of src-dst (wrapping a list)
    drl_paths = {}
    for str_src, dsts in k_paths.items():
        drl_paths[str_src] = {}
        for str_dst, paths in dsts.items():
            if paths:
                drl_paths[str_src][str_dst] = [paths[0]]  # Package into a list (only one item)

    # Step 3: Save as drl_paths.json to the specified folder
    algs_name = algs_config["algs_name"]
    output_dir = os.path.join("results", algs_name)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "drl_paths.json")

    with open(output_path, "w") as f:
        json.dump(drl_paths, f, indent=2)
