# run_drl.py
import argparse, json
from loader import train_loader
from algs.drsir import DRL_paths_threading

def main():
    parser = argparse.ArgumentParser(description="Launch DRL agent in train or test mode")
    parser.add_argument("--merged_cfg", required=True,
                        help="Path to JSON file dumped by main.py that holds the merged config dict")
    parser.add_argument("--mode", choices=["train", "test_single", "test"], default="train",
                        help="training or testing mode (default: train)")

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load merged config produced by main.py
    # ------------------------------------------------------------------
    with open(args.merged_cfg) as f:
        cfg_dict = json.load(f)

    #merged_cfg = SimpleNamespace(**cfg_dict)

    # ------------------------------------------------------------------
    # Dispatch to training / testing
    # ------------------------------------------------------------------
    if cfg_dict["algs_name"] == "drsir":
        DRL_paths_threading.DRL_thread(cfg_dict)
    else:
        if args.mode == "train":
            train_loader.training(cfg_dict)
        elif args.mode == "test_single":
            train_loader.testing(cfg_dict)
        elif args.mode == "test":
            train_loader.testing_anime(cfg_dict)

if __name__ == "__main__":
    main()
