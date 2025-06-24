# test_single_tm.py
import argparse, config, os, time, subprocess, signal
from loader import env_loader
from utils.init_path import init_paths
import pwd

def _spawn_new_terminal(cmd: str):
    return subprocess.Popen(
        ["gnome-terminal", "--", "bash", "-c", cmd]
    )

def spawn_controller(ctrl_cfg):
    cmd = (
        f"source {ctrl_cfg['conda_sh']} && "
        f"conda activate {ctrl_cfg['conda_env']} && "
        f"cd {ctrl_cfg['controller_dir']} && "
        f"ryu-manager --observe-link {ctrl_cfg['controller_entry']}"
    )
    return _spawn_new_terminal(cmd)

def spawn_drl(merged_cfg, mode):
    import tempfile, json
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as fp:
        json.dump(merged_cfg, fp)
        cfg_path = fp.name
    
    conda_sh  = merged_cfg['conda_sh']
    conda_env = merged_cfg['conda_env']
    project_root = merged_cfg.get('project_root', os.getcwd())

    cmd = (
        f"source {conda_sh} && "
        f"conda activate {conda_env} && "
        f"python {project_root}/run_drl.py "
        f"--merged_cfg {cfg_path} --mode {mode}"
    )
    return _spawn_new_terminal(cmd)

def init_result_dirs(config):
    algs_name = config["algs_name"]
    base_dir = os.path.join("results", algs_name)
    model_dir = os.path.join(base_dir, "model")
    metrics_dir = os.path.join(base_dir, "Metrics")

    if config["algs_name"] == 'drsir':
        stretch_dir = os.path.join(base_dir, "stretch")
        os.makedirs(stretch_dir, exist_ok=True)

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    sudo_uid = os.environ.get("SUDO_UID")
    if sudo_uid:
        try:
            uid = int(sudo_uid)
            user_info = pwd.getpwuid(uid)
            gid = user_info.pw_gid

            os.chown(base_dir, uid, gid)
            os.chown(model_dir, uid, gid)
            os.chown(metrics_dir, uid, gid)

            if algs_name == 'drsir':
                os.chown(stretch_dir, uid, gid)
        except Exception as e:
            pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True,  help="32node_24tm")
    parser.add_argument("--alg", required=True,  help="ls2ic / meanfield ...")
    parser.add_argument("--ctrl", default="simple_monitor", help="controller config key")
    args = parser.parse_args()

    env_cfg, alg_cfg, ctrl_cfg = config.get(args.env, args.alg, args.ctrl)
    
    init_result_dirs(alg_cfg)
    init_paths(env_cfg, alg_cfg)
        
    os.environ["ALG_NAME"] = alg_cfg["algs_name"]
    os.environ["ENV_NAME"] = env_cfg["topology"]
    # --- 1. 先建拓樸 ---------------------------------------------------
    print("Building topology ...")
    net = env_loader.build_topo(env_cfg)

    # --- 2. 開 Ryu controller -----------------------------------------
    ctrl_proc = spawn_controller(ctrl_cfg)

    print("Controller spawned, wait 30 s ...")
    time.sleep(30)

    while(1):
        input_ = input("Enter the traffic matrix ID to be tested (e.g., 06) or type QUIT to exit: ").strip()
    
        if input_.upper() == 'QUIT':
            break
        
        if not input_:
            print("Input cannot be empty. Please enter a valid ID or QUIT.")
            continue

        try:
            tm_id = int(input_)
            if tm_id < 0:
                print("Traffic matrix ID must be non-negative.")
                continue
            formatted_input = f"{tm_id:02}"
        except ValueError:
            print("Invalid input. Please enter a numeric traffic matrix ID (e.g., 06).")
            continue

        # --- 3. 啟動 DRL 訓練 ---------------------------------------------
        print(f"Start DRL for traffic matrix {formatted_input} ...")
        merged_cfg = {**env_cfg, **alg_cfg, **ctrl_cfg}   # SimpleNamespace 給 training()
        merged_cfg["tm_id"] = formatted_input
        drl_proc = spawn_drl(merged_cfg, 'test_single')

        # --- 4. 啟動流量 ---------------------------------------------------
        print("Start traffic ...")
        env_loader.start_single_traffic(net, env_cfg, formatted_input)

        # 關閉 DRL 程序
        drl_proc.terminate()
        drl_proc.wait()

    # --- 5. 收尾 -------------------------------------------------------
    print("Training finished, clean up.")
    net.stop()
    ctrl_proc.terminate()
    ctrl_proc.wait()

if __name__ == "__main__":
    main()