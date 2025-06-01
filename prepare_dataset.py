from utils.iperf3_scripts import generate_traffic as generate_32node
from utils.iperf3_geant import generate_traffic as generate_geant
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--topology', type=str, required=True, help='Input topology name')
    parser.add_argument('--tms', type=str, required=True, help='Output number of tms')
    parser.add_argument('--duration', type=int, default=80000, help='Time duration per flow')
    args = parser.parse_args()

    if args.topology == "32node":
        pkl_file = f"dataset/{args.topology}_traffic/traffic_generator/{args.topology}_tms_info_{args.tms}.pkl"
        out_path = f"dataset/{args.topology}_traffic/{args.topology}_{args.tms}"
        generate_32node(pkl_file, out_path, args.duration)
    elif args.topology == "geant":
        data_file = f"dataset/{args.topology}_traffic/traffic_generator"
        out_path = f"dataset/{args.topology}_traffic/23node"
        generate_geant(data_file, out_path, args.duration)