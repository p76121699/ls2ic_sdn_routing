# Traffic Generator
We only provide the datasets we use. If you want the complete dataset, please refer to the following

## Download

Go to:
https://totem.info.ucl.ac.be/dataset.html
Download the dataset. After the file is downloaded, unzip it. There will be a traffic-matrices folder with traffic matrices for many time periods. Please put the entire traffic-matrices in dataset/geant_taffic/traffic_generator

## Modify the program

If you want to generate traffic matrices for different days, please modify iperf3_geant.py (Located at utils/iperf3_geant.py)

After modifying lines 12 and 14, execute prepare_dataset.py again.