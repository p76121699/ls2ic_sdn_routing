import sys
import os
import statistics
import pickle
from pathlib import Path
import matplotlib
matplotlib.use("agg")


def generate_traffic(pkl_path: str, out_dir: str, duration: int = 80000):
    mean_loads_list = []
    total_load_list = []
    with open(pkl_path, 'rb') as f:
        od_bin, tms = pickle.load(f)

    num_tms = len(tms)

    tms_hours = [f"{i:02}" for i in range(num_tms)]
    if num_tms == 144:
        tms_hours = tms_hours[12:] + tms_hours[:12]
    else:
        tms_hours = tms_hours[4:] + tms_hours[:4]

    for j, tm in enumerate(tms):
        #FOR CREATING FOLDERS PER TRAFFIC MATRIX  
        nameTM = Path(out_dir) / f'TM-{tms_hours[j]}'
        print('------',nameTM)
        if not os.path.exists(nameTM):
            os.mkdir(nameTM)

        #--------------------FLOWS--------------------------
        # FOR CREATING FOLDERS PER NODE
        for i in range(len(tm[0])):
            nameNode = str(nameTM)+'/Clients'
            if not os.path.exists(nameNode):
                    os.mkdir(nameNode)
                    # print("Folder created:", nameNode )

        for i in range(len(tm[0])):
            nameNode = str(nameTM)+'/Servers'
            if not os.path.exists(nameNode):
                    os.mkdir(nameNode)
                    # print("Folder created:", nameNode )

        # Default parameters
        time_duration = duration
        throughput = 0.0 #take it in kbps from TM

        # Obtain parameters from arguments
        for arg in sys.argv[1:]:
            option = arg.split("=")
            if option[0] == "--time":
                time_duration = option[1]

            else:
                print ("Option %s is not valid" % option[0])

        for src in range(len(tm[0])):
            for dst in range(len(tm[0])):
                if src == dst and tm[src][dst] != 0.0:
                    tm[src][dst] = 0.0

        for src in range(len(tm[0])):
            src_ = src+1
            #SCRIPT WITH COMMANDS FOR GENERATE TRAFFIC
            if src_ > 9:
                fileClient = open(str(nameTM)+"/Clients/client_{0}.sh".format(str(src_)), 'w')
            else:
                fileClient = open(str(nameTM)+"/Clients/client_0{0}.sh".format(str(src_)), 'w')

            n=0
            for dst in range(len(tm[0])):
                dst_ = dst+1
                throughput = float(tm[src][dst])
                # throughput_g = throughput / (100) #scale the throughput value to mininet link capacities
                temp1 = ''
                if throughput != 0.0:
                    n = n+1
                    temp1 = ''
                    temp1 += '\n'
                    temp1 += 'stdbuf -o0 iperf3 -c '
                    temp1 += '10.0.0.{0} '.format(str(dst_))
                    if dst_ > 9:   
                        temp1 += '-p {0}0{1} '.format(str(src_),str(dst_))
                    else:
                        temp1 += '-p {0}00{1} '.format(str(src_),str(dst_))
                    temp1 += '-u -b '+str(format(throughput,'.3f'))+'k'
                    temp1 += ' -w 256k -t '+str(time_duration)
                    temp1 += ' -i 0 '
                    
                    temp1 += ' &\n' # & at the end of the line it's for running the process in bkg
                    temp1 += 'sleep 0.4'

                fileClient.write(temp1)
            fileClient.close()
        # print(na)
        for dst in range(len(tm[0])):
            dst_ = dst+1
            
            #SCRIPT FOR COMMANDS TO INITIALIZE SERVERS LISTENING
            if dst_ > 9:
                fileServer = open(str(nameTM)+"/Servers/server_{0}.sh".format(str(dst_)), 'w') 
            else:
                fileServer = open(str(nameTM)+"/Servers/server_0{0}.sh".format(str(dst_)), 'w') 

            for src in range(len(tm[0])):
                src_ = src+1
                temp2 = ''
                if tm[src][dst] != 0:
                    # n = n+1
                    temp2 = ''
                    temp2 += '\n'
                    temp2 += 'iperf3 -s '
                    if dst_ > 9:   
                        temp2 += '-p {0}0{1} '.format(str(src_),str(dst_))
                    else:
                        temp2 += '-p {0}00{1} '.format(str(src_),str(dst_))
                    temp2 += ' -1'
                    
                    temp2 += ' &\n' # & at the end of the line it's for running the process in bkg
                    temp2 += 'sleep 0.3'
                fileServer.write(temp2)
            fileServer.close() 

        list_loads = []
        
        for src in range(len(tm[0])):
            src_ = src+1
            for dst in range(len(tm[0])):
                dst_ = dst+1
                if tm[src][dst]/100 != 0.0: #/100 to scale to mininet and it is in kbps
                    list_loads.append(tm[src][dst])

        total = sum(list_loads)
        mean = statistics.mean(list_loads)

        total_load_list.append(total)
        mean_loads_list.append(mean)

        print('Mean load of TM {0}: {1}'.format(nameTM, total))

        j += 1
    print('List of mean load for each TM:', mean_loads_list)
    print('List of total load for each TM:', total_load_list)
