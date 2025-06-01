import xml.etree.ElementTree as ET
import os
import numpy as np
import statistics
from pathlib import Path

def generate_traffic(data_file: str, out_dir: str, duration: int = 80000):
    for i in range(24):

        tm = np.zeros((24,24))
        if i < 10:
            file = data_file + '/traffic-matrices/IntraTM-2005-01-03-0'+str(i)+'-00.xml'
        else:
            file = data_file + '/traffic-matrices/IntraTM-2005-01-03-'+str(i)+'-00.xml'
        tree = ET.parse(file)
        root = tree.getroot()

        for src in root[1]:
            for dst in src:
                tm[int(src.attrib.get('id'))][int(dst.attrib.get('id'))] = float(dst.text) * 5


        
        nameTM = Path(out_dir) / f'TM-{i}'
        if i < 10:
            nameTM = Path(out_dir) / f'TM-0{i}'
        if not os.path.exists(nameTM):
            os.mkdir(nameTM)

        nameNode = str(nameTM)+'/Clients'
        if not os.path.exists(nameNode):
                os.mkdir(nameNode)

        nameNode = str(nameTM)+'/Servers'
        if not os.path.exists(nameNode):
                os.mkdir(nameNode)

        time_duration = duration


        for src in range(1,24):
            for dst in range(1,24):
                if src == dst and tm[src][dst] != 0.0:
                    tm[src][dst] = 0.0

        for src in range(1,24):
            #SCRIPT WITH COMMANDS FOR GENERATE TRAFFIC
            if src > 9:
                fileClient = open(str(nameTM)+"/Clients/client_{0}.sh".format(str(src)), 'w')
            else:
                fileClient = open(str(nameTM)+"/Clients/client_0{0}.sh".format(str(src)), 'w')
            outputstring_a1 = ''' #!/bin/bash \necho Generating traffic...
            '''
            #fileClient.write(outputstring_a1)
            n=0
            for dst in range(1,24):
                throughput = float(tm[src][dst])
                throughput_g = throughput / (100)  #scale the throughput value to mininet link capacities
                throughput_g = throughput_g 
                temp1 = ''
                if throughput_g >= 0.001:
                    n = n+1
                    temp1 = ''
                    temp1 += '\n'
                    #temp1 += 'taskset -c 0,1 stdbuf -o0 iperf3 -c '
                    temp1 += 'stdbuf -o0 iperf3 -c '
                    temp1 += '10.0.0.{0} '.format(str(dst))
                    if dst > 9:   
                        temp1 += '-p {0}0{1} '.format(str(src),str(dst))
                    else:
                        temp1 += '-p {0}00{1} '.format(str(src),str(dst))
                    temp1 += '-u -b '+str(format(throughput_g,'.3f'))+'k'
                    temp1 += ' -w 256k -t '+str(time_duration)
                    # if n != dst_amounts[src]: #When it is the last command, it does not include &
                    temp1 += ' -i 0 &\n' # & at the end of the line it's for running the process in bkg
                    temp1 += 'sleep 0.4'

                fileClient.write(temp1)
            fileClient.close()
    
        for dst in range(1,24):
            
            #SCRIPT FOR COMMANDS TO INITIALIZE SERVERS LISTENING
            if dst > 9:
                fileServer = open(str(nameTM)+"/Servers/server_{0}.sh".format(str(dst)), 'w') 
            else:
                fileServer = open(str(nameTM)+"/Servers/server_0{0}.sh".format(str(dst)), 'w') 
            outputstring_a2 = ''' #!/bin/bash \necho Initializing server listening...
            '''
            #fileServer.write(outputstring_a2)
            # n=0
            for src in range(1,24):
                temp2 = ''
                # n = n+1
                temp2 = ''
                temp2 += '\n'
                #temp2 += 'taskset -c 0,1 iperf3 -s '
                temp2 += 'iperf3 -s '
                if dst > 9:   
                    temp2 += '-p {0}0{1} '.format(str(src),str(dst))
                else:
                    temp2 += '-p {0}00{1} '.format(str(src),str(dst))
                #temp2 += '-1 -D '
                #temp2 += '-1 '
                temp2 += '-1 &'
                # if n != dst_amounts[src]: #When it is the last command, it does not include &
                #temp2 += ' -i 0 ' # & at the end of the line it's for running the process in bkg
                temp2 += ' \nsleep 0.3'
                fileServer.write(temp2)
            fileServer.close() 

        list_loads = []
        flow = 0
        for src in range(len(tm[0])):
            src_ = src+1
            for dst in range(len(tm[0])):
                dst_ = dst+1
                if tm[src][dst]/100 != 0.0: #/100 to scale to mininet and it is in kbps
                    flow += 1
                    list_loads.append(tm[src][dst])
        total = sum(list_loads)
        mean = statistics.mean(list_loads)
        print(total)
        print(mean)






