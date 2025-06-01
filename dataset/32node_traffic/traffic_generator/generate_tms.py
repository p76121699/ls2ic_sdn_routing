from tmgen.models import modulated_gravity_tm
from tmgen import TrafficMatrix
from numpy import random
import numpy as np
import random
import numpy as np
import pickle
import time
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
'''
This file generates the pkl file with the TMs based on the modulated gravity model 
by using the TMGen python library. 
Please refer to https://github.com/progwriter/TMgen for installing the library.
'''

"""
    Generate a modulated gravity traffic matrix with the given parameters
    :-- num_nodes: number of Points-of-Presence (i.e., origin-destination pairs)
    :-- num_tms: total number of traffic matrices to generate (i.e., time epochs)
    :-- mean_traffic: the average total volume of traffic
    :-- pm_ratio: peak-to-mean ratio. Peak  traffic will be larger by
        this much (must be bigger than 1). Default is 1.5
    :-- t_ratio: trough-to-mean ratio. Default is 0.75
    :-- diurnal_freq: Frequency of modulation. Default is 1/24 (i.e., hourly)
        if you are generating multi-day TMs
    :-- spatial_variance: Variance on the volume of traffic between
        origin-destination pairs.
        Pick something reasonable with respect to your mean_traffic.
        Default is 100
    :-- temporal_variance: Variance on the volume in time. Default is 0.01
"""
random.seed(17)
np.random.seed(17)

num_nodes = 32
num_tms = 144
mean_traffic = 9000 #per OD kbps -- 75% of link capacities in network
pm_ratio = 1.5
diurnal_freq = 1/24 #generates tms per hours of day
t_ratio = 0.85
spatial_variance = 300
temporal_variance = 0.05
     
tm = modulated_gravity_tm(num_nodes, num_tms, mean_traffic, pm_ratio, t_ratio)#, spatial_variance=spatial_variance, temporal_variance=temporal_variance)
print(np.size(tm.at_time(0)))
print(tm.at_time(0))

#choose randomly nodes for final traffic matrix
static_od_bin = [[0 if random.random() < 0.01 else 1 for _ in range(num_nodes)] for _ in range(num_nodes)]

tms = []

for i in range(num_tms):
    high_load_bin = [[1 if random.random() < 0.4 else 0 for _ in range(num_nodes)] for _ in range(num_nodes)]
    
    base_tm = np.array(tm.at_time(i))
    
    final_od_bin = np.array(static_od_bin)
    high_load_matrix = base_tm * 1.3 * np.array(high_load_bin)
    
    final_tm = final_od_bin * high_load_matrix
    tms.append(final_tm.tolist())

event_od = int((num_nodes-1) * num_nodes/20)

for i in range(num_tms):
    for _ in range(event_od):
        src, dst = np.random.randint(15), np.random.randint(15)
        if tms[i][src][dst] + 7500 < 75000:  
            tms[i][src][dst] += random.randint(5000, 7500)


mean_load_tms_final = []
total_load_tms_final = []
print("Tms final:")
for i in range(num_tms):
    # print('Epoch',i,'\n')#,tms[i])
    # print(np.array(tms[i]).mean())
    mean_load_tms_final.append(np.array(tms[i]).mean())
    # print(np.array(tms[i]).sum())
    total_load_tms_final.append(np.array(tms[i]).sum())
    # print()

with open(str(num_nodes)+'Nodes_tms_info_all.pkl','wb') as f:
    pickle.dump([static_od_bin, tms],f)

x = list(range(num_tms))
plt.plot(x,total_load_tms_final,marker = 'o', linestyle = '')
plt.title(str(num_nodes)+"Nodes topology loads (all)")
plt.xlabel('Tms')
plt.ylabel('Load (Mbps)')
plt.xticks(x)
plt.grid()
plt.savefig(str(num_nodes)+"Nodes_load_all.eps",bbox_inches = 'tight') 
plt.close()

#-------------Choosen tms for scripts traffic-------------
print("\n\nTms final train:")
train_percentage = 0.6
train_size = int(num_tms * train_percentage)
train_indices = sorted(random.sample(range(num_tms), train_size))

with open("train_indices.pkl", "wb") as f:
    pickle.dump(train_indices, f)

tms_train = [tms[i] for i in train_indices]
mean_load_tms_train = [np.array(tms_train[i]).mean() for i in range(len(tms_train))]
total_load_tms_train = [np.array(tms_train[i]).sum() for i in range(len(tms_train))]
print("Amount tms train: ",len(tms_train))
print('Mean load tms train:\n', mean_load_tms_train, len(mean_load_tms_train))
print('Total load tms train:\n', total_load_tms_train, len(total_load_tms_train))

with open(str(num_nodes)+'Nodes_tms_info_train.pkl','wb') as f:
    pickle.dump([static_od_bin, tms_train],f)

plt.plot(train_indices, total_load_tms_train, marker = 'o', linestyle = '')
plt.title(str(num_nodes)+"Nodes topology loads (train)")
plt.xlabel('Tms')
plt.ylabel('Load (Mbps)')
plt.xticks(train_indices)
plt.grid()
plt.savefig(str(num_nodes)+"Nodes_load_train.eps",bbox_inches = 'tight') 
plt.close()


#--------------------GEANT----------------------
geant_load = [5488416.834900001, 5043266.933899991, 5095276.160400002, 6880465.770500002, 9581002.970000016, 9705147.848200008, 9725804.7823, 9466881.377099995, 9276812.729499986, 8702496.461300002, 8453294.000400001, 8333976.977400003, 7917113.724099998, 8011548.747299995, 7522307.892499997, 6322587.761599998, 5872382.728099998]
y = [0,1,3,5,7,9,11,12,13,14,15,16,17,18,19,21,23]
plt.plot(y,geant_load)
plt.title("Geant load")
plt.xlabel('Tms')
plt.ylabel('Load (Mbps)')
# plt.legend(fontsize = 14,loc='lower right', fancybox=True, shadow=True)
plt.savefig("loadGeant.eps",bbox_inches = 'tight') 
plt.close()

print('Rec values:\n')
"""with open('48Nodes_tms_info_14.pkl', 'rb') as f:
    rec_od_bin,rec_tms = pickle.load(f)"""
#print(rec_tms, rec_od_bin)

