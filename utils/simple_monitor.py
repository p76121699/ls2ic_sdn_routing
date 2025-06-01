from operator import attrgetter
from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.base.app_manager import lookup_service_brick
from ryu.controller.handler import MAIN_DISPATCHER, DEAD_DISPATCHER
from ryu.controller.handler import CONFIG_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.topology import event, switches
from ryu.ofproto.ether import ETH_TYPE_IP
from ryu.topology.api import get_switch, get_link
from ryu.ofproto import ofproto_v1_3
from ryu.lib import hub
from ryu.lib.packet import packet
from ryu.lib.packet import arp
import time
import simple_awareness
import simple_delay
import manager
import json, ast
import setting
import csv
import time
import os

class simple_Monitor(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]
    _CONTEXTS = {"simple_awareness": simple_awareness.simple_Awareness,
                 "simple_delay": simple_delay.simple_Delay,
                 "manager": manager.Manager}

    def __init__(self, *args, **kwargs):
        super(simple_Monitor, self).__init__(*args, **kwargs)
        self.name = "monitor"
        self.count_monitor = 0
        self.topology_api_app = self
        self.datapaths = {}
        self.port_stats = {}
        self.port_speed = {}
        self.flow_stats = {}
        self.flow_speed = {}
        self.flow_loss = {}
        self.port_loss = {}
        self.stats = {}
        self.port_features = {}
        self.free_bandwidth = {}
        self.paths = {}
        self.installed_paths = {}
        self.alg_name = os.environ.get("ALG_NAME", "ls2ic")
        self.env_name = os.environ.get("ENV_NAME", "32node")
        self.awareness = kwargs["simple_awareness"]
        self.delay = kwargs["simple_delay"]
        self.manager = kwargs["manager"]
        self.shortest_paths = self.get_k_paths() # initial k_paths
        self.monitor_thread = hub.spawn(self.monitor)


    @set_ev_cls(ofp_event.EventOFPStateChange,
                [MAIN_DISPATCHER, DEAD_DISPATCHER])
    def state_change_handler(self, ev): 
        """
            Record datapath information.
        """
        datapath = ev.datapath
        if ev.state == MAIN_DISPATCHER:
            if datapath.id not in self.datapaths:
                self.logger.debug('Datapath registered:', datapath.id) 
                self.datapaths[datapath.id] = datapath
        elif ev.state == DEAD_DISPATCHER:
            if datapath.id in self.datapaths:
                self.logger.debug('Datapath unregistered:', datapath.id)
                del self.datapaths[datapath.id]

    def monitor(self):
        """
            Main entry method of monitoring traffic.
        """
        while True:
            s_time = time.time()  # start timestamp
            self.count_monitor += 1
            self.stats['port'] = {}
            print("[Statistics Module Ok]")
            print("[{0}]".format(self.count_monitor))
            for dp in self.datapaths.values():
                self.port_features.setdefault(dp.id, {}) # setdefault() returns the value of the item with the specified key
                self.paths = None
                self.request_stats(dp)

            if self.awareness.link_to_port:
                    '''
                    Do the Action.
                    '''
                    self.flow_install_monitor()

            if self.stats['port']:
                self.manager.get_port_loss()
                self.manager.get_link_free_bw()
                self.manager.get_link_used_bw()
                '''
                Save [bwd,delay,loss] information to net_info as State.
                '''
                self.manager.write_values(self.alg_name)
                
                if self.manager.link_free_bw and self.shortest_paths:
                    '''
                    Save k_paths_metrics_dic to calculate Reward.
                    '''
                    self.manager.get_k_paths_metrics_dic(self.shortest_paths,self.manager.link_free_bw, self.delay.link_delay, self.manager.link_loss, self.alg_name)

                self.show_stat('link')
            #print(self.awareness.link_to_port)
            d_time = time.time() - s_time
            hub.sleep(setting.MONITOR_PERIOD - d_time)

#------------------------------------------------------------------------------------
#---------------------FLOW INSTALLATION MODULE FUNCTIONS ----------------------------
    def flow_install_monitor(self): 
        print("[Flow Installation Ok]")
        out_time= time.time()
        for dp in self.datapaths.values():   
            for dp2 in self.datapaths.values():
                if dp.id != dp2.id:
                    ip_src = '10.0.0.'+str(dp.id)
                    ip_dst = '10.0.0.'+str(dp2.id)
                    self.forwarding(dp.id, ip_src, ip_dst, dp.id, dp2.id)
                    time.sleep(0.0005)
        end_out_time = time.time()
        out_total_ = end_out_time - out_time
        print("Flow installation time: {0}s".format(out_total_))
        return 

    def forwarding(self, dpid, ip_src, ip_dst, src_sw, dst_sw):
        """
            Get paths and install them into datapaths.
        """
        self.installed_paths.setdefault(dpid, {})
        path = self.get_path(str(src_sw), str(dst_sw))
        self.installed_paths[src_sw][dst_sw] = path 
        #print(str(src_sw), str(dst_sw),path)
        flow_info = (ip_src, ip_dst)
        self.install_flow(self.datapaths, self.awareness.link_to_port, path, flow_info)

    def get_path(self, src, dst):
        
            if self.paths != None:
                path = self.paths.get(src).get(dst)[0]
                return path
            else:
                paths = self.get_dRL_paths()
                path = paths.get(src).get(dst)[0]
                return path

    def install_flow(self, datapaths, link_to_port, path,
                     flow_info, data=None):
        init_time_install = time.time()
        ''' 
            Install flow entires. 
            path=[dpid1, dpid2...]
            flow_info=(src_ip, dst_ip)
        '''
        if path is None or len(path) == 0:
            self.logger.info("Path error!")
            return
        
        in_port = 1
        first_dp = datapaths[path[0]]

        out_port = first_dp.ofproto.OFPP_LOCAL
        back_info = (flow_info[1], flow_info[0])

        # Flow installing por middle datapaths in path
        if len(path) > 2:
            for i in range(1, len(path)-1):
                port = self.get_port_pair_from_link(link_to_port,
                                                    path[i-1], path[i])
                port_next = self.get_port_pair_from_link(link_to_port,
                                                         path[i], path[i+1])
                if port and port_next:
                    src_port, dst_port = port[1], port_next[0]
                    datapath = datapaths[path[i]]
                    #print(datapath.id,flow_info,src_port,dst_port)
                    self.send_flow_mod(datapath, flow_info, src_port, dst_port)
                    #self.send_flow_mod(datapath, back_info, dst_port, src_port)
                    # print("Inter link flow install")
        if len(path) > 1:
            # The last flow entry
            port_pair = self.get_port_pair_from_link(link_to_port,
                                                     path[-2], path[-1])
            if port_pair is None:
                self.logger.info("Port is not found")
                return
            src_port = port_pair[1]
            dst_port = 1 #I know that is the host port --
            last_dp = datapaths[path[-1]]

            self.send_flow_mod(last_dp, flow_info, src_port, dst_port)
            #self.send_flow_mod(last_dp, back_info, dst_port, src_port)

            # The first flow entry
            port_pair = self.get_port_pair_from_link(link_to_port, path[0], path[1])
            if port_pair is None:
                self.logger.info("Port not found in first hop.")
                return
            out_port = port_pair[0]
            self.send_flow_mod(first_dp, flow_info, in_port, out_port)
            #self.send_flow_mod(first_dp, back_info, out_port, in_port)

        # src and dst on the same datapath
        else:
            out_port = 1
            self.send_flow_mod(first_dp, flow_info, in_port, out_port)
            #self.send_flow_mod(first_dp, back_info, out_port, in_port)

        end_time_install = time.time()
        total_install = end_time_install - init_time_install
        # print("Time install", total_install)
#------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------

    def get_k_paths(self):
        file = f"../dataset/{self.env_name}_traffic/k_paths.json"
        with open(file,'r') as json_file:
            k_shortest_paths = json.load(json_file)
            k_shortest_paths = ast.literal_eval(json.dumps(k_shortest_paths))      
        print("[k_paths OK]")
        return k_shortest_paths

    def request_stats(self, datapath):
        self.logger.debug('send stats request: %016x', datapath.id)
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser 
        req = parser.OFPPortDescStatsRequest(datapath, 0) # for port description 
        datapath.send_msg(req)
        req = parser.OFPPortStatsRequest(datapath, 0, ofproto.OFPP_ANY) 
        datapath.send_msg(req)

    

    def send_flow_mod(self, datapath, flow_info, src_port, dst_port):
        """
            Build flow entry, and send it to datapath.
        """
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        actions = []
        actions.append(parser.OFPActionOutput(dst_port))

        match = parser.OFPMatch(
             eth_type=ETH_TYPE_IP, ipv4_src=flow_info[0], 
             ipv4_dst=flow_info[1])

        self.add_flow(datapath, 1, match, actions,
                      idle_timeout=270, hard_timeout=0)
        

    def add_flow(self, dp, priority, match, actions, idle_timeout=0, hard_timeout=0):
        """
            Send a flow entry to datapath.
        """
        ofproto = dp.ofproto
        parser = dp.ofproto_parser
        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]
        mod = parser.OFPFlowMod(datapath=dp, command=dp.ofproto.OFPFC_ADD, priority=priority,
                                idle_timeout=idle_timeout,
                                hard_timeout=hard_timeout,
                                match=match, instructions=inst)
        dp.send_msg(mod)

    def del_flow(self, datapath, dst):
        """
            Deletes a flow entry of the datapath.
        """
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        match = parser.OFPMatch(eth_type=ETH_TYPE_IP, ipv4_src=flow_info[0],ipv4_dst=flow_info[1])
        mod = parser.OFPFlowMod(datapath=datapath, match=match, cookie=0,command=ofproto.OFPFC_DELETE)
        datapath.send_msg(mod)

    def build_packet_out(self, datapath, buffer_id, src_port, dst_port, data):
        """
            Build packet out object.
        """
        actions = []
        if dst_port:
            actions.append(datapath.ofproto_parser.OFPActionOutput(dst_port))

        msg_data = None
        if buffer_id == datapath.ofproto.OFP_NO_BUFFER:
            if data is None:
                return None
            msg_data = data

        out = datapath.ofproto_parser.OFPPacketOut(
            datapath=datapath, buffer_id=buffer_id,
            data=msg_data, in_port=src_port, actions=actions)
        return out

    def arp_forwarding(self, msg, src_ip, dst_ip):
        """
            Send ARP packet to the destination host if the dst host record
            is existed.
            result = (datapath, port) of host
        """
        datapath = msg.datapath
        ofproto = datapath.ofproto

        result = self.awareness.get_host_location(dst_ip)
        if result:
            # Host has been recorded in access table.
            datapath_dst, out_port = result[0], result[1]
            datapath = self.datapaths[datapath_dst]
            out = self.build_packet_out(datapath, ofproto.OFP_NO_BUFFER,
                                         ofproto.OFPP_CONTROLLER,
                                         out_port, msg.data)
            datapath.send_msg(out)
            self.logger.debug("Deliver ARP packet to knew host")
        else:
            # self.flood(msg)
            pass

    def get_port_pair_from_link(self, link_to_port, src_dpid, dst_dpid):
        """
            Get port pair of link, so that controller can install flow entry.
            link_to_port = {(src_dpid,dst_dpid):(src_port,dst_port),}
        """
        if (src_dpid, dst_dpid) in link_to_port:
            return link_to_port[(src_dpid, dst_dpid)]
        else:
            self.logger.info("Link from dpid:%s to dpid:%s is not in links" %
             (src_dpid, dst_dpid))
            return None 

    def get_dRL_paths(self):
        
        file = f"../results/{self.alg_name}/drl_paths.json"
        try:
            with open(file,'r') as json_file:
                paths_dict = json.load(json_file)
                paths_dict = ast.literal_eval(json.dumps(paths_dict))
                self.paths = paths_dict
                return self.paths
        # except ValueError as e: #error excpetion when trying to read the json and is still been updated
        except:
            print("READ FILE EXCEPT!!!!!!!!!!!!!!!!!!!!!!!!!!")
            time.sleep(0.35)
            with open(file,'r') as json_file: #try again
                paths_dict = json.load(json_file)
                paths_dict = ast.literal_eval(json.dumps(paths_dict))
                self.paths = paths_dict
                return self.paths

    #-----------------------STATISTICS MODULE FUNCTIONS -------------------------
    def save_stats(self, _dict, key, value, length=5): #Save values in dics (max len 5)
        if key not in _dict:
            _dict[key] = []
        _dict[key].append(value)
        if len(_dict[key]) > length:
            _dict[key].pop(0)

    def get_speed(self, now, pre, period):
        if period:
            return ((now - pre)*8) / period # byte to bit
        else:
            return 0

    def get_time(self, sec, nsec): #Total time that the flow was alive in seconds
        return sec + nsec / 1000000000.0 

    def get_period(self, n_sec, n_nsec, p_sec, p_nsec):
                                                         # calculates period of time between flows
        return self.get_time(n_sec, n_nsec) - self.get_time(p_sec, p_nsec)
    
    def get_sw_dst(self, dpid, out_port):
        for key in self.awareness.link_to_port:
            src_port = self.awareness.link_to_port[key][0]
            if key[0] == dpid and src_port == out_port:
                dst_sw = key[1]
                dst_port = self.awareness.link_to_port[key][1]
                # print(dst_sw,dst_port)
                return (dst_sw, dst_port)

    def get_link_bw(self, file_bw, src_dpid, dst_dpid):
        fin = open(file_bw, "r")
        bw_capacity_dict = {}
        for line in fin:
            a = line.split(',')
            if a:
                s1 = a[0]
                s2 = a[1]
                # bwd = a[2] #random capacities
                bwd = a[3] #original capacities
                bw_capacity_dict.setdefault(s1,{})
                bw_capacity_dict[str(a[0])][str(a[1])] = bwd
        fin.close()
        bw_link = bw_capacity_dict[str(src_dpid)][str(dst_dpid)]
        return bw_link

    def get_free_bw(self, port_capacity, speed):
        # freebw: Kbit/s
        return max(port_capacity - (speed/ 1000.0), 0)

    @set_ev_cls(ofp_event.EventOFPPortDescStatsReply, MAIN_DISPATCHER)
    def port_desc_stats_reply_handler(self, ev):
        """
            Save port description info.
        """
        msg = ev.msg
        dpid = msg.datapath.id
        ofproto = msg.datapath.ofproto

        config_dict = {ofproto.OFPPC_PORT_DOWN: "Down",
                       ofproto.OFPPC_NO_RECV: "No Recv",
                       ofproto.OFPPC_NO_FWD: "No Farward",
                       ofproto.OFPPC_NO_PACKET_IN: "No Packet-in"}

        state_dict = {ofproto.OFPPS_LINK_DOWN: "Down",
                      ofproto.OFPPS_BLOCKED: "Blocked",
                      ofproto.OFPPS_LIVE: "Live"}

        ports = []
        for p in ev.msg.body:
            if p.port_no != 1:

                ports.append('port_no=%d hw_addr=%s name=%s config=0x%08x '
                             'state=0x%08x curr=0x%08x advertised=0x%08x '
                             'supported=0x%08x peer=0x%08x curr_speed=%d '
                             'max_speed=%d' %
                             (p.port_no, p.hw_addr,
                              p.name, p.config,
                              p.state, p.curr, p.advertised,
                              p.supported, p.peer, p.curr_speed,
                              p.max_speed))
                if p.config in config_dict: # if in key
                    config = config_dict[p.config]
                else:
                    config = "up"

                if p.state in state_dict:
                    state = state_dict[p.state]
                else:
                    state = "up"

                # Recording data.
                port_feature = [config, state]
                self.port_features[dpid][p.port_no] = port_feature

    @set_ev_cls(ofp_event.EventOFPPortStatsReply, MAIN_DISPATCHER)
    def port_stats_reply_handler(self, ev):
        a = time.time()
        body = ev.msg.body
        dpid = ev.msg.datapath.id

        self.stats['port'][dpid] = body
        self.free_bandwidth.setdefault(dpid, {})
        self.port_loss.setdefault(dpid, {})
        """
            Save port's stats information into self.port_stats.
            Calculate port speed and Save it.
            self.port_stats = {(dpid, port_no):[(tx_bytes, rx_bytes, rx_errors, duration_sec,  duration_nsec),],}
            self.port_speed = {(dpid, port_no):[speed,],}
            Note: The transmit performance and receive performance are independent of a port.
            Calculate the load of a port only using tx_bytes.
        
        Replay message content:
            (stat.port_no,
             stat.rx_packets, stat.tx_packets,
             stat.rx_bytes, stat.tx_bytes,
             stat.rx_dropped, stat.tx_dropped,
             stat.rx_errors, stat.tx_errors,
             stat.rx_frame_err, stat.rx_over_err,
             stat.rx_crc_err, stat.collisions,
             stat.duration_sec, stat.duration_nsec))
        """

        for stat in sorted(body, key=attrgetter('port_no')):
            port_no = stat.port_no
            key = (dpid, port_no) # src_dpid, src_port
            value = (stat.tx_bytes, stat.rx_bytes, stat.rx_errors,
                     stat.duration_sec, stat.duration_nsec, stat.tx_errors, stat.tx_dropped, stat.rx_dropped, stat.tx_packets, stat.rx_packets)
            self.save_stats(self.port_stats, key, value, 5) # save switch's port information
            if port_no != ofproto_v1_3.OFPP_LOCAL: # local openflow port       
                if port_no != 1 and self.awareness.link_to_port :
                    # Get port speed and Save it
                    pre = 0
                    tmp = self.port_stats[key]
                    if len(tmp) > 1: # have pre value and now value
                        # Calculate with the tx_bytes and rx_bytes
                        pre = tmp[-2][0] + tmp[-2][1] # post Tx + Rx
                        period = self.get_period(tmp[-1][3], tmp[-1][4], tmp[-2][3], tmp[-2][4])
                    speed = self.get_speed(self.port_stats[key][-1][0] + self.port_stats[key][-1][1], pre, period) #speed in bits/s
                    self.save_stats(self.port_speed, key, speed, 5)
                    file_bw = f"../dataset/{self.env_name}_traffic/bw_r.txt"
                    link_to_port = self.awareness.link_to_port
                    for k in list(link_to_port.keys()): # link_to_port.keys()  --> (src_dpid,dst_dpid)
                        if k[0] == dpid:
                            if link_to_port[k][0] == port_no:
                                dst_dpid = k[1]
                                bw_link = float(self.get_link_bw(file_bw, dpid, dst_dpid)) #23nodos
                                # bw_link = float(100) #resto de topologias todos los links tienen 10mbps de capacidad
                                port_state = self.port_features.get(dpid).get(port_no)

                                if port_state:
                                    #bw_link_kbps = bw_link * 1000.0
                                    bw_link_kbps = bw_link * 1000.0 * 2.0
                                    self.port_features[dpid][port_no].append(bw_link_kbps)                     
                                    free_bw = self.get_free_bw(bw_link_kbps, speed)
                                    # print'free_bw of link ({0}, {1}) is: {2}'.format(dpid,dst_dpid,free_bw)
                                    # print('------------------------------------')
                                    self.free_bandwidth[dpid][port_no] = free_bw  # save free_bandwidth
        # print("stats time {0}".format(time.time()-a))


    # @set_ev_cls(ofp_event.EventOFPPortStatus, MAIN_DISPATCHER)
    # def port_status_handler(self, ev):
    #     """
    #         Handle the port status changed event.
    #     """
    #     msg = ev.msg
    #     ofproto = msg.datapath.ofproto
    #     reason = msg.reason
    #     dpid = msg.datapath.id
    #     port_no = msg.desc.port_no

    #     reason_dict = {ofproto.OFPPR_ADD: "added",
    #                    ofproto.OFPPR_DELETE: "deleted",
    #                    ofproto.OFPPR_MODIFY: "modified", }

    #     if reason in reason_dict:
    #         print ("switch%d: port %s %s" % (dpid, reason_dict[reason], port_no))
    #     else:
    #         print ("switch%d: Illegal port state %s %s" % (dpid, port_no, reason))

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def packet_in_handler(self, ev):
        '''
            In packet_in handler, we need to learn access_table by ARP and IP packets.
            Therefore, the first packet from UNKOWN host MUST be ARP
        '''
        msg = ev.msg
        pkt = packet.Packet(msg.data)
        arp_pkt = pkt.get_protocol(arp.arp)
        if isinstance(arp_pkt, arp.arp):
            self.arp_forwarding(msg, arp_pkt.src_ip, arp_pkt.dst_ip)

    
    def show_stat(self, _type):
        if setting.TOSHOW is False:
            return
        if _type == 'link':
            print('\nnode1  node2  used-bw(Kb/s)   free-bw(Kb/s)    latency(ms)     loss')
            print('-----  -----  --------------   --------------   -----------    ---- ') 
            format_ = '{:>5}  {:>5} {:>14.5f}  {:>14.5f}  {:>12}  {:>12}'
            links_in = []
            for link, values in sorted(self.manager.net_info.items()):
                links_in.append(link)
                tup = (link[1], link[0])
                if tup not in links_in:
                    print(format_.format(link[0],link[1],
                        self.manager.link_used_bw[link]/1000.0,
                        values[0], values[1], values[2]))

            

