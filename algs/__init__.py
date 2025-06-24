REGISTRY = {}

from .ls2ic import ls2ic_agent
from .meanfield import meanfield_agent
from .ps_dqn_a import ps_dqn_a_agent
from .ps_dqn import ps_dqn_agent
from .ospf import ospf_agent
from .adaptive_dijkstra import dijkstraAgent

REGISTRY["ls2ic"] = ls2ic_agent
REGISTRY["meanfield"] = meanfield_agent
REGISTRY["ps_dqn_a"] = ps_dqn_a_agent
REGISTRY["ps_dqn"] = ps_dqn_agent
REGISTRY["ospf"] = ospf_agent
REGISTRY["ospf"] = dijkstraAgent