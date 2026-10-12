FEATURES = [
    "proto","sport","dport","stddev","N_IN_Conn_P_SrcIP","min",
    "state_number","mean","N_IN_Conn_P_DstIP","drate","srate","max",
]

NUMERIC_FEATURES = [
    "sport","dport","stddev","N_IN_Conn_P_SrcIP","min","state_number",
    "mean","N_IN_Conn_P_DstIP","drate","srate","max",
]

PROTO_MAP = {"arp":0,"icmp":1,"ipv6-icmp":2,"tcp":3,"udp":4}

LOG_FEATURES = [
    "sport","dport","stddev","N_IN_Conn_P_SrcIP","min","state_number",
    "mean","N_IN_Conn_P_DstIP","drate","srate","max",
    "NN1_distance","NN3_distance",
]

DIRECT_FEATURES = ["primary_p_attack","leaf_match_rate"]

PRIMARY_THRESHOLD = 0.50
ROUTER_PROB_THRESHOLD = 0.50
ROUTER_NN1_THRESHOLD = 1.145083189010620
VERIFIER_THRESHOLD = 0.40
FINAL_PRIMARY_TREES = 125
FINAL_VERIFIER_EPOCHS = 29
