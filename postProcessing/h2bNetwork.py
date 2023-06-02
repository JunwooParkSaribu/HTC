

def transform_network(histones, clusters):
    networks = []
    for h2b in histones:
        network = []
        traj = histones[h2b].get_trajectory()
        for (x,y), cluster in zip(traj, clusters):
            network.append([x,y, cluster])
        networks.append(network)
    return networks


def explore_net(histones, networks):

    for h2b, network in zip(histones, networks):
        c
        for node in network:


def anomalie_detection(network):
    crossing = []
    prev_cluster = network[0][2]
    for index, node in enumerate(network):
        next_cluster = node[2]

        if prev_cluster != next_cluster:
            connect = (index, prev_cluster, next_cluster)
            crossing.append(connect)


