

def transform_network(histones, clusters):
    networks = []
    for h2b in histones:
        network = []
        traj = histones[h2b].get_trajectory()
        for (x,y), cluster in zip(traj, clusters[h2b]):
            network.append([x,y, cluster])
        networks.append(network)
    return networks


def explore_net(histones, networks):

    for h2b, network in zip(histones, networks):
        connections = anomalie_detection(network)
        print(connections)


def anomalie_detection(network):
    crossing = []
    prev_cluster = network[0][2]
    combs = {}
    for index, node in enumerate(network):
        next_cluster = node[2]
        if prev_cluster != next_cluster:
            if tuple(sorted([prev_cluster, next_cluster])) in combs:
                combs[tuple(sorted([prev_cluster, next_cluster]))] += 1
            else:
                combs[tuple(sorted([prev_cluster, next_cluster]))] = 1
            connect = (index, prev_cluster, next_cluster)
            crossing.append(connect)
        prev_cluster = next_cluster

    anomalie_connections = []
    for connec in crossing:
        index, prev_cluster, next_cluster = connec[0], connec[1], connec[2]
        if combs[tuple(sorted([prev_cluster, next_cluster]))] > 1:
            anomalie_connections.append([index, prev_cluster, next_cluster, 1])
        else:
            anomalie_connections.append([index, prev_cluster, next_cluster, 0])
    return anomalie_connections


