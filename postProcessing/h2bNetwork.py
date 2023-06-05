def transform_network(histones, clusters):
    networks = []
    for h2b in histones:
        network = []
        traj = histones[h2b].get_trajectory()
        for (x,y), cluster in zip(traj, clusters[h2b]):
            network.append([x, y, cluster])
        networks.append(network)
    return networks


def explore_net(histones, networks, cutoff):
    clustered_histones = {}

    for h2b, network in zip(histones, networks):
        connections = anomalie_detection(network)

        merged_group = dict()
        uniques = set()
        merged_group[network[0][2]] = set()
        merged_group[network[0][2]].add(network[0][2])
        uniques.add(network[0][2])

        for conn in connections:
            prev_cluster, next_cluster, anomaly = connections[conn]
            if anomaly == 1:
                if next_cluster not in uniques:
                    merged_group[next_cluster] = set()
                    merged_group[next_cluster].add(next_cluster)
                    uniques.add(next_cluster)
            else:
                for cluster_num in merged_group:
                    main_cluster = cluster_num
                    if prev_cluster not in uniques:
                        merged_group[main_cluster].add(prev_cluster)
                        uniques.add(prev_cluster)
                    if next_cluster not in uniques:
                        merged_group[main_cluster].add(next_cluster)
                        uniques.add(next_cluster)

        new_networks = dict()
        for cluster_num in merged_group:
            new_networks[cluster_num] = []

        for index, node in enumerate(network):
            x, y, cluster = node
            target_cluster = arrow_reverse(merged_group, target=cluster)
            new_networks[target_cluster].append([x, y])

        for cluster_num in new_networks:
            if len(new_networks[cluster_num]) >= cutoff:
                new_id = f'{histones[h2b].get_id()}_cluster{cluster_num}'
                clustered_histones[f'{h2b}_cluster{cluster_num}'] = histones[h2b].copy()
                clustered_histones[f'{h2b}_cluster{cluster_num}'].set_id(new_id)
                clustered_histones[f'{h2b}_cluster{cluster_num}'].set_trajectory(new_networks[cluster_num])
                clustered_histones[f'{h2b}_cluster{cluster_num}'].set_predicted_label(None)  # reset
    del histones
    return clustered_histones


def arrow_reverse(dict_net, target):
    new_dict = {}
    for n in dict_net:
        ac = dict_net[n]
        for a in ac:
            new_dict[a] = n
    return new_dict[target]


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

    anomalie_connections = {}
    for connec in crossing:
        index, prev_cluster, next_cluster = connec[0], connec[1], connec[2]
        if combs[tuple(sorted([prev_cluster, next_cluster]))] > 1:
            #anomalie_connections.append([index, prev_cluster, next_cluster, 1])
            anomalie_connections[index] = [prev_cluster, next_cluster, 1]
        else:
            #anomalie_connections.append([index, prev_cluster, next_cluster, 0])
            anomalie_connections[index] = [prev_cluster, next_cluster, 0]
    return anomalie_connections
