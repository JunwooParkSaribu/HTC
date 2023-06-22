def transform_network(histones, clusters):
    networks = []
    for h2b in histones:
        network = []
        traj = histones[h2b].get_trajectory()
        time = histones[h2b].get_time()
        for (x, y), (cluster, cluster_label), t in zip(traj, clusters[h2b], time):
            network.append([x, y, cluster, t, cluster_label])
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
        new_times = dict()
        for cluster_num in merged_group:
            new_networks[cluster_num] = []
            new_times[cluster_num] = []

        for index, node in enumerate(network):
            x, y, cluster, t, _ = node
            try:
                target_cluster = arrow_reverse(merged_group, target=cluster)
                new_networks[target_cluster].append([x, y])
                new_times[target_cluster].append(t)
            except Exception as e:
                print('@@@@@ ERR on network, check anomaly detection @@@@')
                print(merged_group, x, y, cluster, t)
                print(histones[h2b].get_id())
                print(connections)
                print(e)
                print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
                exit(1)

        for cluster_num, time in zip(new_networks, new_times):
            if len(new_networks[cluster_num]) >= cutoff:
                new_id = f'{histones[h2b].get_id()}_cluster{cluster_num}'
                clustered_histones[f'{h2b}_cluster{cluster_num}'] = histones[h2b].copy()
                clustered_histones[f'{h2b}_cluster{cluster_num}'].set_id(new_id)
                clustered_histones[f'{h2b}_cluster{cluster_num}'].set_trajectory(new_networks[cluster_num])
                clustered_histones[f'{h2b}_cluster{cluster_num}'].set_predicted_label(None)  # reset
                clustered_histones[f'{h2b}_cluster{cluster_num}'].set_time(new_times[time])
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
    prev_label = network[0][4]
    combs = {}
    for index, node in enumerate(network):
        next_cluster = node[2]
        next_label = node[4]
        if prev_cluster != next_cluster:
            if tuple(sorted([prev_cluster, next_cluster])) in combs:
                combs[tuple(sorted([prev_cluster, next_cluster]))] += 1
            else:
                combs[tuple(sorted([prev_cluster, next_cluster]))] = 1
            connect = (index, prev_cluster, next_cluster, (prev_label, next_label))
            crossing.append(connect)
        prev_cluster = next_cluster
        prev_label = next_label

    anomalie_connections = {}
    if len(crossing) == 1:
        anomalie_connections[crossing[0][0]] = [crossing[0][1], crossing[0][2], 0]
    else:
        prev_index, prev_cluster, next_cluster, labels = crossing[0]
        index = crossing[1][0]
        if (combs[tuple(sorted([prev_cluster, next_cluster]))] > 1 and \
            labels[0] == 0 and labels[1] == 0) \
                or (combs[tuple(sorted([prev_cluster, next_cluster]))] > 1 and \
                    abs(index - prev_index) == 1):
            anomalie_connections[prev_index] = [prev_cluster, next_cluster, 1]
        else:
            anomalie_connections[prev_index] = [prev_cluster, next_cluster, 0]

        for connec in crossing[1:]:
            index, prev_cluster, next_cluster, labels = connec
            ## here change.
            if (combs[tuple(sorted([prev_cluster, next_cluster]))] > 1 and\
                    labels[0] == 0 and labels[1] == 0)\
                    or (combs[tuple(sorted([prev_cluster, next_cluster]))] > 1 and\
                        abs(index-prev_index) == 1):
                anomalie_connections[index] = [prev_cluster, next_cluster, 1]
            else:
                anomalie_connections[index] = [prev_cluster, next_cluster, 0]
            prev_index = index
    return anomalie_connections
