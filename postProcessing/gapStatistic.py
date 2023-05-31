import numpy as np
from sklearn.cluster import KMeans


def generate_references(histone, nb, nb_ref_point):
    refs = []
    trajectory = histone.get_trajectory()
    x_min = np.min(trajectory[:, 0]) - 1.5
    x_max = np.max(trajectory[:, 0]) + 1.5
    y_min = np.min(trajectory[:, 1]) - 1.5
    y_max = np.max(trajectory[:, 1]) + 1.5

    for _ in range(nb):
        x_pos = np.random.uniform(x_min, x_max, nb_ref_point).reshape(-1, 1)
        y_pos = np.random.uniform(y_min, y_max, nb_ref_point).reshape(-1, 1)
        pos = np.hstack((x_pos, y_pos))
        refs.append(pos)
    return np.array(refs)


def pooled_mean(cluster):
    pair_dist = 0
    nb = len(cluster)
    for point1 in cluster:
        for point2 in cluster:
            dist = np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)
            pair_dist += dist
    pair_dist = pair_dist / (2 * nb)
    return pair_dist


def cluster_weight_sum(pos, label, nb_cluster):
    tmp = [[] for _ in range(nb_cluster)]
    weight = 0
    for point, lb in zip(pos, label):
        tmp[lb].append(point)
    for cluster in tmp:
        weight += pooled_mean(cluster)
    return weight


def gap_stats(histones, max_cluster_nb=10, nb_reference=100, nb_ref_point=500):
    histone_clusters = {}
    optimal_k = []
    max_nbs = []
    for h2b in histones:
        max_nbs.append(len(histones[h2b].get_trajectory()))
    max_cluster_nb = min(np.min(max_nbs) - 1, max_cluster_nb)

    for h2b in histones:
        gaps = []
        sks = []
        refs = generate_references(histones[h2b], nb=nb_reference, nb_ref_point=nb_ref_point)
        real = histones[h2b].get_trajectory()

        for nb_cluster in range(1, max_cluster_nb+1):
            ref_weights = []

            for ref in refs:
                ref_kmeans = KMeans(nb_cluster, n_init='auto').fit(ref)
                #ref_weight_sum = cluster_weight_sum(ref, ref_kmeans.labels_, nb_cluster)
                ref_weight_sum = ref_kmeans.inertia_
                ref_weights.append(np.log(ref_weight_sum))

            ref_expectation = np.mean(ref_weights)
            ref_std_dev = np.std(ref_weights)
            sk = np.sqrt((1 + 1/nb_reference) * ref_std_dev)

            kmeans = KMeans(nb_cluster, n_init='auto').fit(real)
            #weight_sum = cluster_weight_sum(real, kmeans.labels_, nb_cluster)
            weight_sum = kmeans.inertia_
            if weight_sum == 0:
                print('max number of cluster is bigger than min number of total points')
                raise Exception
            gap = ref_expectation - np.log(weight_sum)
            gaps.append(gap)
            sks.append(sk)

        for k in range(1, max_cluster_nb):
            if gaps[k-1] >= (gaps[k] - sks[k]):
                optimal_k.append(k)
                break

    for h2b, k in zip(histones, optimal_k):
        traj = histones[h2b].get_trajectory()
        kmeans = KMeans(k, n_init='auto').fit(traj)
        label = kmeans.labels_
        histone_clusters[h2b] = label
    return histone_clusters
