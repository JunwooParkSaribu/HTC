import numpy as np
from sklearn.cluster import KMeans, SpectralClustering


def generate_references(histone, nb, nb_ref_point):
    refs = []
    trajectory = histone.get_trajectory()
    x_min = np.min(trajectory[:, 0])-.01
    x_max = np.max(trajectory[:, 0])+.01
    y_min = np.min(trajectory[:, 1])-.01
    y_max = np.max(trajectory[:, 1])+.01

    xy_min = [x_min, y_min]
    xy_max = [x_max, y_max]
    for _ in range(nb):
        pos = np.random.uniform(low=xy_min, high=xy_max, size=(nb_ref_point, 2))
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


def gap_stats(histones, max_cluster_nb=10, nb_reference=100, nb_ref_point=2500):
    histone_clusters = {}
    optimal_k = []
    max_nbs = []
    for h2b in histones:
        max_nbs.append(len(histones[h2b].get_trajectory()))
    #max_cluster_nb = min(np.min(max_nbs) - 1, max_cluster_nb)

    for (h2b_index, h2b), max_nb in zip(enumerate(histones), max_nbs):
        print(f'{h2b_index}/{len(histones)} clustering...')
        gaps = []
        sks = []
        refs = generate_references(histones[h2b], nb=nb_reference, nb_ref_point=nb_ref_point)
        real = histones[h2b].get_trajectory()
        max_nbCluster = min(max_cluster_nb, max_nb)
        print(max_nbCluster, max_nb)
        for nb_cluster in range(1, max_nbCluster+1):
            ref_weights = []

            for ref in refs:
                ref_kmeans = KMeans(nb_cluster, n_init='auto').fit(ref)
                ref_weight_sum = ref_kmeans.inertia_
                #ref_spectral = SpectralClustering(nb_cluster, assign_labels='discretize').fit(ref)
                #ref_weight_sum = cluster_weight_sum(ref, ref_spectral.labels_, nb_cluster)
                ref_weights.append(np.log(ref_weight_sum))

            ref_expectation = np.mean(ref_weights)
            ref_std_dev = np.std(ref_weights)
            sk = ref_std_dev * np.sqrt((1 + 1/nb_reference))

            kmeans = KMeans(nb_cluster, n_init='auto').fit(real)
            print(kmeans.cluster_centers_)
            weight_sum = kmeans.inertia_
            print(weight_sum)
            #spectral = SpectralClustering(nb_cluster, assign_labels='discretize').fit(real)
            #weight_sum = cluster_weight_sum(real, spectral.labels_, nb_cluster)

            if weight_sum == 0:
                gap = np.inf
            else:
                gap = ref_expectation - np.log(weight_sum)
            gaps.append(gap)
            sks.append(sk)
        print('gaps',gaps)
        print('sks',sks)

        for k in range(1, max_nbCluster):
            if gaps[k-1] >= (gaps[k] - sks[k]):
                optimal_k.append(k)
                break

            if k == max_nbCluster-1:
                print(h2b_index, h2b, k)
                print(gaps)
                print(sks)
                optimal_k.append(k)

    for h2b, k in zip(histones, optimal_k):
        traj = histones[h2b].get_trajectory()
        kmeans = KMeans(k, n_init='auto').fit(traj)
        label = kmeans.labels_
        print(kmeans.cluster_centers_, 'optimal k=',k)
        #spectral = SpectralClustering(k, assign_labels='discretize').fit(traj)
        #label = spectral.labels_
        histone_clusters[h2b] = label

    for h2b in histones:
        print(histones[h2b].get_trajectory()[0])
    return histone_clusters
