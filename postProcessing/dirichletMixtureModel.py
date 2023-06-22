import numpy as np
from sklearn.metrics import silhouette_score
from histone.H2B import H2B
from mainPipe import main_pipe
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture


def dpgmm_clustering(histones):
    histone_clusters = {}
    for h2b_index, h2b in enumerate(histones):
        pts = np.array(histones[h2b].get_trajectory())
        dpgmm = BayesianGaussianMixture(n_components=5, weight_concentration_prior=1e-7, init_params='kmeans',
                                        n_init=100, max_iter=1000, covariance_type='spherical')
        y = dpgmm.fit_predict(pts)
        if len(set(y)) != 1:
            silhouette = []
            min_range = min(10, len(pts))
            for k in range(2, min_range):
                gmm = GaussianMixture(n_components=k, covariance_type='spherical', init_params='kmeans', n_init=100,
                                      max_iter=100)
                y = gmm.fit_predict(pts)
                silhouette.append(silhouette_score(pts, y))
            optimal_k = np.argmax(silhouette)+2
            gmm = GaussianMixture(n_components=optimal_k, covariance_type='spherical', init_params='kmeans', n_init=100,
                                  max_iter=100)
            y = gmm.fit_predict(pts)
        histone_clusters[h2b] = y

        ###############
        """
        histone_clusters[h2b]
        sample = dict()
        sample[h2b] = histones[h2b]
        ImagePreprocessor.make_channel(sample, immobile_cutoff=5, hybrid_cutoff=12, nChannel=3)
        histones_imgs, img_size, time_scale = ImagePreprocessor.preprocessing(sample, img_scale=10, amp=2,
                                                                              correction=True)
        zoomed_imgs, scaled_size = ImagePreprocessor.zoom(histones_imgs, size=img_size, to_size=(300, 300))
        MakeImage.make_image(sample, zoomed_imgs, scaled_size, 2, '/Users/junwoopark/Downloads/hybrids', x=1)
        plt.figure()
        plt.title('dpm')
        plt.scatter(pts[:,0], pts[:,1], c=y, alpha=0.5)
        #plt.show()
        plt.savefig(f'/Users/junwoopark/Downloads/hybrids/{histones[h2b].get_file_name()}@{histones[h2b].get_id()}_cluster.png', dpi=600)
        plt.close()
        """
    return histone_clusters


def cluster_prediction(model, histones, histone_clusters, params):
    temp_histones = {}
    for h2b in histones:
        trajectory = histones[h2b].get_trajectory()
        times = histones[h2b].get_time()
        clusters = histone_clusters[h2b].copy()
        cluster_list = set()
        trajectories = {}
        tt = {}
        for point, cluster, time in zip(trajectory, clusters, times):
            if cluster not in cluster_list:
                cluster_list.add(cluster)
                trajectories[cluster] = [point]
                tt[cluster] = [time]
            else:
                trajectories[cluster].append(point)
                tt[cluster].append(time)
        for cluster in trajectories:
            temp_traj = trajectories[cluster]
            partial_time = tt[cluster]
            temp_histones[f'{h2b}_cluster{cluster}'] = H2B()
            temp_histones[f'{h2b}_cluster{cluster}'].set_trajectory(temp_traj)
            temp_histones[f'{h2b}_cluster{cluster}'].set_id(f'{h2b}_cluster{cluster}')
            temp_histones[f'{h2b}_cluster{cluster}'].set_time(partial_time)

    main_pipe(model, temp_histones, params)

    labeled_clusters = {}
    link = {}
    for temp_h2b in temp_histones:
        h2b_id = temp_h2b.split('_cluster')[0]
        cluster_label = temp_histones[temp_h2b].get_predicted_label()
        cluster_index = int(temp_h2b.split('_cluster')[-1])
        link[(h2b_id, cluster_index)] = cluster_label

    for h2b_id in histone_clusters:
        for clus in histone_clusters[h2b_id]:
            if h2b_id in labeled_clusters:
                labeled_clusters[h2b_id].append((clus, link[(h2b_id, clus)]))
            else:
                labeled_clusters[h2b_id] = [(clus, link[(h2b_id, clus)])]
    del temp_histones
    return labeled_clusters
