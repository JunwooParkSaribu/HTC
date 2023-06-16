from sklearn.mixture import BayesianGaussianMixture
import numpy as np
import ProgressBar
from fileIO import DataLoad, DataSave, ReadParam
from imageProcessor import ImagePreprocessor, ImgGenerator, MakeImage
from keras.models import load_model
from tensorflow import device
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def dpgmm_clustering(histones):
    histone_clusters = {}

    for h2b_index, h2b in enumerate(histones):
        pts = np.array(histones[h2b].get_trajectory())
        BGM = BayesianGaussianMixture(n_components=5, weight_concentration_prior=1e-7, init_params='kmeans',
                                      n_init=100, max_iter=1000, covariance_type='spherical')
        y = BGM.fit_predict(pts)
        histone_clusters[h2b] = y

        """
        centers = BGM.means_[:len(set(y))]
        print(centers)
        kmeans = KMeans(n_clusters=len(set(y)), init=centers).fit(pts)
        kmean_y = kmeans.labels_

        sample = dict()
        sample[h2b] = histones[h2b]
        ImagePreprocessor.make_channel(sample, immobile_cutoff=5, hybrid_cutoff=12, nChannel=3)
        histones_imgs, img_size, time_scale = ImagePreprocessor.preprocessing(sample, img_scale=10, amp=2,
                                                                              correction=True)
        zoomed_imgs, scaled_size = ImagePreprocessor.zoom(histones_imgs, size=img_size, to_size=(300, 300))
        MakeImage.make_image(sample, zoomed_imgs, scaled_size, 2, 'show')

        print(set(y))
        plt.figure()
        plt.title('dpm')
        plt.scatter(pts[:,0], pts[:,1], c=y, alpha=0.5)

        plt.figure()
        plt.title('kmeans')
        plt.scatter(pts[:,0], pts[:,1], c=kmean_y, alpha=0.5)
        plt.show()
        """
    return histone_clusters

