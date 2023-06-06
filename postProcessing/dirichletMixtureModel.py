from sklearn.mixture import BayesianGaussianMixture
import numpy as np
import ProgressBar
from fileIO import DataLoad, DataSave, ReadParam
from imageProcessor import ImagePreprocessor, ImgGenerator, MakeImage
from keras.models import load_model
from tensorflow import device

def dpgmm_clustering(histones):
    histone_clusters = {}

    for h2b_index, h2b in enumerate(histones):
        print(f'{h2b_index}/{len(histones)} clustering...')
        pts = histones[h2b].get_trajectory()
        BGM = BayesianGaussianMixture(n_components=5, init_params='k-means++', n_init=10)
        y = BGM.fit_predict(pts)
        """
        sample = dict()
        sample[h2b] = histones[h2b]
        ImagePreprocessor.make_channel(sample, immobile_cutoff=5, hybrid_cutoff=12, nChannel=3)
        histones_imgs, img_size, time_scale = ImagePreprocessor.preprocessing(sample, img_scale=10, amp=2
                                                                              , correction=True)
        zoomed_imgs, scaled_size = ImagePreprocessor.zoom(histones_imgs, size=img_size, to_size=(100, 100))
        MakeImage.make_image(sample, zoomed_imgs, scaled_size, 2, f'/Users/junwoopark/Downloads')
        """
        histone_clusters[h2b] = y
    return histone_clusters

