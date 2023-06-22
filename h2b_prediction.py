import sys
from mainPipe import main_pipe
from fileIO import DataLoad, DataSave, ReadParam
from imageProcessor import ImagePreprocessor, MakeImage
from keras.models import load_model
from tensorflow import device
from postProcessing import h2bNetwork, dirichletMixtureModel, splitHistones


if __name__ == '__main__':
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = '.'
    params = ReadParam.read(config_path)

    with device('/cpu:0'):
        print(f'Main processing...')
        full_data = DataLoad.file_distrib(paths=params['data'], cutoff=params['cut_off'], group_size=params['group_size'])
        HTC_model = load_model(params['model_dir'], compile=False)
        HTC_model.compile()
        """
        sss = {}
        for xx in full_data:
            for h2b in xx:
                if xx[h2b].get_id() == '5148':
                    sss[h2b] = xx[h2b].copy()
        full_data = sss
        """
        main_pipe(HTC_model, full_data, params)

        print(f'Post processing...')
        hybrids, others = splitHistones.split_hybrid_from_otehrs(full_data)
        clusters = dirichletMixtureModel.dpgmm_clustering(hybrids)
        labeled_clusters = dirichletMixtureModel.cluster_prediction(HTC_model, hybrids, clusters, params)
        networks = h2bNetwork.transform_network(hybrids, labeled_clusters)
        clustered_hybrids = h2bNetwork.explore_net(hybrids, networks, params['cut_off'])
        main_pipe(HTC_model, [clustered_hybrids], params)

        ###
        """
        ImagePreprocessor.make_channel(clustered_hybrids, immobile_cutoff=5, hybrid_cutoff=12, nChannel=3)
        histones_imgs, img_size, time_scale = ImagePreprocessor.preprocessing(clustered_hybrids, img_scale=10, amp=2,
                                                                              correction=True)
        zoomed_imgs, scaled_size = ImagePreprocessor.zoom(histones_imgs, size=img_size, to_size=(300, 300))
        MakeImage.make_image(clustered_hybrids, zoomed_imgs, scaled_size, 2, '/Users/junwoopark/Downloads/hybrids', x=2)
        exit(1)
        """
        ###

        print(f'Report saving...')
        post_processed_histones = clustered_hybrids | others
        reports = DataSave.save_report([post_processed_histones], path=params['save_dir'], all=params['all'])
        DataSave.save_diffcoef([post_processed_histones], path=params['save_dir'], all=params['all'])
        MakeImage.make_classified_cell_map(reports, fullh2bs=[post_processed_histones], make=params['makeImage'])
        print(f'Number of histones:{len(post_processed_histones)} where the length of trajectory >= {params["cut_off"]}')
