import sys
from mainPipe import main_pipe
from fileIO import DataLoad, DataSave, ReadParam
from imageProcessor import MakeImage
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
        main_pipe(HTC_model, full_data, params)

        if params['postProcessing']:
            print(f'Post processing...')
            hybrids, others = splitHistones.split_hybrid_from_others(full_data)
            clusters = dirichletMixtureModel.dpgmm_clustering(hybrids)
            labeled_clusters = dirichletMixtureModel.cluster_prediction(HTC_model, hybrids, clusters, params)
            networks = h2bNetwork.transform_network(hybrids, labeled_clusters)
            clustered_hybrids = h2bNetwork.explore_net(hybrids, networks, params['cut_off'])
            main_pipe(HTC_model, [clustered_hybrids], params)
            full_data = [clustered_hybrids | others]

        print(f'Report saving...')
        reports = DataSave.save_report(full_data, path=params['save_dir'], all=params['all'])
        DataSave.save_diffcoef(full_data, path=params['save_dir'], all=params['all'])
        MakeImage.make_classified_cell_map(reports, fullh2bs=full_data, make=params['makeImage'])
        print(f'Number of histones:{len(full_data[0])} where the length of trajectory >= {params["cut_off"]}')
