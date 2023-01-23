import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
import ReadParam
import ImagePreprocessor
import Labeling
import DataLoad
import numpy as np
from sklearn import tree


data_path = 'data/1_WT-H2BHalo_noIR/whole cells/20220301_H2B Halo_Before_Irradiation_entire_Cell'
report_path = 'result/report_all.csv'


if __name__ == '__main__':
    if len(sys.argv) > 1:
        cur_path = sys.argv[1]
        data_path = cur_path + '/' + data_path
    else:
        cur_path = '.'

    """    
    nChannel = 3
    print(f'\nLoading the data...')
    histones = DataLoad.read_files(path=data_path, cutoff=10, chunk=False)
    histones_label = Labeling.make_label(histones, radius=0.45, density=0.4)
    print(f'Image processing...')
    ImagePreprocessor.make_channel(histones, immobile_cutoff=0.3, hybrid_cutoff=10, nChannel=nChannel)
    histones_imgs, img_size, time_scale = ImagePreprocessor.preprocessing(histones, img_scale=10, amp=2)
    zoomed_imgs, scaled_size = ImagePreprocessor.zoom(histones_imgs, size=img_size, to_size=(500, 500))
    print(f'Number of training items:{len(zoomed_imgs)}, processed shape:{scaled_size}, time scale:{time_scale}\n')

    clf = tree.DecisionTreeClassifier()

    X = []
    Y = []
    for histone in histones:
        var = zoomed_imgs[histone]
        var = var.reshape(500 * 500 * 3)
        X.append(var)
        Y.append(histones_label[histone])
    X = np.array(X)

    clf = clf.fit(X, Y)
    tree.plot_tree(clf)
    """

    header, data = DataLoad.read_report(report_path)
    params = ReadParam.read(cur_path)

    print(f'Loading the data...')
    full_histones = DataLoad.read_files(path=data_path, cutoff=params['cut_off'], group_size=params['group_size'])
    ImagePreprocessor.make_gif(full_histones, '20220301_H2B Halo_Field2_no_ir', '387')