import os
import sys

root_path = os.getcwd().split('/scratch')[0]
os.chdir(root_path)
sys.path.append(root_path)

from analysis import DataAnalysis
from physics import DataSimulation, TrajectoryPhy
from imageProcessor import ImagePreprocessor, MakeImage
from fileIO import ReadParam
from fileIO import DataLoad, DataSave
import numpy as np
import pandas as pd
from xgboost import XGBClassifier, plot_tree
import xgboost as xgb
import matplotlib.pyplot as plt
import graphviz
from label import Labeling
from sklearn.model_selection import train_test_split



#DataAnalysis.bootstrapping_mean('./result/before/all.csv', repeat=10000)
#DataAnalysis.confusion_matrix(['./result/pred1_vs_pred2.csv'])
#histones = DataLoad.file_distrib(['./data/1_WT-H2BHalo_noIR/whole cells/20220217_h2b halo_before_irradiation_entire_Cell/20220217_h2b halo_cell6_no_ir003.rpt_tracked.trxyt'], cutoff=5)[0][0]
#ImagePreprocessor.make_gif(histones, '20220217_h2b halo_cell6_no_ir003.rpt_tracked.trxyt', '1846')
#DataAnalysis.cell_radius_map('./result/20220217_h2b halo_cel8_no_ir.rpt_tracked.trxyt.csv', [0])

"""
plt.figure()
immo = []
hyb = []
mob = []
rpnum=1000
for x in range(1, rpnum):
    ratios = bootstrapping_mean('./result/15min/eval_all.csv', repeat=x)
    immo.append(ratios['0'])
    hyb.append(ratios['1'])
    mob.append(ratios['2'])

plt.plot(range(1, rpnum), immo, label='immobile')
plt.plot(range(1, rpnum), hyb, label='hybrid')
plt.plot(range(1, rpnum), mob, label='mobile')
plt.legend()

plt.show()
"""


params = ReadParam.read('.')
reports = [#'./result/pred_wholecells_by_cutoff/cutoff5_model7_lab.csv', ## simulated data
           #'./result/pred_wholecells_by_cutoff/cutoff5_model13.csv',  ## retarined over model7_lab
           #'./result/pred_wholecells_by_cutoff/cutoff5_model14.csv',  ## retrained over model13
           #'./result/pred_wholecells_by_cutoff/cutoff5_model17.csv',  ## simulated data
           #'./result/pred_wholecells_by_cutoff/cutoff5_model18.csv',
           #'./result/pred_wholecells_by_cutoff/cutoff5_model19.csv',  ## 13 + 17
           #'./result/pred_wholecells_by_cutoff/cutoff5_model20.csv',  ## 7_lab + 19
           #'./result/pred_wholecells_by_cutoff/cutoff5_model21.csv',  ## 7_lab + 19
           #'./result/pred_wholecells_by_cutoff/cutoff5_model22.csv',  ## 13 + 17 (1st replica of model19 with same cnn)
           #'./result/pred_wholecells_by_cutoff/cutoff5_model23.csv',  ## 13 + 17 (2nd replica of model19 with same cnn)
           #'./result/pred_wholecells_by_cutoff/cutoff5_model24.csv',  ## 13 + 17 (3rd replica of model19 with same cnn)
           #'./result/pred_wholecells_by_cutoff/cutoff5_model25.csv',  ## manuel label , 1024
           #'./result/pred_wholecells_by_cutoff/cutoff5_model26.csv',  ## manuel label , 1024
           #'./result/pred_wholecells_by_cutoff/cutoff5_model27.csv',  ## manuel label , 300,300,300

           ##'./result/pred_wholecells_by_cutoff/cutoff5_model28.csv',  ## manuel label , 1040
           #'./result/pred_wholecells_by_cutoff/cutoff5_model29.csv',  ## manuel label , 1040
           #'./result/pred_wholecells_by_cutoff/cutoff5_model30.csv',  ## manuel label , 1040
           #'./result/pred_wholecells_by_cutoff/cutoff5_model31.csv',  ## manuel label , 1040
           #'./result/pred_wholecells_by_cutoff/cutoff5_model32.csv',  ## manuel label , 1040
           #'./result/pred_wholecells_by_cutoff/cutoff5_model33.csv',  ## manuel label , 1040
           #'./result/pred_wholecells_by_cutoff/cutoff5_model34.csv',  ## manuel label , 1040
           #'./result/pred_wholecells_by_cutoff/cutoff5_model35.csv',  ## manuel label , 1040
           #'./result/pred_wholecells_by_cutoff/cutoff5_model36.csv',  ## manuel label , 1040
           ##'./result/pred_wholecells_by_cutoff/cutoff5_model37.csv',  ## manuel label , 1040
           #'./scratch/6.mn_tracked.trxyt.csv'

           #'./result/pred_wholecells_by_cutoff/cutoff5_model38.csv',  ## manuel label , 1040
           #'./result/pred_wholecells_by_cutoff/cutoff5_model39.csv',  ## manuel label , 1040
           './result/pred_wholecells_by_cutoff/cutoff5_model40.csv',  ## manuel label , 1040
           ]


#[print(DataAnalysis.ratio_calcul(report)) for report in reports]
#DataAnalysis.confusion_matrix(reports)
#MakeImage.make_classified_cell_map(reports[0], path='./', interpolation=True)
#MakeImage.comparison_from_reports(reports, data_path='.', img_save_path='/shared/home/jpark/jwoo/HTC/40_mobile')
#MakeImage.make_image_from_single_report(reports[0], option=0, img_save_path='/shared/home/jpark/jwoo/HTC/40_immobile')

#histones = DataLoad.file_distrib(paths=params['data'], cutoff=2, group_size=params['group_size'], chunk=False)[0]  # 16GB RAM
#ImagePreprocessor.make_gif(histones, 'tracks.simple.filtered.sos', 240, correction=True)

#histones = DataSimulation.make_simulation_data(30)
#DataSave.save_simulated_data(histones, './data/SimulationData/4500_simulated_data.trxyt')
#histones = TrajectoryPhy.trjaectory_rotation(histones, 4)
#ImagePreprocessor.make_channel(histones, immobile_cutoff=5, hybrid_cutoff=12, nChannel=3)
#histones_imgs, img_size, time_scale = ImagePreprocessor.preprocessing(histones, img_scale=10, amp=2, correction=True)
#zoomed_imgs, scaled_size = ImagePreprocessor.zoom(histones_imgs, size=img_size, to_size=(500, 500))
#MakeImage.make_image(histones, zoomed_imgs, scaled_size, amp=2, img_save_path='./data/SimulationData/images')



#histones = DataSimulation.make_simulation_data(9000)
#DataSave.save_simulated_data(histones, './data/SimulationData/27000_resimulated_data.trxyt')



"""
histones = DataLoad.file_distrib(paths=params['data'], cutoff=params['cut_off'], chunk=False)[0]
histones = Labeling.label_from_report(histones, reports[0], equal=False)
dips = Labeling.displacements(histones)

for histone in histones:
    pred_class = histones[histone].get_manuel_label()
    if pred_class == 2:
        print(pred_class, np.mean(dips[histone]), np.std(dips[histone]))
"""



""""
histones = DataLoad.file_distrib(paths=['./data/SimulationData/27000_resimulated_data.trxyt'], cutoff=2,
                                 chunk=False)[0]
histones = TrajectoryPhy.trjaectory_rotation(histones, 4)

print(f'Channel processing...')
ImagePreprocessor.make_channel(histones, immobile_cutoff=3, hybrid_cutoff=8, nChannel=params['nChannel'])
trainable_histones_X = []
trainable_histones_Y = []
key_list = list(histones.keys())
np.random.shuffle(key_list)
for histone in key_list:
    traj = np.array(histones[histone].get_trajectory())
    time = np.array(histones[histone].get_time())
    speed = []
    prev_xy = traj[0]
    for index in range(len(traj)):
        grad = (traj[index] - prev_xy)
        prev_xy = traj[index]
        speed.append(np.sqrt(grad[0] ** 2 + grad[1] ** 2) / time[index])

    x_std = np.std(traj[:, 0])
    y_std = np.std(traj[:, 1])
    x_mean = np.mean(traj[:, 0])
    y_mean = np.mean(traj[:, 1])
    x_median = np.median(traj[:, 0])
    y_median = np.median(traj[:, 1])
    avg_speed = np.mean(speed)

    trainable_histones_X.append([x_std, y_std, x_mean, y_mean, x_median, y_median,
                                 avg_speed, histones[histone].get_max_radius(), histones[histone].get_time_duration()])
    trainable_histones_Y.append(histones[histone].get_manuel_label())
trainable_histones_X = np.array(trainable_histones_X)
trainable_histones_Y = np.array(trainable_histones_Y)

X_train, X_test, y_train, y_test = train_test_split(trainable_histones_X, trainable_histones_Y, test_size=.2)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# create model instance
bst = XGBClassifier(n_estimators=10, max_depth=2, learning_rate=1, objective='binary:logistic')
# fit model
bst.fit(X_train, y_train)
# make predictions

y_predict = bst.predict(X_test)

acc_train = np.sum([1 if x == 0 else 0 for x in (y_test - y_predict)]) / float(
    y_predict.shape[0])
print('Accuracy in training set = ', acc_train)


bst.save_model('xgb0001.model')
"""


"""
histones = DataLoad.file_distrib(paths=params['data'], cutoff=5, chunk=False)[0]

print(f'Channel processing...')
ImagePreprocessor.make_channel(histones, immobile_cutoff=3, hybrid_cutoff=8, nChannel=params['nChannel'])
x_test = []
key_list = list(histones.keys())
for histone in key_list:
    traj = np.array(histones[histone].get_trajectory())
    time = np.array(histones[histone].get_time())
    speed = []
    prev_xy = traj[0]
    for index in range(len(traj)):
        grad = (traj[index] - prev_xy)
        prev_xy = traj[index]
        speed.append(np.sqrt(grad[0] ** 2 + grad[1] ** 2) / time[index])

    x_std = np.std(traj[:, 0])
    y_std = np.std(traj[:, 1])
    x_mean = np.mean(traj[:, 0])
    y_mean = np.mean(traj[:, 1])
    x_median = np.median(traj[:, 0])
    y_median = np.median(traj[:, 1])
    avg_speed = np.mean(speed)

    x_test.append([x_std, y_std, x_mean, y_mean, x_median, y_median,
                                 avg_speed, histones[histone].get_max_radius(), histones[histone].get_time_duration()])
trainable_histones_X = np.array(x_test)

bst = xgb.Booster({'nthread': 4})
bst.load_model('xgb0001.model')  # load data

pred_result = bst.predict(xgb.DMatrix(x_test))
y_predict_proba = np.array([np.max(x) for x in pred_result])
y_predict = np.array([np.argmax(x) for x in pred_result])

for index, histone in enumerate(key_list):
    histones[histone].set_predicted_label(y_predict[index])
    histones[histone].set_predicted_proba(y_predict_proba[index])
#print(y_predict[0])
print(y_predict.shape)
DataSave.save_report(histones, path=params['save_dir'], all=params['all'])

"""
"""
acc = [0.9875801, 0.9878472, 0.98517627, 0.9807692, 0.984375, 0.98504275, 0.98170406, 0.98183763, 0.99038464, 0.9885149]
ratio = [[],  # immobile
         [],  # hybrid
         []]  # mobile
for report in reports:
    ratio_result = DataAnalysis.ratio_calcul(report)
    ratio[0].append(ratio_result[0])
    ratio[1].append(ratio_result[1])
    ratio[2].append(ratio_result[2])
fig, axs = plt.subplots(1, 1)
axs.boxplot(ratio)
plt.show()
"""
