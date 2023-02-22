import DataAnalysis
import MakeImage
import DataSimulation
import ImagePreprocessor

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


#reports = ['./result/15min/old_eval_all.csv']
#MakeImage.comparison_from_reports(reports, img_save_path='./result')
#MakeImage.make_image_from_single_report('./result/1.csv', option=0, img_save_path='./result/image')



histones, _ = DataSimulation.make_simulation_data(150)
ImagePreprocessor.make_channel(histones, immobile_cutoff=0.3, hybrid_cutoff=10, nChannel=3)
histones_imgs, img_size, time_scale = ImagePreprocessor.preprocessing(histones, img_scale=10, amp=2, correction=True)
zoomed_imgs, scaled_size = ImagePreprocessor.zoom(histones_imgs, size=img_size, to_size=(500, 500))
MakeImage.make_image(histones, zoomed_imgs, scaled_size, amp=2, img_save_path='./data/SimulationData/images')
