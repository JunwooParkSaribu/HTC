import DataAnalysis
import MakeImage

#bootstrapping_mean('./result/before/all.csv', repeat=10000)
#confusion_matrix(['./result/pred1_vs_pred2.csv'])
#histones = DataLoad.file_distrib(['./data/1_WT-H2BHalo_noIR/whole cells/20220217_h2b halo_before_irradiation_entire_Cell/20220217_h2b halo_cell6_no_ir003.rpt_tracked.trxyt'], cutoff=5)[0][0]
#ImagePreprocessor.make_gif(histones, '20220217_h2b halo_cell6_no_ir003.rpt_tracked.trxyt', '1846')
#cell_radius_map('./result/20220217_h2b halo_cel8_no_ir.rpt_tracked.trxyt.csv', [0])

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



reports = ['./result/15min/old_eval_all.csv']
#comparison_from_reports(reports, img_save_path='./result')
MakeImage.make_image_from_single_report(reports[0], option=1, img_save_path='./result')


"""
histones, _ = make_simulation_data(150)
ImagePreprocessor.make_channel(histones, immobile_cutoff=0.3, hybrid_cutoff=10, nChannel=3)
histones_imgs, img_size, time_scale = ImagePreprocessor.preprocessing(histones, img_scale=10, amp=2, correction=True)
zoomed_imgs, scaled_size = ImagePreprocessor.zoom(histones_imgs, size=img_size, to_size=(500, 500))
for histone in histones:
    trajectory = histones[histone].get_trajectory()
    histone_first_pos = [int(trajectory[0][0] * (10 ** 2)),
                         int(trajectory[0][1] * (10 ** 2))]
    ImagePreprocessor.img_save(zoomed_imgs[histone], histones[histone], scaled_size,
                               histone_first_pos=histone_first_pos, amp=2, path='./data/SimulationData/images')
"""
