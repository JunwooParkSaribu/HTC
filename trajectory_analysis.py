import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import os
import read_data
import img_preprocess
import make_label
import trajectory_phy

path = 'data/1_WT-H2BHalo_noIR/whole cells/sample'


"""
def get_available_devices():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos]
    """


def mean_slope(histones):
    # histone_velocity = velocity(histones)

    histone_velocity = img_preprocess.displacement(histones)
    for histone in histone_velocity:
        histone_velocity[histone] = [histone_velocity[histone][0] / histone_velocity[histone][1],
                                     histone_velocity[histone][1]]

    # histone_acc_vel = {}
    # for histone in histones:
    #    histone_acc_vel[histone] = accumulate(histone_velocity[histone])

    histone_slope = {}
    for histone in histones:
        histone_slope[histone] = []

        """    
        for histone in histones:
        mean_acc = (histone_acc_vel[histone][-1][0] - histone_acc_vel[histone][0][0]) \
                   / (histones[histone][-1][2] - histones[histone][0][2])
        histone_slope[histone].append(mean_acc)
        histone_slope[histone].append(histones[histone][-1][2] - histones[histone][0][2])
        """

    return histone_velocity


def acceleration(histones):
    histone_acc = {}
    for histone in histones:
        if len(histones[histone]) < 2:
            continue
        histone_acc[histone] = []

    for histone in histones:
        if len(histones[histone]) < 2:
            continue
        for trajec_num in range(len(histones[histone]) - 1):
            acc = histones[histone][trajec_num + 1][0] - histones[histone][trajec_num][0]
            histone_acc[histone].append([acc])

    return histone_acc


    # plt.imshow(img, cmap='coolwarm', extent=(0, img_size, 0, img_size))
    # plt.savefig('img/toal')



# preprocessing(histones)
# displacements = displacement(histones)
# histone_velocity = velocity(histones)
# slope = mean_slope(histones)
# histone_acc = acceleration(histone_distance)


"""
fig, axs = plt.subplots(1, 1, figsize=(15,15))
hist = []
for histone in displacements:
    hist.append(displacements[histone][0])
plt.hist(hist, bins=50)
plt.savefig('img/displacements_histo.png')
"""

"""
fig, axs = plt.subplots(1, 1, figsize=(15,15))
hist = []
for histone in distances:
    hist.append(distances[histone][0])
plt.hist(hist, bins=50)
plt.savefig('img/distances_all_histo.png')
"""

"""
fig, axs = plt.subplots(1, 1, figsize=(15,15))
for histone in displacements:
    axs.scatter(displacements[histone][0], displacements[histone][1])
    #axs.set_xlim(0, 2.5)
    #axs.set_ylim(0, 2.5)
plt.savefig('img/displacements.png')
"""

"""
fig, axs = plt.subplots(1, 1, figsize=(15,15))
for histone in distances:
    axs.scatter(distances[histone][0], distances[histone][1])
    #axs.set_xlim(0, 2.5)
    #axs.set_ylim(0, 2.5)
plt.savefig('img/distances_all.png')
"""

"""
fig, axs = plt.subplots(1, 1, figsize=(15,15))
for histone in slope:
    axs.scatter(slope[histone][0], slope[histone][1])
    axs.set_xlim(0, 1)
    axs.set_ylim(0, 1)
    #axs[ax_num].legend()
plt.savefig('img/scatter.png')
"""

"""
fig, axs = plt.subplots(2, 1, figsize=(15,15))
i = 0
ax_num = 0
for histone in histone_velocity:
    axs[ax_num].plot(accumulate(histone_velocity[histone]), alpha=0.5, label=str(histone))
    axs[ax_num].legend()
    i += 1
    if i%200 == 0:
        ax_num += 1
    if i==10:
        break

plt.savefig('img/velocity_2.png')
"""
"""
fig, axs = plt.subplots(10, 1, figsize=(15,15))
i = 0
ax_num = 0
for histone in histone_acc:
    axs[ax_num].plot(histone_acc[histone], alpha=0.5, label='A')
    i += 1
    if i%50 == 0:
        ax_num += 1
    if i==500:
        break

plt.savefig('img/acc.png')
"""

"""
fig, axs = plt.subplots(1, 1, figsize=(15,15))
for histone in distances:
    axs.scatter(distances[histone][0] / displacements[histone][0], displacements[histone][1])
    #axs.set_xlim(0, 4)
    #axs.set_ylim(0, 2.5)
plt.savefig('img/mobile_by_fraction.png')
"""

"""
for histone in distances:
    if distances[histone][0] / displacements[histone][0] > 1000:
        print(histone, distances[histone][0] , displacements[histone][0])
        preprocessing(histones[histone], 1)
"""

data_path = 'data/1_WT-H2BHalo_noIR/whole cells/sample'
amplif = 2
print(f'Loading the data...')
histones = read_data.read_files(path=data_path)
histones_label = make_label.make_label(histones, radius=0.2, density=0.5)
print(f'Image processing...')
histones_channel, nChannel = img_preprocess.make_channel(histones, immobile_cutoff=0.5, hybrid_cutoff=30)
histones_imgs, img_size, time_scale = \
    img_preprocess.preprocessing(histones, histones_channel, img_size=8, amplif=amplif, channel=nChannel)
print(f'Making imgs...')
i = 0
for histone in histones:
    histone_first_pos = [int(histones[histone][0][0] * (10 ** amplif)),
                        int(histones[histone][0][1] * (10 ** amplif))]
    channels = histones_channel[histone]
    if 0 in channels or 2 in channels:
        print(f'i={i}')
        i += 1
        zoomed_img, to_size = img_preprocess.zoom(histones_imgs[histone], size=img_size, to_size=(300, 300))
        img_preprocess.img_save(zoomed_img, histone, to_size, label=histones_label[histone],
                                histone_first_pos=histone_first_pos, amplif=amplif, path='.')
