import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import os
import read_data

import img_preprocess

path = 'data/1_WT-H2BHalo_noIR/whole cells/20220217_h2b halo_before_irradiation_entire_Cell'
# file_name = 'data/1_WT-H2BHalo_noIR/whole cells/20220217_h2b halo_before_irradiation_entire_Cell/20220217_h2b halo_cel8_no_ir.rpt_tracked.trxyt'
# file_name = 'data/1_WT-H2BHalo_noIR/whole cells/20220301_H2B Halo_Before_Irradiation_entire_Cell/20220301_H2B Halo_Field1_no_IR.rpt_tracked copy.trxyt'
amplif = 8

"""
def get_available_devices():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos]
    """


def distance(histones):
    distances = {}
    for histone in histones:
        distances[histone] = []

    for histone in histones:
        dist = 0
        for i in range(len(histones[histone]) - 1):
            x_distance = histones[histone][i + 1][0] - histones[histone][i][0]
            y_distance = histones[histone][i + 1][1] - histones[histone][i][1]
            dist += np.sqrt(x_distance ** 2 + y_distance ** 2)
        t = histones[histone][-1][2] - histones[histone][0][2]
        distances[histone].append(dist)
        distances[histone].append(t)
    return distances


def displacement(histones):
    displacements = {}
    for histone in histones:
        displacements[histone] = []

    for histone in histones:
        x_displacement = histones[histone][-1][0] - histones[histone][0][0]
        y_displacement = histones[histone][-1][1] - histones[histone][0][1]
        t = histones[histone][-1][2] - histones[histone][0][2]
        displacements[histone].append(np.sqrt(x_displacement ** 2 + y_displacement ** 2))
        displacements[histone].append(t)
    return displacements


def velocity(histones):
    histone_velocity = {}
    for histone in histones:
        histone_velocity[histone] = []

    for histone in histones:
        for trajec_num in range(len(histones[histone]) - 1):
            x_distance = histones[histone][trajec_num + 1][0] - histones[histone][trajec_num][0]
            y_distance = histones[histone][trajec_num + 1][1] - histones[histone][trajec_num][1]
            t = histones[histone][trajec_num + 1][2] - histones[histone][trajec_num][2]
            histone_velocity[histone].append([np.sqrt(x_distance ** 2 + y_distance ** 2) / t])

    return histone_velocity


def accumulate(histone):
    acc_histone = []
    acc = 0
    for velocity in histone:
        acc += velocity[0]
        acc_histone.append([acc])
    return acc_histone


def mean_slope(histones):
    # histone_velocity = velocity(histones)

    histone_velocity = displacement(histones)
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


def preprocessing(trajecs, amplif):
    img_size = 25 * (10 ** amplif)
    # img = np.zeros((img_size, img_size))
    # for histone in histones:
    img = np.zeros((img_size, img_size))
    for trajec in trajecs:
        x_val = int(trajec[0] * (10 ** amplif))
        y_val = int(trajec[1] * (10 ** amplif))
        t = trajec[2]

        img[x_val][y_val] = 1

    plt.imshow(img, cmap='coolwarm', extent=(0, img_size, 0, img_size))
    plt.savefig('img/' + str(histone))
    # plt.imshow(img, cmap='coolwarm', extent=(0, img_size, 0, img_size))
    # plt.savefig('img/toal')


# histones, a,b,c,d,e = read_file(file_name)
histones = read_data.read_files(path)
# preprocessing(histones)
# distances = distance(histones)
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

histones_imgs, img_size, time_scale = img_preprocess.preprocessing3D(histones, img_size=8, amplif=2, channel=False)
print(histones_imgs)
