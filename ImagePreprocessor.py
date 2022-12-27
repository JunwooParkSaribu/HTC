import numpy as np
from matplotlib import pyplot as plt
import TrajectoryPhy


def preprocessing(histones, histones_channel, img_size=None, amplif=2, channel=3, interpolation=True):
    if img_size is None:
        img_size = 5 * (10 ** amplif)
    else:
        img_size = img_size * (10 ** amplif)
    central_point = [int(img_size / 2), int(img_size / 2)]
    imgs = {}
    for histone in histones:
        current_xval = central_point[0]
        current_yval = central_point[1]
        if not channel:
            img = np.zeros((img_size, img_size))
        else:
            img = np.zeros((img_size, img_size, channel))
        x_shift = central_point[0] - int(histones[histone][0][0] * (10 ** amplif))
        y_shift = central_point[1] - int(histones[histone][0][1] * (10 ** amplif))
        for index, trajectory in enumerate(histones[histone]):
            if index < len(histones_channel[histone]):
                trajec_channel = histones_channel[histone][index]
            else:
                trajec_channel = histones_channel[histone][index-1]

            x_val = x_shift + int(trajectory[0] * (10 ** amplif))
            y_val = y_shift + int(trajectory[1] * (10 ** amplif))
            if not interpolation:

                # Forcing the scailing to reduce the memory
                if y_val >= img_size:
                    y_val = img_size-1
                if y_val < 0:
                    y_val = 0
                if x_val >= img_size:
                    x_val = img_size-1
                if x_val < 0:
                    x_val = 0

                if not channel:
                    img[y_val][x_val] = 1
                else:
                    img[y_val][x_val][trajec_channel] = 1
            else:
                interpolate_pos = interpolate([current_xval, current_yval], [x_val, y_val])
                current_xval = x_val
                current_yval = y_val
                for inter_pos in interpolate_pos:

                    # Forcing the scailing to reduce the memory
                    if inter_pos[0] < 0:
                        inter_pos[0] = 0
                    if inter_pos[0] >= img_size:
                        inter_pos[0] = img_size-1
                    if inter_pos[1] < 0:
                        inter_pos[1] = 0
                    if inter_pos[1] >= img_size:
                        inter_pos[1] = img_size-1

                    if not channel:
                        img[inter_pos[1]][inter_pos[0]] = 1
                    else:
                        img[inter_pos[1]][inter_pos[0]][trajec_channel] = 1
        imgs[histone] = img
    return imgs, img_size, None


def preprocessing3D(histones, img_size=None, amplif=3, channel=1, time_scale=500, interpolation=True):
    channel_vals = 0
    if img_size is None:
        img_size = 5 * (10 ** amplif)
    else:
        img_size = img_size * (10 ** amplif)
    central_point = [int(img_size / 2), int(img_size / 2)]
    imgs = {}
    for histone in histones:
        current_xval = central_point[0]
        current_yval = central_point[1]
        start_time =  int(histones[histone][0][2] * 100)
        current_time = int(histones[histone][0][2] * 100)
        if not channel:
            img = np.zeros((img_size, img_size, time_scale))
        else:
            img = np.zeros((img_size, img_size, time_scale, channel))
        x_shift = central_point[0] - int(histones[histone][0][0] * (10 ** amplif))
        y_shift = central_point[1] - int(histones[histone][0][1] * (10 ** amplif))

        for trajectory in histones[histone]:
            x_val = x_shift + int(trajectory[0] * (10 ** amplif))
            y_val = y_shift + int(trajectory[1] * (10 ** amplif))
            t_time = int(trajectory[2] * 100)
            if not interpolation:


                # Forcing the scailing to reduce the memory
                shifted_time = t_time - current_time
                scaled_y_val = img_size - y_val
                if shifted_time >= time_scale:
                    shifted_time = time_scale-1
                if scaled_y_val >= img_size:
                    scaled_y_val = img_size-1
                if scaled_y_val < 0:
                    scaled_y_val = 0
                if x_val >= img_size:
                    x_val = img_size-1
                if x_val < 0:
                    x_val = 0


                if not channel:
                    img[scaled_y_val][x_val][shifted_time] = 1
                else:
                    img[scaled_y_val][x_val][shifted_time][channel_vals] = 1
            else:
                interpolate_pos = interpolate3D([current_xval, current_yval, current_time-start_time],
                                                [x_val, y_val, t_time-start_time])
                current_xval = x_val
                current_yval = y_val
                current_time = t_time
                for inter_pos in interpolate_pos:


                    # Forcing the scailing to reduce the memory
                    if inter_pos[2] >= time_scale:
                        inter_pos[2] = time_scale-1
                    if inter_pos[0] < 0:
                        inter_pos[0] = 0
                    if inter_pos[0] >= img_size:
                        inter_pos[0] = img_size-1
                    if img_size - inter_pos[1] < 0:
                        inter_pos[1] = img_size
                    if img_size - inter_pos[1] >= img_size:
                        inter_pos[1] = 1


                    if not channel:
                        img[img_size - inter_pos[1]][inter_pos[0]][inter_pos[2]] = 1
                    else:
                        img[img_size - inter_pos[1]][inter_pos[0]][inter_pos[2]][channel_vals] = 1
        imgs[histone] = img
    return imgs, img_size, time_scale


def interpolate(current_pos, next_pos):  # 2D interpolation
    pos = []
    current_xval = current_pos[0]
    current_yval = current_pos[1]
    next_xval = next_pos[0]
    next_yval = next_pos[1]

    # if slope is 0 (vertical)
    if (next_xval - current_xval) == 0:
        if next_yval < current_yval:
            for yval in range(current_yval, next_yval, -1):
                pos.append([next_xval, yval])
        else:
            for yval in range(current_yval, next_yval):
                pos.append([next_xval, yval])
        return pos

    slope = (next_yval - current_yval) / (next_xval - current_xval)
    if next_xval < current_xval:
        xorder = -1
        if next_yval < current_yval:
            yorder = -1
        else:
            yorder = 1
    else:
        xorder = 1
        if next_yval < current_yval:
            yorder = -1
        else:
            yorder = 1
    xrange = range(current_xval, next_xval, xorder)
    yrange = range(current_yval, next_yval, yorder)

    if len(xrange) > len(yrange):
        for xval in range(current_xval, next_xval, xorder):
            yval = int(slope * (xval - current_xval)) + current_yval
            pos.append([xval, yval])
    else:
        for yval in range(current_yval, next_yval, yorder):
            xval = int((yval - current_yval) / slope) + current_xval
            pos.append([xval, yval])
    pos.append([next_xval, next_yval])
    return pos


def interpolate3D(current_pos, next_pos):  # 3D interpolation
    pos = []
    current_xval = current_pos[0]
    current_yval = current_pos[1]
    current_time = current_pos[2]
    next_xval = next_pos[0]
    next_yval = next_pos[1]
    next_time = next_pos[2]

    if (next_xval - current_xval) == 0:
        ## need to add time slope
        if next_yval < current_yval:
            for yval in range(current_yval, next_yval, -1):
                pos.append([next_xval, yval, next_time]) ## this is wrong
        else:
            for yval in range(current_yval, next_yval):
                pos.append([next_xval, yval, next_time]) ## this is wrong
        return pos

    xy_slope = (next_yval - current_yval) / (next_xval - current_xval)
    z_slope = np.around(np.linspace(current_time, next_time+1, num=abs(current_xval - next_xval))).astype(int)
    if next_xval < current_xval:
        for xval, time in zip(range(current_xval, next_xval, -1), z_slope):
            yval = int(xy_slope * (xval - current_xval)) + current_yval
            pos.append([xval, yval, time])
    else:
        for xval, time in zip(range(current_xval, next_xval), z_slope):
            yval = int(xy_slope * (xval - current_xval)) + current_yval
            pos.append([xval, yval, time])
    pos.append([next_xval, next_yval, next_time])
    return pos


def make_channel(histones, immobile_cutoff=0.5, hybrid_cutoff=25):
    num_channel = 3
    hist_velos = TrajectoryPhy.velocity(histones)
    hist_channel = {}
    for histone in histones:
        hist_channel[histone] = []

    for histone in hist_velos:
        for trajec in hist_velos[histone]:
            if trajec < immobile_cutoff:
                hist_channel[histone].append(0)
            elif trajec < hybrid_cutoff:
                hist_channel[histone].append(1)
            else:
                hist_channel[histone].append(2)
    del hist_velos
    return hist_channel, num_channel


def zoom(imgs, size=800, to_size=(300, 300)):
    zoomed_imgs = {}
    for histone in imgs:
        if type(size) is not int:
            center_pos = [int(size[0]/2), int(size[1]/2)]
        else:
            center_pos = [int(size/2), int(size/2)]

        x_start = center_pos[0] - int(to_size[0] / 2)
        x_end = center_pos[0] + int(to_size[0] / 2)
        y_start = center_pos[1] - int(to_size[1] / 2)
        y_end = center_pos[1] + int(to_size[1] / 2)
        zoomed_imgs[histone] = imgs[histone][x_start:x_end, y_start:y_end].copy()
    return zoomed_imgs, to_size[0]


def img_save(img, img_name, img_size, label=None, pred=None, histone_first_pos=None, amplif=2, path=''):
    ps = ''
    if len(path) > 0:
        path = path + '/'
    if label is not None:
        if label==0:
            label = 'immobile'
        if label==1:
            label = 'hybrid'
        if label==2:
            label = 'mobile'
    if pred is not None:
        if pred==0:
            pred = 'immobile'
        if pred==1:
            pred = 'hybrid'
        if pred==2:
            pred = 'mobile'
    if label is not None:
        ps += 'label = ' + label
    if pred is not None:
        ps += '\nprediction = ' + pred

    if histone_first_pos is None:
        plt.imshow(img, cmap='coolwarm', origin='lower')
    else:
        plt.imshow(img, cmap='coolwarm', origin='lower',
                   extent=[int((histone_first_pos[0] - (img_size/2))/(10**amplif)),
                           int((histone_first_pos[0] + int(img_size/2))/(10**amplif)),
                           int((histone_first_pos[1] - int(img_size/2))/(10**amplif)),
                           int((histone_first_pos[1] + int(img_size/2))/(10**amplif))])
    plt.title(ps)
    plt.savefig(path + str(img_name))

