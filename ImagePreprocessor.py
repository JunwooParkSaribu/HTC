import imageio
import numpy as np
from matplotlib import pyplot as plt
import TrajectoryPhy


def preprocessing(histones, img_scale=None, amp=2, interpolation=True, correction=False):
    if img_scale is None:
        img_size = 5 * (10 ** amp)
    else:
        img_size = img_scale * (10 ** amp)
    central_point = [int(img_size / 2), int(img_size / 2)]
    imgs = {}
    for histone in histones:
        histones_channel = histones[histone].get_channel()
        channel = histones[histone].get_channel_size()
        current_xval = central_point[0]
        current_yval = central_point[1]
        if not channel:
            img = np.zeros((img_size, img_size))
        else:
            img = np.zeros((img_size, img_size, channel))
        histone_trajectory = histones[histone].get_trajectory()
        x_shift = central_point[0] - int(histone_trajectory[0][0] * (10 ** amp))
        y_shift = central_point[1] - int(histone_trajectory[0][1] * (10 ** amp))
        for index, trajectory in enumerate(histone_trajectory):
            if index < len(histones_channel):
                trajec_channel = histones_channel[index]
            else:
                trajec_channel = histones_channel[index-1]

            x_val = x_shift + int(trajectory[0] * (10 ** amp))
            y_val = y_shift + int(trajectory[1] * (10 ** amp))
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
                        # add channels or not (val in float 0.0 ~ 1.0)
                        img[inter_pos[1]][inter_pos[0]][trajec_channel] = 1
                        if correction:
                            img[inter_pos[1]][inter_pos[0]][0] = 1
        imgs[histone] = img
    return imgs, (img_size, img_size), None


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


def make_channel(histones, immobile_cutoff=3, hybrid_cutoff=8, nChannel=3):
    histones_velocity = TrajectoryPhy.velocity(histones)

    hist_channel = {}
    for histone in histones:
        hist_channel[histone] = []

    for histone in histones:
        temp = []
        for velocity in histones_velocity[histone]:
            if velocity < immobile_cutoff:
                temp.append(0)
            elif velocity < hybrid_cutoff:
                temp.append(1)
            else:
                temp.append(2)
        histones[histone].set_channel(temp)
        histones[histone].set_channel_size(nChannel)
    del histones_velocity


def zoom(imgs, size=1000, to_size=(300, 300)):
    zoomed_imgs = {}
    keys = list(imgs.keys())
    for histone in keys:
        if type(size) is not int:
            center_pos = [int(size[0]/2), int(size[1]/2)]
        else:
            center_pos = [int(size/2), int(size/2)]
        x_start = center_pos[0] - int(to_size[0] / 2)
        x_end = center_pos[0] + int(to_size[0] / 2)
        y_start = center_pos[1] - int(to_size[1] / 2)
        y_end = center_pos[1] + int(to_size[1] / 2)
        zoomed_imgs[histone] = imgs[histone][x_start:x_end, y_start:y_end].copy()
        del imgs[histone]
    return zoomed_imgs, to_size


def img_save(img, h2b, img_size, histone_first_pos=None, amp=2, path='.'):
    ps = ''
    label = h2b.get_manuel_label()
    pred = h2b.get_predicted_label()
    proba = h2b.get_predicted_proba()

    if type(img_size) is tuple:
        img_size = img_size[0]

    if type(pred) is not list:
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
        if proba is not None:
            ps += '\nprobability = ' + str(proba)
        ps += f'\nDuration:{str(round(h2b.get_time_duration(), 5))}sec'
    else:
        for index, prediction in enumerate(pred):
            ps += f'Model{str(index+1)}:{prediction}\n'
        ps += f'Duration:{str(round(h2b.get_time_duration(), 5))}sec'

    if histone_first_pos is None:
        plt.imshow(img, cmap='coolwarm', origin='lower', label='a')
    else:
        plt.imshow(img, cmap='coolwarm', origin='lower',
                   extent=[(histone_first_pos[0] - img_size/2)/(10**amp),
                           (histone_first_pos[0] + img_size/2)/(10**amp),
                           (histone_first_pos[1] - img_size/2)/(10**amp),
                           (histone_first_pos[1] + img_size/2)/(10**amp)], label='a')
    plt.legend(title=ps)
    plt.savefig(f'{path}/{h2b.get_file_name()}@{h2b.get_id()}.png')


def make_gif(full_histones, filename, id, immobile_cutoff=3,
             hybrid_cutoff=8, nChannel=3, img_scale=5, amp=2):
    try:
        histones = {}
        if type(full_histones) is list:
            for h in full_histones:
                histones |= h
        elif type(full_histones) is dict:
            histones = full_histones
        else:
            raise Exception

        gif = []
        key = f'{filename}@{id}'
        make_channel(histones, immobile_cutoff=immobile_cutoff, hybrid_cutoff=hybrid_cutoff, nChannel=nChannel)
        if img_scale is None:
            img_size = 5 * (10 ** amp)
        else:
            img_size = img_scale * (10 ** amp)
        central_point = [int(img_size / 2), int(img_size / 2)]
        histones_channel = histones[key].get_channel()
        channel = histones[key].get_channel_size()
        current_xval = central_point[0]
        current_yval = central_point[1]
        if not channel:
            img = np.zeros((img_size, img_size))
        else:
            img = np.zeros((img_size, img_size, channel))
        histone_trajectory = histones[key].get_trajectory()
        x_shift = central_point[0] - int(histone_trajectory[0][0] * (10 ** amp))
        y_shift = central_point[1] - int(histone_trajectory[0][1] * (10 ** amp))

        for index, trajectory in enumerate(histone_trajectory):
            if index < len(histones_channel):
                trajec_channel = histones_channel[index]
            else:
                trajec_channel = histones_channel[index - 1]

            x_val = x_shift + int(trajectory[0] * (10 ** amp))
            y_val = y_shift + int(trajectory[1] * (10 ** amp))

            interpolate_pos = interpolate([current_xval, current_yval], [x_val, y_val])
            current_xval = x_val
            current_yval = y_val

            for mod, inter_pos in enumerate(interpolate_pos):
                if trajec_channel == 0:
                    if mod%2 == 0:
                        gif.append(img.copy())
                elif trajec_channel == 1:
                    if mod%3 == 0:
                        gif.append(img.copy())
                else:
                    if mod%5 == 0:
                        gif.append(img.copy())

                # Forcing the scailing to reduce the memory
                if inter_pos[0] < 0:
                    inter_pos[0] = 0
                if inter_pos[0] >= img_size:
                    inter_pos[0] = img_size - 1
                if inter_pos[1] < 0:
                    inter_pos[1] = 0
                if inter_pos[1] >= img_size:
                    inter_pos[1] = img_size - 1

                if not channel:
                    img[img_size - inter_pos[1]][inter_pos[0]] = 1
                else:
                    img[img_size - inter_pos[1]][inter_pos[0]][trajec_channel] = 1

        ps = ''
        label = histones[key].get_manuel_label()
        pred = histones[key].get_predicted_label()
        proba = histones[key].get_predicted_proba()

        if label is not None:
            if label == 0:
                label = 'immobile'
            if label == 1:
                label = 'hybrid'
            if label == 2:
                label = 'mobile'
        if pred is not None:
            if pred == 0:
                pred = 'immobile'
            if pred == 1:
                pred = 'hybrid'
            if pred == 2:
                pred = 'mobile'
        if label is not None:
            ps += 'label = ' + label
        if pred is not None:
            ps += '\nprediction = ' + pred
        if proba is not None:
            ps += '\nprobability = ' + str(proba)

        with imageio.get_writer(f'./gif/{filename}@{id}.gif', mode='I') as writer:
            for i in range(len(gif)):
                writer.append_data(np.array(gif[i]))

    except Exception as e:
        print(e)
        print(f'There is no matching filename and id in the data')
