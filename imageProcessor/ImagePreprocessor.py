import numpy as np
from physics import TrajectoryPhy


def zoom(imgs, to_size: tuple) -> (dict, tuple):
    """
    @params : imgs(dict), to_size(tuple)
    @return : zoomed images and scaled size
    Scaling of the images, read a dictionary containing 2D matrices and change its size.
    """
    zoomed_imgs = {}
    keys = list(imgs.keys())
    size_y = len(imgs[keys[0]])
    size_x = len(imgs[keys[0]][0])
    for histone in keys:
        center_pos = [int(size_x / 2), int(size_y / 2)]
        x_start = center_pos[0] - int(to_size[0] / 2)
        x_end = center_pos[0] + int(to_size[0] / 2)
        y_start = center_pos[1] - int(to_size[1] / 2)
        y_end = center_pos[1] + int(to_size[1] / 2)
        zoomed_imgs[histone] = imgs[histone][x_start:x_end, y_start:y_end].copy()
        del imgs[histone]  # delete original images to save the memory
    return zoomed_imgs, to_size


def preprocessing(histones: dict, img_scale: int, amp: int, interpolation=True, correction=False):
    """
    @params : histones(dict), img_scale(Integer), amp(Integer), interpolation(boolean), correction(boolean)
    @return : processed images(dict) and the size(tuple)
    Processing of a trajectory data(x, y positions) into images.
    Only for a 2-dimensional data.
    """
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
            if index == 0:
                trajec_channel = histones_channel[index]
            else:
                trajec_channel = histones_channel[index - 1]

            x_val = x_shift + int(trajectory[0] * (10 ** amp))
            y_val = y_shift + int(trajectory[1] * (10 ** amp))
            if not interpolation:

                # Forcing the scailing to reduce the memory
                if y_val >= img_size:
                    y_val = img_size - 1
                if y_val < 0:
                    y_val = 0
                if x_val >= img_size:
                    x_val = img_size - 1
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
                        inter_pos[0] = img_size - 1
                    if inter_pos[1] < 0:
                        inter_pos[1] = 0
                    if inter_pos[1] >= img_size:
                        inter_pos[1] = img_size - 1

                    if not channel:
                        img[inter_pos[1]][inter_pos[0]] = 1
                    else:
                        # add channels or not (val in float 0.0 ~ 1.0)
                        img[inter_pos[1]][inter_pos[0]][trajec_channel] = 1
                        if correction:
                            img[inter_pos[1]][inter_pos[0]][0] = 1
        imgs[histone] = img
    return imgs, (img_size, img_size)


def interpolate(current_pos: int, next_pos: int) -> list:
    """
    @params : current_pos(Integer), next_pos(Integer)
    @return : interpolated x, y positions
    Directional interpolation of two extrema.
    """
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


def make_channel(histones, immobile_cutoff=5, hybrid_cutoff=12, nChannel=3):
    """
    @params : histones(dict), cutoff values, number of channels.
    Calculate the speed of each trajectory and set the numbers(colors) with given cutoff values.
    if the speed is slower than immobile cutoff, set to 0
    if the speed is faster than immobile cutoff but slower than hybrid cutoff, set to 1
    if the speed is faster than hybrid cutoff, set to 2
    These cutoff values are to maximize the differences of each class (optimized for the h2b),
    in the case of other molecules, it needs to change by inspecting its diffusion coefficient.
    The values are stored in the H2B object.
    """
    histones_velocity = TrajectoryPhy.velocity(histones)
    immobile_cutoff = float(immobile_cutoff)
    hybrid_cutoff = float(hybrid_cutoff)

    for histone in histones:
        temp = []
        if len(histones_velocity[histone]) == 0:
            temp.append(0)
        else:
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
