import numpy as np
from matplotlib import pyplot as plt
import matplotlib.colors


def preprocessing(histones, img_size=None, amplif=3, channel=1, interpolation=True):
    channel_vals = 0
    if img_size == None:
        img_size = 5 * (10 ** amplif)
    else:
        img_size = img_size * (10 ** amplif)
    central_point = [int(img_size / 2), int(img_size / 2)]
    current_xval = central_point[0]
    current_yval = central_point[1]
    imgs = {}
    for histone in histones:
        if not channel:
            img = np.zeros((img_size, img_size))
        else:
            img = np.zeros((img_size, img_size, channel))
        x_shift = central_point[0] - int(histones[histone][0][0] * (10 ** amplif))
        y_shift = central_point[1] - int(histones[histone][0][1] * (10 ** amplif))
        for trajectory in histones[histone]:
            x_val = x_shift + int(trajectory[0] * (10 ** amplif))
            y_val = y_shift + int(trajectory[1] * (10 ** amplif))
            if not interpolation:
                if not channel:
                    img[img_size - y_val][x_val] = 1
                else:
                    img[img_size - y_val][x_val][channel_vals] = 1
            else:
                interpolate_pos = interpolate([current_xval, current_yval], [x_val, y_val])
                current_xval = x_val
                current_yval = y_val
                for inter_pos in interpolate_pos:
                    if not channel:
                        img[img_size - inter_pos[1]][inter_pos[0]] = 1
                    else:
                        img[img_size - inter_pos[1]][inter_pos[0]][channel_vals] = 1
        imgs[histone] = img
    return imgs, img_size


def interpolate(current_pos, next_pos):  # 2D interpolation
    current_xval = current_pos[0]
    current_yval = current_pos[1]
    next_xval = next_pos[0]
    next_yval = next_pos[1]
    if (next_xval - current_xval) == 0:
        return [current_pos]
    slope = (next_yval - current_yval) / (next_xval - current_xval)

    pos = []
    if next_xval < current_xval:
        for xval in range(current_xval, next_xval, -1):
            yval = int(slope * (xval - current_xval)) + current_yval
            pos.append([xval, yval])
    else:
        for xval in range(current_xval, next_xval):
            yval = int(slope * (xval - current_xval)) + current_yval
            pos.append([xval, yval])
    pos.append([next_xval, next_yval])
    return pos


def zoom(img, size=800, to_size=(300, 300)):
    if type(size) is not int:
        center_pos = [int(size[0]/2), int(size[1]/2)]
    else:
        center_pos = [int(size/2), int(size/2)]

    x_start = center_pos[0] - int(to_size[0] / 2)
    x_end = center_pos[0] + int(to_size[0] / 2)
    y_start = center_pos[1] - int(to_size[1] / 2)
    y_end = center_pos[1] + int(to_size[1] / 2)
    return img[x_start:x_end, y_start:y_end], to_size[0]


def img_save(img, img_name, img_size, label=None, pred=None):
    ps = ''
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
    plt.imshow(img, cmap=matplotlib.colors.ListedColormap(['black', 'white']),
               extent=(0, img_size, 0, img_size))
    plt.title(ps)
    plt.savefig('img/training_imgs/' + str(img_name))


"""
hist = {}
hist[203_1] = [[21.148162, 11.869134], [20.514558,	11.477142],
            [19.937856, 11.609182],[19.641905,	11.49018],[20.626125,	11.929091],
            [20.514558,	11.477142],[19.937856,	11.609182],[19.641905,	11.49018]]
kk, imz_size = preprocessing(hist, img_size=5, amplif=2, channel=False)
kkk = zoom(kk[2031], size=imz_size)
img_save(kkk[0], '2031', kkk[1], 1, 2)
"""
