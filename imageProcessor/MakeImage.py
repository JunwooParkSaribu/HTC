import os
import numpy as np
import imageio
import matplotlib.pyplot as plt
from imageProcessor import ImagePreprocessor
from physics import TrajectoryPhy
from fileIO import DataLoad


def make_image(histones, zoomed_imgs, scaled_size, amp, img_save_path='.', x=1):
    for histone in histones:
        trajectory = histones[histone].get_trajectory()
        histone_first_pos = [int(trajectory[0][0] * (10 ** amp)),
                             int(trajectory[0][1] * (10 ** amp))]
        img_save(zoomed_imgs[histone], histones[histone], scaled_size,
                 histone_first_pos=histone_first_pos, amp=amp, path=img_save_path, x=x)


def recursive_filesearch(path, filename, params, h2b_ids, cls, img_save_path, lbs: list | None, img_option=0):
    f_dirs = os.listdir(path)
    files = []
    dirs = []
    for f in f_dirs:
        if os.path.isdir(f'{path}/{f}'):
            dirs.append(f)
        if f.strip().split('.')[-1] == 'trxyt' or f.strip().split('.')[-1] == 'sos':
            files.append(f)

    if filename in files:
        if 'trxyt' in filename:
            histones = DataLoad.read_file(f'{path}/{filename}', cutoff=0, filetype='trxyt')
        else:
            histones = DataLoad.read_file(f'{path}/{filename}', cutoff=0, filetype='sos')
        temp = {}
        for hist in histones:
            for index, hid in enumerate(h2b_ids):
                if histones[hist].get_id() == hid:
                    temp[hist] = histones[hist]
                    if lbs is None:
                        temp[hist].set_predicted_label(cls)
                    else:
                        temp[hist].set_predicted_label(cls[index])
                    if img_option == 1 and len(lbs) != 0:
                        temp[hist].set_manuel_label(lbs[index])
        ImagePreprocessor.make_channel(temp, immobile_cutoff=params['immobile_cutoff'],
                                       hybrid_cutoff=params['hybrid_cutoff'], nChannel=params['nChannel'])
        histones_imgs, img_size, time_scale = ImagePreprocessor.preprocessing(temp, img_scale=10, amp=params['amp']
                                                                              , correction=True)
        zoomed_imgs, scaled_size = ImagePreprocessor.zoom(histones_imgs, size=img_size, to_size=(500, 500))
        make_image(temp, zoomed_imgs, scaled_size, params['amp'], img_save_path)
        return 0

    if len(dirs) > 0:
        for dir in dirs:
            px = recursive_filesearch(f'{path}/{dir}', filename, params, h2b_ids, cls, img_save_path, lbs, img_option)
            if px == 0:
                return 0
    return 1


def comparison_from_reports(reports: list, data_path='.', img_save_path='.') -> None:
    data_list = [DataLoad.read_report(rp)[1] for rp in reports]
    params = DataLoad.read_params(data_path)

    print("Reading the reports...")
    img_list = {}
    for dt in data_list[0]:
        flag = 0
        pred_class_list = []
        for data in data_list:
            for comp_dt in data:
                if dt['filename'] == comp_dt['filename'] and dt['h2b_id'] == comp_dt['h2b_id']:
                    flag += 1
                    pred_class_list.append(comp_dt['predicted_class_id'])
        if flag == len(data_list):
            if dt['filename'] not in img_list:
                img_list[dt['filename']] = [[dt['h2b_id'], pred_class_list]]
            else:
                img_list[dt['filename']].append([dt['h2b_id'], pred_class_list])

    print('Generating the images...')
    # simple recursive search
    total = 0
    for filename in list(img_list.keys()):
        for i, dt in enumerate(img_list[filename]):
            h2b_id, cls = dt[0], dt[1]

            # add conditions of classes (cls)
            cls_sum = sum([int(x) for x in cls])
            if cls_sum % len(cls) != 0:
            #if int(cls[0]) == 2 and int(cls[1]) == 1:
            #if (int(cls[0]) == 0 and int(cls[1]) == 1) or (int(cls[0]) == 1 and int(cls[1]) == 0):
            #if i % 100 == 0:
            #if int(cls[0]) != int(cls[1]):
            #if int(cls[0]) == 1:
                recursive_filesearch(data_path, filename, params, [h2b_id], cls, img_save_path, lbs=None)


def make_image_from_single_report(report: str, option=1, data_path='.', img_save_path='.',
                                  filename=None, h2b_id=None) -> None:
    header, data = DataLoad.read_report(report)
    params = DataLoad.read_params(data_path)
    img_list = {}

    match option:
        case 0:
            for dt in data:
                if dt['filename'] in img_list:
                    img_list[dt['filename']][0].append(dt['h2b_id'])
                    img_list[dt['filename']][1].append(dt['predicted_class_id'])
                else:
                    img_list[dt['filename']] = [[dt['h2b_id']], [dt['predicted_class_id']]]

        case 1:
            if 'labeled_class_id' not in header:
                print("There is no label in the report")
                raise Exception
            for dt in data:
                if dt['predicted_class_id'] != dt['labeled_class_id']:
                    if dt['filename'] in img_list:
                        img_list[dt['filename']][0].append(dt['h2b_id'])
                        img_list[dt['filename']][1].append(dt['predicted_class_id'])
                        img_list[dt['filename']][2].append(dt['labeled_class_id'])
                    else:
                        img_list[dt['filename']] = [[dt['h2b_id']], [dt['predicted_class_id']], [dt['labeled_class_id']]]
        case 2:
            if filename is None or h2b_id is None:
                print("Empty filename or h2b id")
                raise Exception
            for dt in data:
                if dt['filename'] == filename and dt['h2b_id'] == h2b_id:
                    img_list[dt['filename']] = [[dt['h2b_id']], [dt['predicted_class_id']]]
                else:
                    print("There is no matching filename or h2b id")
                    raise Exception

    # simple recursive search
    for filename in list(img_list.keys()):
        print(filename)
        if option == 1:
            h2b_ids, cls, lbs = img_list[filename]
        else:
            h2b_ids, cls = img_list[filename]
            lbs = []
        recursive_filesearch(data_path, filename, params, h2b_ids, cls, img_save_path, lbs, img_option=option)


def make_classified_cell_map(reports, fullh2bs, interpolation=True, make='true'):
    if make:
        search_file_names = set()
        datas = [DataLoad.read_report(report)[1] for report in reports]
        for data in datas:
            for dt in data:
                search_file_names.add(dt['filename'])

        histones = {}
        try:
            if type(fullh2bs) is list:
                for hs in fullh2bs:
                    histones |= hs
            elif type(fullh2bs) is dict:
                histones = fullh2bs
            else:
                raise Exception
        except Exception as e:
            print(e)
            print('histone container type err')

        for data, report in zip(datas, reports):
            new_histones = {}
            for dt in data:
                selected_histone = histones[f'{dt["filename"]}@{dt["h2b_id"]}']
                selected_histone.set_predicted_label(dt['predicted_class_id'])
                new_histones[f'{dt["filename"]}@{dt["h2b_id"]}'] = selected_histone.copy()
            classified_cellmap(histones=new_histones, report_name=report,
                               interpolation=interpolation)


def img_save(img, h2b, img_size, histone_first_pos=None, amp=2, path='.', x=1):
    ps = ''
    label = h2b.get_manuel_label()
    pred = h2b.get_predicted_label()
    proba = h2b.get_predicted_proba()

    if type(img_size) is tuple:
        img_size = img_size[0]

    if type(pred) is not list:
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
        ps += f'\nDuration:{str(round(h2b.get_time_duration(), 5))}sec'
    else:
        for index, prediction in enumerate(pred):
            ps += f'Model{str(index + 1)}:{prediction}\n'
        ps += f'Duration:{str(round(h2b.get_time_duration(), 5))}sec'

    plt.figure()
    if histone_first_pos is None:
        plt.imshow(img, cmap='coolwarm', origin='lower', label='a')
    else:
        plt.imshow(img, cmap='coolwarm', origin='lower',
                   extent=[(histone_first_pos[0] - img_size / 2) / (10 ** amp),
                           (histone_first_pos[0] + img_size / 2) / (10 ** amp),
                           (histone_first_pos[1] - img_size / 2) / (10 ** amp),
                           (histone_first_pos[1] + img_size / 2) / (10 ** amp)], label='a')
    plt.legend(title=ps)
    if path != 'show':
        plt.savefig(f'{path}/{h2b.get_file_name()}@{h2b.get_id()}_{x}.png', dpi=600)


def make_gif(full_histones, filename, id, immobile_cutoff=5,
             hybrid_cutoff=12, nChannel=3, img_scale=5, amp=2, correction=False):
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
        ImagePreprocessor.make_channel(histones, immobile_cutoff=immobile_cutoff, hybrid_cutoff=hybrid_cutoff, nChannel=nChannel)
        histones_velocity = TrajectoryPhy.velocity(histones)
        if img_scale is None:
            img_size = 5 * (10 ** amp)
        else:
            img_size = img_scale * (10 ** amp)
        central_point = [int(img_size / 2), int(img_size / 2)]
        histones_channel = histones[key].get_channel()
        channel = histones[key].get_channel_size()
        histone_velocity = histones_velocity[key]
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
            if index == 0:
                trajec_channel = histones_channel[index]
            else:
                trajec_channel = histones_channel[index - 1]
            velocity = histone_velocity[index - 1] if index > 0 else 0

            x_val = x_shift + int(trajectory[0] * (10 ** amp))
            y_val = y_shift + int(trajectory[1] * (10 ** amp))

            interpolate_pos = ImagePreprocessor.interpolate([current_xval, current_yval], [x_val, y_val])
            current_xval = x_val
            current_yval = y_val
            for mod, inter_pos in enumerate(interpolate_pos):
                frame = int(velocity)/2
                if frame == 0:
                    gif.append(img.copy())
                else:
                    if mod % (int(velocity)/2) == 0:
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
                    if correction:
                        img[img_size - inter_pos[1]][inter_pos[0]][0] = 1

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


def classified_cellmap(histones, report_name: str, amp=2, interpolation=True):
    channel = 3  # nb of classes
    min_x = 0
    max_x = 0
    min_y = 0
    max_y = 0

    for histone in histones:
        trajectory = histones[histone].get_trajectory()
        for traj in trajectory:
            x, y = traj[0], traj[1]
            max_x = max(max_x, x)
            max_y = max(max_y, y)

    max_x += 1
    max_y += 1
    max_x = max_x * (10 ** amp)
    max_y = max_y * (10 ** amp)

    img_x_size = int(max_x - min_x)
    img_y_size = int(max_y - min_y)
    img = np.zeros((img_y_size, img_x_size, channel))
    for histone in histones:
        histone_trajectory = histones[histone].get_trajectory()
        current_xval, current_yval = int(histone_trajectory[0][0] * (10 ** amp)), int(histone_trajectory[0][1] * (10 ** amp))
        for index, trajectory in enumerate(histone_trajectory):
            x_val = int(trajectory[0] * (10 ** amp))
            y_val = int(trajectory[1] * (10 ** amp))
            label = int(histones[histone].get_predicted_label())

            if not interpolation:
                img[y_val][x_val][label] = 1
            else:
                interpolate_pos = ImagePreprocessor.interpolate([current_xval, current_yval], [x_val, y_val])
                current_xval = x_val
                current_yval = y_val
                for inter_pos in interpolate_pos:
                    # add channels or not (val in float 0.0 ~ 1.0)
                    img[inter_pos[1]][inter_pos[0]][label] = 1
    plt.imshow(img, cmap='coolwarm', origin='lower', label='cellmap',
               extent=[min_x/(10**amp), max_x/(10**amp), min_y/(10**amp), max_y/(10**amp)])
    plt.savefig(f'{report_name}_cellmap.png', dpi=1200)