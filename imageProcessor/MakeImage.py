import os
from fileIO import DataLoad, ReadParam
from imageProcessor import ImagePreprocessor


def make_image(histones, zoomed_imgs, scaled_size, amp, img_save_path='.'):
    for histone in histones:
        trajectory = histones[histone].get_trajectory()
        histone_first_pos = [int(trajectory[0][0] * (10 ** amp)),
                             int(trajectory[0][1] * (10 ** amp))]
        ImagePreprocessor.img_save(zoomed_imgs[histone], histones[histone], scaled_size,
                                   histone_first_pos=histone_first_pos, amp=amp, path=img_save_path)


def recursive_filesearch(path, filename, params, h2b_ids, cls, img_save_path, lbs: list | None, img_option=0):
    f_dirs = os.listdir(path)
    files = []
    dirs = []
    for f in f_dirs:
        if os.path.isdir(f'{path}/{f}'):
            dirs.append(f)
        if f.strip().split('.')[-1] == 'trxyt':
            files.append(f)

    if filename in files:
        histones = DataLoad.read_file(f'{path}/{filename}', cutoff=0)
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
        ImagePreprocessor.make_channel(temp, immobile_cutoff=3, hybrid_cutoff=8, nChannel=params['nChannel'])
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
    params = ReadParam.read(data_path)

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
    for filename in list(img_list.keys()):
        for dt in img_list[filename]:
            h2b_id, cls = dt[0], dt[1]

            # add conditions of classes (cls)
            cls_sum = sum([int(x) for x in cls])
            if cls_sum % len(cls) != 0:
                recursive_filesearch(data_path, filename, params, [h2b_id], cls, img_save_path, lbs=None)


def make_image_from_single_report(report: str, option=1, data_path='.', img_save_path='.',
                                  filename=None, h2b_id=None) -> None:
    header, data = DataLoad.read_report(report)
    params = ReadParam.read(data_path)
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
        if option == 1:
            h2b_ids, cls, lbs = img_list[filename]
        else:
            h2b_ids, cls = img_list[filename]
            lbs = []
        recursive_filesearch(data_path, filename, params, h2b_ids, cls, img_save_path, lbs, img_option=option)
