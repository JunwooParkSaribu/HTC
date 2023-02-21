import sys
import os
import ReadParam
import DataLoad
import ImagePreprocessor


def making_image(histones, zoomed_imgs, scaled_size, amp, img_save_path='.'):
    for histone in histones:
        trajectory = histones[histone].get_trajectory()
        histone_first_pos = [int(trajectory[0][0] * (10 ** amp)),
                             int(trajectory[0][1] * (10 ** amp))]
        ImagePreprocessor.img_save(zoomed_imgs[histone], histones[histone], scaled_size,
                                   histone_first_pos=histone_first_pos, amp=amp, path=img_save_path)


def recursive_filesearch(path, filename, h2b_ids, cls, img_save_path, lbs=0, img_option=0):
    f_dirs = os.listdir(path)
    files = []
    dirs = []
    for f in f_dirs:
        if os.path.isdir(f'{path}/{f}'):
            dirs.append(f)
        if f.strip().split('.')[-1] == 'trxyt':
            files.append(f)

    if filename in files:
        params = ReadParam.read('.')
        histones = DataLoad.read_file(f'{path}/{filename}', 0)
        temp = {}
        for hist in histones:
            for index, hid in enumerate(h2b_ids):
                if histones[hist].get_id() == hid:
                    temp[hist] = histones[hist]
                    if type(cls) is list:
                        temp[hist].set_predicted_label(cls)
                    else:
                        temp[hist].set_predicted_label(cls[index])
                    if img_option == 1:
                        temp[hist].set_manuel_label(lbs[index])
        ImagePreprocessor.make_channel(temp, immobile_cutoff=0.3, hybrid_cutoff=10, nChannel=params['nChannel'])
        histones_imgs, img_size, time_scale = ImagePreprocessor.preprocessing(temp, img_scale=10, amp=params['amp'])
        zoomed_imgs, scaled_size = ImagePreprocessor.zoom(histones_imgs, size=img_size, to_size=(500, 500))
        making_image(temp, zoomed_imgs, scaled_size, params['amp'], img_save_path)
        return 0

    if len(dirs) > 0:
        for dir in dirs:
            px = recursive_filesearch(f'{path}/{dir}', filename, h2b_ids, cls, img_save_path, lbs, img_option)
            if px == 0:
                return 0
    return 1


def comparison_from_reports(reports: list, data_path='.', img_save_path='.'):
    data_list = [DataLoad.read_report(rp)[1] for rp in reports]

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

            recursive_filesearch(data_path, filename, [h2b_id], cls, img_save_path)


reports = ['./result/15min/old_eval_all.csv']
comparison_from_reports(reports, img_save_path='./result')

"""
if __name__ == '__main__':
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
        report_path = sys.argv[2]
        img_option = sys.argv[3]
        if img_option == '-all':
            img_option = 0
        elif img_option == '-diff':
            img_option = 1
        else:
            fn = sys.argv[3]
            h2bid = sys.argv[4]
            img_option = 2
    else:
        img_option = 1

    data_path = './data'
    report_path = './result/retrain_label(1).csv'
    img_save_path = './result/image'
    header, data = DataLoad.read_report(report_path)

    img_list = {}
    if img_option == 0:
        for dt in data:
            if dt['filename'] in img_list:
                img_list[dt['filename']][0].append(dt['h2b_id'])
                img_list[dt['filename']][1].append(dt['predicted_class_id'])
            else:
                img_list[dt['filename']] = [[dt['h2b_id']], [dt['predicted_class_id']]]
    elif img_option == 1:
        if 'labeled_class_id' not in header:
            raise Exception
        for dt in data:
            if dt['predicted_class_id'] != dt['labeled_class_id']:
                if dt['filename'] in img_list:
                    img_list[dt['filename']][0].append(dt['h2b_id'])
                    img_list[dt['filename']][1].append(dt['predicted_class_id'])
                    img_list[dt['filename']][2].append(dt['labeled_class_id'])
                else:
                    img_list[dt['filename']] = [[dt['h2b_id']], [dt['predicted_class_id']], [dt['labeled_class_id']]]
    else:
        for dt in data:
            if dt['filename'] == fn and dt['h2b_id'] == h2bid:
                img_list[dt['filename']] = [[dt['h2b_id']], [dt['predicted_class_id']]]
            else:
                raise Exception

    # simple recursive search
    for filename in list(img_list.keys()):
        if img_option==1:
            h2b_ids, cls, lbs = img_list[filename]
        else:
            h2b_ids, cls = img_list[filename]
            lbs = 0
        recursive_filesearch(data_path, filename, h2b_ids, cls, img_save_path, lbs, img_option=img_option)
"""


