import sys
import os
import ReadParam
import DataLoad
import ImagePreprocessor


def making_image(histones, zoomed_imgs, scaled_size, amp):
    for histone in histones:
        trajectory = histones[histone].get_trajectory()
        histone_first_pos = [int(trajectory[0][0] * (10 ** amp)),
                             int(trajectory[0][1] * (10 ** amp))]
        ImagePreprocessor.img_save(zoomed_imgs[histone], histones[histone], scaled_size,
                                   histone_first_pos=histone_first_pos, amp=amp, path='result/image')


def recursive_fileSearch(path, filename, h2b_ids, cls):
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
        histones, _ = DataLoad.read_file(f'{path}/{filename}', 0)
        temp = {}
        for hist in histones:
            for index, hid in enumerate(h2b_ids):
                if histones[hist].get_id() == hid:
                    temp[hist] = histones[hist]
                    temp[hist].set_predicted_label(cls[index])
        ImagePreprocessor.make_channel(temp, immobile_cutoff=0.3, hybrid_cutoff=10, nChannel=params['nChannel'])
        histones_imgs, img_size, time_scale = ImagePreprocessor.preprocessing(temp, img_scale=10, amp=params['amp'])
        zoomed_imgs, scaled_size = ImagePreprocessor.zoom(histones_imgs, size=img_size, to_size=(500, 500))
        making_image(temp, zoomed_imgs, scaled_size, params['amp'])
        return 0

    if len(dirs) > 0:
        for dir in dirs:
            px = recursive_fileSearch(f'{path}/{dir}', filename, h2b_ids, cls)
            if px == 0:
                return 0
    return 1


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
    report_path = './result/before/eval_all.csv'
    header, data = DataLoad.read_report(report_path)

    img_list = {}
    if img_option == 0:
        for dt in data:
            if dt['filename'] in img_list:
                img_list[dt['filename']][0].append(dt['h2b_id'])
                img_list[dt['filename']][1].append(dt['predicted_class_name'])
            else:
                img_list[dt['filename']] = [[dt['h2b_id']], [dt['predicted_class_name']]]
    elif img_option == 1:
        if 'labeled_class_id' not in header:
            raise Exception
        for dt in data:
            if dt['predicted_class_id'] != dt['labeled_class_id']:
                if dt['filename'] in img_list:
                    img_list[dt['filename']][0].append(dt['h2b_id'])
                    img_list[dt['filename']][1].append(dt['predicted_class_name'])
                else:
                    img_list[dt['filename']] = [[dt['h2b_id']], [dt['predicted_class_name']]]
    else:
        for dt in data:
            if dt['filename'] == fn and dt['h2b_id'] == h2bid:
                img_list[dt['filename']] = [[dt['h2b_id']], [dt['predicted_class_name']]]
            else:
                raise Exception

    # simple recursive search
    for filename in list(img_list.keys()):
        h2b_ids, cls = img_list[filename]
        recursive_fileSearch(data_path, filename, h2b_ids, cls)



