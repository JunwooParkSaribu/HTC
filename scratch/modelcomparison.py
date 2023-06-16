import os
import sys
import csv
from fileIO import ReadParam
from imageProcessor import MakeImage
from fileIO import DataLoad

root_path = os.getcwd().split('/scratch')[0]
os.chdir(root_path)
sys.path.append(root_path)

params = ReadParam.read('.')
reports = [
    '/Users/junwoopark/Downloads/abc/cell8_model41.trxyt.csv',
    '/Users/junwoopark/Downloads/abc/cell8_model42.trxyt.csv',
]

predictions = {}
for report in reports:
    header, lines = DataLoad.read_report(report)
    for line in lines:
        if line['h2b_id'] in predictions:
            predictions[line['h2b_id']].append(line['predicted_class_id'])
        else:
            predictions[line['h2b_id']] = [line['predicted_class_id']]

count = 0
for h2b in predictions:
    cls_list = predictions[h2b]
    if cls_list[0] != cls_list[1]:
        count += 1

print('simliarity:', (len(predictions)-count) / len(predictions), '  NB:', len(predictions))
MakeImage.comparison_from_reports(reports, data_path='.', img_save_path='/Users/junwoopark/Downloads/abc')