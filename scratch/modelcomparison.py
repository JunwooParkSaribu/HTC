import os
import sys
import csv
from fileIO import ReadParam
from imageProcessor import MakeImage

root_path = os.getcwd().split('/scratch')[0]
os.chdir(root_path)
sys.path.append(root_path)

params = ReadParam.read('.')
reports = [
           #'./result/pred_wholecells_by_cutoff/cutoff5_model38.csv',  ## manuel label , 1040
           #'./result/pred_wholecells_by_cutoff/cutoff5_model39.csv',  ## manuel label , 1040
           #'./result/pred_wholecells_by_cutoff/cutoff5_model40.csv',  ## manuel label , 1040
           ]

MakeImage.comparison_from_reports(reports, data_path='.', img_save_path='/shared/home/jpark/jwoo/HTC/40_hybrid')