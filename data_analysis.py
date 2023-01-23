import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
import ReadParam
import ImagePreprocessor
import Labeling
import DataLoad
import numpy as np
from sklearn import tree


data_path = 'data/1_WT-H2BHalo_noIR/whole cells/20220301_H2B Halo_Before_Irradiation_entire_Cell'
report_path = 'result/report_all.csv'


if __name__ == '__main__':
    if len(sys.argv) > 1:
        cur_path = sys.argv[1]
        data_path = cur_path + '/' + data_path
    else:
        cur_path = '.'

    header, data = DataLoad.read_report(report_path)
    params = ReadParam.read(cur_path)

    print(f'Loading the data...')
    histones = DataLoad.read_files(path=data_path, cutoff=params['cut_off'], group_size=params['group_size'])
    #histones = DataLoad.read_file('./data/scratch/1/20220217_h2b halo_cel9_no_ir.rpt_tracked.trxyt', cutoff=params['cut_off'])

    ImagePreprocessor.make_gif(histones, '20220301_H2B Halo_Field28_no_ir', '902')
