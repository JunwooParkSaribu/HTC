from fileIO import DataLoad
import csv
filepath = '/Users/junwoopark/Desktop/Junwoo/Faculty/Master/M2/HTC/data/1_WT-H2BHalo_noIR/whole cells/20220217_h2b halo_before_irradiation_entire_Cell/20220217_h2b halo_cel8_no_ir.rpt_tracked.trxyt'
histones = DataLoad.read_file(filepath, cutoff=0)
write_file_name = f'/Users/junwoopark/Downloads/tractor.csv'

with open(write_file_name, 'w', newline='') as f:
    fieldnames = ['x', 'y', 't', 'n']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    for i, key in enumerate(histones):
        trajectory = histones[key].get_trajectory()
        h2b_id = histones[key].get_id()
        time = histones[key].get_time()

        for t, (x, y) in zip(time, trajectory):
            writer.writerow({'x':x,'y':y,'t':t,'n':h2b_id})
