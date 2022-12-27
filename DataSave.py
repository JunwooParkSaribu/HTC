def save_report(pred, histone_names, path=''):
    write_file_name = path + '/report.csv'
    with open(write_file_name, 'w') as f:
        num_immobile = 0
        num_hybrid = 0
        num_mobile = 0
        input_string = ''
        for i, histone_name in enumerate(histone_names):
            histone_name = histone_name.strip().split('@')
            file_name = histone_name[0]
            trajectory_num = histone_name[1]
            class_num = pred[i]
            class_id = 'unidentified'
            if class_num == 0:
                class_id = 'Immobile'
                num_immobile += 1
            if class_num == 1:
                class_id = 'Hybrid'
                num_hybrid += 1
            if class_num == 2:
                class_id = 'Mobile'
                num_mobile += 1

            input_string += f'{file_name : <25}\t{trajectory_num : <6}\t{class_num : <4}\t{class_id : <9}'
            input_string += '\n'
        f.write(f'Immobile:{num_immobile}, Hybrid:{num_hybrid}, Mobile:{num_mobile}\n')
        f.write(input_string)

