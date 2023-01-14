def save_report(full_histones, pred, histone_names, path=''):
    write_file_name = path + '/report.csv'
    histones = {}
    for h in full_histones:
        histones |= h

    with open(write_file_name, 'w') as f:
        num_immobile = 0
        num_hybrid = 0
        num_mobile = 0
        input_string = ''
        for i, histone_name in enumerate(histone_names):
            trajectory = histones[histone_name].get_trajectory()
            first_x_pos = trajectory[0][0]
            first_y_pos = trajectory[0][1]
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

            input_string += f'{file_name : <25}\t{trajectory_num : <6}\t{class_num : <4}\t' \
                            f'{class_id : <9}\t{first_x_pos : <9}\t{first_y_pos : <9}'
            input_string += '\n'
        total_num = num_mobile + num_hybrid + num_immobile
        f.write(f'Immobile:{num_immobile/total_num}, Hybrid:{num_hybrid/total_num}, Mobile:{num_mobile/total_num}\n')
        f.write(input_string)

