import csv


def save_report(full_histones, path=''):
    write_file_name = path + '/report.csv'
    histones = {}
    for h in full_histones:
        histones |= h
    histone_names = list(histones.keys())

    with open(write_file_name, 'w', newline='') as f:
        fieldnames = ['filename', 'id', 'class_id', 'class_name', 'probability',
                      'maximum_radius', 'first_x_position', 'first_y_position']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, histone_name in enumerate(histone_names):
            trajectory = histones[histone_name].get_trajectory()
            first_x_pos = trajectory[0][0]
            first_y_pos = trajectory[0][1]
            file_name = histones[histone_name].get_file_name()
            h2b_id = histones[histone_name].get_id()
            class_id = histones[histone_name].get_predicted_label()
            class_name = 'unidentified'
            if class_id == 0:
                class_name = 'Immobile'
            if class_id == 1:
                class_name = 'Hybrid'
            if class_id == 2:
                class_name = 'Mobile'
            max_r = histones[histone_name].get_max_radius()
            proba = histones[histone_name].get_predicted_proba()

            writer.writerow({'filename':file_name, 'id':h2b_id, 'class_id':class_id, 'class_name':class_name,
                             'probability':proba, 'maximum_radius':max_r, 'first_x_position':first_x_pos,
                             'first_y_position':first_y_pos})
            """
            input_string += f'{file_name : <25}\t{trajectory_num : <6}\t{class_num : <4}\t' \
                            f'{class_id : <9}\t{first_x_pos : <9}\t{first_y_pos : <9}'
            input_string += '\n'
            """
