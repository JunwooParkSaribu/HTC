import csv


def save_report(full_histones, filename='', path=''):
    write_file_name = path + '/' + filename
    histones = {}
    for h in full_histones:
        histones |= h
    histone_names = list(histones.keys())

    with open(write_file_name, 'w', newline='') as f:
        fieldnames = ['filename', 'h2b_id', 'predicted_class_id', 'predicted_class_name', 'probability',
                      'maximum_radius','labeled_class_id', 'first_x_position', 'first_y_position']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, histone_name in enumerate(histone_names):
            trajectory = histones[histone_name].get_trajectory()
            first_x_pos = trajectory[0][0]
            first_y_pos = trajectory[0][1]
            file_name = histones[histone_name].get_file_name()
            h2b_id = histones[histone_name].get_id()
            pred_class_id = histones[histone_name].get_predicted_label()
            manuel_label_id = histones[histone_name].get_manuel_label()
            max_r = histones[histone_name].get_max_radius()
            proba = histones[histone_name].get_predicted_proba()

            pred_class_name = 'unidentified'
            if pred_class_id == 0:
                pred_class_name = 'Immobile'
            if pred_class_id == 1:
                pred_class_name = 'Hybrid'
            if pred_class_id == 2:
                pred_class_name = 'Mobile'

            writer.writerow({'filename':file_name, 'h2b_id':h2b_id, 'predicted_class_id':pred_class_id,
                             'predicted_class_name':pred_class_name, 'probability':proba, 'maximum_radius':max_r,
                             'labeled_class_id':manuel_label_id,
                             'first_x_position':first_x_pos, 'first_y_position':first_y_pos})
            """
            input_string += f'{file_name : <25}\t{trajectory_num : <6}\t{class_num : <4}\t' \
                            f'{class_id : <9}\t{first_x_pos : <9}\t{first_y_pos : <9}'
            input_string += '\n'
            """
