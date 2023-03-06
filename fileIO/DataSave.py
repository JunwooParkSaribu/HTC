import csv


def save_report(full_data, path='', all=False, eval=False):
    histones = {}

    if type(full_data) is list:
        for chunked_data in full_data:
            histones |= chunked_data
    else:
        histones = full_data

    # Accuracy only for evaluation
    if eval == True:
        miss_classfied = 0
        for i, histone in enumerate(histones):
            if histones[histone].get_predicted_label() != histones[histone].get_manuel_label():
                miss_classfied += 1
        print(f'Accuracy = {(i - miss_classfied) / i}')

    if all == False:
        histone_names = list(histones.keys())
        filenames = set()
        for histone in histone_names:
            filenames.add(histone.split('@')[0])

        for filename in filenames:
            h = {}
            for histone in histone_names:
                if filename in histone:
                    h[histone] = histones[histone]
            if eval == True:
                write_file_name = f'{path}/eval_{filename}.csv'
            else:
                write_file_name = f'{path}/{filename}.csv'
            with open(write_file_name, 'w', newline='') as f:
                if eval:
                    fieldnames = ['filename', 'h2b_id', 'predicted_class_id', 'predicted_class_name', 'probability',
                                  'maximum_radius', 'labeled_class_id', 'first_x_position', 'first_y_position']
                else:
                    fieldnames = ['filename', 'h2b_id', 'predicted_class_id', 'predicted_class_name', 'probability',
                                  'maximum_radius', 'first_x_position', 'first_y_position']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

                for i, key in enumerate(h):
                    trajectory = histones[key].get_trajectory()
                    first_x_pos = trajectory[0][0]
                    first_y_pos = trajectory[0][1]
                    file_name = histones[key].get_file_name()
                    h2b_id = histones[key].get_id()
                    pred_class_id = histones[key].get_predicted_label()
                    manuel_label_id = histones[key].get_manuel_label()
                    max_r = histones[key].get_max_radius()
                    proba = histones[key].get_predicted_proba()

                    pred_class_name = 'unidentified'
                    if pred_class_id == 0:
                        pred_class_name = 'Immobile'
                    if pred_class_id == 1:
                        pred_class_name = 'Hybrid'
                    if pred_class_id == 2:
                        pred_class_name = 'Mobile'

                    if eval:
                        writer.writerow({'filename':file_name, 'h2b_id':h2b_id, 'predicted_class_id':pred_class_id,
                                         'predicted_class_name':pred_class_name, 'probability':proba, 'maximum_radius':max_r,
                                         'labeled_class_id':manuel_label_id,
                                         'first_x_position':first_x_pos, 'first_y_position':first_y_pos})
                    else:
                        writer.writerow({'filename':file_name, 'h2b_id':h2b_id, 'predicted_class_id':pred_class_id,
                                         'predicted_class_name':pred_class_name, 'probability':proba, 'maximum_radius':max_r,
                                         'first_x_position':first_x_pos, 'first_y_position':first_y_pos})
    else:
        if eval == True:
            write_file_name = f'{path}/eval_all.csv'
        else:
            write_file_name = f'{path}/pred_all.csv'
        with open(write_file_name, 'w', newline='') as f:
            if eval:
                fieldnames = ['filename', 'h2b_id', 'predicted_class_id', 'predicted_class_name', 'probability',
                              'maximum_radius', 'labeled_class_id', 'first_x_position', 'first_y_position']
            else:
                fieldnames = ['filename', 'h2b_id', 'predicted_class_id', 'predicted_class_name', 'probability',
                              'maximum_radius', 'first_x_position', 'first_y_position']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for i, key in enumerate(histones):
                trajectory = histones[key].get_trajectory()
                first_x_pos = trajectory[0][0]
                first_y_pos = trajectory[0][1]
                file_name = histones[key].get_file_name()
                h2b_id = histones[key].get_id()
                pred_class_id = histones[key].get_predicted_label()
                manuel_label_id = histones[key].get_manuel_label()
                max_r = histones[key].get_max_radius()
                proba = histones[key].get_predicted_proba()

                pred_class_name = 'unidentified'
                if pred_class_id == 0:
                    pred_class_name = 'Immobile'
                if pred_class_id == 1:
                    pred_class_name = 'Hybrid'
                if pred_class_id == 2:
                    pred_class_name = 'Mobile'

                if eval:
                    writer.writerow({'filename': file_name, 'h2b_id': h2b_id, 'predicted_class_id': pred_class_id,
                                     'predicted_class_name': pred_class_name, 'probability': proba,
                                     'maximum_radius': max_r,
                                     'labeled_class_id': manuel_label_id,
                                     'first_x_position': first_x_pos, 'first_y_position': first_y_pos})
                else:
                    writer.writerow({'filename': file_name, 'h2b_id': h2b_id, 'predicted_class_id': pred_class_id,
                                     'predicted_class_name': pred_class_name, 'probability': proba,
                                     'maximum_radius': max_r,
                                     'first_x_position': first_x_pos, 'first_y_position': first_y_pos})


def save_simulated_data(histones, filepath):
    try:
        with open(filepath, 'w', encoding="utf-8") as f:
            for histone in histones:
                for traj, time in zip(histones[histone].get_trajectory(), histones[histone].get_time()):
                    line = ''
                    line += histones[histone].get_id()
                    line += '\t'
                    line += str(traj[0])
                    line += '\t'
                    line += str(traj[1])
                    line += '\t'
                    line += str(time)
                    line += '\t'
                    line += str(histones[histone].get_manuel_label())
                    f.write(f'{line}\n')

    except Exception as e:
        print('Simulated data save err')
        print(e)
