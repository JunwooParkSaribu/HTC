import csv
from analysis import DataAnalysis


def save_report(full_data, path='', all=False) -> list:
    histones = {}
    report_names = []

    if type(full_data) is list:
        for chunked_data in full_data:
            histones |= chunked_data
    elif type(full_data) is dict:
        histones = full_data
    else:
        raise Exception

    if not all:
        histone_names = list(histones.keys())
        filenames = set()
        for histone in histone_names:
            filenames.add(histone.split('\\')[-1].split('@')[0])

        for filename in filenames:
            h = {}
            for histone in histone_names:
                if filename in histone:
                    h[histone] = histones[histone]
            write_file_name = f'{path}/{filename}.csv'
            with open(write_file_name, 'w', newline='') as f:
                report_names.append(write_file_name)
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
                    max_r = histones[key].get_max_radius()
                    proba = histones[key].get_predicted_proba()

                    pred_class_name = 'unidentified'
                    if pred_class_id == 0:
                        pred_class_name = 'Immobile'
                    if pred_class_id == 1:
                        pred_class_name = 'Hybrid'
                    if pred_class_id == 2:
                        pred_class_name = 'Mobile'

                    writer.writerow({'filename':file_name, 'h2b_id':h2b_id, 'predicted_class_id':pred_class_id,
                                         'predicted_class_name':pred_class_name, 'probability':proba, 'maximum_radius':max_r,
                                         'first_x_position':first_x_pos, 'first_y_position':first_y_pos})
    else:
        write_file_name = f'{path}/prediction_all.csv'
        with open(write_file_name, 'w', newline='') as f:
            report_names.append(write_file_name)
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
                max_r = histones[key].get_max_radius()
                proba = histones[key].get_predicted_proba()

                pred_class_name = 'unidentified'
                if pred_class_id == 0:
                    pred_class_name = 'Immobile'
                if pred_class_id == 1:
                    pred_class_name = 'Hybrid'
                if pred_class_id == 2:
                    pred_class_name = 'Mobile'

                writer.writerow({'filename': file_name, 'h2b_id': h2b_id, 'predicted_class_id': pred_class_id,
                                     'predicted_class_name': pred_class_name, 'probability': proba,
                                     'maximum_radius': max_r,
                                     'first_x_position': first_x_pos, 'first_y_position': first_y_pos})
    return report_names


def save_diffcoef(full_data, path='', all=False, exel=False) -> list:
    histones = {}

    if type(full_data) is list:
        for chunked_data in full_data:
            histones |= chunked_data
    elif type(full_data) is dict:
        histones = full_data
    else:
        raise Exception

    if not all:
        histone_names = list(histones.keys())
        filenames = set()
        for histone in histone_names:
            filenames.add(histone.split('\\')[-1].split('@')[0])

        for filename in filenames:
            h = {}
            for histone in histone_names:
                if filename in histone:
                    h[histone] = histones[histone]
            if exel:
                write_file_name = f'{path}/{filename}_diffcoef.xlsx'
            else:
                write_file_name = f'{path}/{filename}_diffcoef.csv'

            with open(f'{path}/{filename}_ratio.txt', 'w', newline='') as ratio_file:
                report_name = f'{path}/{filename}.csv'
                ratio = DataAnalysis.ratio_calcul(report_name)
                ratio_file.write(f'(immobile:hybrid:mobile):{ratio}\n')
                ratio_file.close()

            if exel:
                """
                # Workbook() takes one, non-optional, argument
                # which is the filename that we want to create.
                workbook = xlsxwriter.Workbook(write_file_name)

                # The workbook object is then used to add new
                # worksheet via the add_worksheet() method.
                worksheet = workbook.add_worksheet()

                for i, key in enumerate(h):
                    file_name = histones[key].get_file_name()
                    h2b_id = histones[key].get_id()
                    diff_coefs = histones[key].get_diff_coef()
                    for j, diff_coef in enumerate(diff_coefs):
                        if j == 0:
                            worksheet.write(file_name, h2b_id, diff_coef)
                        else:
                            worksheet.write('', '', diff_coef)
                workbook.close()
                """

            else:
                with open(write_file_name, 'w', newline='') as f:
                    fieldnames = ['filename', 'h2b_id', 'diffusion_coef']
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()

                    for i, key in enumerate(h):
                        file_name = histones[key].get_file_name()
                        h2b_id = histones[key].get_id()
                        diff_coefs = histones[key].get_diff_coef()
                        for j, diff_coef in enumerate(diff_coefs):
                            if j==0:
                                writer.writerow({'filename': file_name, 'h2b_id': h2b_id, 'diffusion_coef': diff_coef})
                            else:
                                writer.writerow({'diffusion_coef': diff_coef})

    else:
        write_file_name = f'{path}/prediction_all_diffcoef.csv'
        with open(f'{path}/prediction_all_ratio.txt', 'w', newline='') as ratio_file:
            report_name = f'{path}/prediction_all.csv'
            ratio = DataAnalysis.ratio_calcul(report_name)
            ratio_file.write(f'(immobile:hybrid:mobile):{ratio}\n')
            ratio_file.close()

        with open(write_file_name, 'w', newline='') as f:
            report_name = f'{path}/prediction_all.csv'
            ratio = DataAnalysis.ratio_calcul(report_name)
            f.write(f'(immobile:hybrid:mobile):{ratio}\n')

            fieldnames = ['filename', 'h2b_id', 'diffusion_coef']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for i, key in enumerate(histones):
                file_name = histones[key].get_file_name()
                h2b_id = histones[key].get_id()
                diff_coefs = histones[key].get_diff_coef()
                for j, diff_coef in enumerate(diff_coefs):
                    if j == 0:
                        writer.writerow({'filename': file_name, 'h2b_id': h2b_id, 'diffusion_coef': diff_coef})
                    else:
                        writer.writerow({'diffusion_coef': diff_coef})


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
