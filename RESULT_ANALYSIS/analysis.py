import os
import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.patches import Polygon
from modules.fileIO import DataLoad


def read_files_by_each_time(path: str, plotList: list) -> dict:
    """
    Read histone files and return the stored H2B object by each time.
    """
    histones = {}
    print("Number of immobile h2b for each time")
    for time in plotList:
        histones[time] = []
        h = DataLoad.read_files([f'{path}/{time}'], cutoff=1, chunk=False)[0]
        for h2b in h:
            histones[time].append(h[h2b])
        print(f'{time}: {len(histones[time])}')
    return histones


def read_ratio_file(file):
    ratio = []
    with open(file) as f:
        line = f.readline()
        str_ratio = line.split(':')[-1].split('(')[-1].split(')')[0].strip().split(',')
        try:
            for r in str_ratio:
                ratio.append(float(r.strip()))
        except Exception as e:
            print(e)
            print('Err while reading ratio files')
    return ratio


def read_coef_file(file):
    coef = []
    with open(file) as f:
        lines = f.readlines()
        for line in lines[1:]:  # first line is fieldname
            try:
                val = line.strip().split(',')[-1].strip()
                coef.append(float(val))
            except Exception as e:
                print(e)
                print('Err while reading diff_coef files')
    return coef


def dir_search(path):
    ratio_dict = {}
    diffcoef_dict = {}
    for root, dirs, files in os.walk(path, topdown=False):
        root = root.replace('\\', '/')
        hint = root.strip().split('/')[-1]
        if hint == path.split('/')[-1]:
            continue
        ratio_dict[hint] = []
        diffcoef_dict[hint] = []

        for file in files:
            if 'ratio.txt' in file:
                ratio = read_ratio_file(f'{root}/{file}')
                ratio_dict[hint].append(ratio)
            if 'diffcoef.csv' in file:
                coef = read_coef_file(f'{root}/{file}')
                diffcoef_dict[hint].append(coef)
    return ratio_dict, diffcoef_dict


def MSD(histones, plotList, fontSize):
    """
    Mean Squared Displacement.
    The average displacement of immobile H2B for each time,
    Formula(1) is in the attached image.
    ref: https://iopscience.iop.org/article/10.1088/1367-2630/15/4/045011.
    """
    msd = dict()
    x_axis = dict()
    for time in plotList:
        histone_list = histones[time]
        disps = []
        for h2b in histone_list:
            trajectory = h2b.get_trajectory()
            t_seq = h2b.get_time()
            ref_position = trajectory[0]
            ref_t = t_seq[0]
            displacement = dict()
            for pos, t in zip(trajectory, t_seq):
                displacement[np.round(t - ref_t, 5)] = \
                    np.sqrt((pos[0] - ref_position[0])**2 + (pos[1] - ref_position[1])**2) ** 2
            disps.append(displacement)
        all_possible_times = set()
        for disp in disps:
            tmp = list(disp.keys())
            for tp in tmp:
                all_possible_times.add(tp)
        all_possible_times = list(all_possible_times)
        all_possible_times.sort()
        x_axis[time] = all_possible_times.copy()
        msd[time] = dict()
        for t in all_possible_times:
            msd[time][t] = []
            for disp in disps:
                if t in disp:
                    msd[time][t].append(disp[t])

    plt.figure()
    for time in plot_list:
        y_vals = []
        for t in x_axis[time]:
            y_vals.append(np.mean(msd[time][t]))
        plt.plot(x_axis[time], y_vals, label=str(time), alpha=0.7)
    plt.ylabel('MSD($um^{2}$)', fontsize=fontSize)
    plt.xlabel('Time(sec)', fontsize=fontSize)
    plt.ylim(0, 0.15)
    plt.legend()
    plt.show()


def TAMSD(histones, plotList, fontSize):
    """
    Time Averaged Mean Squared Displacement.
    The time averaged displacement of immobile H2B for each time.
    Formula(2) is in the attached image.
    ref: https://iopscience.iop.org/article/10.1088/1367-2630/15/4/045011
    """
    tamsd = dict()
    x_vals_mean = {}
    y_vals_mean = {}
    for time in plotList:
        tamsd[time] = {}
        histone_list = histones[time]
        max_time_gap = -999
        for h2b in histone_list:
            tamsd[time][h2b] = {}
            t = h2b.get_time()
            max_t = t[-1]
            min_t = t[0]
            max_time_gap = max(max_time_gap, np.round(max_t - min_t, 5))

        time_gaps = np.arange(0.00, max_time_gap+0.01, 0.01)
        for h2b in histone_list:
            for delta_t in time_gaps:
                delta_t = np.round(delta_t, 5)
                tamsd[time][h2b][delta_t] = []

        for h2b in histone_list:
            trajectory = h2b.get_trajectory()
            t_seq = h2b.get_time()
            displacement = {}
            for i in range(len(t_seq)):
                for j in range(i, len(t_seq)):
                    t_gap = np.round(t_seq[j] - t_seq[i], 5)
                    disp = (trajectory[j][0] - trajectory[i][0]) ** 2 \
                           + (trajectory[j][1] - trajectory[i][1]) ** 2
                    if t_gap in displacement:
                        displacement[t_gap].append(disp)
                    else:
                        displacement[t_gap] = [disp]

            for t_gap in displacement:
                tamsd[time][h2b][t_gap].append(np.mean(displacement[t_gap]))

        plt.figure(f'{time}')
        x_vals_mean[time] = time_gaps.copy()
        y_vals_mean[time] = []
        tmp_y_vals_mean = {}
        for h2b in histone_list:
            single_x_vals = []
            single_y_vals = []
            for t_gap in tamsd[time][h2b]:
                if t_gap in tmp_y_vals_mean:
                    tmp_y_vals_mean[t_gap].extend(tamsd[time][h2b][t_gap].copy())
                else:
                    tmp_y_vals_mean[t_gap] = tamsd[time][h2b][t_gap].copy()
                if len(tamsd[time][h2b][t_gap]) > 0:
                    single_x_vals.append(t_gap)
                    y_val = tamsd[time][h2b][t_gap][0]
                    single_y_vals.append(y_val)

            # TAMSD of single particle
            plt.plot(single_x_vals, single_y_vals, c='red', alpha=0.3)

        for t_gap in tmp_y_vals_mean:
            y_vals_mean[time].append(np.mean(tmp_y_vals_mean[t_gap]))

        # Average of particle's TAMSD
        plt.plot(x_vals_mean[time], y_vals_mean[time], c='blue', alpha=0.7)
        blue_line = mlines.Line2D([], [], color='blue', label='Avg')
        red_line = mlines.Line2D([], [], color='red', label='TAMSD of single particle')
        plt.legend(handles=[blue_line, red_line])
        plt.ylabel('TAMSD($um^{2}$)', fontsize=fontSize)
        plt.xlabel('Time(sec)', fontsize=fontSize)
        plt.title(f'{time}')

    plt.figure(f'Avgs of each time')
    for time in plot_list:
        plt.plot(x_vals_mean[time], y_vals_mean[time], label=str(time), alpha=0.7)
    plt.ylabel('TAMSD($um^{2}$)', fontsize=fontSize)
    plt.xlabel('Time(sec)', fontsize=fontSize)
    plt.ylim(0, 0.07)
    plt.legend()
    plt.show()


def box_plots(path, plotList, boxColors, fontSize):
    plotList = plotList.copy()
    # Read classification result files (ratio, diffusion coef)
    ratio, coefs = dir_search(path)
    key_list = list(ratio.keys())
    for key in key_list:
        if key not in plotList:
            del ratio[key]
            del coefs[key]

    nb_time = len(plotList)
    data = []
    for time in plotList:
        immobile = np.array(ratio[time])[:, 0]
        hybrid = np.array(ratio[time])[:, 1]
        mobile = np.array(ratio[time])[:, 2]
        data.append(immobile)
        data.append(hybrid)
        data.append(mobile)

    # T-test between Before and each class
    print('###  ttest  ###')
    for time in range(1, len(plotList)):
        print(f'result between {plotList[0]} and {plotList[time]}')
        print('Immobile: ',
              scipy.stats.ttest_ind(np.array(ratio[plotList[0]])[:, 0], np.array(ratio[plotList[time]])[:, 0]))
        print('Hybrid: ',
              scipy.stats.ttest_ind(np.array(ratio[plotList[0]])[:, 1], np.array(ratio[plotList[time]])[:, 1]))
        print('Mobile: ',
              scipy.stats.ttest_ind(np.array(ratio[plotList[0]])[:, 2], np.array(ratio[plotList[time]])[:, 2]))
        print()

    """
    Box plots of the each class in plot_list
    """
    fig, ax1 = plt.subplots(figsize=(10, 8))
    fig.canvas.manager.set_window_title('H2B class boxplot')
    fig.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)

    bp = ax1.boxplot(data, sym='+', vert=True, whis=1.5, notch=False)
    plt.setp(bp['boxes'], color='black')
    plt.setp(bp['whiskers'], color='black', linestyle='dashdot')
    plt.setp(bp['fliers'], color='red', marker='+')

    # Add a horizontal grid to the plot
    ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                   alpha=0.4)
    ax1.set(
        axisbelow=True,  # Hide the grid behind plot objects
        title='Change of H2B type over time',
    )
    ax1.set_xlabel('Time', fontsize=fontSize)
    ax1.set_ylabel('Percentage', fontsize=fontSize)

    # Now fill the boxes with desired colors
    immobile_coord = []
    hybrid_coord = []
    mobile_coord = []
    avgs = []
    num_boxes = len(data)
    medians = np.empty(num_boxes)
    for i in range(num_boxes):
        box = bp['boxes'][i]
        box_x = []
        box_y = []
        for j in range(len(box.get_xdata())):
            box_x.append(box.get_xdata()[j])
            box_y.append(box.get_ydata()[j])
        box_coords = np.column_stack([box_x, box_y])
        ax1.add_patch(Polygon(box_coords, facecolor=boxColors[i % 3], alpha=0.8))
        # Now draw the median lines back
        med = bp['medians'][i]
        median_x = []
        median_y = []
        for j in range(2):
            median_x.append(med.get_xdata()[j])
            median_y.append(med.get_ydata()[j])
        medians[i] = median_y[0]
        if i % 3 == 0:
            immobile_coord.append(np.average(med.get_xdata()))
        elif i % 3 == 1:
            hybrid_coord.append(np.average(med.get_xdata()))
        else:
            mobile_coord.append(np.average(med.get_xdata()))
        mean_val = np.mean(data[i])
        avgs.append(mean_val)
        ax1.plot(np.average(med.get_xdata()), mean_val,
                 color='w', marker='.', markeredgecolor='k')

    ax1.set_xlim(0.5, num_boxes + 0.5)
    top = 1
    bottom = 0
    ax1.set_ylim(bottom, top)
    ax1.set_xticklabels(['immobile', 'hybrid', 'mobile'] * nb_time, rotation=45, fontsize=fontSize)
    ax1.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1.0], rotation=0, fontsize=fontSize)
    pos = np.arange(num_boxes) + 1
    upper_labels = [str(round(s, 3)) for s in avgs]
    weights = ['bold', 'semibold', 'bold']
    for tick, label in zip(range(num_boxes), ax1.get_xticklabels()):
        k = tick % 3
        ax1.text(pos[tick], .95, upper_labels[tick],
                 transform=ax1.get_xaxis_transform(),
                 horizontalalignment='center', fontsize=fontSize,
                 weight=weights[k], color=boxColors[k])
    immobile_line = []
    hybrid_line = []
    mobile_line = []
    for i in range(len(data)):
        if i % 3 == 0:
            immobile_line.append([i, np.average(data[i])])
        elif i % 3 == 1:
            hybrid_line.append([i, np.average(data[i])])
        else:
            mobile_line.append([i, np.average(data[i])])
    immobile_line = np.array(immobile_line)
    hybrid_line = np.array(hybrid_line)
    mobile_line = np.array(mobile_line)

    plt.plot(immobile_coord, immobile_line[:, 1], color=boxColors[0], alpha=0.5)
    plt.plot(hybrid_coord, hybrid_line[:, 1], color=boxColors[1], alpha=0.5)
    plt.plot(mobile_coord, mobile_line[:, 1], color=boxColors[2], alpha=0.5)

    ax2 = ax1.twiny()
    ax2_tick_col = list((np.array(hybrid_coord) - 0.5) / mobile_coord[-1]) + [1]
    ax2.set_xticks(ax2_tick_col)
    plotList.append('')
    ax2.set_xticklabels(plotList, fontsize=fontSize)
    plt.show()


if __name__ == '__main__':
    current_path = os.getcwd()
    current_path = current_path.replace('\\', '/')
    result_files_path = f'{current_path}/zone_results'
    merged_immobile_trxyt_path = f'{current_path}/immobile_files/merged_files'
    separated_immobile_trxyt_path = f'{current_path}/immobile_files/separated_files'

    # Element names of plot_list must be same as the folder names
    plot_list = ['before', '30s', '1min', '2min']
    # box colors for immobile, hybrid, mobile respectively, only for boxplot
    box_colors = ['red', 'green', 'royalblue']
    # font size for plots (axis and labels)
    FONTSIZE = 14
    # Register font style
    axis_font = {'family': 'serif', 'size': FONTSIZE}
    plt.rc('font', **axis_font)

    # Read trajectory files from the path
    histones = read_files_by_each_time(separated_immobile_trxyt_path, plotList=plot_list)

    """
    Functions.
    To change the details, check the functions body.
    For example:
        alpha(0.0 ~ 1.0): the opacity of plot
    """
    box_plots(result_files_path, plot_list, box_colors, FONTSIZE)
    MSD(histones, plot_list, FONTSIZE)
    TAMSD(histones, plot_list, FONTSIZE)
