import os
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
import scipy


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


if __name__ == '__main__':
    # Must change the backslash(\) to slash(/) or double backlash(\\) on WindowsOS
    path = "/Users/junwoopark/Downloads/zone_results"

    path = path.replace('\\', '/')
    # Element names of plot_list must be same as the folder names
    plot_list = ['before', '30s', '1min', '2min']
    box_colors = ['red', 'green', 'royalblue']
    FONTSIZE = 14

    # Read classification result files (ratio, diffusion coef)
    ratio, coefs = dir_search(path)
    nb_time = len(plot_list)
    data = []
    for time in plot_list:
        immobile = np.array(ratio[time])[:, 0]
        hybrid = np.array(ratio[time])[:, 1]
        mobile = np.array(ratio[time])[:, 2]
        data.append(immobile)
        data.append(hybrid)
        data.append(mobile)

    # T-test between Before and each class
    print('###  ttest  ###')
    for time in range(1, len(plot_list)):
        print(f'result between {plot_list[0]} and {plot_list[time]}')
        print('Immobile: ',scipy.stats.ttest_ind(np.array(ratio[plot_list[0]])[:, 0], np.array(ratio[plot_list[time]])[:, 0]))
        print('Hybrid: ',scipy.stats.ttest_ind(np.array(ratio[plot_list[0]])[:, 1], np.array(ratio[plot_list[time]])[:, 1]))
        print('Mobile: ',scipy.stats.ttest_ind(np.array(ratio[plot_list[0]])[:, 2], np.array(ratio[plot_list[time]])[:, 2]))
        print()

    """
    Box plots of the each class in plot_list
    """
    fig, ax1 = plt.subplots(figsize=(10, 8))
    fig.canvas.manager.set_window_title('H2B class boxplot')
    fig.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)

    bp = ax1.boxplot(data, notch=False, sym='+', vert=True, whis=1.5)
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
    ax1.set_xlabel('Time', fontsize=FONTSIZE)
    ax1.set_ylabel('Percentage', fontsize=FONTSIZE)

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
        for j in range(nb_time):
            box_x.append(box.get_xdata()[j])
            box_y.append(box.get_ydata()[j])
        box_coords = np.column_stack([box_x, box_y])
        ax1.add_patch(Polygon(box_coords, facecolor=box_colors[i % 3]))
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
    ax1.set_xticklabels(['immobile', 'hybrid', 'mobile'] * nb_time, rotation=45, fontsize=FONTSIZE)
    ax1.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1.0], rotation=0, fontsize=FONTSIZE)
    pos = np.arange(num_boxes) + 1
    upper_labels = [str(round(s, 3)) for s in avgs]
    weights = ['bold', 'semibold', 'bold']
    for tick, label in zip(range(num_boxes), ax1.get_xticklabels()):
        k = tick % 3
        ax1.text(pos[tick], .95, upper_labels[tick],
                 transform=ax1.get_xaxis_transform(),
                 horizontalalignment='center', fontsize=FONTSIZE,
                 weight=weights[k], color=box_colors[k])
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

    plt.plot(immobile_coord, immobile_line[:, 1], color=box_colors[0], alpha=0.5)
    plt.plot(hybrid_coord, hybrid_line[:, 1], color=box_colors[1], alpha=0.5)
    plt.plot(mobile_coord, mobile_line[:, 1], color=box_colors[2], alpha=0.5)

    ax2 = ax1.twiny()
    ax2_tick_col = list((np.array(hybrid_coord) - 0.5) / mobile_coord[-1]) + [1]
    ax2.set_xticks(ax2_tick_col)
    plot_list.append('')
    ax2.set_xticklabels(plot_list, fontsize=FONTSIZE)


    """
    ######## IMMOBILE PLOT #############
    plot_list = ['before', '15s', '30s', '1min', '2min']
    fig, ax1 = plt.subplots(figsize=(10, 8))
    fig.canvas.manager.set_window_title('Immobile boxplot')
    fig.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)

    immobile_data = []
    for i in range(nb_time*3):
        if i % 3 == 0:
            immobile_data.append(data[i])
        else:
            immobile_data.append(0)

    bp = ax1.boxplot(immobile_data, notch=False, sym='+', vert=True, whis=1.5)
    plt.setp(bp['boxes'], color='black')
    plt.setp(bp['whiskers'], color='black', linestyle='dashdot')
    plt.setp(bp['fliers'], color='red', marker='+')

    # Add a horizontal grid to the plot, but make it very light in color
    # so we can use it for reading data values but not be distracting
    ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                   alpha=0.4)

    ax1.set(
        axisbelow=True,  # Hide the grid behind plot objects
        title='Change of H2B type over time',
    )
    ax1.set_xlabel('Time', fontsize=14)
    ax1.set_ylabel('Percentage', fontsize=14)

    # Now fill the boxes with desired colors
    immobile_coord = []
    hybrid_coord = []
    mobile_coord = []
    avgs = []
    box_colors = ['red', 'green', 'royalblue']
    num_boxes = len(data)
    medians = np.empty(num_boxes)
    for i in range(num_boxes):
        box = bp['boxes'][i]
        box_x = []
        box_y = []
        for j in range(5):
            box_x.append(box.get_xdata()[j])
            box_y.append(box.get_ydata()[j])
        box_coords = np.column_stack([box_x, box_y])
        # Alternate between Dark Khaki and Royal Blue
        ax1.add_patch(Polygon(box_coords, facecolor=box_colors[i % 3]))
        # Now draw the median lines back over what we just filled in
        med = bp['medians'][i]
        median_x = []
        median_y = []
        for j in range(2):
            median_x.append(med.get_xdata()[j])
            median_y.append(med.get_ydata()[j])
            # ax1.plot(median_x, median_y, 'k')
        medians[i] = median_y[0]
        # Finally, overplot the sample averages, with horizontal alignment
        # in the center of each box
        if i % 3 == 0:
            immobile_coord.append(np.average(med.get_xdata()))
        elif i % 3 == 1:
            hybrid_coord.append(np.average(med.get_xdata()))
        else:
            mobile_coord.append(np.average(med.get_xdata()))
        mean_val = np.mean(data[i])
        avgs.append(mean_val)
        if i % 3 == 0:
            ax1.plot(np.average(med.get_xdata()), mean_val,
                     color='w', marker='.', markeredgecolor='k')

    # Set the axes ranges and axes labels
    ax1.set_xlim(0.5, num_boxes + 0.5)
    top = 1
    bottom = 0
    ax1.set_ylim(bottom, top)
    ax1.set_xticklabels(['immobile', '', '']*nb_time, rotation=45, fontsize=14)
    ax1.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1.0], rotation=0, fontsize=14)

    # Due to the Y-axis scale being different across samples, it can be
    # hard to compare differences in medians across the samples. Add upper
    # X-axis tick labels with the sample medians to aid in comparison
    # (just use two decimal places of precision)
    pos = np.arange(num_boxes) + 1
    upper_labels = [str(round(s, 3)) for s in avgs]
    weights = ['bold', 'semibold', 'bold']
    for tick, label in zip(range(num_boxes), ax1.get_xticklabels()):
        k = tick % 3
        if tick % 3 == 0:
            ax1.text(pos[tick], .95, upper_labels[tick],
                     transform=ax1.get_xaxis_transform(),
                     horizontalalignment='center', size='x-large',
                     weight=weights[k], color=box_colors[k])

    immobile_line = []
    hybrid_line = []
    mobile_line = []
    for i in range(len(data)):
        if i % 3 == 0:
            immobile_line.append([i, np.average(data[i])])

    immobile_line = np.array(immobile_line)
    plt.plot(immobile_coord, immobile_line[:, 1], color=box_colors[0], alpha=0.5)
    ax2 = ax1.twiny()
    ax2.set_xticks([0.1, 0.3, 0.5, 0.7, 0.9, 1])
    plot_list.append('')
    ax2.set_xticklabels(plot_list, fontsize=14)

    ######## HYBRID PLOT #############
    plot_list = ['before', '15s', '30s', '1min', '2min']
    fig, ax1 = plt.subplots(figsize=(10, 8))
    fig.canvas.manager.set_window_title('Hybrid boxplot')
    fig.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)

    hybrid_data = []
    for i in range(15):
        if i % 3 == 1:
            hybrid_data.append(data[i])
        else:
            hybrid_data.append(0)

    bp = ax1.boxplot(hybrid_data, notch=False, sym='+', vert=True, whis=1.5)
    plt.setp(bp['boxes'], color='black')
    plt.setp(bp['whiskers'], color='black', linestyle='dashdot')
    plt.setp(bp['fliers'], color='red', marker='+')

    # Add a horizontal grid to the plot, but make it very light in color
    # so we can use it for reading data values but not be distracting
    ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                   alpha=0.4)

    ax1.set(
        axisbelow=True,  # Hide the grid behind plot objects
        title='Change of H2B type over time'
    )
    ax1.set_xlabel('Time', fontsize=14)
    ax1.set_ylabel('Percentage', fontsize=14)

    # Now fill the boxes with desired colors
    immobile_coord = []
    hybrid_coord = []
    mobile_coord = []
    avgs = []
    box_colors = ['red', 'green', 'royalblue']
    num_boxes = len(data)
    medians = np.empty(num_boxes)
    for i in range(num_boxes):
        box = bp['boxes'][i]
        box_x = []
        box_y = []
        for j in range(5):
            box_x.append(box.get_xdata()[j])
            box_y.append(box.get_ydata()[j])
        box_coords = np.column_stack([box_x, box_y])
        # Alternate between Dark Khaki and Royal Blue
        ax1.add_patch(Polygon(box_coords, facecolor=box_colors[i % 3]))
        # Now draw the median lines back over what we just filled in
        med = bp['medians'][i]
        median_x = []
        median_y = []
        for j in range(2):
            median_x.append(med.get_xdata()[j])
            median_y.append(med.get_ydata()[j])
            # ax1.plot(median_x, median_y, 'k')
        medians[i] = median_y[0]
        # Finally, overplot the sample averages, with horizontal alignment
        # in the center of each box
        if i % 3 == 0:
            immobile_coord.append(np.average(med.get_xdata()))
        elif i % 3 == 1:
            hybrid_coord.append(np.average(med.get_xdata()))
        else:
            mobile_coord.append(np.average(med.get_xdata()))
        mean_val = np.mean(data[i])
        avgs.append(mean_val)
        if i % 3 == 1:
            ax1.plot(np.average(med.get_xdata()), mean_val,
                     color='w', marker='.', markeredgecolor='k')

    # Set the axes ranges and axes labels
    ax1.set_xlim(0.5, num_boxes + 0.5)
    top = 1
    bottom = 0
    ax1.set_ylim(bottom, top)
    ax1.set_xticklabels(['', 'hybrid', '',
                         '', 'hybrid', '',
                         '', 'hybrid', '',
                         '', 'hybrid', '',
                         '', 'hybrid', '', ], rotation=45, fontsize=14)
    ax1.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1.0], rotation=0, fontsize=14)

    # Due to the Y-axis scale being different across samples, it can be
    # hard to compare differences in medians across the samples. Add upper
    # X-axis tick labels with the sample medians to aid in comparison
    # (just use two decimal places of precision)
    pos = np.arange(num_boxes) + 1
    upper_labels = [str(round(s, 3)) for s in avgs]
    weights = ['bold', 'semibold', 'bold']
    for tick, label in zip(range(num_boxes), ax1.get_xticklabels()):
        k = tick % 3
        if tick % 3 == 1:
            ax1.text(pos[tick], .95, upper_labels[tick],
                     transform=ax1.get_xaxis_transform(),
                     horizontalalignment='center', size='x-large',
                     weight=weights[k], color=box_colors[k])

    immobile_line = []
    hybrid_line = []
    mobile_line = []
    for i in range(len(data)):
        if i % 3 == 1:
            hybrid_line.append([i, np.average(data[i])])

    hybrid_line = np.array(hybrid_line)
    plt.plot(hybrid_coord, hybrid_line[:, 1], color=box_colors[1], alpha=0.5)
    ax2 = ax1.twiny()
    ax2.set_xticks([0.1, 0.3, 0.5, 0.7, 0.9, 1])
    plot_list.append('')
    ax2.set_xticklabels(plot_list, fontsize=14)

    ######## MOBILE PLOT #############
    plot_list = ['before', '15s', '30s', '1min', '2min']
    fig, ax1 = plt.subplots(figsize=(10, 8))
    fig.canvas.manager.set_window_title('Mobile boxplot')
    fig.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)

    mobile_data = []
    for i in range(15):
        if i % 3 == 2:
            mobile_data.append(data[i])
        else:
            mobile_data.append(0)

    bp = ax1.boxplot(mobile_data, notch=False, sym='+', vert=True, whis=1.5)
    plt.setp(bp['boxes'], color='black')
    plt.setp(bp['whiskers'], color='black', linestyle='dashdot')
    plt.setp(bp['fliers'], color='red', marker='+')

    # Add a horizontal grid to the plot, but make it very light in color
    # so we can use it for reading data values but not be distracting
    ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                   alpha=0.4)
    ax1.set(
        axisbelow=True,  # Hide the grid behind plot objects
        title='Change of H2B type over time'
    )
    ax1.set_xlabel('Time', fontsize=14)
    ax1.set_ylabel('Percentage', fontsize=14)

    # Now fill the boxes with desired colors
    immobile_coord = []
    hybrid_coord = []
    mobile_coord = []
    avgs = []
    box_colors = ['red', 'green', 'royalblue']
    num_boxes = len(data)
    medians = np.empty(num_boxes)
    for i in range(num_boxes):
        box = bp['boxes'][i]
        box_x = []
        box_y = []
        for j in range(5):
            box_x.append(box.get_xdata()[j])
            box_y.append(box.get_ydata()[j])
        box_coords = np.column_stack([box_x, box_y])
        # Alternate between Dark Khaki and Royal Blue
        ax1.add_patch(Polygon(box_coords, facecolor=box_colors[i % 3]))
        # Now draw the median lines back over what we just filled in
        med = bp['medians'][i]
        median_x = []
        median_y = []
        for j in range(2):
            median_x.append(med.get_xdata()[j])
            median_y.append(med.get_ydata()[j])
            # ax1.plot(median_x, median_y, 'k')
        medians[i] = median_y[0]
        # Finally, overplot the sample averages, with horizontal alignment
        # in the center of each box
        if i % 3 == 0:
            immobile_coord.append(np.average(med.get_xdata()))
        elif i % 3 == 1:
            hybrid_coord.append(np.average(med.get_xdata()))
        else:
            mobile_coord.append(np.average(med.get_xdata()))
        mean_val = np.mean(data[i])
        avgs.append(mean_val)
        if i % 3 == 2:
            ax1.plot(np.average(med.get_xdata()), mean_val,
                     color='w', marker='.', markeredgecolor='k')

    # Set the axes ranges and axes labels
    ax1.set_xlim(0.5, num_boxes + 0.5)
    top = 1
    bottom = 0
    ax1.set_ylim(bottom, top)
    ax1.set_xticklabels(['', '', 'mobile',
                         '', '', 'mobile',
                         '', '', 'mobile',
                         '', '', 'mobile',
                         '', '', 'mobile', ], rotation=45, fontsize=14)
    ax1.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1.0], rotation=0, fontsize=14)

    # Due to the Y-axis scale being different across samples, it can be
    # hard to compare differences in medians across the samples. Add upper
    # X-axis tick labels with the sample medians to aid in comparison
    # (just use two decimal places of precision)
    pos = np.arange(num_boxes) + 1
    upper_labels = [str(round(s, 3)) for s in avgs]
    weights = ['bold', 'semibold', 'bold']
    for tick, label in zip(range(num_boxes), ax1.get_xticklabels()):
        k = tick % 3
        if tick % 3 == 2:
            ax1.text(pos[tick], .95, upper_labels[tick],
                     transform=ax1.get_xaxis_transform(),
                     horizontalalignment='center', size='x-large',
                     weight=weights[k], color=box_colors[k])

    immobile_line = []
    hybrid_line = []
    mobile_line = []
    for i in range(len(data)):
        if i % 3 == 2:
            mobile_line.append([i, np.average(data[i])])

    mobile_line = np.array(mobile_line)
    plt.plot(mobile_coord, mobile_line[:, 1], color=box_colors[2], alpha=0.5)
    ax2 = ax1.twiny()
    ax2.set_xticks([0.1, 0.3, 0.5, 0.7, 0.9, 1])
    plot_list.append('')
    ax2.set_xticklabels(plot_list, fontsize=14)
    """

    ######### DIFF COEF PLOT ###########
    coef_data = []
    plot_list = ['before', '30s', '1min', '2min']
    for time in plot_list:
        tmp = []
        for nb in range(len(coefs[time])):
            tmp.extend(coefs[time][nb])
        tmp = np.array(tmp)
        coef_data.append(tmp * 10)

    print('ttest of diffusion coefficient')
    for time in range(1, len(plot_list)):
        print(scipy.stats.ttest_ind(coef_data[0], coef_data[time]))
    print()

    fig, ax1 = plt.subplots(figsize=(10, 8))
    fig.canvas.manager.set_window_title('H2B diff_coef boxplot')
    fig.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)
    bp = ax1.boxplot(coef_data, notch=False, sym='', vert=True, whis=1.5, widths=0.25)
    plt.setp(bp['boxes'], color='black')
    plt.setp(bp['whiskers'], color='black', linestyle='dashdot')
    plt.setp(bp['fliers'], color='red', marker='+')

    # Add a horizontal grid to the plot, but make it very light in color
    # so we can use it for reading data values but not be distracting
    ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                   alpha=0.4)

    ax1.set(
        axisbelow=True,  # Hide the grid behind plot objects
        title='H2B diffusion coefficient over time'
    )
    ax1.set_xlabel('Time', fontsize=14)
    ax1.set_ylabel('D ($\u03bcm^{%d}/s$)' % (2), fontsize=14)

    # Now fill the boxes with desired colors
    coef_coord = []
    avgs = []
    box_colors = ['grey']
    num_boxes = len(coef_data)
    medians = np.empty(num_boxes)
    for i in range(num_boxes):
        box = bp['boxes'][i]
        box_x = []
        box_y = []
        for j in range(5):
            box_x.append(box.get_xdata()[j])
            box_y.append(box.get_ydata()[j])
        box_coords = np.column_stack([box_x, box_y])
        # Alternate between Dark Khaki and Royal Blue
        ax1.add_patch(Polygon(box_coords, facecolor=box_colors[i % 1]))
        # Now draw the median lines back over what we just filled in
        med = bp['medians'][i]
        median_x = []
        median_y = []
        for j in range(2):
            median_x.append(med.get_xdata()[j])
            median_y.append(med.get_ydata()[j])
            # ax1.plot(median_x, median_y, 'k')
        medians[i] = median_y[0]
        # Finally, overplot the sample averages, with horizontal alignment
        # in the center of each box
        coef_coord.append(np.average(med.get_xdata()))
        mean_val = np.mean(coef_data[i])
        avgs.append(mean_val)
        ax1.plot(np.average(med.get_xdata()), mean_val,
                 color='w', marker='.', markeredgecolor='k')

    # Set the axes ranges and axes labels
    ax1.set_xlim(0.5, num_boxes + 0.5)
    top = 8
    bottom = 0
    ax1.set_ylim(bottom, top)
    ax1.set_xticklabels(plot_list, rotation=0, fontsize=14)
    ax1.set_yticklabels([0, 1, 2, 3, 4, 5, 6, 7, 8], rotation=0, fontsize=14)

    # Due to the Y-axis scale being different across samples, it can be
    # hard to compare differences in medians across the samples. Add upper
    # X-axis tick labels with the sample medians to aid in comparison
    # (just use two decimal places of precision)
    pos = np.arange(num_boxes) + 1
    upper_labels = [str(round(s, 3)) for s in avgs]
    weights = ['bold', 'semibold', 'bold']
    for tick, label in zip(range(num_boxes), ax1.get_xticklabels()):
        k = tick % 1
        ax1.text(pos[tick], .95, upper_labels[tick],
                 transform=ax1.get_xaxis_transform(),
                 horizontalalignment='center', size='x-large',
                 weight=weights[k], color=box_colors[k])

    coef_line = []
    for i in range(len(coef_data)):
        coef_line.append([i, np.average(coef_data[i])])
    coef_line = np.array(coef_line)
    plt.plot(coef_coord, coef_line[:, 1], color=box_colors[0], alpha=0.5)

    plt.show()

    """
    ###### MERGED INTO ONE FILE VERSION (CUTOFF = 8)######
    plot_list = ['before', '15s', '30s', '1min', '2min']

    data = [0.8308571428571428, 0.10228571428571429, 0.06685714285714285,
            0.7718120805369127, 0.15268456375838926, 0.07550335570469799,
            0.7561349693251533, 0.16411042944785276, 0.07975460122699386,
            0.76, 0.14666666666666667, 0.09333333333333334,
            0.7722132471728594, 0.14378029079159935, 0.0840064620355412]

    new_data = []
    immobile_tmp = []
    hybrid_tmp = []
    mobile_tmp = []
    for i, dt in enumerate(data):
        if i % 3 == 0:
            immobile_tmp.append(dt)
        elif i % 3 == 1:
            hybrid_tmp.append(dt)
        else:
            mobile_tmp.append(dt)
    new_data.append(immobile_tmp)
    new_data.append(hybrid_tmp)
    new_data.append(mobile_tmp)
    new_data = np.array(new_data)

    fig, ax1 = plt.subplots(figsize=(10, 8))
    fig.canvas.manager.set_window_title('H2B class ratio bar')
    fig.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)

    # Add a horizontal grid to the plot, but make it very light in color
    # so we can use it for reading data values but not be distracting
    ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                   alpha=0.4)

    ax1.set(
        axisbelow=True,  # Hide the grid behind plot objects
        title='Change of H2B type over time',
        xlabel='Time',
        ylabel='Percentage',
    )

    # Now fill the boxes with desired colors
    box_colors = ['red', 'green', 'royalblue']

    X = np.arange(5)
    ax1.bar(X + 0.00, new_data[0], color='red', width=0.23, label='immobile', alpha=0.75)
    ax1.bar(X + 0.25, new_data[1], color='green', width=0.23, label='hybrid', alpha=0.75)
    ax1.bar(X + 0.50, new_data[2], color='royalblue', width=0.23, label='mobile', alpha=0.75)
    x_coord = np.array([0.25, 1.25, 2.25, 3.25, 4.25])
    top = 1
    bottom = 0
    ax1.set_ylim(bottom, top)
    ax1.set_xticks(x_coord, plot_list)
    ax1.legend(loc='center right')
    ax1.plot(x_coord-.25, immobile_tmp, color='red', alpha=0.8)
    ax1.plot(x_coord, hybrid_tmp, color='green', alpha=0.8)
    ax1.plot(x_coord+.25, mobile_tmp, color='royalblue', alpha=0.8)

    ax1.scatter(x_coord-.25, immobile_tmp, color='red', alpha=0.8, s=8)
    ax1.scatter(x_coord, hybrid_tmp, color='green', alpha=0.8, s=8)
    ax1.scatter(x_coord+.25, mobile_tmp, color='royalblue', alpha=0.8, s=8)

    pos = np.arange(15) + 1
    upper_labels = np.array([str(round(s, 3)) for s in data])
    weights = ['bold', 'semibold', 'bold']
    for i, x_pos in enumerate((x_coord-.25)):
        val = upper_labels[int(i * 3)]
        ax1.text(x_pos, .95, val,
                 transform=ax1.get_xaxis_transform(),
                 horizontalalignment='center', size='x-small',
                 color='red')
    for i, x_pos in enumerate((x_coord)):
        val = upper_labels[int(i * 3 + 1)]
        ax1.text(x_pos, .95, val,
                 transform=ax1.get_xaxis_transform(),
                 horizontalalignment='center', size='x-small',
                 color='green')
    for i, x_pos in enumerate((x_coord+.25)):
        val = upper_labels[int(i * 3 + 2)]
        ax1.text(x_pos, .95, val,
                 transform=ax1.get_xaxis_transform(),
                 horizontalalignment='center', size='x-small',
                 color='royalblue')
    plt.show()
    """
