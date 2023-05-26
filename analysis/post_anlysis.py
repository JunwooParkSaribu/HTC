import os
import sys
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np


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
                diffcoef_dict[hint].append(file)
    return ratio_dict, diffcoef_dict


if __name__ == '__main__':
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = '.'

    path = '/Users/junwoopark/Downloads/h2b_zone'
    ratio, diffcoef_files = dir_search(path)
    types = list(ratio.keys())
    print(diffcoef_files)

    plot_list = ['before', '15s', '30s', '1min', '2min']

    data = []
    for time in plot_list:
        immobile = np.array(ratio[time])[:, 0]
        hybrid = np.array(ratio[time])[:, 1]
        mobile = np.array(ratio[time])[:, 2]

        data.append(immobile)
        data.append(hybrid)
        data.append(mobile)

    fig, ax1 = plt.subplots(figsize=(10, 8))
    fig.canvas.manager.set_window_title('H2B class boxplot')
    fig.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)

    bp = ax1.boxplot(data, notch=False, sym='+', vert=True, whis=1.5)
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
        xlabel='Time',
        ylabel='Value',
    )

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
            #ax1.plot(median_x, median_y, 'k')
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
        ax1.plot(np.average(med.get_xdata()), mean_val,
                 color='w', marker='.', markeredgecolor='k')

    # Set the axes ranges and axes labels
    ax1.set_xlim(0.5, num_boxes + 0.5)
    top = 1
    bottom = 0
    ax1.set_ylim(bottom, top)
    ax1.set_xticklabels(['immobile', 'hybrid', 'mobile',
                         'immobile', 'hybrid', 'mobile',
                         'immobile', 'hybrid', 'mobile',
                         'immobile', 'hybrid', 'mobile',
                         'immobile', 'hybrid', 'mobile',], rotation=45, fontsize=8)

    # Due to the Y-axis scale being different across samples, it can be
    # hard to compare differences in medians across the samples. Add upper
    # X-axis tick labels with the sample medians to aid in comparison
    # (just use two decimal places of precision)
    pos = np.arange(num_boxes) + 1
    upper_labels = [str(round(s, 3)) for s in avgs]
    weights = ['bold', 'semibold', 'bold']
    for tick, label in zip(range(num_boxes), ax1.get_xticklabels()):
        k = tick % 3
        ax1.text(pos[tick], .95, upper_labels[tick],
                 transform=ax1.get_xaxis_transform(),
                 horizontalalignment='center', size='x-small',
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

    ax2.set_xticks([0.1, 0.3, 0.5, 0.7, 0.9, 1])
    plot_list.append('')
    ax2.set_xticklabels(plot_list)

    plt.show()
