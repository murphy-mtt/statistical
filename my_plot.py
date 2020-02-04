import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches  # needed for waffle Charts


def my_plotter(ax, data1, data2, para):
    out = ax.plot(data1, data2, **para)
    return out


def plot_angle(ax, x, y, angle, style):
    phi = np.radians(angle)
    xx = [x + .5, x, x + .5 * np.cos(phi)]
    yy = [y, y, y + .5 * np.sin(phi)]
    ax.plot(xx, yy, lw=12, color='tab:blue', solid_joinstyle=style)
    ax.plot(xx, yy, lw=1, color='black')
    ax.plot(xx[1], yy[1], 'o', color='tab:red', markersize=3)


def my_scatter(ax, data1, data2, para):
    out = ax.scatter(data1, data2, **para)
    return out


# Step 8: pack everything into a function
def create_waffle_chart(dataset, categories, values, height, width, colormap, value_sign=''):
    # compute the proportion of each category with respect to the total
    total_values = sum(values)
    category_proportions = [(float(value) / total_values) for value in values]

    # compute the total number of tiles
    total_num_tiles = width * height  # total number of tiles
    print('Total number of tiles is', total_num_tiles)

    # compute the number of tiles for each catagory
    tiles_per_category = [round(proportion * total_num_tiles) for proportion in category_proportions]

    # print out number of tiles per category
    for i, tiles in enumerate(tiles_per_category):
        print(dataset.index.values[i] + ': ' + str(tiles))

    # initialize the waffle chart as an empty matrix
    waffle_chart = np.zeros((height, width))

    # define indices to loop through waffle chart
    category_index = 0
    tile_index = 0

    # populate the waffle chart
    for col in range(width):
        for row in range(height):
            tile_index += 1

            # if the number of tiles populated for the current category
            # is equal to its corresponding allocated tiles...
            if tile_index > sum(tiles_per_category[0:category_index]):
                # ...proceed to the next category
                category_index += 1

                # set the class value to an integer, which increases with class
            waffle_chart[row, col] = category_index

    # instantiate a new figure object
    fig = plt.figure()

    # use matshow to display the waffle chart
    colormap = plt.cm.coolwarm
    plt.matshow(waffle_chart, cmap=colormap)
    plt.colorbar()

    # get the axis
    ax = plt.gca()

    # set minor ticks
    ax.set_xticks(np.arange(-.5, (width), 1), minor=True)
    ax.set_yticks(np.arange(-.5, (height), 1), minor=True)

    # add dridlines based on minor ticks
    ax.grid(which='minor', color='w', linestyle='-', linewidth=2)

    plt.xticks([])
    plt.yticks([])

    # compute cumulative sum of individual categories to match color schemes between chart and legend
    values_cumsum = np.cumsum(values)
    total_values = values_cumsum[len(values_cumsum) - 1]

    # create legend
    legend_handles = []
    for i, category in enumerate(categories):
        if value_sign == '%':
            label_str = category + ' (' + str(values[i]) + value_sign + ')'
        else:
            label_str = category + ' (' + value_sign + str(values[i]) + ')'

        color_val = colormap(float(values_cumsum[i]) / total_values)
        legend_handles.append(mpatches.Patch(color=color_val, label=label_str))

    # add legend to chart
    plt.legend(
        handles=legend_handles,
        loc='lower center',
        ncol=len(categories),
        bbox_to_anchor=(0., -0.2, 0.95, .1)
    )


def stacked_plot(data):
    """
    条形堆积图，输入pandas dataframe
    :param data: pandas dataframe
    :return:
    """
    df = data
    columns = df.columns
    rows = ['%d case' % x for x in df.index]

    colors = plt.cm.BuPu(np.linspace(0, 0.5, len(rows)))
    n_rows = len(df)
    index = np.arange(len(columns)) + 0.3
    bar_width = 0.4
    y_offset = np.zeros(len(columns))
    cell_text = []

    for row in range(n_rows):
        plt.bar(index, df.iloc[row, :], bar_width, bottom=y_offset, color=colors[row])
        y_offset = y_offset + df.iloc[row, :]
        cell_text.append(['%1.1f' % x for x in y_offset])
    colors = colors[::-1]
    cell_text.reverse()

    the_table = plt.table(
        cellText=cell_text,
        rowLabels=rows,
        rowColours=colors,
        colLabels=columns,
        loc='bottom'
    )
    plt.subplots_adjust(left=0.2, bottom=0.2)
    plt.ylabel("Y amount")
    plt.xticks([])
    plt.savefig("/home/murphy/django/static/images/stat.png")


def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value


def set_axis_style(ax, labels):
    ax.get_xaxis().set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_xlim(0.25, len(labels) + 0.75)
    ax.set_xlabel('Sample name')


def violin_plot(data):
    fig, ax = plt.subplots(nrows=1, ncols=1)

    ax.set_title('Customized violin plot')
    parts = ax.violinplot(
        data, showmeans=False, showmedians=False,
        showextrema=False)

    for pc in parts['bodies']:
        pc.set_facecolor('#D43F3A')
        pc.set_edgecolor('black')
        pc.set_alpha(1)

    quartile1, medians, quartile3 = np.percentile(data, [25, 50, 75], axis=1)
    whiskers = np.array([
        adjacent_values(sorted_array, q1, q3)
        for sorted_array, q1, q3 in zip(data, quartile1, quartile3)])
    whiskersMin, whiskersMax = whiskers[:, 0], whiskers[:, 1]

    inds = np.arange(1, len(medians) + 1)
    ax.scatter(inds, medians, marker='o', color='white', s=30, zorder=3)
    ax.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
    ax.vlines(inds, whiskersMin, whiskersMax, color='k', linestyle='-', lw=1)

    # set style for the axes
    # labels = ['A', 'B', 'C', 'D']
    # set_axis_style(ax, labels)

    plt.subplots_adjust(bottom=0.15, wspace=0.05)
    plt.savefig("/home/murphy/django/static/images/stat.png")


def stacked_bar_chart(ax, data):
    p1 = ax.bar(ind, df1.loc[:, 'M_mean'], width, yerr=df1.loc[:, 'M_std'])
    p2 = ax.bar(ind, df1.loc[:, 'F_mean'], width, bottom=df1.loc[:, 'M_mean'], yerr=df1.loc[:, 'F_std'])


def grouped_bar_chart(ax, data):
    pass


def auto_label(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(
            "{}".format(height),
            xy=(rect.get_x() + rect.get_width()/2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha='center',
            va='bottom'
        )


if __name__ == "__main__":
    fig, ax = plt.subplots(2, 1, sharex=True)
    t = np.arange(0.0, 2.0, 0.01)
    s1 = np.sin(2*np.pi*t)
    s2 = np.sin(4*np.pi*t)
    my_plotter(ax[0], t, s1, {})
    my_plotter(ax[1], t, s2, {'marker': 's'})
    # sf = "/home/murphy/stats/statistical/school.txt"
    # df = pd.read_csv(sf, sep=',')
    # d = {}
    # for sch in range(len(df)):
    #     name = df.iloc[sch, 0]
    #     d[name] = {
    #         'M': [np.random.randint(40, 80, np.random.randint(40, 50, 1))],
    #         'F': [np.random.randint(40, 70, np.random.randint(40, 50, 1))]
    #     }
    #
    # for k, v in d.items():
    #     d[k]['M_mean'] = np.int(np.mean(d[k]['M']))
    #     d[k]['F_mean'] = np.int(np.mean(d[k]['F']))
    #     d[k]['M_std'] = np.int(np.std(d[k]['M']))
    #     d[k]['F_std'] = np.int(np.std(d[k]['F']))
    # df = pd.DataFrame(d)
    # df = df.T
    # df1 = df.loc[:, ['F_mean', 'F_std', 'M_mean', 'M_std']]
    # N = len(df1)
    # ind = np.arange(N)
    # width = 0.35
    # rects1 = ax.bar(ind-width/2, df1.loc[:, 'M_mean'], width, label='Male')
    # rects2 = ax.bar(ind+width/2, df1.loc[:, 'F_mean'], width, label='Female')
    # # stacked_bar_chart(ax, df1)
    # plt.ylabel("Score")
    # plt.xlabel("School")
    # plt.xticks(ind, df1.index, rotation=-30)
    # plt.title("Bar Plot Example")
    # auto_label(rects1)
    # auto_label(rects2)
    # plt.legend()

    # fig, ax = plt.subplots(2, 1)
    # t = np.arange(0.0, 1.0+0.01, 0.01)
    # s = np.cos(2 * 2*np.pi * t)
    # t[41:60] = np.nan
    # my_plotter(ax[0], t, s, {'lw': 2, 'marker': 'o'})
    # plt.xlabel('time(s)')
    # plt.ylabel('voltage(V)')
    # plt.title('A sine wave with a gap of NaNs between 0.4 and 0.6')
    # plt.grid(True)
    #
    # t[0] = np.nan
    # t[-1] = np.nan
    # my_plotter(ax[1], t, s, {"marker": "o", 'lw': 2})
    plt.title('Also with NaN in first and last point')
    plt.xlabel('time(s)')
    plt.ylabel('voltage(V)')
    plt.title('Another')
    # plt.grid(True)
    plt.tight_layout()

    # f = "/home/murphy/stats/diabetes.csv"
    # df = pd.read_csv(f, sep=',')
    # stacked_plot(data=df)
    # np.random.seed(19680801)
    # data = [sorted(np.random.normal(0, std, 100)) for std in range(1, 9)]
    # violin_plot(data=data)

    # y = ["sugar"]
    # x = ["auxin", "insulin"]
    # fig, ax = plt.subplots(len(x), df.shape[1]-len(x), figsize=(14, 7))
    # if len(x) == 1:
    #     for j in range(df.shape[1]-len(x)):
    #         print(ax[j])
    #         print(j)
    # elif len(y) == 1:
    #     for j in range(len(x)):
    #         my_scatter(ax[j], df.loc[:, x[j]], df.loc[:, y[0]], {})
    # else:
    #     pass
    plt.savefig("/home/murphy/django/static/images/stat.png")
