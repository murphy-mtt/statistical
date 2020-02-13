import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches  # needed for waffle Charts
import mpl_toolkits.axisartist as axisartist
import matplotlib.dates as mdates
from datetime import datetime
from matplotlib.ticker import MaxNLocator
from collections import namedtuple


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


def hist(ax, data, para):
    return ax.hist(data, **para)


def hist2d(ax, data1, data2, para):
    return ax.hist2d(data1, data2, **para)


def setup_axes(fig, rect):
    ax = axisartist.Subplot(fig, rect)
    fig.add_axes(ax)

    ax.set_ylim(-0.1, 1.5)
    ax.set_yticks([0, 1])

    ax.axis[:].set_visible(False)

    ax.axis["x"] = ax.new_floating_axis(1, 0.5)
    ax.axis["x"].set_axisline_style("->", size=1.5)

    return ax


def get_github_data():
    try:
        import urllib.request
        import json
        url = 'https://api.github.com/repos/matplotlib/matplotlib/releases'
        url += '?per_page=100'
        data = json.loads(urllib.request.urlopen(url).read().decode())
        dates = []
        names = []
        for item in data:
            if 'rc' not in item['tag_name'] and 'b' not in item['tag_name']:
                dates.append(item['published_at'].split("T")[0])
                names.append(item['tag_name'])
        dates = [datetime.strptime(d, "%Y-%m-%d") for d in dates]
    except Exception:
        names = ['v2.2.4', 'v3.0.3', 'v3.0.2', 'v3.0.1', 'v3.0.0', 'v2.2.3',
                 'v2.2.2', 'v2.2.1', 'v2.2.0', 'v2.1.2', 'v2.1.1', 'v2.1.0',
                 'v2.0.2', 'v2.0.1', 'v2.0.0', 'v1.5.3', 'v1.5.2', 'v1.5.1',
                 'v1.5.0', 'v1.4.3', 'v1.4.2', 'v1.4.1', 'v1.4.0']

        dates = ['2019-02-26', '2019-02-26', '2018-11-10', '2018-11-10',
                 '2018-09-18', '2018-08-10', '2018-03-17', '2018-03-16',
                 '2018-03-06', '2018-01-18', '2017-12-10', '2017-10-07',
                 '2017-05-10', '2017-05-02', '2017-01-17', '2016-09-09',
                 '2016-07-03', '2016-01-10', '2015-10-29', '2015-02-16',
                 '2014-10-26', '2014-10-18', '2014-08-26']
        dates = [datetime.strptime(d, "%Y-%m-%d") for d in dates]

    return names, dates


def timeline(names, dates):
    levels = np.tile([-5, 5, -3, 3, -1, 1], int(np.ceil(len(dates)/6)))[:len(dates)]
    fix, ax = plt.subplots(figsize=(8.8, 4), constrained_layout=True)
    ax.set(title='Example')
    markerline, stemline, baseline = ax.stem(
        dates,  # 当仅给出levels时timeline产生图示，但是间距相等
        levels,
        linefmt='C3-',
        basefmt='k-',
    )
    plt.setp(markerline, mec='k', mfc='w', zorder=3)
    markerline.set_ydata(np.zeros(len(dates)))  # 删掉默认的圈

    vert = np.array(['top', 'bottom'])[(levels>0).astype(int)]
    for d, l, r, va in zip(dates, levels, names, vert):
        ax.annotate(r, xy=(d, l), va=va, ha='right', xytext=(-3, np.sign(l)*3), textcoords="offset points")

    ax.get_xaxis().set_major_locator(mdates.MonthLocator(interval=4))
    ax.get_xaxis().set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')

    ax.get_yaxis().set_visible(False)

    for spine in ['left', 'top', 'right']:
        ax.spines[spine].set_visible(False)

    ax.margins(y=0.1)

    plt.savefig("/home/murphy/django/static/images/stat.png")


def attach_ordinal(num):
    """helper function to add ordinal string to integers

    1 -> 1st
    56 -> 56th
    """
    suffixes = {str(i): v
                for i, v in enumerate(['th', 'st', 'nd', 'rd', 'th',
                                       'th', 'th', 'th', 'th', 'th'])}

    v = str(num)
    # special case early teens
    if v in {'11', '12', '13'}:
        return v + 'th'
    return v + suffixes[v[-1]]


def format_score(scr, test):
    """
    Build up the score labels for the right Y-axis by first
    appending a carriage return to each string and then tacking on
    the appropriate meta information (i.e., 'laps' vs 'seconds'). We
    want the labels centered on the ticks, so if there is no meta
    info (like for pushups) then don't add the carriage return to
    the string
    """
    md = testMeta[test]
    if md:
        return '{0}\n{1}'.format(scr, md)
    else:
        return scr


def format_ycursor(y):
    y = int(y)
    if y < 0 or y >= len(testNames):
        return ''
    else:
        return testNames[y]


def plot_student_results(student, scores, cohort_size):
    #  create the figure
    fig, ax1 = plt.subplots(figsize=(9, 7))
    fig.subplots_adjust(left=0.115, right=0.88)
    fig.canvas.set_window_title('Eldorado K-8 Fitness Chart')

    pos = np.arange(len(testNames))

    rects = ax1.barh(pos, [scores[k].percentile for k in testNames],
                     align='center',
                     height=0.5,
                     tick_label=testNames)

    ax1.set_title(student.name)

    ax1.set_xlim([0, 100])
    ax1.xaxis.set_major_locator(MaxNLocator(11))
    ax1.xaxis.grid(True, linestyle='--', which='major',
                   color='grey', alpha=.25)

    # Plot a solid vertical gridline to highlight the median position
    ax1.axvline(50, color='grey', alpha=0.25)

    # Set the right-hand Y-axis ticks and labels
    ax2 = ax1.twinx()

    scoreLabels = [format_score(scores[k].score, k) for k in testNames]

    # set the tick locations
    ax2.set_yticks(pos)
    # make sure that the limits are set equally on both yaxis so the
    # ticks line up
    ax2.set_ylim(ax1.get_ylim())

    # set the tick labels
    ax2.set_yticklabels(scoreLabels)

    ax2.set_ylabel('Test Scores')

    xlabel = ('Percentile Ranking Across {grade} Grade {gender}s\n'
              'Cohort Size: {cohort_size}')
    ax1.set_xlabel(xlabel.format(grade=attach_ordinal(student.grade),
                                 gender=student.gender.title(),
                                 cohort_size=cohort_size))

    rect_labels = []
    # Lastly, write in the ranking inside each bar to aid in interpretation
    for rect in rects:
        # Rectangle widths are already integer-valued but are floating
        # type, so it helps to remove the trailing decimal point and 0 by
        # converting width to int type
        width = int(rect.get_width())

        rankStr = attach_ordinal(width)
        # The bars aren't wide enough to print the ranking inside
        if width < 40:
            # Shift the text to the right side of the right edge
            xloc = 5
            # Black against white background
            clr = 'black'
            align = 'left'
        else:
            # Shift the text to the left side of the right edge
            xloc = -5
            # White on magenta
            clr = 'white'
            align = 'right'

        # Center the text vertically in the bar
        yloc = rect.get_y() + rect.get_height() / 2
        label = ax1.annotate(rankStr, xy=(width, yloc), xytext=(xloc, 0),
                            textcoords="offset points",
                            ha=align, va='center',
                            color=clr, weight='bold', clip_on=True)
        rect_labels.append(label)

    # make the interactive mouse over give the bar title
    ax2.fmt_ydata = format_ycursor
    # return all of the artists created
    return {'fig': fig,
            'ax': ax1,
            'ax_right': ax2,
            'bars': rects,
            'perc_labels': rect_labels}


if __name__ == "__main__":
    np.random.seed(42)

    Student = namedtuple('Student', ['name', 'grade', 'gender'])
    sale = namedtuple('Sale', ['name', 'team', 'gender'])
    Score = namedtuple('Score', ['score', 'percentile'])


    # GLOBAL CONSTANTS
    testNames = ['Pacer Test', 'Flexed Arm\n Hang', 'Mile Run', 'Agility',
                 'Push Ups']
    product_names = ['肺癌专项版90基因检测',
                    'HapOnco 肿瘤605基因临床用药守护计划（再次检测）',
                    'HapOnco 肺癌专项版90基因检测',
                    'HapOnco 肺癌基础版11基因检测',
                    '肺癌耐药检测(T790M)',
                    '结直肠癌临床用药28基因检测',
                    '肿瘤临床用药451基因检测',
                    'HapOnco 肿瘤605基因临床用药守护计划（首次检测）',
                    'WESPlus',
                    '遗传性肿瘤58基因检测',
                    'HapOnco 肺癌临床用药11基因检测（赠送PD-L1和MSI）',
                    '消化道肿瘤51基因检测',
                    'HapOnco 肿瘤临床用药605基因（单样本单次检测）',
                    'HapOnco\xa0 消化道肿瘤51基因检测',
                    '肺癌靶向用药31基因检测',
                    'HapOnco 肿瘤临床用药605基因（双样本单次检测）',
                    'HapOnco WESPlus 单次检测',
                    'HapOnco 肺癌临床用药90基因检测（赠送PD-L1和MSI）',
                    '肺癌基础版11基因检测',
                    '实体瘤605基因检测']
    testMeta = dict(zip(testNames, ['laps', 'sec', 'min:sec', 'sec', '']))
    productMeta = dict(zip(testNames, ['', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '']))

    student = Student('Johnny Doe', 2, 'boy')
    scores = dict(zip(testNames,
                      (Score(v, p) for v, p in
                       zip(['7', '48', '12:52', '17', '14'], np.round(np.random.uniform(0, 1, len(testNames)) * 100, 0)))))
    cohort_size = 62  # The number of other 2nd grade boys

    arts = plot_student_results(student, scores, cohort_size)

    # fig = plt.figure(figsize=(3, 2.5))
    # fig.subplots_adjust(top=0.8)
    # ax1 = setup_axes(fig, "111")
    #
    # ax1.axis["x"].set_axis_direction("left")
    # mu, sigma = 100, 15
    # mu2, sigma2 = 130, 20
    # x = mu + sigma * np.random.randn(10000)
    # x1 = mu2 + sigma2 * np.random.randn(10000)
    # hist2d(ax, x, x1, {
    #     'bins': 100,
    #     'facecolor': 'g',
    # })
    #
    # t = np.arange(0.0, 2.0, 0.01)
    # s1 = np.sin(2*np.pi*t)
    # s2 = np.sin(4*np.pi*t)
    # s3 = np.exp(-t)
    # my_plotter(ax[0], t, s1, {})
    # my_plotter(ax[1], t, s2, {})
    # my_plotter(ax[2], t, s3, {})
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
    # np.random.seed(1985)
    #
    # fig, ax = plt.subplots(2, 2)
    # data1 = np.random.random([6, 50])
    # data2 = np.random.gamma(1, size=[60, 50])
    # colors1 = ['C{}'.format(i) for i in range(6)]
    # colors2 = 'black'
    #
    # lineoffsets1 = np.array([-15, -3, 1, 1.5, 6, 8])
    # lineoffsets2 = 1
    # linelengths1 = [5, 2, 1, 1, 3, 1.5]
    # linelengths2 = 1
    #
    # ax[0][0].eventplot(data1, colors=colors1, lineoffsets=linelengths1, linelengths=linelengths1)
    # ax[1][0].eventplot(
    #     data1, colors=colors1, lineoffsets=linelengths1, linelengths=linelengths1, orientation='vertical'
    # )
    # ax[0][1].eventplot(data2, colors=colors2, lineoffsets=lineoffsets2, linelengths=linelengths2)
    # ax[1][1].eventplot(data2[0:10], colors=colors2, lineoffsets=lineoffsets2, linelengths=linelengths2)
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
    # plt.title('Also with NaN in first and last point')
    # plt.xlabel('time(s)')
    # plt.ylabel('voltage(V)')
    # plt.title('Another')
    # ax[1].set_title("test")
    # plt.setp(ax[2].get_xticklabels(), fontsize=6)
    # plt.setp(ax[1].get_xticklabels(), visible=False)
    # # plt.grid(True)
    # plt.tight_layout()

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
