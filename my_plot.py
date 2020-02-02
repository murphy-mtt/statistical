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


if __name__ == "__main__":
    fig, ax = plt.subplots(2, 1)
    t = np.arange(0.0, 1.0+0.01, 0.01)
    s = np.cos(2 * 2*np.pi * t)
    t[41:60] = np.nan
    my_plotter(ax[0], t, s, {'lw': 2, 'marker': 'o'})
    plt.xlabel('time(s)')
    plt.ylabel('voltage(V)')
    plt.title('A sine wave with a gap of NaNs between 0.4 and 0.6')
    plt.grid(True)

    t[0] = np.nan
    t[-1] = np.nan
    my_plotter(ax[1], t, s, {"marker": "o", 'lw': 2})
    plt.title('Also with NaN in first and last point')
    plt.xlabel('time(s)')
    plt.ylabel('voltage(V)')
    plt.title('Another')
    plt.grid(True)

    plt.tight_layout()
    # f = "/home/murphy/stats/diabetes.csv"
    # df = pd.read_csv(f, sep=',')
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
