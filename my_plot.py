import numpy as np
import matplotlib.pyplot as plt


def my_plotter(ax, data1, data2, para):
    out = ax.plot(data1, data2, **para)
    return out


if __name__ == "__main__":
    data1, data2, data3, data4 = np.random.randn(4, 100)
    fig, ax = plt.subplots(1, 2, figsize=(14, 7))
    my_plotter(ax[0], data1, data2, {"marker": 'x', 'color': 'red'})
    my_plotter(ax[0], data3, data4, {"marker": "+"})
    ax[0].set_title("Plot1", fontsize=18)
    ax[0].set_xlabel('xlabel', fontsize=18, fontfamily='sans-serif', fontstyle='italic')
    ax[0].set_ylabel('ylabel', fontsize='x-large', fontstyle='oblique')
    ax[0].minorticks_on()
    ax[0].grid(which='minor', axis='both')
    ax[0].legend(('plot1', 'plot2'))
    my_plotter(ax[1], data1, data3, {"marker": "o"})
    plt.savefig("/home/murphy/django/static/images/stat.png")
