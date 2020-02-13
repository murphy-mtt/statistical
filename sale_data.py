import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from matplotlib.ticker import MaxNLocator
from collections import namedtuple
import scipy.stats as stats


plt.rcParams['font.sans-serif'] = ['SimSun']
plt.rcParams['axes.unicode_minus'] = False


Student = namedtuple('Student', ['name', 'grade', 'gender'])
Score = namedtuple('Score', ['score', 'percentile'])


class Chandler:
    def __init__(self, file_list, period='month', region=["华东"]):
        """
        Whate is Chandler's job?
        :param file_list: Excel files to be analysed
        :param period: On what time span the analysis base on, default month
        :param region: On which region the analysis base on, default East of China
        """
        self.files = file_list
        self.column = ['癌种', '送检医生', '医院', '大区', '大区经理', '地区经理', '地区', '订单类型', '订单价格', '样本方式', '产品名称', '受检者', '创建订单时间', '送检单号', '销售姓名', '科室']
        self.period = period
        self.region = region

    def __callback(self, process, *args):
        method = getattr(self, process, None)
        if callable(method):
            method(*args)

    @staticmethod
    def read_file(file):
        return pd.read_excel(file, sheet_name='Sheet1')

    @staticmethod
    def transfer_date(df):
        df['创建订单时间'] = pd.to_datetime(df['创建订单时间'], format="%Y-%m-%d")
        return df

    def integration(self):
        temp = []
        for f in self.files:
            df = self.read_file(f)
            dd = df.loc[:, self.column]
            temp.append(dd)
        result = pd.concat(temp)
        result = self.transfer_date(result)
        self.make_category(result, ['样本方式', '订单类型'])
        return result

    @staticmethod
    def make_category(df, args):
        for i in args:
            df[i] = df[i].astype('category')

    def cancer_type_analysis(self):
        df_tmp = self.region_analysis(region=self.region)
        df_tmp.癌种.fillna(value='unknown', inplace=True)
        df_filled = df_tmp.pivot_table(df_tmp, index=['癌种'], columns=['地区']).fillna(value=0.0)
        return df_filled

    def region_analysis(self, region):
        df = self.integration()
        df_r = df[df['大区'].isin(region)]
        return df_r

    def date_analysis(self):
        if self.period not in ['month', 'quarter', 'day', 'week']:
            raise ValueError("Choose from ['month', 'quarter', 'day', 'week']")
        df_tmp = self.region_analysis(region=self.region)
        if self.period == "month":
            p = ["%s-%s" % (a.year, a.month) for a in df_tmp['创建订单时间']]
        elif self.period == "quarter":
            p = ["%s-%s" % (a.year, a.quarter) for a in df_tmp['创建订单时间']]
        elif self.period == "week":
            p = ["%s-%s" % (a.year, a.week) for a in df_tmp['创建订单时间']]
        elif self.period == "day":
            p = ["%s-%s" % (a.year, a.day) for a in df_tmp['创建订单时间']]
        else:
            p = df_tmp['创建订单时间']
        df_tmp['period'] = pd.to_datetime(p, format='%Y-%m')
        df_filled = df_tmp.pivot_table(df_tmp, index=['period'], columns=['地区']).fillna(value=0.0)
        return df_filled

    def client_analysis(self, client_type):
        client_type_list = ['doctor', 'hospital', 'department']
        if client_type not in client_type_list:
            raise ValueError("Choose from %s" % "/".join(client_type_list))
        df_tmp = self.region_analysis(region=self.region)
        df_tmp.送检医生.fillna(value='unknown', inplace=True)
        df_tmp.科室.fillna(value='unknown', inplace=True)
        df_tmp.医院.fillna(value='unknown', inplace=True)
        if client_type == "department":
            df_filled = df_tmp.pivot_table(df_tmp, index=["科室"], columns=["地区"]).fillna(value=0.0)
        elif client_type == "hospital":
            df_filled = df_tmp.pivot_table(df_tmp, index=["医院"], columns=["地区"]).fillna(value=0.0)
        else:
            df_filled = df_tmp.pivot_table(df_tmp, index=["送检医生"], columns=["地区"]).fillna(value=0.0)
        return df_filled

    def stack_plot(self, type, df):
        data_list = ['date', 'cancer', 'department']
        if type not in data_list:
            raise ValueError("Choose from %s" % ("/".join(data_list)))
        if type == "date":
            xtickslabel = df.index.strftime(date_format='%Y-%m')
        else:
            xtickslabel = df.index
        fig, ax = plt.subplots()
        x = np.arange(len(df.index))
        y = []
        for i in range(len(df.columns)):
            y.append(df.iloc[:, i])
        xticks = range(0, len(df.index), 1)
        ax.set_xticks(xticks)
        ax.stackplot(x, y, labels=df.columns)
        ax.set_xticklabels(xtickslabel, rotation=45)
        plt.title("{}区域销量（{}）".format(''.join(self.region), self.period))
        plt.legend(loc='upper left')
        plt.savefig("/home/murphy/django/static/images/stat.png")

    def grouped_bar(self, dataframe):
        """
        Ref: https://chrisalbon.com/python/data_visualization/matplotlib_grouped_bar_plot/
        :param dataframe:
        :return:
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        xticker, width = self.group_bar_ticker(len(dataframe.columns), len(dataframe.index))
        pos = list(range(len(df.index)))
        for i in range(len(dataframe.columns)):
            rect = ax.bar(
                [(p + width*i - width*len(dataframe.columns)/2.0) for p in pos],
                df.iloc[:, i],
                width,
                alpha=0.5,
                label=dataframe.columns[i]
            )
            self.autolabel(ax, rect)

        ax.set_xticks([p + 1.5 * width for p in pos])
        ax.set_xticklabels(dataframe.index)

        plt.xlim(min(pos) - width * len(dataframe.columns), max(pos) + width * len(dataframe.columns))
        plt.legend()

        plt.savefig("/home/murphy/django/static/images/stat.png")

    def stacked_bar(self, dataframe):
        dataframe.plot.bar(stacked=True, rot=45)
        plt.savefig("/home/murphy/django/static/images/stat.png")

    def pie_chart(self, data, label=None):
        fig, ax = plt.subplots()
        ax.pie(data, autopct='%1.1f%%', startangle=90, labels=data.index)
        ax.axis('equal')
        plt.legend()
        plt.title("Pie of Sales")
        plt.savefig("/home/murphy/django/static/images/stat.png")

    def region_distribution(self, region=None):
        df = self.integration()
        df_bp = pd.pivot_table(df, index=['地区', '销售姓名'], aggfunc=np.sum).fillna(0)
        counts1 = []
        lables1 = df_bp.index.levels[0].values.tolist()
        region_index = lables1.index(region)
        item = lables1.pop(region_index)
        lables1.insert(0, item)
        for i in lables1:
            counts1.append(df_bp.loc[i].sum()[0])
        counts1 = self.percent_convert(np.array(counts1))
        count2 = self.percent_convert(df_bp.loc[region].values)
        labels2 = df_bp.loc[region].index.tolist()
        self.bar_of_pie(counts1, lables1, count2, labels2)

    def bar_of_pie(self, ratios1, label1, ratios2, label2):
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.subplots_adjust(wspace=0)

        # pie chart parameters
        explode = np.zeros(len(ratios1))
        explode[0] = 0.1
        # rotate so that first wedge is split by the x-axis
        angle = -180 * ratios1[0]
        ax1.pie(ratios1, autopct='%1.1f%%', startangle=angle,
                labels=label1, explode=explode)

        # bar chart parameters

        xpos = 0
        bottom = 0
        width = .2
        # colors = [[.1, .3, .5], [.1, .3, .3], [.1, .3, .7], [.1, .3, .9]]

        for j in range(len(ratios2)):
            height = ratios2[j]
            ax2.bar(xpos, height, width, bottom=bottom)
            ypos = bottom + ax2.patches[j].get_height() / 2
            bottom += height
            ax2.text(xpos, ypos, "%d%%" % (ax2.patches[j].get_height() * 100),
                     ha='center')


        ax2.legend(label2)
        ax2.axis('off')
        ax2.set_xlim(- 2.5 * width, 2.5 * width)

        # use ConnectionPatch to draw lines between the two plots
        # get the wedge data
        theta1, theta2 = ax1.patches[0].theta1, ax1.patches[0].theta2
        center, r = ax1.patches[0].center, ax1.patches[0].r
        bar_height = sum([item.get_height() for item in ax2.patches])

        # draw top connecting line
        x = r * np.cos(np.pi / 180 * theta2) + center[0]
        y = np.sin(np.pi / 180 * theta2) + center[1]
        con = ConnectionPatch(xyA=(- width / 2, bar_height), xyB=(x, y),
                              coordsA="data", coordsB="data", axesA=ax2, axesB=ax1)
        con.set_color([0, 0, 0])
        con.set_linewidth(1)
        ax2.add_artist(con)

        # draw bottom connecting line
        x = r * np.cos(np.pi / 180 * theta1) + center[0]
        y = np.sin(np.pi / 180 * theta1) + center[1]
        con = ConnectionPatch(xyA=(- width / 2, 0), xyB=(x, y), coordsA="data",
                              coordsB="data", axesA=ax2, axesB=ax1)
        con.set_color([0, 0, 0])
        ax2.add_artist(con)
        con.set_linewidth(1)

        plt.title('{}销量分布'.format("/".join(self.region)))

        plt.savefig("/home/murphy/django/static/images/stat.png")

    def individual_sale_data(self):
        """
        区域销售销量分布
        :return:
        """
        df = self.integration()
        dfs = pd.pivot_table(df, index=['销售姓名'], aggfunc=([np.sum, np.mean]))
        dfs['order'] = np.arange(len(dfs.index))
        r = (dfs.loc[:, 'sum'] / 1000) / 2 * np.pi
        x_loc = r.iloc[:, 0] + np.arange(len(dfs))
        dfs['r'] = r
        fig, ax = plt.subplots(figsize=(12, 9))
        ax.scatter(dfs.index, dfs.loc[:, 'mean'], s=dfs.loc[:, 'sum'] / 1000, alpha=0.5)
        ax.annotate('te', xy=(x_loc[2], dfs.iloc[2, 1]))
        for i in range(len(dfs)):
            ax.annotate(
                dfs.iloc[i, 0],
                xy=(dfs.iloc[i, 2], dfs.iloc[i, 1]),
            )
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        plt.xlabel("销售人员")
        plt.ylabel("销量均值")
        plt.grid(True)
        plt.title('{}销售销量分布'.format("/".join(self.region)))
        plt.savefig("/home/murphy/django/static/images/stat.png")

    def bubble(self, dataframe):
        pass

    def get_product_list(self):
        df_ps = pd.pivot_table(self.integration(), index=['产品名称'], columns=['销售姓名'], aggfunc=sum).fillna(0)
        product_name = df_ps.index.values.tolist()
        testMeta = []
        for i in range(len(product_name)):
            testMeta.append("")
        return product_name, testMeta

    def sale_ranking(self, s):
        df = self.integration()
        df_ps = pd.pivot_table(df, index=['产品名称'], columns=['销售姓名'], aggfunc=sum).fillna(0)
        result = {}

        for sale in df_ps.columns:
            result[sale[1]] = {}
            scoreList = []
            percList = []
            result[sale[1]]['s'] = []
            result[sale[1]]['p'] = []
            for product in df_ps.index:
                product_total_sale = df_ps.loc[product, :].values.tolist()
                product_individual_sale = df_ps.loc[product, sale]
                percentile = stats.percentileofscore(product_total_sale, product_individual_sale)
                scoreList.append(product_individual_sale)
                percList.append(percentile)
                result[sale[1]]['s'] = scoreList
                result[sale[1]]['p'] = percList

        testNames, testMeta = self.get_product_list()
        student = Student(s, 2, 'boy')
        cohort_size = len(df_ps.columns.levels[1])
        scores = dict(zip(testNames, (Score(v, p) for v, p in zip(result[s]['s'], result[s]['p']))))
        arts = self.plot_student_results(student, scores, cohort_size)
        plt.savefig("/home/murphy/django/static/images/stat.png")

    def plot_student_results(self, student, scores, cohort_size):
        testNames, testMeta = self.get_product_list()
        for i in range(len(testNames)):
            testMeta.append("")

        #  create the figure
        fig, ax1 = plt.subplots(figsize=(14, 7))
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

        scoreLabels = [self.format_score(scores[k].score, k) for k in testNames]

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
        ax1.set_xlabel(xlabel.format(grade=self.attach_ordinal(student.grade),
                                     gender=student.gender.title(),
                                     cohort_size=cohort_size))

        rect_labels = []
        # Lastly, write in the ranking inside each bar to aid in interpretation
        for rect in rects:
            # Rectangle widths are already integer-valued but are floating
            # type, so it helps to remove the trailing decimal point and 0 by
            # converting width to int type
            width = int(rect.get_width())

            rankStr = self.attach_ordinal(width)
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
        ax2.fmt_ydata = self.format_ycursor
        # return all of the artists created

        # return {'fig': fig,
        #         'ax': ax1,
        #         'ax_right': ax2,
        #         'bars': rects,
        #         'perc_labels': rect_labels}
        print("test")
        plt.savefig("/home/murphy/django/static/images/stat.png")

    @staticmethod
    def group_bar_ticker(cols, rows, gap=0.2):
        width = (1 - gap)/cols
        tickers = []
        x = np.arange(rows)
        for j in x:
            tmp = []
            for i in np.arange(cols):
                t = x[j] - width * rows / 2 + width / 2 + i * width
                tmp.append(t)
            tickers.append(tmp)
        return tickers, width

    @staticmethod
    def group_bar_ticker_modify(dataframe, x_axis='columns', gap=0.2):
        if x_axis == "columns":
            nticks, nbars = len(dataframe.columns), len(dataframe.index)
        else:
            nbars, nticks = len(dataframe.columns), len(dataframe.index)
        width = (1 - gap) / nbars
        tickers = []
        return [p + width*i for p in pos]

    @staticmethod
    def autolabel(ax, rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    @staticmethod
    def percent_convert(ser):
        return ser / float(ser.sum())

    @staticmethod
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

    def format_score(self, scr, test):
        """
        Build up the score labels for the right Y-axis by first
        appending a carriage return to each string and then tacking on
        the appropriate meta information (i.e., 'laps' vs 'seconds'). We
        want the labels centered on the ticks, so if there is no meta
        info (like for pushups) then don't add the carriage return to
        the string
        """
        testNames, testMeta = self.get_product_list()
        # md = testMeta[test]
        md = ""
        if md:
            return '{0}\n{1}'.format(scr, md)
        else:
            return scr

    @staticmethod
    def format_ycursor(y):
        y = int(y)
        if y < 0 or y >= len(testNames):
            return ''
        else:
            return testNames[y]


class Picasso:
    def __init__(self, nrows=1, ncols=1, data=None):
        self.fig, self.ax = plt.subplots(nrows, ncols)
        self.data = data

    def plot(self, ax, x=None, y=None, para=None):
        return ax.plot(x, y, **para)


if __name__ == "__main__":
    d = "/home/murphy/stats/tables"
    for r, d, f in os.walk(d):
        files = [os.path.join(r, x) for x in f]
    monica = Chandler(file_list=files, period='quarter')
    monica.sale_ranking(s='施燕琳')

    # monica.region_distribution(region='江浙沪皖I')
    # df_total = monica.integration()
    # df = monica.date_analysis().round(2)
    # t = df_total.groupby('地区').sum()
    # monica.stack_plot(type='date', df=df)
    # monica.pie_chart(data=t)
    # labels = ['G1', 'G2', 'G3', 'G4', 'G5']
    # men_means = [20, 34, 30, 35, 27]
    # women_means = [25, 32, 34, 20, 25]
    # df = pd.DataFrame([men_means, women_means], index=['Male', 'Female'], columns=labels)
    # monica.grouped_bar_modify(df)
