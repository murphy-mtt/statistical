import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']


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

    def grouped_bar(self, type, dataframe):
        """
        Discard
        :param type:
        :param dataframe:
        :return:
        """
        data_list = ['date', 'cancer', 'department']
        if type not in data_list:
            raise ValueError("Choose from %s" % ("/".join(data_list)))
        if type == "date":
            xtickslabel = df.index.strftime(date_format='%Y-%m')
        else:
            xtickslabel = df.index
        fig, ax = plt.subplots()
        xticker, width = self.group_bar_ticker(len(dataframe.columns), len(dataframe.index))
        for i in range(len(dataframe.index)):
            rect = ax.bar(xticker[i], dataframe.iloc[i, :], width, label=dataframe.index[i])
            self.autolabel(ax, rect)
        xticks = range(0, len(df.index), 1)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xtickslabel, rotation=45)
        plt.legend()
        plt.savefig("/home/murphy/django/static/images/stat.png")

    def grouped_bar_modify(self, dataframe):
        fig, ax = plt.subplots(figsize=(12, 8))
        xticker, width = self.group_bar_ticker(len(dataframe.columns), len(dataframe.index))
        pos = list(range(len(df.index)))
        for i in range(len(dataframe.columns)):
            rect = ax.bar([(p + width*i - width*len(dataframe.columns)/2.0) for p in pos], df.iloc[:, i], width, alpha=0.5, label=dataframe.columns[i])
            self.autolabel(ax, rect)

        ax.set_xticks([p + 1.5 * width for p in pos])
        ax.set_xticklabels(dataframe.index)

        plt.xlim(min(pos) - width * len(dataframe.columns), max(pos) + width * len(dataframe.columns))
        plt.legend()
        plt.savefig("/home/murphy/django/static/images/stat.png")

    def stackplot(self, ax, x, y, para):
        return ax.stackplot(x, y, **para)

    def plot(self, ax, x, y, para):
        return ax.plot(x, y, **para)

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
    df = monica.cancer_type_analysis().round(2)
    # # monica.stack_plot(type='date', df=df, figure_type="plot")
    monica.grouped_bar_modify(dataframe=df)
    # labels = ['G1', 'G2', 'G3', 'G4', 'G5']
    # men_means = [20, 34, 30, 35, 27]
    # women_means = [25, 32, 34, 20, 25]
    # df = pd.DataFrame([men_means, women_means], index=['Male', 'Female'], columns=labels)
    # monica.grouped_bar_modify(df)
