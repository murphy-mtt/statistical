import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class Chandler:
    def __init__(self, file_list, period='month', region=["华东"]):
        """
        Whate is Chandler's job?
        :param file_list: Excel files to be analysed
        :param period: On what time span the analysis base on, default month
        :param region: On which region the analysis base on, default East of China
        """
        self.files = file_list
        self.column = ['癌种', '送检医生', '医院', '大区', '大区经理', '地区经理', '地区', '订单类型', '订单价格', '样本方式', '产品名称', '受检者', '创建订单时间', '送检单号', '销售姓名']
        self.period = period
        self.region = region

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
        pass

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

    def stack_plot(self):
        df = self.date_analysis()
        fig, ax = plt.subplots()
        return None

    def my_plotter(ax, data1, data2, para):
        out = ax.plot(data1, data2, **para)
        return out


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
    monica = Chandler(file_list=files)
    df = monica.stack_plot()
