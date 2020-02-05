import pandas as pd
import numpy as np
import os


class Chandler:
    def __init__(self, file_list):
        self.files = file_list
        self.column = ['癌种', '送检医生', '医院', '大区经理', '地区经理', '地区', '订单类型', '订单价格', '样本方式', '产品名称', '受检者', '创建订单时间', '送检单号', '销售姓名']

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
        return self.transfer_date(pd.concat(temp))


if __name__ == "__main__":
    d = "/home/murphy/stats/tables"
    for r, d, f in os.walk(d):
        files = [os.path.join(r, x) for x in f]
    monica = Chandler(file_list=files)
    print(monica.integration())
