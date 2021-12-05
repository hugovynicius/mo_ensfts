import pandas as pd


class DataSetUtil():
    def __init__(self):
        # debug attributes
        self.name = 'DataSet Util'
        self.shortname = 'dfutil'

    def get_samples_data_frame(self, data, perc):
        df = pd.DataFrame(data)
        df_sample = df.head(int(len(df) * perc))
        df_complement = df.iloc[max(df_sample.index):]
        return df_sample, df_complement, max(df_sample.index)