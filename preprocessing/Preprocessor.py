import glob
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


class Preprocessor:
    def __init__(self, data=None, path=None, file_prefix=None):
        self.master_df = pd.DataFrame()
        if data is not None:
            self.master_df = data
        elif path is not None and file_prefix is not None:
            self.load_data(path, file_prefix)

    def load_data(self, path, file_prefix):
        pickle_files = glob.glob(f"{path}/{file_prefix}*.pkl")

        for file in pickle_files:
            df = pd.read_pickle(file)

            if type(df) == list:
                df = pd.DataFrame(df)

            df.columns = ['time', 'value']
            df['time'] = pd.to_datetime(df['time'], unit='s')

            try:
                self.master_df = pd.concat([self.master_df, df], axis=0, join='outer')
            except:
                print(f"An error occurred with file {file}")

    def fix_missing_data(self):
        return True

    def remove_duplicates(self):
        print("Found Duplicates: " + str(self.master_df.index.duplicated().any()))
        print("Remove Duplicate Rows")
        self.master_df = self.master_df.drop_duplicates(subset='time', keep='first')
        self.master_df = self.master_df.reset_index(drop=True)
        print("Found Duplicates: " + str(self.master_df.index.duplicated().any()))

    def normalize(self):
        scaler = MinMaxScaler()
        self.master_df['value'] = scaler.fit_transform(np.array(self.master_df['value']).reshape(-1, 1))

    def preprocess(self):
        self.fix_missing_data()
        self.remove_duplicates()
        self.normalize()
        return self.master_df
