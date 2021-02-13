import os
import pandas as pd
from constants import COLUMN_DROP, WINE_DATA, DROP_NA


class GetData:
    def __init__(self, opertaing_systemt,nrows=None):
        if opertaing_system not in ['mac','windows']:
             raise ValueError(f"{opertaing_system} please provide mac or windows")
        self.opertaing_system = opertaing_system
        self.all_data = self.__import_raw_data(self.opertaing_system)
        self.nrows = nrows

    def __import_raw_data(self, operating_system):
        """generate a dictionary of raw dataframes

        parameters
        -----------
        type of operating system used windows or mac
        default mac

        """
        if operating_system == "mac":
            base_file_path = r"/Users/{}/Desktop/data".format(os.getlogin())
        if operating_system == "windows":
            base_file_path = r"C:\Users\{}\Desktop\data".format(os.getlogin())
        df_dict = dict()
        for file in os.listdir(base_file_path):
            if file.endswith(".csv"):
                df_dict[file.split(".")[0]] = pd.read_csv(
                    os.path.join(base_file_path, file)
                )
        return df_dict

    def select_data_type(self, data_key=WINE_DATA):
        """get data within dictionary

        parameters
        -----------
        data_key, the filename within the data folder

        raises
        -----------
        if data key not in data dictionary

        """
        if data_key not in self.all_data.keys():
            raise ValueError(f"data key does not match {list(data.keys())}")
            if self.nrows is not None & data_key == WINE_DATA:
                return self.all_data.get(data_key).iloc[:self.nrows,:]
            return self.all_data.get(data_key)

    def clean_data(self):
        wine_df = self.select_data_type()
        wine_df = wine_df.drop(columns=COLUMN_DROP)
        # remove duplicates based on unique values in the df
        wine_df = wine_df.drop_duplicates(subset=["description", "title"])
        wine_df["points"] = pd.cut(wine_df["points"], bins=5, labels=[1, 2, 3, 4, 5])
        return wine_df


if __name__ == "__main__":
    d = GetData("mac")
    df = d.clean_data()
