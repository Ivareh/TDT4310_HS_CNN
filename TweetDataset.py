from torch.utils.data import Dataset
from typing import Callable
from sklearn.model_selection import train_test_split

class TweetDatasetSplitter(Dataset):
    def __init__(self, dataframe, train_pct: float, val_pct: float, test_pct: float):
        self.train_pct = train_pct
        self.val_pct = val_pct
        self.test_pct = test_pct
        self.dataframe = dataframe

        self.train_data = None
        self.val_data = None
        self.test_data = None

    def __len__(self):
        return len(self.dataframe)

    def split_data(self, preprocess_fn: Callable = None):
        if preprocess_fn is not None:
            self.dataframe = preprocess_fn(self.dataframe)

        print("DATAFRAME TYPE " + str(type(self.dataframe)))

        # Split data into training and test sets
        train_data, test_data = train_test_split(self.dataframe, test_size=self.test_pct)

        # Further split training data into training and validation sets
        train_data, val_data = train_test_split(train_data, test_size=self.val_pct/(1-self.test_pct))

        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        print("TRAIN DATA TYPE ", str(type(self.train_data)))

    def get_sets(self):
        self.split_data()
        return self.train_data, self.val_data, self.test_data
