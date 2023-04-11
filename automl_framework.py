from ModelTrainer import ModelTrainer
from preprocessing import Preprocessor


class AutoMLFramework:
    """
         AutoMLFramework is a class that provides a high-level interface for
         automating the process of applying machine learning algorithms to
         univariate time-series forecasting tasks. It simplifies the process of
         preprocessing, model selection, hyperparameter tuning, and evaluation.

         Args:
         data (pandas.DataFrame, optional): A DataFrame containing the time-series
                                            data. Must have columns 'time' and 'value'.
                                            Defaults to None.

         path (str, optional): A string representing the path to the directory
                               containing data files. Defaults to None.

         file_prefix (str, optional): A string representing the prefix of the data
                                      files in the specified directory. The class
                                      will load all files with names starting with
                                      this prefix. Defaults to None.

         Usage:
         # Using a DataFrame
         automl = AutoMLFramework(data=dataframe)

         # Using a path and file prefix
         automl = AutoMLFramework(path="path/to/your/files", file_prefix="avgC")

         Methods:
         preprocess_data(): Preprocesses the input data, including handling missing
                           values, removing duplicates, and normalization.


         Example usage:
            def main():
                # Using the library with a DataFrame
                dataframe = pd.read_csv('your_data.csv')
                automl = AutoMLFramework(data=dataframe)
                automl.preprocess_data()
                print(automl.processed_data.head())

                # Or, using the library with a path and file prefix
                path = "path/to/your/files"
                file_prefix = "avgC"
                automl = AutoMLFramework(path=path, file_prefix=file_prefix)
                automl.preprocess_data()
                print(automl.processed_data.head())
     """


def __init__(self, data=None, path=None, file_prefix=None):
    self.data = data
    self.path = path
    self.file_prefix = file_prefix
    self.processed_data = None
    self.trained_model = None
    self.mean_cv_score = None

def preprocess_data(self):
    preprocessor = Preprocessor(data=self.data, path=self.path, file_prefix=self.file_prefix)
    self.processed_data = preprocessor.preprocess()

def train_and_tune_model(self, model_type, hyperparameters=None):
    if self.processed_data is None:
        raise ValueError("Data has not been preprocessed. Call preprocess_data() first.")

    model_trainer = ModelTrainer(data=self.processed_data, model_type=model_type, hyperparameters=hyperparameters)
    self.trained_model, self.mean_cv_score = model_trainer.train_and_tune()
