from preprocessing import Preprocessor

def main():
    # Load data from a DataFrame or a path with a file prefix
    # Example: Using a path and file prefix
    path = "path/to/your/files"
    file_prefix = "avgC"
    preprocessor = Preprocessor(path=path, file_prefix=file_prefix)

    # Preprocess the data
    processed_data = preprocessor.preprocess()

    # Display the preprocessed data
    print(processed_data.head())

if __name__ == "__main__":
    main()
