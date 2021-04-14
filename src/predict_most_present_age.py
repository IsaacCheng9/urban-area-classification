"""
An implementation of a decision tree classifier to predict the most present age
in each urban area in the data set.
"""
import pandas


def main():
    # Loads the data set from the CSV file.
    data = pandas.read_csv("resources/data.csv")
    data.head()
    # Gets all column names from the data set except the most_present_age.
    feature_columns = list(data.columns)
    feature_columns.remove("most_present_age")


if __name__ == "__main__":
    main()
