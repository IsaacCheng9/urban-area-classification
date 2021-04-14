"""
An implementation of a decision tree classifier to predict the most present age
in each urban area in the data set.
"""
import json

import pandas
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def main():
    data = read_data_set()
    feature_columns, features, target = prepare_data_set(data)

    # Checks the accuracy of the decision tree classifier with different depths
    # and test sizes.
    for depth in range(1, 11):
        train_decision_tree(depth, features, target)
    # Identifies influence of features in most present age of an urban area.
    visualise_feature_importance(feature_columns, features, target)


def read_data_set():
    # Loads the data set from the CSV file.
    data = pandas.read_csv("resources/data.csv")
    data.head()
    return data


def prepare_data_set(data):
    # Gets all column names from the data set except the most_present_age.
    feature_columns = list(data.columns)
    feature_columns.remove("most_present_age")
    # Splits the data set in feature variables and the target variable.
    features = data[feature_columns]
    target = data.most_present_age
    return feature_columns, features, target


def train_decision_tree(depth, features, target):
    # Uses 40% of the data set for the test.
    (feature_train, feature_test,
     target_train, target_test) = train_test_split(features, target,
                                                   test_size=0.4,
                                                   random_state=1)
    # Create the decision tree classifier and train it.
    classifier = DecisionTreeClassifier(max_depth=depth)
    classifier = classifier.fit(feature_train, target_train)
    target_prediction = classifier.predict(feature_test)
    # Evaluates the accuracy of the model.
    print("Depth: {}\nAccuracy: {}%\n".format(
        depth, metrics.accuracy_score(target_test, target_prediction) * 100))
    return classifier


def visualise_feature_importance(feature_columns, features, target):
    # Generates a bar chart to visualise feature importance.
    classifier = train_decision_tree(10, features, target)
    feature_importance = classifier.feature_importances_
    columns = [column for column in feature_columns]
    plt.bar(columns, feature_importance)
    # Pretty prints the percentage influence of features.
    importance_dict = dict()
    for index, importance in enumerate(feature_importance):
        importance_dict[columns[index]] = importance
    print(json.dumps(importance_dict, indent=4, sort_keys=True))
    # Generates a graph to display the percentage influence of features.
    plt.xticks(rotation="vertical")
    plt.title("Influence of Features in Most Present Age of Urban Areas")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
