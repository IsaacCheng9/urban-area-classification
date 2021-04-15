"""
An implementation of a decision tree classifier to predict the most present age
in each urban area in the data set. It performs the prediction, and evaluates
the accuracy of the classifier per maximum depth level, as well as the
importance of features of urban areas from the data set.
"""
import json
from statistics import mean

import pandas
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def main():
    """
    Runs the decision tree classification algorithm and evaluation.
    """
    data = read_data_set()
    feature_columns, features, target = prepare_data_set(data)
    evaluate_classifier_accuracy(features, target)
    evaluate_feature_importance(feature_columns, features, target)


def read_data_set():
    """
    Loads the data set from the CSV file.

    Returns:
        The data set of Rome from the CSV file.
    """
    data = pandas.read_csv("resources/data.csv")
    data.head()
    return data


def prepare_data_set(data):
    """
    Prepares the data set to use with the decision tree classifier.

    Args:
        data: The data set of Rome from the CSV file.

    Returns:
        Attributes to use in the decision tree and the attribute to predict.
    """
    # Gets all column names from the data set except cell_id and
    # most_present_age.
    feature_columns = list(data.columns)
    feature_columns.remove("cell_id")
    feature_columns.remove("most_present_age")
    features = data[feature_columns]
    target = data.most_present_age
    return feature_columns, features, target


def train_decision_tree(depth, features, target):
    """
    Creates the decision tree classifier and trains it.

    Args:
        depth: Maximum depth level for the decision tree.
        features: The attributes to use in the decision tree.
        target: The attribute to predict.

    Returns:
        The decision tree classifier, and its accuracy.
    """
    # Uses 40% of the data set for the test.
    (feature_train, feature_test,
     target_train, target_test) = train_test_split(features, target,
                                                   test_size=0.4,
                                                   random_state=1)
    classifier = DecisionTreeClassifier(max_depth=depth)
    classifier = classifier.fit(feature_train, target_train)
    target_prediction = classifier.predict(feature_test)
    # Evaluates the accuracy of the model.
    accuracy = metrics.accuracy_score(target_test, target_prediction) * 100
    return classifier, accuracy


def evaluate_classifier_accuracy(features, target):
    """
    Checks the accuracy of the decision tree classifier with different depths
    and test sizes.

    Args:
        features: The attributes to use in the decision tree.
        target: The attribute to predict.
    """
    depth_accuracy_dict = {}
    # Gets the accuracy of five analysis runs for each maximum depth of
    # decision tree classifier.
    for _ in range(5):
        for depth in range(1, 21):
            _, accuracy = train_decision_tree(depth, features, target)
            key = "Average Accuracy for Maximum Depth {}".format(depth)
            try:
                depth_accuracy_dict[key].append(accuracy)
            except KeyError:
                depth_accuracy_dict[key] = [accuracy]
    # Calculates the average accuracy for each maximum depth setting.
    for key in depth_accuracy_dict:
        depth_accuracy_dict[key] = mean(depth_accuracy_dict[key])
    # Pretty prints the average accuracy values.
    print(json.dumps(depth_accuracy_dict, indent=4) + "\n\n")

    # Generates a graph to display the average accuracy per depth setting.
    columns = [num for num in range(1, 21)]
    accuracies = [value for value in depth_accuracy_dict.values()]
    plt.bar(columns, accuracies)
    plt.xticks(columns)
    plt.xlabel("Maximum Depth Level")
    plt.ylabel("Average Classification Accuracy (%)")
    plt.title(
        "Average Accuracy Per Maximum Depth Level of Decision Tree (5 runs)")
    plt.tight_layout()
    plt.show()


def evaluate_feature_importance(feature_columns, features, target):
    """
    Identifies influence of features for most present age of an urban area.

    Args:
        feature_columns: Listed attributes used in the decision tree.
        features: The attributes to use in the decision tree.
        target: The attribute to predict.
    """
    # Generates a bar chart to visualise feature importance.
    classifier, _ = train_decision_tree(10, features, target)
    feature_importance = classifier.feature_importances_ * 100

    # Pretty prints the percentage influence of features.
    columns = [column for column in feature_columns]
    importance_dict = {}
    for index, importance in enumerate(feature_importance):
        importance_dict[columns[index]] = importance
    print(json.dumps(importance_dict, indent=4, sort_keys=True))

    # Generates a graph to display the percentage influence of features.
    plt.bar(columns, feature_importance * 100)
    plt.xticks(rotation="vertical")
    plt.xlabel("Urban Feature")
    plt.ylabel("Influence (%)")
    plt.title("Influence of Features in Most Present Age of Urban Areas")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
