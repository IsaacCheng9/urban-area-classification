"""
An implementation of a decision tree classifier to predict the most present age
in each urban area in the data set.
"""
import pandas
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def main():
    # Loads the data set from the CSV file.
    data = pandas.read_csv("resources/data.csv")
    data.head()
    # Gets all column names from the data set except the most_present_age.
    feature_columns = list(data.columns)
    feature_columns.remove("most_present_age")

    # Splits the data set in feature variables and the target variable.
    features = data[feature_columns]
    target = data.most_present_age
    (feature_train, feature_test,
     target_train, target_test) = train_test_split(features, target,
                                                   test_size=0.3,
                                                   random_state=1)
    # Create the decision tree classifier and train it.
    classifier = DecisionTreeClassifier()
    classifier = classifier.fit(feature_train, target_train)
    target_prediction = classifier.predict(feature_test)
    # Evaluates the accuracy of the model.
    print("Accuracy: {}".format(
        metrics.accuracy_score(target_test, target_prediction)))


if __name__ == "__main__":
    main()
