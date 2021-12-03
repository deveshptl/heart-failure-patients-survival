import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def trainModel(filename, iteration):
    data = pd.read_csv(f'data/{filename}')
    analysis = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'fScore': []
    }

    for _ in range(iteration):
        # shuffle the rows
        data = data.sample(frac=1)

        # get labels and features
        labels = data['DEATH_EVENT']
        features = data.drop('DEATH_EVENT', axis=1)

        # perform train test split
        train_features, test_features, train_labels, test_labels = train_test_split(
            features, labels, test_size=0.30)

        # train the model
        model = DecisionTreeClassifier(criterion='entropy')
        model.fit(train_features, train_labels)
        predictions = model.predict(test_features)

        # store the results for analysis
        analysis['accuracy'].append(accuracy_score(test_labels, predictions))
        analysis['precision'].append(precision_score(test_labels, predictions))
        analysis['recall'].append(recall_score(test_labels, predictions))
        analysis['fScore'].append(f1_score(test_labels, predictions))
    return analysis
