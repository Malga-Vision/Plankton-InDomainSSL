import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.neural_network import MLPClassifier

def train_and_evaluate_classifier(features_path, labels_path, test_features_path, test_labels_path, classifier_type='SVM', test_size=0.2, random_state=31, multi_class='ovr'):
    """
    Trains and evaluates a classifier on pre-extracted features for multi-class classification.

    Parameters:
    file_path (str): Path to the npy file containing features and labels.
    classifier_type (str): Type of classifier to use.
    test_size (float): Proportion of the dataset to include in the test split.
    random_state (int): Controls the shuffling applied to the data before applying the split.
    multi_class (str): Multi-class strategy for LogisticRegression ('ovr' for One-vs-Rest, 'multinomial' for Multinomial).

    Returns:
    float: Accuracy of the model on the test set.
    str: Classification report.
    """
    # Load features and labels from .npy file
    features = np.load(features_path)
    labels = np.load(labels_path)

    print(f"Features shape: {features.shape[1]}")
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=random_state, stratify=labels)

    # Initialize the classifier
    if classifier_type == 'SVM':
        classifier = SVC(kernel='linear')
    elif classifier_type == 'LogisticRegression':
        # classifier = LogisticRegression(multi_class=multi_class, solver='lbfgs', max_iter=5000, warm_start=True)
        classifier = LogisticRegression(class_weight='balanced', multi_class=multi_class, solver='lbfgs', max_iter=5000, warm_start=True)
    elif classifier_type == 'RandomForest':
        classifier = RandomForestClassifier(n_estimators=200)
    elif classifier_type == 'MLP':
        classifier = MLPClassifier(hidden_layer_sizes=(features.shape[1],), max_iter=5000, activation='relu', n_iter_no_change=50,
                                    solver='sgd', alpha=0.0001, learning_rate_init=0.001, random_state=42, verbose=True)
    else:
        raise ValueError("Unsupported classifier type.")

    
    # Train the classifier
    classifier.fit(X_train, y_train)

    # Make predictions and evaluate
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    # Output the results
    print(f"Validation Accuracy:  {accuracy * 100:.2f}%")
    print("Classification Report:")
    print(report)

    # Load features and labels from .npy file
    test_features = np.load(test_features_path)
    testlabels = np.load(test_labels_path)

    # Make predictions
    new_predictions = classifier.predict(test_features)
    accuracy = accuracy_score(testlabels, new_predictions)
    report = classification_report(testlabels, new_predictions)

    # Output the results
    print(f"Test Accuracy:  {accuracy * 100:.2f}%")
    print("Classification Report:")
    print(report)

    return classifier, accuracy, report

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train and evaluate a classifier on pre-extracted features.")
    parser.add_argument("--features_path", type=str, help="Path to the npy file containing features.")
    parser.add_argument("--labels_path", type=str, help="Path to the npy file containing  labels.")
    parser.add_argument("--classifier_type", type=str, default='SVM', 
                        help="Type of classifier to use ('SVM' or 'LogisticRegression').")
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--test_size", type=float, default=0.2, help="Proportion of the dataset to include in the test split.")
    parser.add_argument("--random_state", type=int, default=42, help="Controls the shuffling applied to the data before applying the split.")
    parser.add_argument("--multi_class", type=str, default='ovr', choices=['ovr', 'multinomial'], 
                        help="Multi-class strategy for LogisticRegression ('ovr' for One-vs-Rest, 'multinomial' for Multinomial).")
    
    # Additional argument for testing
    parser.add_argument("--test", action='store_true', help="Test the trained classifier on new data.")
    parser.add_argument("--classifier_path", type=str, help="Path to the saved classifier model for testing.")
    parser.add_argument("--test_features_path", type=str, help="Path to the npy file containing new features for testing.")
    parser.add_argument("--test_labels_path", type=str, help="Path to the npy file containing new labels for testing.")

    
    args = parser.parse_args()
  
    # Call the function with parsed arguments
    classifier, accuracy, report = train_and_evaluate_classifier(
        features_path=args.features_path,
        labels_path=args.labels_path,
        test_features_path=args.test_features_path,
        test_labels_path=args.test_labels_path,
        classifier_type=args.classifier_type, 
        test_size=args.test_size, 
        random_state=args.random_state, 
        multi_class=args.multi_class
    )