class Evaluator:
    """Utility class for evaluating classifier performance."""

    def evaluate_sample(self, classifier, record, true_label):
        """Evaluate a single classification.

        Args:
            classifier (NaiveBayesClassifier): Trained classifier.
            record (dict): Feature values to classify.
            true_label (str): Expected label for the record.

        Returns:
            bool: ``True`` if prediction matches ``true_label``.

        Usage:
            evaluator.evaluate_sample(clf, sample, "yes")
        """
        predicted = classifier.classify(record)
        if predicted == true_label:
            print(f"Correctly classified: {record} as {predicted}")
            return True
        else:
            print(
                f"Incorrectly classified: {record} as {predicted}, expected {true_label}"
            )
            return False

    def evaluate(self, classifier, test_df, target_column):
        """Evaluate classifier accuracy on a dataset.

        Args:
            classifier (NaiveBayesClassifier): Trained classifier.
            test_df (pandas.DataFrame): Dataset containing features and labels.
            target_column (str): Name of the label column.

        Returns:
            float: Classification accuracy as a value between 0 and 1.

        Usage:
            accuracy = evaluator.evaluate(clf, df_test, "label")
        """
        correct_predictions = 0
        total_predictions = len(test_df)
        for _, row in test_df.iterrows():
            record = row.drop(target_column).to_dict()
            predicted = classifier.classify(record)
            if predicted == row[target_column]:
                correct_predictions += 1
        accuracy = correct_predictions / total_predictions
        print(
            f"Accuracy: {correct_predictions}/{total_predictions} ({(correct_predictions / total_predictions) * 100:.2f}%)"
        )
        return accuracy
