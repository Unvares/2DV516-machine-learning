class ROCAnalysis:
    """
    Class to calculate various metrics for Receiver Operating Characteristic (ROC) analysis.

    Attributes:
        y_pred (list): Predicted labels.
        y_true (list): True labels.
        tp (int): Number of true positives.
        tn (int): Number of true negatives.
        fp (int): Number of false positives.
        fn (int): Number of false negatives.
    """

    def __init__(self, y_predicted, y_true):
        """
        Initialize ROCAnalysis object.

        Parameters:
            y_predicted (list): Predicted labels (0 or 1).
            y_true (list): True labels (0 or 1).
        """
        self.y_pred = y_predicted
        self.y_true = y_true

    def tp_rate(self):
        """
        Calculate True Positive Rate (Sensitivity, Recall).

        Returns:
            float: True Positive Rate.
        """
        tp = sum(
            (self.y_pred[i] == 1) and (self.y_true[i] == 1)
            for i in range(len(self.y_true))
        )
        fn = sum(
            (self.y_pred[i] == 0) and (self.y_true[i] == 1)
            for i in range(len(self.y_true))
        )
        return tp / (tp + fn) if (tp + fn) > 0 else 0

    def fp_rate(self):
        """
        Calculate False Positive Rate.

        Returns:
            float: False Positive Rate.
        """
        fp = sum(
            (self.y_pred[i] == 1) and (self.y_true[i] == 0)
            for i in range(len(self.y_true))
        )
        tn = sum(
            (self.y_pred[i] == 0) and (self.y_true[i] == 0)
            for i in range(len(self.y_true))
        )
        return fp / (fp + tn) if (fp + tn) > 0 else 0

    def precision(self):
        """
        Calculate Precision.

        Returns:
            float: Precision.
        """
        tp = self.tp_rate()
        fp = self.fp_rate()
        return tp / (tp + fp) if (tp + fp) > 0 else 0

    def f_score(self, beta=1):
        """
        Calculate the F-score.

        Parameters:
            beta (float, optional): Weighting factor for precision in the harmonic mean. Defaults to 1.

        Returns:
            float: F-score.
        """
        precision = self.precision()
        recall = self.tp_rate()
        if (precision + recall) == 0:
            return 0
        return (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall)
