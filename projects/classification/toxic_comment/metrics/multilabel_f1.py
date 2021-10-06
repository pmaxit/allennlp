from typing import Optional

import torch
from allennlp.training.metrics.metric import Metric

@Metric.register('multilabel-f1')
class MultiLabelF1Measure(Metric):
    """ Computers multilabel F1. Assumes that predictions are 0 or 1
    """

    def __init__(self)->None:
        self._true_positives = 0.0
        self._true_negatives = 0.0
        self._false_positives = 0.0
        self._false_negatives = 0.0

    def __call__(self,
            predictions: torch.LongTensor,
            gold_labels: torch.LongTensor):

        """ Parameters """

        self._true_positives += (predictions * gold_labels).sum().item()
        self._false_positives += (predictions * (1 - gold_labels)).sum().item()
        self._true_negatives += ((1-predictions)* ( 1- gold_labels)).sum().item()
        self._false_negatives += ((1-predictions) * gold_labels).sum().item()

    def get_metric(self, reset:bool = False):
        predicted_positives = self._true_positives + self._false_positives
        actual_positives = self._true_positives + self._false_negatives
        accuracy = self._true_positives + self._true_negatives 
        total_items = (self._true_positives + self._true_negatives + self._false_positives + self._false_negatives)

        precision = self._true_positives / predicted_positives if predicted_positives > 0 else 0
        recall = self._true_positives / actual_positives if actual_positives > 0 else 0
        accuracy = accuracy / total_items if total_items > 0 else 0

        if precision + recall > 0:
            f1_measure = 2 * precision * recall / (precision + recall)
        else:
            f1_measure = 0

        if reset:
            self.reset()
        return precision, recall, f1_measure, accuracy

    def reset(self):
        self._true_positives = 0.0
        self._true_negatives = 0.0
        self._false_positives = 0.0
        self._false_negatives = 0.0
