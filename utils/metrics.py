import torch
import torcheval.metrics as tm
from utils.func import print_msg


class Estimator():
    def __init__(self, metrics, num_classes, criterion, average='macro', thresholds=None):
        self.criterion = criterion
        self.num_classes = num_classes
        self.thresholds = [-0.5 + i for i in range(num_classes)] if not thresholds else thresholds

        if criterion in regression_based_metrics and 'auc' in metrics:
            metrics.remove('auc')
            print_msg('AUC is not supported for regression based metrics {}.'.format(criterion), warning=True)

        self.metrics = metrics
        self.metrics_fn = {m: metrics_fn[m](num_classes=num_classes, average=average) for m in metrics}
        self.conf_mat_fn = tm.MulticlassConfusionMatrix(num_classes=num_classes)

    def update(self, predictions, targets):
        targets = targets.data.cpu().long()
        logits = predictions.data.cpu()
        predictions = self.to_prediction(logits)

        # update metrics
        self.conf_mat_fn.update(predictions, targets)
        for m in self.metrics_fn.keys():
            if m in logits_required_metrics:
                self.metrics_fn[m].update(logits, targets)
            else:
                self.metrics_fn[m].update(predictions, targets)

    def get_scores(self, digits=-1):
        scores = {m: self._compute(m, digits) for m in self.metrics}
        return scores

    def _compute(self, metric, digits=-1):
        score = self.metrics_fn[metric].compute().item()
        score = score if digits == -1 else round(score, digits)
        return score
    
    def get_conf_mat(self):
        return self.conf_mat_fn.compute().numpy().astype(int)

    def reset(self):
        for m in self.metrics_fn.keys():
            self.metrics_fn[m].reset()
        self.conf_mat_fn.reset()
    
    def to_prediction(self, predictions):
        if self.criterion in regression_based_metrics:
            predictions = torch.tensor([self.classify(p.item()) for p in predictions]).long()
        else:
            predictions = torch.argmax(predictions, dim=1).long()

        return predictions

    def classify(self, predict):
        thresholds = self.thresholds
        predict = max(predict, thresholds[0])
        for i in reversed(range(len(thresholds))):
            if predict >= thresholds[i]:
                return i


metrics_fn = {
    'acc': tm.MulticlassAccuracy,
    'f1': tm.MulticlassF1Score,
    'auc': tm.MulticlassAUROC,
    'precision': tm.MulticlassPrecision,
    'recall': tm.MulticlassRecall
}
available_metrics = metrics_fn.keys()
logits_required_metrics = ['auc']
regression_based_metrics = ['mean_square_error', 'mean_absolute_error', 'smooth_L1']
