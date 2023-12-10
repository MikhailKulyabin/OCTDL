from utils.const import regression_loss


class WarpedLoss():
    def __init__(self, loss_function, criterion):
        self.loss_function = loss_function
        self.criterion = criterion

        self.squeeze = True if self.criterion in regression_loss else False

    def __call__(self, pred, target):
        if self.squeeze:
            pred = pred.squeeze()

        return self.loss_function(pred, target)
