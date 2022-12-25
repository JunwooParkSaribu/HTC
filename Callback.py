import numpy as np


class EarlyStoppingAtMinLoss():
    def __init__(self, patience=0):
        self.patience = patience
        # best_weights to store the weights at which the minimum loss occurs.
        self.best_weights = None

    def on_train_begin(self):
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0
        # Initialize the best as infinity.
        self.best = np.Inf

    def on_epoch_end(self, epoch, weights, loss=None):
        current = loss
        if np.less(current, self.best):
            self.best = current
            self.wait = 0
            # Record the best weights if current results is better (less).
            self.best_weights = weights
            return None
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                print("Restoring model weights from the end of the best epoch.")
                return self.best_weights
            return None

    def on_train_end(self):
        if self.stopped_epoch > 0:
            print("Epoch %4d: early stopping" % (self.stopped_epoch + 1))
        return self.best_weights