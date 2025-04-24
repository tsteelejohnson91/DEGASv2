
# I refer to https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch for early stopping funcion
class EarlyStopper:
    def __init__(self, patience=10, min_delta=1e-1, epsilon = 0.05):
        # check if the loss difference larger than min_delta * max(abs(loss)) or epsilon
        self.patience = patience
        self.min_delta = min_delta
        self.epsilon = epsilon
        self.counter = 0

    def early_stop(self, validation_loss):
        if len(validation_loss) <= 2:
            return False
        current_loss = validation_loss.iloc[-1]
        previous_loss = validation_loss.iloc[-2]
        abs_diff = (current_loss - previous_loss).abs()
        max_abs_loss = validation_loss.abs().max()
        if (abs_diff < max_abs_loss * self.min_delta).all() or (max(abs_diff) < self.epsilon):
            self.counter += 1
        if (self.counter >= self.patience):
            return True
        return False
    