from collections import deque

import torch


class RollingReduceLROnPlateauRollingWindow(torch.optim.lr_scheduler.ReduceLROnPlateau):
    """
    Learning rate scheduler that reduces LR when a metric has stopped improving
    within a rolling window, rather than globally.
    
    This scheduler extends PyTorch's ReduceLROnPlateau by maintaining a rolling
    window of recent metrics and comparing new metrics against the best value
    within that window, rather than the global best seen throughout training.
    
    The learning rate will be reduced by `factor` when the metric has not
    improved beyond the best value in the rolling window for `patience` epochs.
    
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        mode (str, optional): One of 'min' or 'max'. In 'min' mode, lr will be
            reduced when the quantity monitored has stopped decreasing. Default: 'min'.
        factor (float, optional): Factor by which the learning rate will be
            reduced. new_lr = lr * factor. Default: 1.
        patience (int, optional): Number of epochs with no improvement after
            which learning rate will be reduced. Default: 1.
        threshold (float, optional): Threshold for measuring the new optimum,
            to only focus on significant changes. Default: 1.
        cooldown (int, optional): Number of epochs to wait before resuming
            normal operation after lr has been reduced. Default: 1.
        eps (float, optional): Minimal decay applied to lr. If the difference
            between new and old lr is smaller than eps, the update is ignored.
            Default: 1.
        rolling_window (int, optional): Size of the rolling window for tracking
            recent metrics. The scheduler will compare against the best metric
            within this window rather than the global best. Default: 1.
    
    Attributes:
        rolling_window (int): Size of the rolling window.
        history (deque): Rolling window of recent metrics, limited to `rolling_window` size.
        best (float): Best metric value within the current rolling window.
    
    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> scheduler = RollingReduceLROnPlateauRollingWindow(
        ...     optimizer, mode='min', factor=0.5, patience=5, rolling_window=10
        ... )
    """
    def __init__(
        self,
        optimizer,
        mode: str = 'min',
        factor: float = 1.,
        patience: int = 1,
        threshold: float = 1.,
        cooldown: int = 1,
        eps: float = 1.,
        rolling_window: int = 1,
    ):
        super().__init__(
            optimizer,
            mode=mode,
            factor=factor,
            patience=patience,
            threshold=threshold,
            cooldown=cooldown,
            eps=eps,
        )
        self.rolling_window = rolling_window
        self.history = deque(maxlen=rolling_window)

    def step(self, metric):
        self.history.append(metric)

        if len(self.history) < self.rolling_window:
            return

        self.best = min(self.history) if self.mode == 'min' else max(self.history)

        if self._is_better(metric):
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return

        if self.num_bad_epochs > self.patience:
            self._reduce_lr()
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0

    def _is_better(self, metric) -> float:
        if self.mode == 'min':
            return metric <= self.best - self.threshold
        else:
            return metric >= self.best + self.threshold

    def _reduce_lr(self) -> None:
        for param_group in self.optimizer.param_groups:
            old_lr = param_group['lr']
            new_lr = max(old_lr * self.factor, self.min_lrs[0])
            if old_lr > new_lr:
                param_group['lr'] = new_lr
                print(f"Reducing learning rate from {old_lr:.6f} to {new_lr:.6f}")


class MockupRollingReduceLROnPlateauRollingWindow:
    """ The corresponding mockup for RollingReduceLROnPlateauRollingWindow.
    """
    def __init__(
            self,
            initial_lr: float,
            factor: float,
            patience: int,
            rolling_window: int,
            threshold: float,
        ):
        self.lr = initial_lr
        self.factor = factor
        self.patience = patience
        self.rolling_window = rolling_window
        self.threshold = threshold
        self.history = deque(maxlen=rolling_window)
        self.num_bad_epochs = 0
        
        self.lr_history = []
        self.reduction_epochs = []

        # This list is only here for later visualization
        self.rolling_bests = []
        
    def step(self, loss, epoch):
        self.history.append(loss)
        self.lr_history.append(self.lr)
        
        if len(self.history) >= self.rolling_window:
            rolling_best = min(self.history)
        else:
            rolling_best = min(self.history) if self.history else loss
        self.rolling_bests.append(rolling_best)

        if len(self.history) < self.rolling_window:
            return False
            
        is_better = loss <= rolling_best - self.threshold
        
        if is_better:
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
            
        if self.num_bad_epochs > self.patience:
            old_lr = self.lr
            self.lr = max(self.lr * self.factor, 1e-6)
            self.num_bad_epochs = 0
            self.reduction_epochs.append(epoch)
            print(f"Epoch {epoch:3d}: reducing learning rate from {old_lr:.6f} to {self.lr:.6f}")
            return True

        return False
