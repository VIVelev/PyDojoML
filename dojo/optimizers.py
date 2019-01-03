from abc import abstractmethod
import numpy as np

__all__ = [
    "Optimizer",
    "GradientDescent",
    "Momentum",
    "RMSprop",
    "Adam",
    "NesterovAcceleratedGradient",
]


class Optimizer:
    def __init__(self, alpha):
        self.alpha = alpha

    @abstractmethod
    def update(self, W, dW):
        pass

# ====================================================================================================
# ====================================================================================================

class GradientDescent(Optimizer):
    def __init__(self, alpha=0.01):
        super().__init__(alpha)

    def update(self, W, dW):
        return W - self.alpha * dW

class Momentum(Optimizer):
    def __init__(self, alpha=0.01, beta=0.9):
        super().__init__(alpha)
        self.beta = beta
        self.v = None

    def update(self, W, dW):
        if self.v is None:
            self.v = np.zeros_like(W)

        # Exponentially Weighted Moving Average
        self.v = self.beta * self.v + (1 - self.beta) * dW
        # Update
        return W - self.alpha * self.v

class RMSprop(Optimizer):
    def __init__(self, alpha=0.01, beta=0.999):
        super().__init__(alpha)
        self.beta = beta
        self.eps = 1e-8
        self.s = None

    def update(self, W, dW):
        if self.s is None:
            self.s = np.zeros_like(W)

        # Exponentially Weighted Moving Average
        self.s = self.beta * self.s + (1 - self.beta) * np.square(dW)
        # Update
        return W - self.alpha * dW / (np.sqrt(self.s) + self.eps)

class Adam(Optimizer):
    def __init__(self, alpha=0.01, beta1=0.9, beta2=0.999):
        super().__init__(alpha)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = 1e-8
        self.t = 1
        self.v = None
        self.s = None

    def update(self, W, dW):
        if self.v is None:
            self.v = np.zeros_like(W)
            self.s = np.zeros_like(W)

        # Exponentially Weighted Moving Average
        self.v = self.beta1 * self.v + (1 - self.beta1) * dW
        self.s = self.beta2 * self.s + (1 - self.beta2) * np.square(dW)

        # Bias correction
        v_corrected = self.v / (1 - self.beta1**self.t)
        s_corrected = self.s / (1 - self.beta2**self.t)
        self.t += 1

        # Update
        return W - self.alpha * v_corrected / (np.sqrt(s_corrected) + self.eps)

class NesterovAcceleratedGradient(Optimizer):
    def __init__(self, alpha=0.001, beta=0.4):
        super().__init__(alpha)
        self.beta = beta
        self.v = None

    def update(self, W, dW_func):
        if self.v is None:
            self.v = np.zeros_like(W)

        # Approximate the future gradient
        approx_future_grad = dW_func(W - self.alpha * self.beta * self.v)
        # Exponentially Weighted Moving Average
        self.v = self.beta * self.v + (1 - self.beta) * approx_future_grad

        # Update
        return W - self.alpha * self.v
