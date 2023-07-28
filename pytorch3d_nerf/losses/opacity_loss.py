import torch
import torch.nn as nn

def opacity_loss_alpha(rendered_silhouettes, alpha):
    opacity_err = torch.mean(
        (rendered_silhouettes * (1.0 - rendered_silhouettes)) ** alpha
    )
    return opacity_err


class OpacityLoss(nn.Module):
    def __init__(self, alpha, start_factor, start_epoch, max_epochs):
        super().__init__()
        self.loss_func = opacity_loss_alpha
        self.alpha = alpha
        self.start_factor = start_factor
        self.start_epoch = start_epoch
        self.max_epochs = max_epochs

    def forward(self,
                rendered_silhouettes,
                current_epoch,
                ):
        err_unconstrained = self.loss_func(
            rendered_silhouettes, self.alpha
        )
        # decrease loss by factor
        if self.max_epochs > 0 and current_epoch > self.start_epoch:
            loss_factor = self.start_factor *\
                          max(0, 1 - (current_epoch / (self.start_epoch + self.max_epochs)))
        else:
            loss_factor = 0
        err = err_unconstrained * loss_factor

        return err, err_unconstrained, loss_factor


if __name__ == '__main__':
    penalty_0_1_square = lambda x: (x * (1.0 - x)) ** 2
    penalty_0_1_abs = lambda x: torch.abs(x * (1.0 - x))
    penalty_0_1_alpha = lambda x, alpha: ((x * (1.0 - x)) ** alpha)

    # create 2 plots, plot 1: penalty_0_1_square, plot 2: penalty_0_1_abs
    import matplotlib.pyplot as plt

    x = torch.linspace(0, 1, 100)
    y_0_1_square = penalty_0_1_square(x)
    y_0_1_abs = penalty_0_1_abs(x)
    y_0_1_alpha = penalty_0_1_alpha(x, 0.5)

    # plt.plot(x, y_0_1_square, label='penalty_0_1_square')
    # plt.plot(x, y_0_1_abs, label='penalty_0_1_abs')

    # for i in range(30, 41, 5):
    #     plt.plot(x, penalty_0_1_alpha(x, i/10), label=f'penalty_0_1_alpha {i/10}')

    plt.legend()
    plt.show()



