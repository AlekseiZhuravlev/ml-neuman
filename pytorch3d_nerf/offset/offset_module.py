import torch

class OffsetModule(torch.nn.Module):
    def __init__(self,
                 offset_net,
                 offset_scale,
                 offset_delay,
                 ):
        super().__init__()

        self.offset_net = offset_net
        self.offset_scale = offset_scale
        self.offset_delay = offset_delay
        self.mean_offset = 0.0

    def forward(self, input_pts, input_dirs, curr_epoch):
        if curr_epoch < self.offset_delay:
            offset = torch.zeros_like(input_pts[..., 0:3])
        else:
            offset = self.offset_net.forward(input_pts, input_dirs)
            # apply tanh and scale to offset
            offset = torch.tanh(offset) * self.offset_scale

            # with torch.no_grad():
            self.mean_offset = torch.norm(offset, dim=-1).mean()

        return offset

    def batched_forward(self, input_pts, input_dirs, curr_epoch):
        if curr_epoch < self.offset_delay:
            offset = torch.zeros_like(input_pts[..., 0:3])
        else:
            offset = self.offset_net.batched_forward(input_pts, input_dirs)
            # apply tanh and scale to offset
            offset = torch.tanh(offset) * self.offset_scale

            with torch.no_grad():
                self.mean_offset = offset.abs().mean()

        return offset