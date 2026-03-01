import torch
import torch.nn.functional as F

from tools.registery import LOSS_REGISTRY


def _event_reliability_map(events, eps=1e-6):
    if events.dim() == 5:
        n, t, c, h, w = events.shape
        events = events.reshape(n * t, c, h, w)
    mag = torch.sum(torch.abs(events), dim=1, keepdim=True)
    return mag / (mag.amax(dim=(-2, -1), keepdim=True) + eps)


def _event_consistency_core(pred, target, events, gamma):
    if pred.dim() == 5:
        n, t, c, h, w = pred.shape
        pred = pred.reshape(n * t, c, h, w)
        target = target.reshape(n * t, c, h, w)
    w_event = _event_reliability_map(events)
    if w_event.shape[-2:] != pred.shape[-2:]:
        w_event = F.interpolate(w_event, size=pred.shape[-2:], mode='bilinear', align_corners=False)

    abs_err = torch.abs(pred - target).mean(dim=1, keepdim=True)
    supported = (w_event * abs_err).mean()
    unsupported_penalty = ((1.0 - w_event) * abs_err).mean()
    return supported + gamma * unsupported_penalty


@LOSS_REGISTRY.register()
class EventConsistencyScore:
    """Event-consistent anomaly score for robust VAD thresholding."""

    def __init__(self, loss_dict):
        self.weight = loss_dict.weight
        self.gamma = 0.2 if 'gamma' not in loss_dict.keys() else loss_dict.gamma

    def forward(self, pred, target, events):
        return self.weight * _event_consistency_core(pred, target, events, self.gamma)


@LOSS_REGISTRY.register()
class EventConsistencyLoss:
    """Training loss variant of event-consistency scoring."""

    def __init__(self, loss_dict):
        self.weight = loss_dict.weight
        self.gamma = 0.2 if 'gamma' not in loss_dict.keys() else loss_dict.gamma
        self.as_loss = True if 'as_loss' not in loss_dict.keys() else loss_dict.as_loss

    def forward(self, pred, target, events):
        return self.weight * _event_consistency_core(pred, target, events, self.gamma)
