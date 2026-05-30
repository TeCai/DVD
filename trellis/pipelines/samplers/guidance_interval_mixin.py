from typing import *
import torch.nn.functional as F

class GuidanceIntervalSamplerMixin:
    """
    A mixin class for samplers that apply classifier-free guidance with interval.
    """

    def _inference_model(self, model, x_t, t, cond, neg_cond, cfg_strength, cfg_interval, **kwargs):
        if cfg_interval[0] <= t <= cfg_interval[1]:
            pred = super()._inference_model(model, x_t, t, cond, **kwargs)
            neg_pred = super()._inference_model(model, x_t, t, neg_cond, **kwargs)
            return (1 + cfg_strength) * pred - cfg_strength * neg_pred
        else:
            return super()._inference_model(model, x_t, t, cond, **kwargs)

class DiscreteGuidanceIntervalSamplerMixin:
    """
    A mixin class for samplers that apply classifier-free guidance with interval.
    """

    def _inference_model(self, model, x_t, t, cond, neg_cond, cfg_strength, cfg_interval, **kwargs):
        if cfg_interval[0] <= t <= cfg_interval[1]:
            pred_logits = super()._inference_model(model, x_t, t, cond, **kwargs)
            neg_pred_logits = super()._inference_model(model, x_t, t, neg_cond, **kwargs)
            pred_logits_normalized = F.log_softmax(pred_logits, dim=-1)
            neg_pred_logits_normalized = F.log_softmax(neg_pred_logits, dim=-1)

            return (1 + cfg_strength) * pred_logits_normalized - cfg_strength * neg_pred_logits_normalized
        else:
            return super()._inference_model(model, x_t, t, cond, **kwargs)
        