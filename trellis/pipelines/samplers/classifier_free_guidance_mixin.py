from typing import *
import torch.nn.functional as F

class ClassifierFreeGuidanceSamplerMixin:
    """
    A mixin class for samplers that apply classifier-free guidance.
    """

    def _inference_model(self, model, x_t, t, cond, neg_cond, cfg_strength, **kwargs):
        pred = super()._inference_model(model, x_t, t, cond, **kwargs)
        neg_pred = super()._inference_model(model, x_t, t, neg_cond, **kwargs)
        return (1 + cfg_strength) * pred - cfg_strength * neg_pred



class DiscreteClassifierFreeGuidanceSamplerMixin:
    """
    A mixin class for samplers that apply classifier-free guidance. No need to clamp after train with truncate.
    However, if one would like to apply large cfg, consider a very small cfg (or 0) when t->0 to aviod artifacts. 
    """

    def _inference_model(self, model, x_t, t, cond, neg_cond, cfg_strength, **kwargs):
        # the output of inference_model is the unnormalized logit
        pred_logits = super()._inference_model(model, x_t, t, cond, **kwargs)
        neg_pred_logits = super()._inference_model(model, x_t, t, neg_cond, **kwargs)
        pred_logits_normalized = F.log_softmax(pred_logits, dim=-1)
        neg_pred_logits_normalized = F.log_softmax(neg_pred_logits, dim=-1)

        return (1 + cfg_strength) * pred_logits_normalized - cfg_strength * neg_pred_logits_normalized
