from typing import *
import torch.nn.functional as F
from torch import clamp, where, cat, logit
from math import exp, log

class GuidanceFunctionSamplerMixin:
    """
    A mixin class for samplers that apply classifier-free guidance.
    """

    def _inference_model(self, model, x_t, t, cond, neg_cond, cfg_strength, **kwargs):
        pred = super()._inference_model(model, x_t, t, cond, **kwargs)
        neg_pred = super()._inference_model(model, x_t, t, neg_cond, **kwargs)
        return (1 + cfg_strength(t)) * pred - cfg_strength(t) * neg_pred


class DiscreteGuidanceFunctionSamplerMixin:
    """
    A mixin class for samplers that apply classifier-free guidance.
    """

    def _inference_model(self, model, x_t, t, cond, neg_cond, cfg_strength, **kwargs):
        # the output of inference_model is the unnormalized logit
        pred_logits = super()._inference_model(model, x_t, t, cond, **kwargs)
        neg_pred_logits = super()._inference_model(model, x_t, t, neg_cond, **kwargs)
        pred_logits_normalized = F.log_softmax(pred_logits, dim=-1)
        neg_pred_logits_normalized = F.log_softmax(neg_pred_logits, dim=-1)
        

        return (1 + cfg_strength(t)) * pred_logits_normalized - cfg_strength(t) * neg_pred_logits_normalized
    

class DiscreteGuidanceFunctionSamplerMixinReport:
    """
    A mixin class for samplers that apply classifier-free guidance.
    """

    def _inference_model(self, model, x_t, t, cond, neg_cond, cfg_strength, **kwargs):
        # the output of inference_model is the unnormalized logit
        pred_logits = super()._inference_model(model, x_t, t, cond, **kwargs)
        neg_pred_logits = super()._inference_model(model, x_t, t, neg_cond, **kwargs)
        pred_logits_normalized = F.log_softmax(pred_logits, dim=-1)
        neg_pred_logits_normalized = F.log_softmax(neg_pred_logits, dim=-1)
    
        return (1 + cfg_strength(t)) * pred_logits_normalized - cfg_strength(t) * neg_pred_logits_normalized, pred_logits, neg_pred_logits
    

