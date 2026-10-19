"""Step12: reuse proven signed18 arithmetic. Native16 layer5 FFN2 weights
sign-extend without scaling; int32 Q11 activations are never narrowed.
"""
from reference.layer2_fixed import linear_w18,ffn_w18
from reference.layer2_fixed import transformer_layer2_fixed as transformer_layer5_fixed
