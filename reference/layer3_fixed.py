"""Step10: no new arithmetic. Signed16-native layer3 weights sign-extend to
unchanged signed18 kernel; int32 Q11 activations are never narrowed.
"""
from reference.layer2_fixed import linear_w18,ffn_w18
from reference.layer2_fixed import transformer_layer2_fixed as transformer_layer3_fixed
