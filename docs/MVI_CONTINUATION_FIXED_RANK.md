# Fixed rank MVI continuation across load steps and Newmark time steps

Previous accepted X and Y profile arrays are carried to the next time/load
step; they are not reconstructed by QR from the complete nodal field.
This prevents a perfectly representable rank-r previous state from suddenly
failing a new decomposition or changing its separated coordinates.

The initial affine trace lift is reconstructed at the NEW physical time.
The previous free separated variation is preserved, adjusted only for
componentwise scale conversion. The variation is checked against the
current essential constraints. An incompatible trace space gives an
explicit error; no fallback to full-grid FDM or automatic rank growth.

The step driver passes previous_mvi_result only after a fully accepted
static/Newmark step; it never transfers a failed iterate. The previous
result must have a passing independent MVI certificate.

Cold starts and old explicit calls still work with the default
previous_mvi_result=None. For them, the projection of the initial state
onto the admissible kernel now correctly subtracts the nonzero affine
lift BEFORE projecting: B*(B.T*(state-lift)), not B*(B.T*state).

JSON summary and X/Y sweep histories record mvi_initialization as
COLD_START or CONTINUATION. Fixed ranks 1-8, G3D/Gz, all seven
kinematics and full coupled residual acceptance are unchanged.
No SVD, regularization or adaptively increased rank were introduced.
