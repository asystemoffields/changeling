"""changeling — breeding transferable in-context learners. See SPEC.md."""

OBS_DIM = 64
N_ACT = 8
IN_DIM = OBS_DIM + N_ACT + 2  # obs, onehot(last action), last reward, boundary bit
