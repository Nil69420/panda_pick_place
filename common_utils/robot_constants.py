from __future__ import annotations

import numpy as np

PANDA_NUM_JOINTS = 7
PANDA_EE_LINK = 11
PANDA_JOINT_INDICES = list(range(7)) + [9, 10]

PANDA_LOWER_LIMITS = np.array([-2.9671, -1.8326, -2.9671, -3.1416, -2.9671, -0.0873, -2.9671])
PANDA_UPPER_LIMITS = np.array([ 2.9671,  1.8326,  2.9671,  0.0,     2.9671,  3.8223,  2.9671])
PANDA_JOINT_RANGES = PANDA_UPPER_LIMITS - PANDA_LOWER_LIMITS
PANDA_REST_POSES   = np.array([0.0, 0.41, 0.0, -1.85, 0.0, 2.26, 0.79])

PANDA_JOINT_FORCES = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 120.0, 120.0, 170.0, 170.0])
