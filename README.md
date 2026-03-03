# Panda Control

Vision-guided pick-and-place for the Franka Emika Panda 7-DOF robot arm in a PyBullet simulation. The system uses an overhead RGB-D camera for 3-D object detection, an Artificial Potential Field for obstacle-aware motion, and the Hungarian algorithm for optimal multi-cube placement.

## Project Structure

```
panda_control/
    camera.py               # Overhead and wrist-mounted RGB-D capture
    perception.py            # Segmentation-based 3-D object detection
    robot.py                 # IK solving, APF motion, grasp/place state machines
    scene.py                 # Random cube spawning and tracking
    task_runner.py           # High-level task orchestrator
    main.py                  # Command-line entry point
    config.py                # YAML configuration loader
    config/
        default.yaml         # All tuneable parameters in one file
    common_utils/
        apf.py               # Attractive/repulsive potential field functions
        jit_kernels.py       # Numba JIT-accelerated inner loops
        robot_constants.py   # Panda joint limits and rest poses
    tests/                   # Pytest suite (73 tests)
```

## Installation

The project depends on a local fork of panda-gym that must be installed first.

```bash
# 1. Install the panda-gym fork
cd panda-gym
pip install -e .

# 2. Install project dependencies
cd ../panda_control
pip install -r requirements.txt

# 3. Set the Python path so the package can be found
export PYTHONPATH=/path/to/parent/folder:$PYTHONPATH
```

Replace ``/path/to/parent/folder`` with the directory that contains the ``panda_control`` folder.

## Running the Demo

The demo opens a PyBullet GUI window, spawns coloured cubes on the table, and runs the requested pick-and-place task.

```bash
# Pick and place one cube (default)
python -m panda_control.main

# Pick and place all cubes into an optimally-assigned row
python -m panda_control.main --all

# Stack all cubes on top of one another
python -m panda_control.main --stack
```

### Options

| Flag            | Description                              | Default |
|-----------------|------------------------------------------|---------|
| ``--one``       | Pick and place a single cube             | yes     |
| ``--all``       | Pick and place every visible cube        |         |
| ``--stack``     | Stack all cubes                          |         |
| ``--n-cubes N`` | Number of cubes to spawn                 | 5       |
| ``--seed N``    | Random seed for reproducibility          | random  |
| ``--delay S``   | Seconds to sleep per simulation step     | 0.033   |

## Running the Tests

```bash
python -m pytest panda_control/tests/ -q
```

All 73 tests run headless and take roughly 15 seconds.

## Configuration

Every tuneable parameter -- camera settings, cube sizes, APF gains, gripper widths, motion limits -- lives in a single YAML file:

```
panda_control/config/default.yaml
```

Any module can read a value with:

```python
from panda_control.config import get as cfg

width = cfg("camera", "width")          # 640
eta   = cfg("robot", "apf", "eta")      # 0.005
```

There is no need to edit source code to change parameters. Open the YAML file, adjust the value, and re-run.

## How It Works

1. **Scene generation** -- Random cubes are placed on the table with guaranteed minimum separation.
2. **Perception** -- An overhead camera captures an RGB-D frame. The segmentation mask identifies each cube, and the depth map back-projects its centroid to a 3-D world coordinate.
3. **Assignment** -- For multi-cube tasks, the system builds an XY-distance cost matrix and solves the optimal cube-to-slot assignment using the Hungarian algorithm.
4. **Motion** -- The arm moves to each target using an Artificial Potential Field that steers around neighbouring cubes. Vertical descents bypass the field so the gripper lands precisely on the target.
5. **Grasp and place** -- A finite state machine drives the arm through approach, descend, grasp, lift, transit, lower, release, and retract.

## Requirements

- Python 3.8 or later
- PyBullet
- Gymnasium
- NumPy
- SciPy
- PyYAML
- Numba (optional)
- panda-gym (local fork)

See ``requirements.txt`` for pinned minimum versions.

## JIT Acceleration

When Numba is installed, the following hot loops are JIT-compiled:

- APF attractive force
- APF repulsive force (obstacle loop)
- Fused total APF force (single compiled call per simulation step)
- Depth-buffer linearisation (element-wise, no intermediate arrays)

If Numba is not available, equivalent NumPy implementations are used automatically.
