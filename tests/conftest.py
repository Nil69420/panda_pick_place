import pytest
import numpy as np

from panda_control.task_runner import TaskRunner


@pytest.fixture(scope="module")
def pipeline():
    p = TaskRunner(render_mode="rgb_array", n_cubes=5, seed=42)
    yield p
    p.close()
