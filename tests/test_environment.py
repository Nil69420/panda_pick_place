import numpy as np
import pytest


class TestEnvironmentSetup:

    def test_env_created(self, pipeline):
        assert pipeline.env is not None

    def test_sim_accessible(self, pipeline):
        assert pipeline.sim is not None
        assert pipeline.physics_client is not None

    def test_robot_accessible(self, pipeline):
        assert pipeline.robot is not None
        assert pipeline.robot_body_id >= 0

    def test_ee_position_shape(self, pipeline):
        pos = pipeline.ee_position
        assert pos.shape == (3,)
        assert np.all(np.isfinite(pos))
