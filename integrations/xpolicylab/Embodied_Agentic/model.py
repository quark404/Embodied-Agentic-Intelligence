"""Minimal policy server required by RoboDojo's reset-time WebSocket client."""

import os

from XPolicyLab.model_template import ModelTemplate


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        super().__init__()
        action_type = model_cfg.get("action_type")
        mode = os.environ.get("EMBODIED_ROBODOJO_CONTROL_MODE", "ee_delta")
        if (mode, action_type) not in (("joint", "joint"), ("ee_delta", "ee")):
            raise ValueError("Native action_type must be joint for joint mode or ee for ee_delta mode")

    def reset(self):
        pass

    def update_obs(self, obs):
        raise RuntimeError("Embodied_Agentic actions run in RoboDojo's simulator process")

    def get_action(self):
        raise RuntimeError("Embodied_Agentic actions run in RoboDojo's simulator process")

    def update_obs_batch(self, obs_list):
        raise RuntimeError("Batch mode is unsupported")

    def get_action_batch(self, env_idx_list=None):
        raise RuntimeError("Batch mode is unsupported")
