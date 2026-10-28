"""Official simulator-side XPolicyLab entry for one RoboDojo episode."""

from pathlib import Path
import sys


def eval_one_episode(TASK_ENV, model_client):
    # Install this policy directory as a symlink to this repository. Resolving
    # the symlink locates the project without requiring simulator-side pip.
    project_root = Path(__file__).resolve().parents[3]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from embodied_agentic.robodojo_native import run_official_episode

    # RoboDojo has already called model_client.reset() during scene reset.
    # This model client is a transport placeholder; the harness owns its Qwen
    # and Astra providers and executes actions through TASK_ENV in this process.
    del model_client
    return run_official_episode(TASK_ENV)


def eval_one_episode_batch(TASK_ENV, model_client):
    raise ValueError("Embodied_Agentic requires eval_batch: false and one native environment")
