"""Train EDU3 tasks without modifying the upstream task registry."""

from gym.envs import *  # noqa: F401,F403
from gym.envs.edu3_12 import edu3_tasks  # noqa: F401
from gym.scripts.train import train
from gym.utils import get_args


if __name__ == "__main__":
    train(get_args())
