from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

from robolab.tasks.direct.base import BaseAgentCfg


@configclass
class Edu3FlatAgentCfg(BaseAgentCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "edu3_flat_phase_rl-knee-upright-20260720"
        self.wandb_project = "edu3_flat_phase_rl-knee-upright-20260720"
        self.seed = 42
        self.num_steps_per_env = 24
        self.max_iterations = 6000
        self.save_interval = 500
        # EDU3 21-DoF layout differs from Mini; keep symmetry off until mirror indices exist.
        self.policy = RslRlPpoActorCriticCfg(
            class_name="ActorCritic",
            init_noise_std=0.5,
            noise_std_type="scalar",
            actor_hidden_dims=[512, 256, 128],
            critic_hidden_dims=[512, 256, 128],
            activation="elu",
        )
        self.algorithm = RslRlPpoAlgorithmCfg(
            class_name="PPO",
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.005,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
            normalize_advantage_per_mini_batch=False,
            symmetry_cfg=None,
            rnd_cfg=None,
        )
        self.clip_actions = 1.0
