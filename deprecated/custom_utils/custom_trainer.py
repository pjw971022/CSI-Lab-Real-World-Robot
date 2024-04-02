from fsrl.data import FastCollector
from fsrl.trainer import OffpolicyTrainer
from fsrl.utils import BaseLogger
from fsrl.policy import BasePolicy
from typing import Any, Callable, Dict, Optional, Tuple, Union
import numpy as np

def get_linear_fn(start: float, end: float, end_fraction: float):
    """
    Create a function that interpolates linearly between start and end
    between ``progress_remaining`` = 1 and ``progress_remaining`` = ``end_fraction``.
    This is used in DQN for linearly annealing the exploration fraction
    (epsilon for the epsilon-greedy strategy).

    :params start: value to start with if ``progress_remaining`` = 1
    :params end: value to end with if ``progress_remaining`` = 0
    :params end_fraction: fraction of ``progress_remaining``
        where end is reached e.g 0.1 then end is reached after 10%
        of the complete training process.
    :return: Linear schedule function.
    """

    def func(progress_remaining: float) -> float:
        if (1 - progress_remaining) > end_fraction:
            return end
        else:
            return start + (1 - progress_remaining) * (end - start) / end_fraction

    return func


class CustomTrainer(OffpolicyTrainer):
    def __init__(
        self,
        policy: BasePolicy,
        train_collector: FastCollector,
        test_collector: Optional[FastCollector] = None,
        max_epoch: int = 1000,
        batch_size: int = 512,
        cost_limit: float = np.inf,
        step_per_epoch: int = 10000,
        update_per_step: float = 0.1,
        episode_per_collect: int = 1,
        save_model_interval: int = 1,
        episode_per_test: Optional[int] = None,
        stop_fn: Optional[Callable[[float, float], bool]] = None,
        resume_from_log: bool = False,
        logger: BaseLogger = BaseLogger(),
        verbose: bool = True,
        show_progress: bool = True,
        exploration_fraction: float = 0.25,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.05,
    ):
        super().__init__(
            policy=policy,
            train_collector= train_collector,
            test_collector=test_collector,
            max_epoch=max_epoch,
            batch_size=batch_size,
            cost_limit=cost_limit,
            step_per_epoch=step_per_epoch,
            update_per_step=update_per_step,
            episode_per_collect=episode_per_collect,
            save_model_interval=save_model_interval,
            episode_per_test=episode_per_test,
            stop_fn=stop_fn,
            resume_from_log=resume_from_log,
            logger=logger,
            verbose=verbose,
            show_progress=show_progress,
        )
        self.gradient_steps = 0
        self.exploration_rate = 0.0
        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_final_eps = exploration_final_eps
        self.exploration_fraction = exploration_fraction

        self.exploration_schedule = get_linear_fn(
            self.exploration_initial_eps,
            self.exploration_final_eps,
            self.exploration_fraction,
        )
    def _update_current_progress_remaining(self, num_timesteps: int, total_timesteps: int) -> None:
        self._current_progress_remaining = 1.0 - float(num_timesteps) / float(total_timesteps)

    def train_step(self) -> Dict[str, Any]:
        """Perform one training step."""
        assert self.episode_per_test is not None
        stats_train = self.train_collector.collect(self.episode_per_collect,  exploration_rate=self.exploration_rate)
        # print(f"@@@ Exploration_rate :{self.exploration_rate}")
        self.env_step += int(stats_train["n/st"])
        self.cum_cost += stats_train["total_cost"]
        self.cum_episode += int(stats_train["n/ep"])
        self._update_current_progress_remaining(self.env_step, self.step_per_epoch * self.max_epoch)
        self.exploration_rate = self.exploration_schedule(self._current_progress_remaining)

        self.logger.store(
            **{
                "update/episode": self.cum_episode,
                "update/cum_cost": self.cum_cost,
                "train/reward": stats_train["rew"],
                "train/cost": stats_train["cost"],
                "train/length": int(stats_train["len"]),
            }
        )
        return stats_train