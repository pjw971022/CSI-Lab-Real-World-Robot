from fsrl.data import FastCollector
import time
from typing import Any, Callable, Dict, List, Optional
import numpy as np
import torch
from tianshou.data import (
    Batch,
    to_numpy,
)


class CustomCollector(FastCollector):
    def collect(
        self,
        n_episode: int = 1,
        random: bool = False,
        render: bool = False,
        no_grad: bool = True,
        gym_reset_kwargs: Optional[Dict[str, Any]] = None,
        exploration_rate : float = 0.0,
    ) -> Dict[str, Any]:
       
        if n_episode is not None:
            assert n_episode > 0
            ready_env_ids = np.arange(min(self.env_num, n_episode))
            self.data = self.data[:min(self.env_num, n_episode)]
        else:
            raise TypeError("Please specify n_episode"
                            "in FastCollector.collect().")

        start_time = time.time()

        step_count = 0
        total_cost = 0
        termination_count = 0
        truncation_count = 0
        episode_count = 0
        episode_rews = []
        episode_lens = []
        episode_start_indices = []
        
        while True:
            assert len(self.data) == len(ready_env_ids)
            # restore the state: if the last state is None, it won't store
            last_state = self.data.policy.pop("hidden_state", None)

            # get the next action
            if random:
                try:
                    act_sample = [self._action_space[i].sample() for i in ready_env_ids]
                except TypeError:  # envpool's action space is not for per-env
                    act_sample = [self._action_space.sample() for _ in ready_env_ids]
                act_sample = self.policy.map_action_inverse(act_sample)  # type: ignore
                self.data.update(act=act_sample)
            else:
                if no_grad:
                    with torch.no_grad():  # faster than retain_grad version
                        # self.data.obs will be used by agent to get result
                        result = self.policy(self.data, last_state)
                else:
                    result = self.policy(self.data, last_state)
                # update state / act / policy into self.data
                policy = result.get("policy", Batch())
                assert isinstance(policy, Batch)
                state = result.get("state", None)
                if state is not None:
                    policy.hidden_state = state  # save state into buffer
                act = to_numpy(result.act)
                if self.exploration_noise:
                    act = self.policy.exploration_noise(act, self.data, exploration_rate)
                    
                self.data.update(policy=policy, act=act)

            # get bounded and remapped actions first (not saved into buffer)
            action_remap = self.policy.map_action(self.data.act)
            # step in env
            result = self.env.step(action_remap, ready_env_ids)
            if len(result) == 5:
                obs_next, rew, terminated, truncated, info = result
                done = np.logical_or(terminated, truncated)
            elif len(result) == 4:
                obs_next, rew, done, info = result
                if isinstance(info, dict):
                    truncated = info["TimeLimit.truncated"]
                else:
                    truncated = np.array(
                        [
                            info_item.get("TimeLimit.truncated", False)
                            for info_item in info
                        ]
                    )
                terminated = np.logical_and(done, ~truncated)
            else:
                raise ValueError()

            self.data.update(
                obs_next=obs_next,
                rew=rew,
                terminated=terminated,
                truncated=truncated,
                done=done,
                info=info
            )
            if self.preprocess_fn:
                self.data.update(
                    self.preprocess_fn(
                        obs_next=self.data.obs_next,
                        rew=self.data.rew,
                        done=self.data.done,
                        info=self.data.info,
                        policy=self.data.policy,
                        env_id=ready_env_ids,
                    )
                )

            cost = self.data.info.get("cost", np.zeros(rew.shape))
            total_cost += np.sum(cost)
            self.data.update(cost=cost)

            if render:
                self.env.render()

            # add data into the buffer
            ptr, ep_rew, ep_len, ep_idx = self.buffer.add(
                self.data, buffer_ids=ready_env_ids
            )

            # collect statistics
            step_count += len(ready_env_ids)

            if np.any(done):
                env_ind_local = np.where(done)[0]
                env_ind_global = ready_env_ids[env_ind_local]
                episode_count += len(env_ind_local)
                episode_lens.append(ep_len[env_ind_local])
                episode_rews.append(ep_rew[env_ind_local])
                episode_start_indices.append(ep_idx[env_ind_local])
                termination_count += np.sum(terminated)
                truncation_count += np.sum(truncated)
                # now we copy obs_next to obs, but since there might be finished
                # episodes, we have to reset finished envs first.
                self._reset_env_with_ids(env_ind_local, env_ind_global, gym_reset_kwargs)
                for i in env_ind_local:
                    self._reset_state(i)

                # remove surplus env id from ready_env_ids to avoid bias in selecting
                # environments
                if n_episode:
                    surplus_env_num = len(ready_env_ids) - (n_episode - episode_count)
                    if surplus_env_num > 0:
                        mask = np.ones_like(ready_env_ids, dtype=bool)
                        mask[env_ind_local[:surplus_env_num]] = False
                        ready_env_ids = ready_env_ids[mask]
                        self.data = self.data[mask]

            self.data.obs = self.data.obs_next

            if n_episode and episode_count >= n_episode:
                break

        # generate statistics
        self.collect_step += step_count
        self.collect_episode += episode_count
        self.collect_time += max(time.time() - start_time, 1e-9)

        if n_episode:
            self.data = Batch(
                obs={},
                act={},
                rew={},
                cost={},
                terminated={},
                truncated={},
                done={},
                obs_next={},
                info={},
                policy={}
            )
            self.reset_env()

        if episode_count > 0:
            rews, lens = list(map(np.concatenate, [episode_rews, episode_lens]))
            rew_mean = rews.mean()
            len_mean = lens.mean()
        else:
            rew_mean = len_mean = 0

        done_count = termination_count + truncation_count

        return {
            "n/ep": episode_count,
            "n/st": step_count,
            "rew": rew_mean,
            "len": len_mean,
            "total_cost": total_cost,
            "cost": total_cost / episode_count,
            "truncated": truncation_count / done_count,
            "terminated": termination_count / done_count,
        }