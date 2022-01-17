"""
Module containing classes that subclass :code:`ray.rllib.agents.callbacks.DefaultCallbacks`
used in :code:`rllib` experiments.
"""
import logging
from typing import Any, Dict, Mapping

import mercury as me
import numpy as np
import phantom as ph
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.tune.logger import TBXLoggerCallback
from ray.tune.result import TIMESTEPS_TOTAL, TRAINING_ITERATION
from ray.tune.trial import Trial

from .metrics import Metric


class MetricsLoggerCallbacks(DefaultCallbacks):
    def __init__(self, logger_id: me.ID, metrics: Mapping[Any, Metric]) -> None:
        super().__init__()
        self.logger_id: me.ID = logger_id
        self.metrics: Mapping[Any, Metric] = metrics

    def on_episode_start(
        self, *, worker, base_env, policies, episode, env_index, **kwargs
    ) -> None:
        """Callback run on the rollout worker before each episode starts."""

        episode.user_data[self.logger_id] = ph.logging.Logger(self.metrics)

    def on_episode_step(
        self, *, worker, base_env, episode, env_index, **kwargs
    ) -> None:
        """Runs on each episode step."""

        env = base_env.envs[0]
        episode.user_data[self.logger_id].log(env)

    def on_episode_end(
        self, *, worker, base_env, policies, episode, env_index, **kwargs
    ) -> None:
        """Runs when an episode is done."""

        logger = episode.user_data[self.logger_id]

        episode.custom_metrics.update(logger.to_reduced_dict())

    def __call__(self) -> "MetricsLoggerCallbacks":
        return self


# TODO: This should move to a utility script
#
def _fig_to_ndarray(fig):
    import io
    import matplotlib.pyplot as plt

    with io.BytesIO() as buff:
        fig.savefig(buff, format="raw")
        buff.seek(0)
        data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    im = data.reshape((int(h), int(w), -1)).transpose((2, 0, 1))
    im = np.expand_dims(im, axis=0)
    plt.close(fig)
    return im


class NetworkPlotCallbacks(DefaultCallbacks):
    def on_episode_start(
        self, *, worker, base_env, policies, episode, env_index, **kwargs
    ) -> None:
        """Callback run on the rollout worker before each episode starts."""

        # TODO: maybe we should move this to the Graph class
        #
        try:
            import networkx as nx
            import matplotlib.pyplot as plt

            env = base_env.envs[env_index]
            graph = env.network.graph

            fig, ax = plt.subplots(1)
            G = nx.Graph()

            G.add_edges_from(list(graph.edges))
            # TODO: after graph-rewrite merge: G.add_edges_from(list(graph.edges.keys()))

            # remove the EnvironmentActor as it is artificially added
            if G.has_node(ph.env.EnvironmentActor.ID):
                G.remove_node(ph.env.EnvironmentActor.ID)

            pos = nx.spring_layout(G)
            nx.draw(G, pos, ax=ax)
            nx.draw_networkx_labels(G, pos, font_size=20, font_family="sans-serif")

            img = _fig_to_ndarray(fig)
            episode.media.setdefault(f"network/worker{worker.worker_index}", []).append(
                img
            )
        except ImportError:
            logging.warning(
                "`networkx` and `matplotlib` must be installed to be able to log the network"
            )


class TBXExtendedLoggerCallback(TBXLoggerCallback):  # pragma: no cover
    """Extension of the default `TBXLoggerCallback` to support images"""

    def log_trial_result(self, iteration: int, trial: "Trial", result: "Dict"):

        if trial not in self._trial_writer:
            self.log_trial_start(trial)

        step = result.get(TRAINING_ITERATION) or result[TIMESTEPS_TOTAL]

        for name, episodes_imgs in result["episode_media"].items():
            episode_imgs = np.concatenate(episodes_imgs.pop(-1), axis=0)
            self._trial_writer[trial].add_images(name, episode_imgs, global_step=step)

        return super().log_trial_result(iteration, trial, result)
