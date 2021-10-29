import gif
from gym.envs.toy_text.frozen_lake import FrozenLakeEnv
import numpy as np
import os
import gif
import matplotlib.pyplot as plt
import tqdm
import time


class ProgressBar:
    def __init__(self, num_iterations: int, verbose: bool = True):
        if verbose:  # create a nice little progress bar
            self.scalar_tracker = tqdm.tqdm(total=num_iterations, desc="Scalars", bar_format="{desc}",
                                            position=0, leave=True)
            progress_bar_format = '{desc} {n_fmt:' + str(
                len(str(num_iterations))) + '}/{total_fmt}|{bar}|{elapsed}<{remaining}'
            self.progress_bar = tqdm.tqdm(total=num_iterations, desc='Iteration', bar_format=progress_bar_format,
                                          position=1, leave=True)
        else:
            self.scalar_tracker = None
            self.progress_bar = None

    def __call__(self, **kwargs):
        if self.progress_bar is not None:
            formatted_scalars = {key: "{:.3e}".format(value[-1] if isinstance(value, list) else value)
                                 for key, value in kwargs.items()}
            description = ("Scalars: " + "".join([str(key) + "=" + value + ", "
                                                  for key, value in formatted_scalars.items()]))[:-2]
            self.scalar_tracker.set_description(description)
            self.progress_bar.update(1)


# specify the path to save the recordings of this run to.
data_path = 'data'
data_path = os.path.join(data_path, time.strftime("%d-%m-%Y_%H-%M"))
if not (os.path.exists(data_path)):
    os.makedirs(data_path)


# this function will automatically save your figure
def save_figure(save_name: str) -> None:
    assert save_name is not None, "Need to provide a filename to save to"
    plt.savefig(os.path.join(data_path, save_name + ".png"))


def save_gif(frames: list, save_name: str) -> None:
    assert save_name is not None, "Need to provide a filename to save to"
    gif.save(frames, os.path.join(data_path, save_name + ".gif"), duration=3.5, unit="s", between="startend")


class LearningCurvePlotter:
    def __init__(self, env: FrozenLakeEnv, n_eval: int):
        """This class takes a policy and evaluates it on an environment.
        The quality of the policy is recorded in terms of successfully reaching the goal state.

        :param env: FozenLake env
        :param n_eval: How often to evaluate the policy
        """
        self.n_eval = n_eval
        self.env = env
        self.sr = list()

    def eval_policy(self, pi: np.ndarray, render: bool = False) -> np.ndarray:
        """Evaluate a policy on an environment and return success rate

        :param pi: A policy for env
        :param render: Render the policy
        :return: The mean success rate of the given policy
        """

        successes = []

        for ep in range(self.n_eval):
            s = self.env.reset()
            for i in range(100):
                a = int(pi[s])

                s_dash, reward, done, info = self.env.step(a)

                if render:
                    self.env.render()

                if done:
                    successes.append(reward)
                    break
                else:
                    s = s_dash
        mean_sr = np.mean(successes)
        self.sr.append(mean_sr)

        return mean_sr

    def plot_lc(self):
        """Plot the learning curve"""
        _, ax = plt.subplots()
        ax.plot(self.sr)
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Success Rate")


@gif.frame
def value_f_gif_frame(vf: np.ndarray, env: FrozenLakeEnv, iteration = None):
    plot_value_f(vf, env, iteration)


def plot_value_f(vf: np.ndarray, env: FrozenLakeEnv, iteration = None):
    """Plot the value function for a given environment.

    :param vf: A value function
    :param env: An environment
    :return:
    """
    shape = (env.nrow, env.ncol)

    fig, ax = plt.subplots()
    im = ax.imshow(vf.reshape(shape))
    if iteration is not None:
        plt.title(f"State values V(s) - iteration {iteration}")
    else:
        plt.title(f"State values V(s)")
    #
    for i in range(shape[0]):
        for j in range(shape[1]):
            text = ax.text(j, i, env.desc[i, j].decode(),
                           ha="center", va="center", color="w")
