import gif
from gym.envs.toy_text.frozen_lake import FrozenLakeEnv
import numpy as np
import matplotlib.pyplot as plt


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
