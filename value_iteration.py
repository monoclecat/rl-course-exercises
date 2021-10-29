import gym
from gym.envs.toy_text.frozen_lake import FrozenLakeEnv
import numpy as np
from gridworld_utilities import *

np.random.seed(0)
gif_frames = []


class Args:
    # Boilerplate for properly accessing the args
    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, val):
        setattr(self, key, val)

    gamma = 0.99
    alpha = 0.1

    vi_iters = 100


args = Args()


class ValueIteration:
    def __init__(self, env: FrozenLakeEnv, gamma: float):
        """A class for performing Value Iteration

        :param env: A FrozenLakeEnv environment
        :param gamma: Discount factor
        """
        self.env = env
        self.gamma = gamma

        self.n_states = env.nS
        self.n_actions = env.nA

        self.lc_plotter = LearningCurvePlotter(env, 100)

    def value_iteration(self, n_iter: int) -> (list, list):
        """Perform Value Iteration given a DiscreteEnv environment and a discount factor gamma for n_iter iterations.

        :param n_iter: Number of VI iterations
        :return: A tuple (final value_function, optimal policy)
        """

        v = np.zeros(shape=self.n_states)
        pi = np.zeros(shape=self.n_states)

        print("Iteration |  max|V-Vprev|  | # chg actions | V[0]    ")
        print("----------+----------------+---------------+---------")
        for it in range(0, n_iter):
            pi_old = np.copy(pi)
            v_old = np.copy(v)

            # your code here
            # hint: Here, you will need to fill the new policy pi and value function v for all states s.
            # I.e., you need to update pi[s], v[s] for all s.
            for s in range(env.nS):
                q = np.zeros(env.nA)
                for a in range(env.nA):
                    next_state_tuples = env.P[s][a]  # P[s][a] == [(probability, nextstate, reward, done), ...]
                    bellman_backup = np.sum([p*(r + args.gamma*v_old[next_s]) for p, next_s, r, _ in next_state_tuples])
                    q[a] = bellman_backup
                v[s] = np.max(q)
                pi[s] = np.argmax(q)

            # Evaluate policy success rate
            self.lc_plotter.eval_policy(pi)

            max_diff = np.abs(v - v_old).max()
            n_chg_actions = 0 if pi_old is None else (pi != pi_old).sum()
            print("{:4d}      | {:12.5f}   |   {:4d}        | {:8.3f}".format(it, max_diff, n_chg_actions, v[0]))
            gif_frames.append(value_f_gif_frame(v, env, it+1))

            if max_diff < 1e-5:
                # assume convergence if the difference is small enough
                break

        return v, pi


if __name__ == "__main__":
    env = gym.make("FrozenLake-v1")

    v_iteration = ValueIteration(env, args['gamma'])
    v_v, pi_v = v_iteration.value_iteration(args['vi_iters'])

    save_gif(gif_frames, "v_func__value_iteration")
    plot_value_f(v_v, env)
    save_figure("vf_vi")
    v_iteration.lc_plotter.plot_lc()
    save_figure("lc_vi")
