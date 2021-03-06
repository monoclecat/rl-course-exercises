import gym
from gym.envs.toy_text.frozen_lake import FrozenLakeEnv
from typing import Tuple
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

    pi_iters = 100


args = Args()


class PolicyIteration:
    def __init__(self, env: FrozenLakeEnv, gamma: float):
        """A class for performing Policy Iteration

        :param env:
        :param gamma:
        """
        self.env = env
        self.gamma = gamma

        self.n_states = env.nS
        self.n_actions = env.nA

        self.lc_plotter = LearningCurvePlotter(env, 100)

    def policy_evaluation(self, pi_prob: np.ndarray, v=None) -> np.ndarray:
        """Perform the Policy Evaluation step given a policy pi and an environment.

        :param pi_prob: Action probabilities
        :param v: Initial value function (optional)
        :return: value function of the provided policy
        """

        if v is None:
            # Initialize value function for policy evaluation
            v = np.zeros(shape=self.n_states)
        for pe_iter in range(10000):
            # save current estimate
            v_prev = np.copy(v)

            # your code here
            # hint: you will need to iterate over states and actions here
            for s in range(env.nS):
                v[s] = 0
                for a in range(env.nA):
                    next_state_tuples = env.P[s][a]  # P[s][a] == [(probability, nextstate, reward, done), ...]
                    bellman_backup = np.sum([p*(r + args.gamma*v_prev[next_s]) for p, next_s, r, _ in next_state_tuples])
                    v[s] += pi_prob[s][a] * bellman_backup

            # run policy evaluation until convergence
            if np.allclose(v, v_prev):
                break

        return v

    def policy_improvement(self, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Perform the Policy Improvement step given a value function.

        :param v: Value function of a policy
        :return: New policy and distribution over action probabilities
        """

        # initialize policy
        pi = np.zeros(shape=self.n_states)  # contains the actual actions
        pi_prob = np.ones(shape=(self.n_states, self.n_actions))  # contains the action probabilities for each state

        # your code here
        # hint: This again requires you to iterate over states and actions in some way.
        # You can use np.argmax() to get the index of the biggest value in an array.
        for s in range(env.nS):
            q = np.zeros(env.nA)
            for a in range(env.nA):
                next_state_tuples = env.P[s][a]  # P[s][a] == [(probability, nextstate, reward, done), ...]
                bellman_backup = np.sum([p * (r + args.gamma * v[next_s]) for p, next_s, r, _ in next_state_tuples])
                q[a] = bellman_backup
            best_a = np.argmax(q)

            pi[s] = best_a
            pi_prob[s] = np.zeros(env.nA)
            pi_prob[s][best_a] = 1.0

        return pi, pi_prob

    def policy_iteration(self, n_iter: int) -> (np.array, np.array):
        """Perform Policy Iteration given a DiscreteEnv environment and a discount factor gamma for n_iter iterations

        :param env: An openAI Gym DiscreteEnv object
        :param gamma: Discount factor
        :param n_iter: Number of PI iterations
        :return: Final value function and optimal policy

        """

        v = np.zeros(shape=self.n_states)
        pi = np.zeros(shape=self.n_states)  # contains the actual actions
        pi_prob = np.ones(
            shape=(self.n_states, self.n_actions)) / self.n_actions  # contains the action probabilities for each state

        print("Iteration |  max|V-Vprev|  | # chg actions | V[0]    ")
        print("----------+----------------+---------------+---------")
        for pi_iter in range(n_iter):
            pi_old = np.copy(pi)
            v_old = np.copy(v)

            # run policy evaluation
            v_init = np.copy(v_old)
            v = self.policy_evaluation(pi_prob, v_init)

            # run policy improvement
            pi, pi_prob = self.policy_improvement(v)

            # evaluate policy success rate
            self.lc_plotter.eval_policy(pi)

            max_diff = np.abs(v - v_old).max()

            n_chg_actions = 0 if pi_old is None else (pi != pi_old).sum()
            print("{:4d}      | {:12.5f}   |   {:4d}        | {:8.3f}".format(pi_iter, max_diff, n_chg_actions, v[0]))
            gif_frames.append(value_f_gif_frame(v, env, pi_iter+1))

            if max_diff < 1e-5:
                break

        return v, pi


if __name__ == "__main__":
    env = gym.make("FrozenLake-v1")

    p_iteration = PolicyIteration(env, args['gamma'])
    v_p, pi_p = p_iteration.policy_iteration(args['pi_iters'])

    save_gif(gif_frames, "v_func__policy_iteration")
    plot_value_f(v_p, env)
    save_figure("vf_pi")
    p_iteration.lc_plotter.plot_lc()
    save_figure("lc_pi")
