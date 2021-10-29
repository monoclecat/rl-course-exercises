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

    ql_iters = 20000
    t_decay = 10000


args = Args()


class QLearning:
    def __init__(self, env: FrozenLakeEnv, gamma: float, alpha: float, t_decay: int = 10000):
        """A class for tabular Q-Learning with epsilon-greedy exploration.
         The policy is defined implicitly by the Q-function attribute.

        :param env: A FrozenLakeEnv environment
        :param gamma: Discount factor
        :param alpha: Learning rate
        """
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.t_decay = t_decay

        self.n_states = env.nS
        self.n_actions = env.nA
        self.q = np.zeros(shape=(self.n_states, self.n_actions))
        self.t = 0

        self.lc_plotter = LearningCurvePlotter(env, 100)

    def get_greedy_action(self, s: int) -> int:
        """Return the greedy action given a state. Can be used to evaluate the policy after learning.

        :param s: A state
        :return: An action
        """
        return int(np.argmax(self.q[s, :]))

    def sample_action(self, s: int) -> int:
        """Sample an action based on the epsilon greedy strategy with time decaying exploration.

        :param s: A state
        :return: An action
        """
        if np.random.random() < np.minimum(0.99, self.t / self.t_decay):
            action = self.get_greedy_action(s)

        else:
            action = np.random.randint(0, self.n_actions)

        self.t += 1
        return int(action)

    def learn(self, n_iter):
        """Perform n_iter iterations of Q-Learning

        :param n_iter: Number of learning iterations
        :return: The learned Value function and policy
        """
        s = self.env.reset()

        for i in range(n_iter):
            a = self.sample_action(s)

            s_dash, reward, done, info = self.env.step(a)

            # your code here
            # hint: use an if-else to handle terminal states (done==True) differently from non-terminal ones.
            if done:
                td_delta = reward - self.q[s][a]
            else:
                td_delta = reward + self.gamma * np.max(self.q[s_dash]) - self.q[s][a]
            self.q[s][a] = self.q[s][a] + self.alpha * td_delta

            if done:
                s = self.env.reset()
            else:
                s = s_dash

            # evaluate policy success rate
            if i % 100 == 0:
                self.lc_plotter.eval_policy(np.argmax(self.q, axis=1))

        v = np.max(self.q, axis=1)
        pi = np.argmax(self.q, axis=1)

        return v, pi


if __name__ == "__main__":
    env = gym.make("FrozenLake-v1")
    # If you are interested in this, you may also try the 8x8 version and adjust the parameters for Q-Learning
    # This is *not* part of the exercise however. Your figure should still show the 4x4 version
    # for the 8x8 version, run the following code instead:
    # env = gym.make('FrozenLake8x8-v0')

    np.random.seed(0)  # reset the seed in case of multiple cell executions
    q_learning = QLearning(env, args['gamma'], args['alpha'], args['t_decay'])
    v_q, pi_q = q_learning.learn(args['ql_iters'])

    plot_value_f(v_q, env)
    save_figure("vf_q")
    q_learning.lc_plotter.plot_lc()
    save_figure("lc_q")
