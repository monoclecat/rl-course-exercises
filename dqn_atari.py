import random
from collections import deque
from collections import namedtuple

# Progress bar
from tqdm import tqdm

import matplotlib.pyplot as plt

# Although we use Deep Learning, we still need to use some CPU memory for the
# Replay Buffer, since Image data normally takes a lot of space and GPU memory
# is more expensive than CPU memory
import numpy as np

# Deep Learning Platform
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# We use stable baseline to pre-process our Atari game data
# detail Stable Baseline is a RL benchmark library with plenty of Algorithms,
# here we borrow their wrappers for data preprocessing and video recording
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.vec_env import VecVideoRecorder

# Set random seeds
np.random.seed(0)
torch.manual_seed(0)
env_seed = 0

# Use GPU to speed up your training
# detail: the code below will detect the hardware you have and set the
# device to Nvidia "cuda" instead of "cpu".
assert torch.cuda.is_available(), "To save your time, please use cuda!"
device = torch.device("cuda")


def show_img(state):
    """
    This is a helper function to plot the environment state.
    Args:
        state: environment state

    Returns:
        None
    """
    if isinstance(state, np.ndarray):
        state_numpy = state[0]
    elif isinstance(state, torch.Tensor):
        state_numpy = state[0].cpu().numpy()
    else:
        raise NotImplementedError
    fig = plt.figure()
    for i in range(4):
        fig.add_subplot(2, 2, i + 1)
        plt.imshow(state_numpy[..., i])
    plt.show()


def normalize(data):
    """
    Image Normalizer, normalize image data from the range [0, 255] to [0, 1]

    Args:
        data: integer data in either tensor or Numpy array, bounded by [0, 255]

    Returns:
        Float data bounded by [0, 1]
    """
    return data / 255.0


def show_avg_reward(list_num_game, list_avg_game_score):
    """
    Plot average game score
    Args:
        list_num_game: x axis for different number of games
        list_avg_game_score: y axis for average game scores

    Returns:
        None
    """
    # clear_output(True)  # For Jupyter
    plt.figure()
    plt.plot(list_num_game, list_avg_game_score)
    plt.xlabel("Num of games")
    plt.ylabel("Avg_game_score")
    plt.show()


# Get our environment and use some helper wrappers to pre-process the data.
# The wrapper will return a 4-frames-long game screenshots sequence as a game state.
# For more preprocessing details, please refer:
# https://danieltakeshi.github.io/2016/11/25/frame-skipping-and-preprocessing-for-deep-q-networks-on-atari-2600-games/
env = make_atari_env('BreakoutDeterministic-v4', n_envs=1, seed=env_seed)
env = VecFrameStack(env, n_stack=4)
num_actions = env.action_space.n

# Record a 500 steps long video in every 50000 training steps
env = VecVideoRecorder(env, './DQN/video',
                       record_video_trigger=lambda x: x % 50000 == 0,
                       video_length=500,
                       name_prefix="DQN_Atari")


class Args:
    """
    Boilerplate for properly accessing the args
    """

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, val):
        setattr(self, key, val)

    BATCH_SIZE = 32
    GAMMA = 0.99
    EPS_START = 1.0
    EPS_END = 0.05

    EPS_FRACTION = 0.1
    # during the first X% of the entire training process

    TARGET_UPDATE_INTERVAL = 10000
    # updating the target network

    LEARNING_RATE = 1e-4
    BUFFER_SIZE = 100000
    TOTAL_STEPS = 500000

    MAX_GRAD_NORM = 10
    # the gradient if it is greater than this value

    TRAIN_FREQ = 4
    # rollouts

    NUM_ACTIONS = num_actions


# Instantiate
args = Args()


def update_eps(total_time_steps):
    """
    This is a helper function to update exploration rate, the exploration rate
    will decrease from 1 to 0.05 during the first 10% of the training steps

    Args:
        total_time_steps: total time steps from the start of the training

    Returns:
        eps: exploration rate

    """
    return max(args.EPS_START - total_time_steps / (args.TOTAL_STEPS *
                                                    args.EPS_FRACTION),
               args.EPS_END)


# Definition of transition stored by the reply buffer
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))


class ReplayMemory(object):
    """
    Definition of reply buffer used by DQN
    """

    def __init__(self, capacity):
        """Initialize the ReplayMemory with certain capacity"""
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        """
        Sample transitions and transfer them from numpy array to PyTorch tensor
        Args:
            batch_size: mini batch size

        Returns:
            5 batches data
        """
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # a detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*random.sample(self.memory, batch_size)))

        # Get each individual batch prepared
        done_b = torch.tensor(np.concatenate(batch.done), device=device,
                              dtype=torch.bool).squeeze(-1)
        state_b = torch.tensor(np.concatenate(batch.state),
                               device=device, dtype=torch.float)
        next_state_b = torch.tensor(np.concatenate(batch.next_state),
                                    device=device, dtype=torch.float)
        action_b = torch.tensor(np.concatenate(batch.action), device=device)
        reward_b = torch.tensor(np.concatenate(batch.reward),
                                device=device).squeeze(-1)
        return state_b, action_b, next_state_b, reward_b, done_b

    def __len__(self):
        """Length of the ReplayMemory"""
        return len(self.memory)


# Instantiate
memory = ReplayMemory(args.BUFFER_SIZE)


class DQN(nn.Module):
    """
    Q-Network with CNN
    """

    def __init__(self, h, w, outputs):
        """
        Initialize the network, which contains 2D convolution layers (image
        process), Normalization layers (offer better numerical stability),
        and Fully Connected layers
        Args:
            h: height of the image in pixel
            w: width of the image in pixel
            outputs: number of actions
        """
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(4, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    def forward(self, x):
        """
        Forward pass function, given a state s, compute value of all actions
        i.e. Q(s, a)
        Called with either one element to determine next action, or a batch
        during optimization.
        Args:
            x: state

        Returns:
            q_s_a = value of actions given this state
        """
        # Shape of x:
        # [batch_size or 1, height=84, width=84, channel=4]
        #
        # Shape of x after swapping the order of the data's axis:
        # [batch_size or 1, channel=4, height=84, width=84]
        #
        # Shape of q_s_a:
        # [batch_size or 1, num_actions=4]

        x = x.to(device)  # to GPU

        # Swap the order of the data's axis, from
        # [batch, height, width, channel] to [batch, channel, height, width]
        x = torch.einsum('...hwc->...chw', x)

        # Forward pass
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        q_s_a = self.head(x.reshape([x.size(0), -1]))
        return q_s_a


# Instantiate policy net and its optimizer, note we send the networks to GPU
policy_net = DQN(h=84, w=84, outputs=num_actions).to(device)
optimizer = optim.Adam(policy_net.parameters(), lr=args.LEARNING_RATE)

# Instantiate target net using NN parameters of the policy net
target_net = DQN(h=84, w=84, outputs=num_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

# We do not need any gradient from target net, turn it into evaluation mode
target_net = target_net.eval()


def select_action(state: torch.Tensor, eps: float):
    """

    Args:
        state: state mini batch
        eps: exploration rate

    Returns:
        selected_actions: Discrete action tensor batch, each action is one
        integer from [0, 1, 2, 3]

    """

    # Shape of state:
    # [batch_size, height=84, width=84, num_channels=4]
    #
    # Shape of selected_actions:
    # [batch_size, 1]

    # Your code starts here
    if random.random() < eps:
        selected_actions = torch.randint(low=0, high=4, size=(state.shape[0], 1), dtype=torch.long, device=device)
    else:
        with torch.no_grad():
            q = policy_net(normalize(state))  # or target net?
        selected_actions = torch.argmax(q, dim=1).unsqueeze(dim=1)
    # Your code ends here

    # Some code to help check the validity of the output
    assert selected_actions.dtype == torch.long
    assert selected_actions.ndim == 2
    assert selected_actions.shape[0] == state.shape[0]
    assert selected_actions.shape[1] == 1

    return selected_actions


def optimize_model():

    # Train until we have enough data in the buffer
    if len(memory) < args.BATCH_SIZE:
        return

    # Sample mini-batch
    state_b, action_b, next_state_b, reward_b, done_b = memory.sample(args.BATCH_SIZE)

    # Your code starts here
    policy_Q_over_a = policy_net.forward(normalize(state_b))
    policy_Q = torch.gather(policy_Q_over_a, 1, action_b)
    target_Q = torch.max(target_net.forward(normalize(next_state_b)), dim=1).values
    target_Q = torch.logical_not(done_b) * target_Q
    bellman_backup = reward_b + args.GAMMA * target_Q
    loss = F.huber_loss(input=policy_Q.squeeze(), target=bellman_backup)
    # Your code ends here

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), args.MAX_GRAD_NORM)
    optimizer.step()


list_num_game = []
list_avg_game_score = []

# Main Loop of training
num_game = 0
state = env.reset()
remaining_lives = 5
game_score = 0
game_scores = deque([], maxlen=10)  # Store the score of the latest 10 games
game_length = 0
num_time_steps = 0

# Progress bar
with tqdm(total=args.TOTAL_STEPS, position=0, leave=True, unit="steps") as pbar:
    # Loop until total time steps has been reached
    while num_time_steps < args.TOTAL_STEPS:

        # Loop until one life is wasted
        while True:

            # Get exploration rate
            eps = update_eps(num_time_steps)

            # Rollout
            with torch.no_grad():

                # Your code starts here
                state = torch.as_tensor(state)
                old_state = state.detach().clone()
                action = select_action(state, eps).cpu().numpy()
                state, reward, done, info = env.step(action)
                memory.push(old_state, action, state, reward, done)
                # Your code ends here

                # Update game score and length and total time steps
                game_score += reward
                game_length += 1
                num_time_steps += 1

            # Optimize model
            if num_time_steps % args.TRAIN_FREQ == 0:
                optimize_model()

            # Update the target network
            if num_time_steps % args.TARGET_UPDATE_INTERVAL == 0:
                target_net.load_state_dict(policy_net.state_dict())

            # One life is wasted
            if done:
                remaining_lives = info[0]["ale.lives"]
                break

        # When one game is over, i.e. all 5 lives are wasted
        if remaining_lives == 0:
            num_game += 1
            pbar.update(game_length)

            # Print some result in the progress bar for every 10 games
            if num_game % 10 == 0:
                pbar.set_description(
                    "Game #{}-{}, Avg_score: {:.3f},"
                    " eps: {:.3f}".format(num_game - 9, num_game, np.asarray(
                        game_scores).mean(), eps))

            # Plot the average reward curve for every 50 games
            if num_game % 50 == 0:
                list_num_game.append(num_game)
                list_avg_game_score.append(np.asarray(game_scores).mean())
                show_avg_reward(list_num_game, list_avg_game_score)

            # Reset game score and length
            game_scores.append(game_score)
            game_score = 0
            game_length = 0
    pbar.update(game_length)
    print("Finished!")