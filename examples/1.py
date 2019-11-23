from utils import *
# from models.mlp_policy import Policy
# from models.mlp_critic import Value
from models.policy import Policy
from models.critic import Value
from models.mlp_policy_disc import DiscretePolicy
from core.a2c import a2c_step
from core.common import estimate_advantages
from core.agent import Agent
import argparse
import gym
import os
import sys
import pickle
import time


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch A2C example')
    parser.add_argument('--env-name', default="BipedalWalker-v2", metavar='G',  # Hopper-v2 CartPole-v1 BipedalWalker-v2
                        help='name of the environment to run')
    parser.add_argument('--model-path', metavar='G',
                        help='path of pre-trained model')
    parser.add_argument('--render', action='store_true', default=False,
                        help='render the environment')
    parser.add_argument('--log-std', type=float, default=-1.0, metavar='G',
                        help='log std for the policy (default: -1.0)')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.95, metavar='G',
                        help='gae (default: 0.95)')
    parser.add_argument('--l2-reg', type=float, default=1e-3, metavar='G',
                        help='l2 regularization regression (default: 1e-3)')
    parser.add_argument('--num-threads', type=int, default=1, metavar='N',
                        help='number of threads for agent (default: 4)')
    parser.add_argument('--seed', type=int, default=1, metavar='N',
                        help='random seed (default: 1)')
    parser.add_argument('--min-batch-size', type=int, default=2048, metavar='N',
                        help='minimal batch size per A2C update (default: 2048)')
    parser.add_argument('--max-iter-num', type=int, default=50000, metavar='N',
                        help='maximal number of main iterations (default: 500)')
    parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                        help='interval between training status logs (default: 1)')
    parser.add_argument('--save-model-interval', type=int, default=100, metavar='N',
                        help="interval between saving model (default: 0, means don't save)")
    parser.add_argument('--gpu-index', type=int, default=0, metavar='N')
    args = parser.parse_args()

    dtype = torch.float64
    torch.set_default_dtype(dtype)
    device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_index)

    """environment"""
    env = gym.make(args.env_name)
    state_dim = env.observation_space.shape[0]
    is_disc_action = len(env.action_space.shape) == 0
    running_state = ZFilter((state_dim,), clip=5)
    # running_reward = ZFilter((1,), demean=False, clip=10)

    """seeding"""
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env.seed(args.seed)

    """define actor and critic"""
    policy_net, value_net, running_state = pickle.load(open(args.model_path, "rb"))

    """create agent"""
    agent = Agent(env, policy_net, device, running_state=running_state, render=args.render,
                  num_threads=args.num_threads)

    while True:
        state = env.reset()
        state = running_state(state)
        for t in range(10000):
            env.render()
            state_var = tensor(state).unsqueeze(0)
            action = policy_net.select_action(state_var)[0].numpy()
            action = action.astype(np.float64)
            next_state, reward, done, _ = env.step(action)