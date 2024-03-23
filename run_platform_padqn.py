import os
import time
from datetime import datetime
import gym
import gym_platform
import pygame
import torch.cuda
from gym.wrappers import Monitor

from wrappers.platform_domain import PlatformFlattenedActionWrapper
from wrappers.wrappers import ScaledStateWrapper, ScaledParameterisedActionWrapper

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import argparse
import json


# os.environ["SDL_VIDEODRIVER"] = "dummy"
# %-----------------------------------------------------------------------------------%
# %--------- FUNCTION FOR PADDING DISCRETE ACTION AND ACTION PARAMETERS --------------%
# %-----------------------------------------------------------------------------------%
def pad_action(act, act_param):
    """
    :purpose: to form the complete action-parameters and pad it with the chosen action (act).
    :param act: {int}
        - the discrete action of the agent (takes values 0, 1, and 2).
    :param act_param: {ndarray(1,)}
        - contains the parameter corresponding to act.
    :return: padded_action: {tuple(2)}
        - first element is act.
        - second element is params which is a list with three arrays (each of the form ndarray(1,))
            the array in the position corresponding to act equals act_param (parameter corresponding to act)
        - padded_action will be used for env.step().
    """
    params = [np.zeros((1,), dtype=np.float32),
              np.zeros((1,), dtype=np.float32),
              np.zeros((1,), dtype=np.float32)]
    params[act][:] = act_param
    padded_action = tuple((act, params))
    return padded_action


# %-----------------------------------------------------------------------------------%
# %-------------------------- FUNCTION FOR EVALUATION --------------------------------%
# %-----------------------------------------------------------------------------------%
def evaluate(env, agent, eval_episodes=1000, render_freq=None):
    """
    :purpose: evaluate a (trained) agent on the given environment (platform-v0) for a total of eval_episodes
    :param env: {ScaledParameterisedActionWrapper}
    :param agent: {PADQNAgent}
    :param eval_episodes: {int}
        - Number of evaluation episodes
    :param render_freq {int}
        - shows all frames of episode when episode_index % render_freq == 0
    :return: np.array(returns)
        - returns is the list of (undiscounted) returns (one for each evaluation episode)
    """
    returns = []
    time_steps = []
    for i in range(eval_episodes):
        state, _ = env.reset()  # state is ndarray(9,) with dtype of float64
        terminal = False
        t = 0
        total_reward = 0.

        while not terminal:
            if render_freq > 0 and i % render_freq == 0:
                env.render()
            t += 1
            state = np.array(state, dtype=np.float32, copy=False)  # changing dtype from float64 to np.float32
            act, act_param, all_action_parameters = agent.act(state)
            # act(int) is the chosen action (0, 1, or 2)
            # act_param(ndarray(1,)) is the parameter corresponding to act
            # all_action_parameters(ndarray(3,)) contains parameters for all actions
            action = pad_action(act, act_param)
            (state, _), reward, terminal, _ = env.step(action)
            # env.step returns a tuple(4) of the form
            # ((state, _), reward, terminal, steps
            total_reward += reward
        returns.append(total_reward)
        time_steps.append(t)
    return np.array(returns)


# %-----------------------------------------------------------------------------------%
# %----------------------------------- PARSER ----------------------------------------%
# %-----------------------------------------------------------------------------------%
def get_parser(add_help=True):
    parser = argparse.ArgumentParser(description='Train and evaluate an RL agent on platform-v0 environment.',
                                     add_help=add_help)
    parser.add_argument('--title', help='Prefix of output files.', type=str,
                        # default="PADQN",
                        default="temp"
                        )
    parser.add_argument('--multipass', default=True,
                        help='Separate action-parameter inputs using multiple Q-network passes.', type=bool)
    parser.add_argument('--seed', default=1, help='Random seed.', type=int)
    parser.add_argument('--evaluation_episodes', help='Episodes over which to evaluate after training.',
                        type=int,
                        # default=1000,
                        default=100
                        )
    parser.add_argument('--episodes', help='Number of episodes for training.', type=int,
                        # default=0,
                        default=200
                        )
    parser.add_argument('--batch_size', help='Minibatch size.', type=int,
                        # default=128,
                        default=20
                        )
    parser.add_argument('--gamma', default=0.9, help='Discount factor.', type=float)
    parser.add_argument('--replay_memory_size', default=10000, help='Replay memory size in transitions.', type=int)
    parser.add_argument('--initial_memory_threshold',
                        help='Number of transitions required to start learning.', type=int,
                        # default=500,
                        default=20
                        )
    parser.add_argument('--epsilon_initial', default=1., help='Initial epsilon value.', type=float)
    parser.add_argument('--epsilon_steps', default=1000,
                        help='Number of episodes over which to linearly anneal epsilon.', type=int)
    parser.add_argument('--epsilon_final', default=0.01, help='Final epsilon value.', type=float)
    parser.add_argument('--alpha_actor', default=0.001, help="QActor's learning rate.", type=float)
    parser.add_argument('--alpha_param_actor', default=0.0001, help="ActorParam's learning rate.", type=float)

    parser.add_argument('--tau_actor', default=0.1, help="Averaging factor for soft update of QActor's target.",
                        type=float)
    parser.add_argument('--tau_param_actor', default=0.001,
                        help="Averaging factor for soft update of ParamActor\'s target.", type=float)
    parser.add_argument('--loss', default='l1_smooth', help='L1 Smooth loss or MSE loss for QActor', type=str)
    parser.add_argument('--use_ornstein_noise', default=True,
                        help='Use Ornstein noise instead of epsilon-greedy with uniform random exploration.', type=bool)
    parser.add_argument('--scale_actions', default=True, help="Scale actions.", type=bool)
    parser.add_argument('--clip_grad', default=10., help="Parameter gradient clipping limit.", type=float)
    parser.add_argument('--results_dir', default="results", help='Results directory.', type=str)
    parser.add_argument('--save_freq', default=1000, help='How often to save models (0 = never).', type=int)
    parser.add_argument('--render_freq', default=100, help='How often to render frames of an episode (0 = never).',
                        type=int)
    parser.add_argument('--check_point', help='Directory specification of model files upto the index. '
                                              'Do not run with episodes > 0.', type=str,
                        default=None
                        # default='./results/PADQN/train/12_21_2022_22_28/models/79999'
                        # default='./results/MPDQN/train/12_21_2022_18_27/models/79999'
                        )
    return parser


def run(args):
    seed = int(args.seed)
    np.random.seed(seed)
    # copy arguments
    title = args.title
    multipass = args.multipass
    seed = args.seed
    evaluation_episodes = args.evaluation_episodes
    episodes = args.episodes
    batch_size, gamma = args.batch_size, args.gamma
    replay_memory_size, initial_memory_threshold = args.replay_memory_size, args.initial_memory_threshold
    epsilon_initial, epsilon_steps, epsilon_final = args.epsilon_initial, args.epsilon_steps, args.epsilon_final
    alpha_actor, alpha_param_actor = args.alpha_actor, args.alpha_param_actor
    tau_actor, tau_param_actor = args.tau_actor, args.tau_param_actor
    loss, use_ornstein_noise = args.loss, args.use_ornstein_noise
    scale_actions, clip_grad = args.scale_actions, args.clip_grad
    results_dir, save_freq = args.results_dir, args.save_freq
    render_freq = args.render_freq
    check_point = args.check_point

    # %-----------------------------------------------------------------------------------%
    # %-------------------------- CREATE DIRECTORIES START -------------------------------%
    # %-----------------------------------------------------------------------------------%
    assert not (check_point is not None and episodes > 0), 'Wrong input. Only use checkpoint for evaluation.'
    timestamp = datetime.now().strftime("%m_%d_%Y_%H_%M")
    # timestamp = ""
    train_dir = os.path.join(results_dir, title, 'train', timestamp)
    test_dir = os.path.join(results_dir, title, 'test', timestamp)
    models_dir = os.path.join(train_dir, 'models')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    vi_dir = os.path.join(train_dir, 'frames')
    os.makedirs(vi_dir, exist_ok=True)
    config = {'title': title, 'multipass': multipass, 'seed': seed,
              'evaluation_episodes': evaluation_episodes, 'episodes': episodes,
              'batch_size': batch_size, 'gamma': gamma,
              'replay_memory_size': replay_memory_size, 'initial_memory_threshold': initial_memory_threshold,
              'epsilon_initial': epsilon_initial, 'epsilon_steps': epsilon_steps, 'epsilon_final': epsilon_final,
              'alpha_actor': alpha_actor, 'alpha_param_actor': alpha_param_actor,
              'tau_actor': tau_actor, 'tau_param_actor': tau_param_actor,
              'loss': loss, 'use_ornstein_noise': use_ornstein_noise,
              'scale_actions': scale_actions, 'clip_grad': clip_grad,
              'results_dir': results_dir, 'save_freq': save_freq,
              'render_freq': render_freq, 'check_point': check_point}
    print('Program Configuration:')
    import pprint
    pretty = pprint.PrettyPrinter(width=30)
    pretty.pprint(config)
    with open(os.path.join(train_dir, 'config.json'), "w") as output_file:
        json.dump(config, output_file, indent=4, sort_keys=True)
    with open(os.path.join(test_dir, 'config.json'), "w") as output_file:
        json.dump(config, output_file, indent=4, sort_keys=True)
    # %-----------------------------------------------------------------------------------%
    # %--------------------------- CREATE DIRECTORIES END --------------------------------%
    # %-----------------------------------------------------------------------------------%

    # %-----------------------------------------------------------------------------------%
    # %--------------------------- SET UP ENVIRONMENT START ------------------------------%
    # %-----------------------------------------------------------------------------------%
    print(f"%--------------------------%")
    print(f"% SET UP ENVIRONMENT START %")
    print(f"%--------------------------%")
    env = gym.make('Platform-v0')
    """
    env has observation and action spaces of the following form
    - observation_space: Tuple(Box(9,), Discrete(200))
    -- low and high of Box(9,) are np.zeros((9,)) and np.ones((9,)) respectively
    - action_space: Tuple(Discrete(3,), Tuple(Box(1,), Box(1,), Box(1,)))
    -- lows and highs are respectively [0] [30] [0] [430] [0] [720] respectively
    """
    # change the environment to always output scaled state and action_parameters
    env = ScaledStateWrapper(env)
    """
    Now, env has observation space of the following form: Tuple(Box(9,), Discrete(200))
     - low and high of Box(9,) are now -np.ones((9,)) and np.ones((9,)) respectively
    """
    env = PlatformFlattenedActionWrapper(env)
    if scale_actions:
        env = ScaledParameterisedActionWrapper(env)
    """
    Now, env has action space of the following form: Tuple(Discrete(3,), Box(1,), Box(1,), Box(1,))
    -- Each Box(1,) has low and high configured to [-1., -1., ..., -1.] and [1., 1., ..., 1.] respectively
    """
    env.seed(seed)  # set up seed
    # wrap in Monitor
    env = Monitor(
        env,
        directory=os.path.join(train_dir, 'env'),  # directory where Monitor wrapper writes its results
        video_callable=lambda episode_index: episode_index % 20 == 0,
        write_upon_reset=False,
        force=True
    )
    # %-----------------------------------------------------------------------------------%
    # %---------------------------- SET UP ENVIRONMENT END -------------------------------%
    # %-----------------------------------------------------------------------------------%

    # %-----------------------------------------------------------------------------------%
    # %----------------------------- SET UP AGENT START ----------------------------------%
    # %-----------------------------------------------------------------------------------%
    print(f"%--------------------------%")
    print(f"%--- SET UP AGENT START ---%")
    print(f"%--------------------------%")
    from agents.padqn import PADQNAgent
    from agents.padqn_multipass import MultiPassPADQNAgent
    agent_class = PADQNAgent
    if multipass:
        agent_class = MultiPassPADQNAgent
    agent = agent_class(
        env.observation_space.spaces[0],
        env.action_space,
        seed=seed,
        batch_size=batch_size,
        gamma=gamma,
        replay_memory_size=replay_memory_size,
        initial_memory_threshold=initial_memory_threshold,
        epsilon_initial=epsilon_initial,
        epsilon_steps=epsilon_steps,
        epsilon_final=epsilon_final,
        alpha_actor=alpha_actor,
        alpha_actor_param=alpha_param_actor,
        tau_actor=tau_actor,
        tau_actor_param=tau_param_actor,
        loss=loss,
        use_ornstein_noise=use_ornstein_noise,
        clip_grad=clip_grad,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    print('Agent Configuration:')
    print(agent)  # printing the configuration of the agent
    # %-----------------------------------------------------------------------------------%
    # %------------------------------- SET UP AGENT END ----------------------------------%
    # %-----------------------------------------------------------------------------------%

    # %-----------------------------------------------------------------------------------%
    # % -------------------------------- TRAIN START ------------------------------------ %
    # %-----------------------------------------------------------------------------------%
    print(f"%--------------------------%")
    print(f"%----- TRAINING START -----%")
    print(f"%--------------------------%")
    print('Training agent over {} episodes'.format(episodes))
    max_steps = 250
    total_reward = 0.
    running_averages = []
    returns = []
    video_index = 0
    start_time = time.time()
    # %------- EPISODES START ----------%
    start, end = 0, episodes
    for i in range(start, end):
        # get initial state
        state, _ = env.reset()  # state is ndarray(9,) with dtype of float64
        state = np.array(state, dtype=np.float32)  # changing dtype to np.float32

        # compute action
        act, act_param, all_action_parameters = agent.act(state)  # {int}, {ndarray(1,)}, {ndarray(3,)}
        # prepare combined action in the form (int, np.array(3,))
        action = pad_action(act, act_param)  #({int}, {ndarray(3,)})
        # set episode_reward to 0
        episode_reward = 0.
        # start episode
        agent.start_episode()
        for j in range(max_steps):
            # render frame
            if render_freq > 0 and i % render_freq == 0:
                env.render()
            # take action and get observation (ret below is of the form ((next_state {ndarray(9,)}, steps {int}),
            # reward {float64}, terminal {bool}, info {dict})
            ret = env.step(action)
            (next_state, steps), reward, terminal, _ = ret
            # update episode_reward
            episode_reward += reward
            # cast next_state to np.float32
            next_state = np.array(next_state, dtype=np.float32)
            # compute next action ({int}, {ndarray(1,)}, {ndarray(3,)})
            next_act, next_act_param, next_all_action_parameters = agent.act(next_state)
            # agent step (learning)
            agent.step(
                state=state,
                action=(act, all_action_parameters),
                reward=reward,
                next_state=next_state,
                next_action=(next_act, next_all_action_parameters),
                terminal=terminal,
                time_steps=steps  # not used.
            )
            #
            act, act_param, all_action_parameters = next_act, next_act_param, next_all_action_parameters
            # prepare next action
            action = pad_action(next_act, next_act_param)
            # state <- next_state
            state = next_state
            if terminal:
                break
        # end episode
        agent.end_episode()
        # update returns, total_reward, and running_averages
        returns.append(episode_reward)
        total_reward += episode_reward
        running_averages.append(total_reward / (i + 1))
        # save simulation results
        # # npy
        np.save(os.path.join(train_dir, agent.name + '_{}_train'.format(str(seed))), returns)
        # # csv
        df = pd.DataFrame([returns, running_averages], index=['returns', 'running_avg'])
        df.T.to_csv(
            os.path.join(train_dir, agent.name + '_{}_train'.format(str(seed)) + '.csv')
        )
        # # png
        fig, axe_s = plt.subplots(nrows=1, ncols=1, figsize=(1.5 * 6.4, 1.5 * 8.0), dpi=300)
        axe_s.plot(range(1, len(running_averages) + 1), running_averages, linestyle='--', linewidth=2.5,
                   label='running_average')
        axe_s.set_title(agent.name + '_{}'.format(str(seed)))
        axe_s.set_xlabel('current episode')
        axe_s.set_ylabel('running average', fontsize=10)
        axe_s.tick_params(axis='x', labelsize=6)
        axe_s.legend(loc='upper left', prop={'size': 6})
        axe_s.grid(True)
        fig.suptitle('Running Average of Returns')
        fig.tight_layout()
        plt.savefig(os.path.join(train_dir, agent.name + '_{}_train'.format(str(seed)) + '.png'))
        plt.close('all')
        # save models
        if save_freq > 0 and i % save_freq == 0:
            agent.save_models(os.path.join(models_dir, str(i)))
        # %---------- print progress ----------%
        if i % 100 == 0:
            print('{0:5s} R:{1:.4f} r100:{2:.4f}'.format(
                str(i), running_averages[-1], np.array(returns[-100:]).mean()))
            # i is current episode, R is running average, r100 is average over last 100 episodes.
        # %---------- print progress and save results ----------%
    end_time = time.time()
    # print training time in minutes.
    print('Training time: {:.2f} minutes'.format((end_time - start_time) / 60.))
    # close environment
    env.close()
    # save models
    if save_freq > 0 and check_point is None:
        agent.save_models(os.path.join(models_dir, str(end - 1)))
        # print average return over all episodes
        if episodes >= 100:
            print('Average return over all episodes: ', sum(returns) / len(returns))
            print('Average return over last 100 episodes: ', sum(returns[-100:]) / 100.)
    # %----------- EPISODES END ----------%
    # %-----------------------------------------------------------------------------------%
    # % ------------------------------- TRAIN END -------------------------------------- %%
    # %-----------------------------------------------------------------------------------%

    # %-----------------------------------------------------------------------------------%
    # %------------------------------- EVALUATION START ---------------------------------%
    # %-----------------------------------------------------------------------------------%
    print(f"%--------------------------%")
    print(f"%----EVALUATION START -----%")
    print(f"%--------------------------%")
    agent.epsilon_final = 0.
    agent.epsilon = 0.
    agent.noise = None
    if check_point is not None and evaluation_episodes > 0:
        agent.load_models(os.path.join(check_point))
        start_time = time.time()
        avg_returns = []
        for s in range(10):
            np.random.seed(s)
            env.seed(s)
            agent._seed(s)
            print('Evaluating agent over {} episodes with seed {}'.format(evaluation_episodes, s))
            evaluation_returns = evaluate(env, agent, eval_episodes=evaluation_episodes, render_freq=render_freq)
            avg_return = sum(evaluation_returns) / len(evaluation_returns)
            avg_returns.append(avg_return)
            print('Average return over all evaluation episodes: ', avg_return)
            df = pd.DataFrame([avg_returns], index=['avg_returns'])
            df.T.to_csv(
                os.path.join(test_dir, agent.name + '.csv')
            )
        end_time = time.time()
        # print evaluation time in minutes.
        print('Evaluation time: {:.2f} minutes'.format((end_time - start_time) / 60.))

    if check_point is None and evaluation_episodes > 0:
        start_time = time.time()
        print('Evaluating agent over {} episodes'.format(evaluation_episodes))
        evaluation_returns = evaluate(env, agent, eval_episodes=evaluation_episodes, render_freq=render_freq)
        print('Average return over all evaluation episodes: ', sum(evaluation_returns) / len(evaluation_returns))
        np.save(os.path.join(train_dir, agent.name + '_{}_eval'.format(str(seed))), evaluation_returns)
        df = pd.DataFrame([evaluation_returns], index=['evaluation_returns'])
        df.T.to_csv(
            os.path.join(train_dir, agent.name + '_{}_eval.csv'.format(str(seed)))
        )
        end_time = time.time()
        # print evaluation time in minutes.
        print('Evaluation time: {:.2f} minutes'.format((end_time - start_time) / 60.))
    # %-----------------------------------------------------------------------------------%
    # %--------------------------------- EVALUATION END ---------------------------------%%
    # %-----------------------------------------------------------------------------------%
    pygame.display.quit()
    pygame.quit()
    return


if __name__ == '__main__':
    args = get_parser().parse_args()
    run(args)
