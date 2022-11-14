from os import truncate
from tarfile import is_tarfile
import numpy as np
import argparse
import torch
from copy import deepcopy

from option_critic import OptionCriticFeatures, OptionCriticConv
from option_critic import critic_loss as critic_loss_fn
from option_critic import actor_loss as actor_loss_fn

from experience_replay import ReplayBuffer
from utils import make_env, to_tensor
from logger import Logger

import time
import wandb

from crafting import MineCraftingEnv
from crafting.task import RewardShaping, TaskObtainItem

from option_graph.metrics.complexity import learning_complexity
from option_graph.metrics.complexity.histograms import nodes_histograms
from option_graph.option import Option

from callbacks import WandbCallback
from plots import save_requirement_graph, save_option_graph

parser = argparse.ArgumentParser(description="Option Critic PyTorch")
parser.add_argument('--agent', default='OptionCritc', help='agent name')
parser.add_argument('--optimal-eps', type=float, default=0.05, help='Epsilon when playing optimally')
parser.add_argument('--frame-skip', default=4, type=int, help='Every how many frames to process')
parser.add_argument('--learning-rate',type=float, default=.0005, help='Learning rate')
parser.add_argument('--gamma', type=float, default=.99, help='Discount rate')
parser.add_argument('--epsilon-start',  type=float, default=1.0, help=('Starting value for epsilon.'))
parser.add_argument('--epsilon-min', type=float, default=.1, help='Minimum epsilon.')
parser.add_argument('--epsilon-decay', type=float, default=20000, help=('Number of steps to minimum epsilon.'))
parser.add_argument('--max-history', type=int, default=10000, help=('Maximum number of steps stored in replay'))
parser.add_argument('--batch-size', type=int, default=32, help='Batch size.')
parser.add_argument('--freeze-interval', type=int, default=200, help=('Interval between target freezes.'))
parser.add_argument('--update-frequency', type=int, default=4, help=('Number of actions before each SGD update.'))
parser.add_argument('--termination-reg', type=float, default=0.01, help=('Regularization to decrease termination prob.'))
parser.add_argument('--entropy-reg', type=float, default=0.01, help=('Regularization to increase policy entropy.'))
parser.add_argument('--num-options', type=int, default=2, help=('Number of options to create.'))
parser.add_argument('--temp', type=float, default=1, help='Action distribution softmax tempurature param.')

parser.add_argument('--reward-shapping', type=int, default=RewardShaping.DIRECT_USEFUL, help=('shapping rewards.'))

parser.add_argument('--max_steps_ep', type=int, default=200, help='number of maximum steps per episode.')
parser.add_argument('--max_steps_total', type=int, default=int(1e5), help='number of maximum steps to take.') # bout 4 million
parser.add_argument('--cuda', type=bool, default=True, help='Enable CUDA training (recommended if possible).')
parser.add_argument('--seed', type=int, default=0, help='Random seed for numpy, torch, random.')
parser.add_argument('--logdir', type=str, default='runs', help='Directory for logging statistics')
parser.add_argument('--exp', type=str, default=None, help='optional experiment name')
parser.add_argument('--switch-goal', type=bool, default=False, help='switch goal after 2k eps')

def run(args):

    # config = {
    #     "agent": "OptionCritic",
    #     'optimal-eps': args.optimal_eps,
    #     'frame-skip': args.frame_skip,
    #     'learninresetn-start': args.epsilon_start,
    #     'epsilon-min': args.epsilon_min,
    #     'epsilon-decay': args.epsilon_decay,
    #     'max-history': args.max_history,
    #     'batch-size': args.batch_size,
    #     'freeze-interval': args.freeze_interval,
    #     'update-frequency': args.update_frequency,
    #     'termination-reg': args.termination_reg,
    #     'entropy-reg': args.entropy_reg,
    #     'num-options': args.num_options,
    #     'temp': args.temp,
    #     'max_steps_ep': args.max_steps_ep,
    #     'max_steps_total': args.max_steps_total,
    #     'cuda': args.cuda,
    #     'seed': args.seed,
    #     'logdir': args.logdir,
    #     'exp': args.exp,
    #     'switch-goal': args.switch_goal
    # }

    run = wandb.init(project="OptionCritic", config={}, monitor_gym=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dirname = f"{timestamp}-{run.id}"
    wandb.config.update(args)
    config = wandb.config

    env = MineCraftingEnv(max_step=config.max_steps_total, seed=config.seed)
    task = TaskObtainItem(env.world,env.world.item_from_name["wood_plank"],reward_shaping=RewardShaping(eval(config.reward_shapping)))
    env.add_task(task)

    is_atari = False
    option_critic = OptionCriticConv if is_atari else OptionCriticFeatures
    device = torch.device('cuda' if torch.cuda.is_available() and config.cuda else 'cpu')

    option_critic = option_critic(
        in_features=env.observation_space.shape[0],
        num_actions=env.action_space.n,
        num_options=config.num_options,
        temperature=config.temp,
        eps_start=config.epsilon_start,
        eps_min=config.epsilon_min,
        eps_decay=config.epsilon_decay,
        eps_test=config.optimal_eps,
        device=device
    )
    # Create a prime network for more stable Q values
    option_critic_prime = deepcopy(option_critic)

    optim = torch.optim.RMSprop(option_critic.parameters(), lr=config.learning_rate)

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    buffer = ReplayBuffer(capacity=config.max_history, seed=config.seed)
    # logger = Logger(logdir=config.logdir, run_name=f"{OptionCriticFeatures.__name__}-{config.env}-{config.exp}-{time.ctime()}")
    logger = Logger(logdir=config.logdir, run_name=f"MineCrafting-{config.exp}-{time.ctime()}")

    # Get & save requirements graph
    requirement_graph_path = save_requirement_graph(
        run_dirname, env.world, title=str(env.world), figsize=(32, 18)
    )

    # Get & save solving option
    all_options = env.world.get_all_options()
    all_options_list = list(all_options.values())
    solving_option: Option = all_options[f"Get {task.goal_item}"]
    solving_option_graph_path = save_option_graph(solving_option, run_dirname)

    # Compute complexities
    used_nodes_all = nodes_histograms(all_options_list)
    lcomp, comp_saved = learning_complexity(solving_option, used_nodes_all)
    print(f"OPTION: {str(solving_option)}: {lcomp} ({comp_saved})")

    wandb.log(
        {
            "task": str(task),
            "solving_option": str(solving_option),
            "learning_complexity": lcomp,
            "total_complexity": lcomp + comp_saved,
            "saved_complexity": comp_saved,
            "requirement_graph": wandb.Image(requirement_graph_path),
            "solving_option_graph": wandb.Image(solving_option_graph_path),
        }
    )
    steps = 0
    if config.switch_goal: print(f"Current goal {env.goal}")
    while steps < config.max_steps_total:

        rewards = 0 ; option_lengths = {opt:[] for opt in range(config.num_options)}

        obs, infos = env.reset()
        invalid_action_mask = torch.tensor(infos["action_is_legal"])
        
        state = option_critic.get_state(to_tensor(obs))
        greedy_option  = option_critic.greedy_option(state)
        current_option = 0

        # Goal switching experiment: run for 1k episodes in fourrooms, switch goals and run for another
        # 2k episodes. In option-critic, if the options have some meaning, only the policy-over-options
        # should be finedtuned (this is what we would hope).
        if config.switch_goal and logger.n_eps == 1000:
            torch.save({'model_params': option_critic.state_dict(),
                        'goal_state': env.goal},
                        f'models/option_critic_seed={config.seed}_1k')
            env.switch_goal()
            print(f"New goal {env.goal}")

        if config.switch_goal and logger.n_eps > 2000:
            torch.save({'model_params': option_critic.state_dict(),
                        'goal_state': env.goal},
                        f'models/option_critic_seed={config.seed}_2k')
            break

        done = False ; ep_steps = 0 ; option_termination = True ; curr_op_len = 0
        while not done and ep_steps < config.max_steps_ep:
            epsilon = option_critic.epsilon

            if option_termination:
                option_lengths[current_option].append(curr_op_len)
                current_option = np.random.choice(config.num_options) if np.random.rand() < epsilon else greedy_option
                curr_op_len = 0

            action, logp, entropy = option_critic.get_action(state, current_option, invalid_action_mask)

            next_obs, reward, done, truncate, infos = env.step(action)
            invalid_action_mask = torch.tensor(infos["action_is_legal"])
            buffer.push(obs, current_option, reward, next_obs, done, invalid_action_mask)
            rewards += reward

            actor_loss, critic_loss = None, None
            if len(buffer) > config.batch_size:
                actor_loss = actor_loss_fn(obs, current_option, logp, entropy, \
                    reward, done, next_obs, option_critic, option_critic_prime, config)
                loss = actor_loss

                if steps % config.update_frequency == 0:
                    data_batch = buffer.sample(config.batch_size)
                    critic_loss = critic_loss_fn(option_critic, option_critic_prime, data_batch, config)
                    loss += critic_loss

                optim.zero_grad()
                loss.backward()
                optim.step()

                if steps % config.freeze_interval == 0:
                    option_critic_prime.load_state_dict(option_critic.state_dict())

            state = option_critic.get_state(to_tensor(next_obs))
            option_termination, greedy_option = option_critic.predict_option_termination(state, current_option)

            # update global steps etc
            steps += 1
            ep_steps += 1
            curr_op_len += 1
            obs = next_obs

            logger.log_data(steps, actor_loss, critic_loss, entropy.item(), epsilon)
            wandb.log({'reward':rewards, 'steps': steps, 'actor_loss':actor_loss, 'critic_loss':critic_loss, 'entropy':entropy, 'epsilon': epsilon})
        
        logger.log_episode(steps, rewards, option_lengths, ep_steps, epsilon)
        
    
    run.finish()

if __name__=="__main__":
    args = parser.parse_args()
    run(args)
    sweep_configuration = {
        'method': 'random',
        'name': 'sweep',
        'metric': {
            'goal': 'maximize',
            'name': 'rewards'
        },
        'parameters':{
            'batch_size': {'values': [16, 32, 64]},
            'learning_rate': {'max': 0.001, 'min': 0.0005},
            'num_options': {'max': 10, 'min': 2},
            'entropy_reg': {'max': 0.1, 'min': 0.01},
            'termination_reg': {'max': 0.1, 'min': 0.01}
        }
    }

    sweep_id = wandb.sweep(sweep=sweep_configuration, project='OptionCritic')
    wandb.agent(sweep_id, function=run(args), count=10)
