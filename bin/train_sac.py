import gym
import numpy as np
import random
import torch
from src.sac import Agent
from src.commons.utils import (
    plot_learning_curve, NormalizedActions, set_seed_everywhere, WandbLogger)
import argparse
import time


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--proj", help="wandb project",
                        type=str, action='store', nargs='?')
    parser.add_argument("--name", help="wandb experiment name",
                        type=str, action='store', nargs='?')
    parser.add_argument("--entity", help="wandb entity name", 
                        type=str, action='store', nargs='?')
    parser.add_argument("--epochs", help="epochs", default=250,
                        type=int)
    parser.add_argument("--seed", help="seed", 
                        type=int, default=42)
    parser.add_argument("--save", help="save frequency", 
                        type=int, default=50)
    parser.add_argument("--batch_size", help="batch_size", 
                        type=int, default=256)
    parser.add_argument("--wandb")

    args = parser.parse_args()

    # Seeding
    SEED = args.seed
    set_seed_everywhere(SEED)

    # Environment
    # env = gym.make('gym_goddard:Goddard-v0')
    env = NormalizedActions(gym.make("Pendulum-v0"))
    action_dim = env.action_space.shape[0]
    state_dim  = env.observation_space.shape[0]

    print(f"Action dim: {action_dim}, State dim: {state_dim}")

    # Method
    method = 'sac'

    # Hyper parameters
    n_games = args.epochs
    batch_size = args.batch_size
    hidden_dim = 128
    reward_scale = 10.
    deterministic = False

    agent = Agent(
        state_dim=state_dim, 
        env=env, 
        batch_size=batch_size, 
        hidden_dim=hidden_dim,
        action_dim=action_dim, 
        action_range=1, 
        reward_scale=reward_scale
    )

    filename = f'{method}.png'
    figure_file = 'plots/' + filename

    best_score = env.reward_range[0]
    score_history = []

    # Logger
    if args.wandb:
        LOGGER = WandbLogger(
                    project=args.proj,
                    name=args.name,
                    entity=args.entity
                )

    time.sleep(1)
    print("Training...")
    start_time = time.time()
    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0

        epoch_time = time.time()
        policy_losses, value_losses, q1_losses, q2_losses = [], [], [], []
        for j in range(200):
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.remember(observation, action, reward, observation_, done)
            if agent.memory.get_length() > agent.batch_size:
                v_, q1_, q2_, p_ = agent.learn(deterministic=deterministic)

                # Log losses
                policy_losses.append(p_.detach().cpu().numpy())
                value_losses.append(v_.detach().cpu().numpy())
                q1_losses.append(q1_.detach().cpu().numpy())
                q2_losses.append(q2_.detach().cpu().numpy())

            observation = observation_
            if done:
                break
        score_history.append(score)
        avg_score = np.mean(score_history[-20:])

        # Logger
        if args.wandb:
            LOGGER.plot_metrics('avg_reward', avg_score)
            LOGGER.plot_epoch_loss('policy_epoch_loss_ave', policy_losses)
            LOGGER.plot_epoch_loss('value_epoch_loss_ave', value_losses)
            LOGGER.plot_epoch_loss('q1_epoch_loss_ave', q1_losses)
            LOGGER.plot_epoch_loss('q2_epoch_loss_ave', q2_losses)

        current_time = time.time()
        elapsed_time = current_time - start_time
        remaining_time = (elapsed_time / (i + 1)) * (n_games - (i + 1))

        if i > 0 and i % args.save == 0 and avg_score > best_score:
            best_score = avg_score
            agent.save_models()
        print("Elapsed time: ", elapsed_time)
        print("Epoch time: ", time.time() - epoch_time)
        print('Episode ', i, 'score %.1f' % score, 'avg_score %.1f' % avg_score)
        print(f'Remaining_time: {remaining_time}s')


    # Plot
    x = [i+1 for i in range(n_games)]
    plot_learning_curve(x, score_history, figure_file)
    
    # Safely close the environment
    env.close()
