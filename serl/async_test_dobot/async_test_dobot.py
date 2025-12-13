#!/usr/bin/env python3
"""
Asynchronous actor-learner loop for Dobot arm (real robot).

This script mirrors async_drq_randomized.py but removes offline demos and
uses the DobotEnv defined in serl_dobot/dobot_gym_env.py. The high-level RL
policy remains unchanged (DrQ), only the environment is swapped.
"""

#真机测试用这个文件夹

from __future__ import annotations

import time
import tqdm
import numpy as np
import jax
import jax.numpy as jnp
from absl import app, flags
from flax.training import checkpoints

import gym

from agentlace.trainer import TrainerServer, TrainerClient
from agentlace.data.data_store import QueuedDataStore

from serl_launcher.agents.continuous.drq import DrQAgent
from serl_launcher.utils.launcher import (
    make_drq_agent,
    make_trainer_config,
    make_wandb_logger,
)
from serl_launcher.data.data_store import MemoryEfficientReplayBufferDataStore
from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.utils.timer_utils import Timer
from serl_launcher.utils.train_utils import concat_batches

from serl_dobot.dobot_gym_env import DobotEnv, DobotEnvConfig

FLAGS = flags.FLAGS

flags.DEFINE_string("env", "DobotEnv-v0", "Name of environment (unused, for logging).")
flags.DEFINE_string("exp_name", None, "Experiment name for wandb logging.")
flags.DEFINE_integer("seed", 42, "Random seed.")

flags.DEFINE_integer("max_traj_length", 100, "Max episode length.")
flags.DEFINE_integer("max_steps", 200000, "Total training steps.")
flags.DEFINE_integer("random_steps", 500, "Steps of random actions before policy.")
flags.DEFINE_integer("training_starts", 500, "Start training after this many steps.")
flags.DEFINE_integer("steps_per_update", 30, "Steps per actor→learner data push.")
flags.DEFINE_integer("log_period", 50, "Logging period (steps).")
flags.DEFINE_integer("eval_period", 0, "Unused; kept for parity.")

flags.DEFINE_integer("critic_actor_ratio", 4, "critic:actor update ratio.")
flags.DEFINE_integer("batch_size", 256, "Batch size.")
flags.DEFINE_integer("replay_buffer_capacity", 200000, "Replay buffer capacity.")

flags.DEFINE_boolean("learner", False, "Run learner node.")
flags.DEFINE_boolean("actor", False, "Run actor node.")
flags.DEFINE_boolean("render", False, "Render flag (unused).")
flags.DEFINE_string("ip", "localhost", "Learner IP for actor to connect.")
flags.DEFINE_string("encoder_type", "resnet-pretrained", "Encoder type.")

flags.DEFINE_integer("eval_checkpoint_step", 0, "Evaluate checkpoint step (actor).")
flags.DEFINE_integer("eval_n_trajs", 5, "Number of eval trajectories.")
flags.DEFINE_boolean("debug", False, "Disable wandb if True.")


def print_green(x):
    return print("\033[92m {}\033[00m".format(x))


# ---------------------------------------------------------------------- #
# Actor loop
# ---------------------------------------------------------------------- #
def actor(agent: DrQAgent, data_store, env, sampling_rng):
    client = TrainerClient(
        "actor_env",
        FLAGS.ip,
        make_trainer_config(),
        data_store,
        wait_for_server=True,
    )

    def update_params(params):
        nonlocal agent
        agent = agent.replace(state=agent.state.replace(params=params))

    client.recv_network_callback(update_params)

    obs, _ = env.reset()
    done = False
    timer = Timer()
    running_return = 0.0

    for step in tqdm.tqdm(range(FLAGS.max_steps), dynamic_ncols=True):
        timer.tick("total")

        with timer.context("sample_actions"):
            if step < FLAGS.random_steps:
                actions = env.action_space.sample()
            else:
                sampling_rng, key = jax.random.split(sampling_rng)
                actions = agent.sample_actions(
                    observations=jax.device_put(obs),
                    seed=key,
                    deterministic=False,
                )
                actions = np.asarray(jax.device_get(actions))

        with timer.context("step_env"):
            next_obs, reward, done, truncated, info = env.step(actions)

            reward = np.asarray(reward, dtype=np.float32)
            running_return += reward
            transition = dict(
                observations=obs,
                actions=actions,
                next_observations=next_obs,
                rewards=reward,
                masks=1.0 - done,
                dones=done,
            )
            data_store.insert(transition)

            obs = next_obs
            if done or truncated:
                stats = {"train": info}
                client.request("send-stats", stats)
                running_return = 0.0
                obs, _ = env.reset()

        if step % FLAGS.steps_per_update == 0:
            client.update()

        timer.tock("total")
        if step % FLAGS.log_period == 0:
            stats = {"timer": timer.get_average_times()}
            client.request("send-stats", stats)


# ---------------------------------------------------------------------- #
# Learner loop
# ---------------------------------------------------------------------- #
def learner(rng, agent: DrQAgent, replay_buffer):
    wandb_logger = make_wandb_logger(
        project="serl_dobot",
        description=FLAGS.exp_name or FLAGS.env,
        debug=FLAGS.debug,
    )

    update_steps = 0

    def stats_callback(type: str, payload: dict) -> dict:
        assert type == "send-stats"
        if wandb_logger is not None:
            wandb_logger.log(payload, step=update_steps)
        return {}

    server = TrainerServer(make_trainer_config(), request_callback=stats_callback)
    server.register_data_store("actor_env", replay_buffer)
    server.start(threaded=True)

    pbar = tqdm.tqdm(
        total=FLAGS.training_starts,
        initial=len(replay_buffer),
        desc="Filling up replay buffer",
        position=0,
        leave=True,
    )
    while len(replay_buffer) < FLAGS.training_starts:
        pbar.update(len(replay_buffer) - pbar.n)
        time.sleep(1)
    pbar.update(len(replay_buffer) - pbar.n)
    pbar.close()

    server.publish_network(agent.state.params)
    print_green("sent initial network to actor")

    replay_iterator = replay_buffer.get_iterator(
        sample_args={
            "batch_size": FLAGS.batch_size,
            "pack_obs_and_next_obs": True,
        },
        device=jax.sharding.PositionalSharding(jax.local_devices()).replicate(),
    )

    timer = Timer()
    for step in tqdm.tqdm(range(FLAGS.max_steps), dynamic_ncols=True, desc="learner"):
        for _ in range(FLAGS.critic_actor_ratio - 1):
            with timer.context("sample_replay_buffer"):
                batch = next(replay_iterator)
            with timer.context("train_critics"):
                agent, critic_info = agent.update_critics(batch)

        with timer.context("train"):
            batch = next(replay_iterator)
            agent, update_info = agent.update_high_utd(batch, utd_ratio=1)

        if step > 0 and step % FLAGS.steps_per_update == 0:
            agent = jax.block_until_ready(agent)
            server.publish_network(agent.state.params)

        if update_steps % FLAGS.log_period == 0 and wandb_logger:
            wandb_logger.log(update_info, step=update_steps)
            wandb_logger.log({"timer": timer.get_average_times()}, step=update_steps)

        update_steps += 1


# ---------------------------------------------------------------------- #
# Main
# ---------------------------------------------------------------------- #
def main(_):
    rng = jax.random.PRNGKey(FLAGS.seed)

    # Environment
    env = DobotEnv(config=DobotEnvConfig(), fake_env=FLAGS.learner, hz=10)
    env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)

    image_keys = [k for k in env.observation_space.keys() if k != "state"]

    rng, agent_rng = jax.random.split(rng)
    agent: DrQAgent = make_drq_agent(
        seed=FLAGS.seed,
        sample_obs=env.observation_space.sample(),
        sample_action=env.action_space.sample(),
        image_keys=image_keys,
        encoder_type=FLAGS.encoder_type,
    )
    agent: DrQAgent = jax.device_put(
        jax.tree_map(jnp.array, agent), jax.sharding.PositionalSharding(jax.local_devices()).replicate()
    )

    if FLAGS.learner:
        rng = jax.device_put(rng, device=jax.sharding.PositionalSharding(jax.local_devices()).replicate())
        replay_buffer = MemoryEfficientReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=FLAGS.replay_buffer_capacity,
            image_keys=image_keys,
        )
        print_green("starting learner loop")
        learner(rng, agent, replay_buffer)

    elif FLAGS.actor:
        sampling_rng = jax.device_put(rng, jax.sharding.PositionalSharding(jax.local_devices()).replicate())
        data_store = QueuedDataStore(2000)
        print_green("starting actor loop")
        actor(agent, data_store, env, sampling_rng)

    else:
        raise NotImplementedError("Must be either a learner or an actor")


if __name__ == "__main__":
    app.run(main)

