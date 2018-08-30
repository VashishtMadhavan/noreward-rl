from __future__ import print_function
from PIL import Image
from gym.spaces.box import Box
import numpy as np
import gym
from gym import spaces
import gym.wrappers.atari_wrappers as atari_wrappers
import logging
import env_wrapper
import time
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def create_env(env_id, client_id, remotes, **kwargs):
    if 'doom' in env_id.lower() or 'labyrinth' in env_id.lower():
        return create_doom(env_id, client_id, **kwargs)
    if 'mario' in env_id.lower():
        return create_mario(env_id, client_id, **kwargs)
    if 'NoFrameskip' in env_id:
        return create_atari_env(env_id, **kwargs)
    else:
        return create_ple_env(env_id, **kwargs)

def create_ple_env(env_id, record=False, outdir=None, **_):
    env = gym.make(env_id)
    env = env_wrapper.BufferedObsEnv(env, skip=4, shape=(42,42))
    env = env_wrapper.SkipEnv(env, skip=4)
    return env

def create_atari_env(env_id, record=False, outdir=None, **_):
    env = atari_wrappers.make_atari(env_id)
    if record and outdir is not None:
        env = gym.wrappers.Monitor(env, outdir, force=True)
    wrapper_kwargs = {'episode_life': False, 'clip_rewards': False, 'frame_stack': True, 'scale': True}
    env = atari_wrappers.wrap_deepmind(env, **wrapper_kwargs)
    return env

def create_doom(env_id, client_id, envWrap=True, record=False, outdir=None,
                    noLifeReward=False, acRepeat=0, **_):
    from ppaquette_gym_doom import wrappers
    if 'labyrinth' in env_id.lower():
        if 'single' in env_id.lower():
            env_id = 'ppaquette/LabyrinthSingle-v0'
        elif 'fix' in env_id.lower():
            env_id = 'ppaquette/LabyrinthManyFixed-v0'
        else:
            env_id = 'ppaquette/LabyrinthMany-v0'
    elif 'very' in env_id.lower():
        env_id = 'ppaquette/DoomMyWayHomeFixed15-v0'
    elif 'sparse' in env_id.lower():
        env_id = 'ppaquette/DoomMyWayHomeFixed-v0'
    elif 'fix' in env_id.lower():
        if '1' in env_id or '2' in env_id:
            env_id = 'ppaquette/DoomMyWayHomeFixed' + str(env_id[-2:]) + '-v0'
        elif 'new' in env_id.lower():
            env_id = 'ppaquette/DoomMyWayHomeFixedNew-v0'
        else:
            env_id = 'ppaquette/DoomMyWayHomeFixed-v0'
    else:
        env_id = 'ppaquette/DoomMyWayHome-v0'

    # VizDoom workaround: Simultaneously launching multiple vizdoom processes
    # makes program stuck, so use the global lock in multi-threading/processing
    client_id = int(client_id)
    time.sleep(client_id * 10)
    env = gym.make(env_id)
    modewrapper = wrappers.SetPlayingMode('algo')
    obwrapper = wrappers.SetResolution('160x120')
    acwrapper = wrappers.ToDiscrete('minimal')
    env = modewrapper(obwrapper(acwrapper(env)))
    # env = env_wrapper.MakeEnvDynamic(env)  # to add stochasticity

    if record and outdir is not None:
        env = gym.wrappers.Monitor(env, outdir, force=True)

    if envWrap:
        fshape = (42, 42)
        frame_skip = acRepeat if acRepeat>0 else 4
        env.seed(None)
        if noLifeReward:
            env = env_wrapper.NoNegativeRewardEnv(env)
        env = env_wrapper.BufferedObsEnv(env, skip=frame_skip, shape=fshape)
        env = env_wrapper.SkipEnv(env, skip=frame_skip)
    elif noLifeReward:
        env = env_wrapper.NoNegativeRewardEnv(env)

    env = Vectorize(env)
    env = DiagnosticsInfo(env)
    env = Unvectorize(env)
    return env

def create_mario(env_id, client_id, envWrap=True, record=False, outdir=None,
                    noLifeReward=False, acRepeat=0, **_):
    import ppaquette_gym_super_mario
    from ppaquette_gym_super_mario import wrappers
    if '-v' in env_id.lower():
        env_id = 'ppaquette/' + env_id
    else:
        env_id = 'ppaquette/SuperMarioBros-1-1-v0'  # shape: (224,256,3)=(h,w,c)

    # Mario workaround: Simultaneously launching multiple vizdoom processes makes program stuck,
    # so use the global lock in multi-threading/multi-processing
    # see: https://github.com/ppaquette/gym-super-mario/tree/master/ppaquette_gym_super_mario
    client_id = int(client_id)
    time.sleep(client_id * 50)
    env = gym.make(env_id)
    modewrapper = wrappers.SetPlayingMode('algo')
    acwrapper = wrappers.ToDiscrete()
    env = modewrapper(acwrapper(env))
    env = env_wrapper.MarioEnv(env)

    if record and outdir is not None:
        env = gym.wrappers.Monitor(env, outdir, force=True)

    if envWrap:
        frame_skip = acRepeat if acRepeat>0 else 6
        fshape = (42, 42)
        env.seed(None)
        if noLifeReward:
            env = env_wrapper.NoNegativeRewardEnv(env)
        env = env_wrapper.BufferedObsEnv(env, skip=frame_skip, shape=fshape, maxFrames=False)
        if frame_skip > 1:
            env = env_wrapper.SkipEnv(env, skip=frame_skip)
    elif noLifeReward:
        env = env_wrapper.NoNegativeRewardEnv(env)

    env = Vectorize(env)
    env = DiagnosticsInfo(env)
    env = Unvectorize(env)
    # env.close() # TODO: think about where to put env.close !
    return env

# def DiagnosticsInfo(env, *args, **kwargs):
#     return vectorized.VectorizeFilter(env, DiagnosticsInfoI, *args, **kwargs)

# class DiagnosticsInfoI(vectorized.Filter):
#     def __init__(self, log_interval=503):
#         super(DiagnosticsInfoI, self).__init__()

#         self._episode_time = time.time()
#         self._last_time = time.time()
#         self._local_t = 0
#         self._log_interval = log_interval
#         self._episode_reward = 0
#         self._episode_length = 0
#         self._all_rewards = []
#         self._num_vnc_updates = 0
#         self._last_episode_id = -1

#     def _after_reset(self, observation):
#         logger.info('Resetting environment logs')
#         self._episode_reward = 0
#         self._episode_length = 0
#         self._all_rewards = []
#         return observation

#     def _after_step(self, observation, reward, done, info):
#         to_log = {}
#         if self._episode_length == 0:
#             self._episode_time = time.time()

#         self._local_t += 1
#         if info.get("stats.vnc.updates.n") is not None:
#             self._num_vnc_updates += info.get("stats.vnc.updates.n")

#         if self._local_t % self._log_interval == 0:
#             cur_time = time.time()
#             elapsed = cur_time - self._last_time
#             fps = self._log_interval / elapsed
#             self._last_time = cur_time
#             cur_episode_id = info.get('vectorized.episode_id', 0)
#             to_log["diagnostics/fps"] = fps
#             if self._last_episode_id == cur_episode_id:
#                 to_log["diagnostics/fps_within_episode"] = fps
#             self._last_episode_id = cur_episode_id
#             if info.get("stats.gauges.diagnostics.lag.action") is not None:
#                 to_log["diagnostics/action_lag_lb"] = info["stats.gauges.diagnostics.lag.action"][0]
#                 to_log["diagnostics/action_lag_ub"] = info["stats.gauges.diagnostics.lag.action"][1]
#             if info.get("reward.count") is not None:
#                 to_log["diagnostics/reward_count"] = info["reward.count"]
#             if info.get("stats.gauges.diagnostics.clock_skew") is not None:
#                 to_log["diagnostics/clock_skew_lb"] = info["stats.gauges.diagnostics.clock_skew"][0]
#                 to_log["diagnostics/clock_skew_ub"] = info["stats.gauges.diagnostics.clock_skew"][1]
#             if info.get("stats.gauges.diagnostics.lag.observation") is not None:
#                 to_log["diagnostics/observation_lag_lb"] = info["stats.gauges.diagnostics.lag.observation"][0]
#                 to_log["diagnostics/observation_lag_ub"] = info["stats.gauges.diagnostics.lag.observation"][1]

#             if info.get("stats.vnc.updates.n") is not None:
#                 to_log["diagnostics/vnc_updates_n"] = info["stats.vnc.updates.n"]
#                 to_log["diagnostics/vnc_updates_n_ps"] = self._num_vnc_updates / elapsed
#                 self._num_vnc_updates = 0
#             if info.get("stats.vnc.updates.bytes") is not None:
#                 to_log["diagnostics/vnc_updates_bytes"] = info["stats.vnc.updates.bytes"]
#             if info.get("stats.vnc.updates.pixels") is not None:
#                 to_log["diagnostics/vnc_updates_pixels"] = info["stats.vnc.updates.pixels"]
#             if info.get("stats.vnc.updates.rectangles") is not None:
#                 to_log["diagnostics/vnc_updates_rectangles"] = info["stats.vnc.updates.rectangles"]
#             if info.get("env_status.state_id") is not None:
#                 to_log["diagnostics/env_state_id"] = info["env_status.state_id"]

#         if reward is not None:
#             self._episode_reward += reward
#             if observation is not None:
#                 self._episode_length += 1
#             self._all_rewards.append(reward)

#         if done:
#             logger.info('True Game terminating: env_episode_reward=%s episode_length=%s', self._episode_reward, self._episode_length)
#             total_time = time.time() - self._episode_time
#             to_log["global/episode_reward"] = self._episode_reward
#             to_log["global/episode_length"] = self._episode_length
#             to_log["global/episode_time"] = total_time
#             to_log["global/reward_per_time"] = self._episode_reward / total_time
#             self._episode_reward = 0
#             self._episode_length = 0
#             self._all_rewards = []

#         if 'distance' in info: to_log['distance'] = info['distance']  # mario
#         if 'POSITION_X' in info:  # doom
#             to_log['POSITION_X'] = info['POSITION_X']
#             to_log['POSITION_Y'] = info['POSITION_Y']
#         return observation, reward, done, to_log