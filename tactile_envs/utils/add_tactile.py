"""Wrapper for resizing observations."""
from __future__ import annotations

import numpy as np

import gymnasium as gym
from gymnasium.error import DependencyNotInstalled
from gymnasium.spaces import Box, Dict


class AddTactile(gym.ObservationWrapper):
   
    def __init__(self, env: gym.Env, use_symlog: bool = True) -> None:
   
        gym.ObservationWrapper.__init__(self, env)

        self.obs_shape = (3, 8 * 16)

        self.use_symlog = use_symlog

        self.observation_space = Dict()
        for key in self.env.observation_space.spaces.keys():
            self.observation_space[key] = self.env.observation_space[key]
        self.observation_space['tactile'] = Box(
            low=-np.inf,
            high=np.inf,
            shape=self.obs_shape,
            dtype=np.float32,
        )

        self.mj_data = self.env.unwrapped.mj_data

    def observation(self, observation):
        
        lh_palm_touch = self.mj_data.sensor('lh_palm_touch').data.reshape((3, 8)) # 3 x nx*ny
        lh_palm_touch = lh_palm_touch[[1, 2, 0]] # zxy -> xyz

        lh_ffproximal_touch = self.mj_data.sensor('lh_ffproximal_touch').data.reshape((3, 8))
        lh_ffproximal_touch = lh_ffproximal_touch[[1, 2, 0]]

        lh_ffmiddle_touch = self.mj_data.sensor('lh_ffmiddle_touch').data.reshape((3, 8))
        lh_ffmiddle_touch = lh_ffmiddle_touch[[1, 2, 0]]

        lh_ffdistal_touch = self.mj_data.sensor('lh_ffdistal_touch').data.reshape((3, 8))
        lh_ffdistal_touch = lh_ffdistal_touch[[1, 2, 0]]

        lh_mfproximal_touch = self.mj_data.sensor('lh_mfproximal_touch').data.reshape((3, 8))
        lh_mfproximal_touch = lh_mfproximal_touch[[1, 2, 0]]

        lh_mfmiddle_touch = self.mj_data.sensor('lh_mfmiddle_touch').data.reshape((3, 8))
        lh_mfmiddle_touch = lh_mfmiddle_touch[[1, 2, 0]]

        lh_mfdistal_touch = self.mj_data.sensor('lh_mfdistal_touch').data.reshape((3, 8))
        lh_mfdistal_touch = lh_mfdistal_touch[[1, 2, 0]]

        lh_rfproximal_touch = self.mj_data.sensor('lh_rfproximal_touch').data.reshape((3, 8))
        lh_rfproximal_touch = lh_rfproximal_touch[[1, 2, 0]]

        lh_rfmiddle_touch = self.mj_data.sensor('lh_rfmiddle_touch').data.reshape((3, 8))
        lh_rfmiddle_touch = lh_rfmiddle_touch[[1, 2, 0]]

        lh_rfdistal_touch = self.mj_data.sensor('lh_rfdistal_touch').data.reshape((3, 8))
        lh_rfdistal_touch = lh_rfdistal_touch[[1, 2, 0]]

        lh_lfmetacarpal_touch = self.mj_data.sensor('lh_lfmetacarpal_touch').data.reshape((3, 8))
        lh_lfmetacarpal_touch = lh_lfmetacarpal_touch[[1, 2, 0]]

        lh_lfproximal_touch = self.mj_data.sensor('lh_lfproximal_touch').data.reshape((3, 8))
        lh_lfproximal_touch = lh_lfproximal_touch[[1, 2, 0]]

        lh_lfmiddle_touch = self.mj_data.sensor('lh_lfmiddle_touch').data.reshape((3, 8))
        lh_lfmiddle_touch = lh_lfmiddle_touch[[1, 2, 0]]
        
        lh_lfdistal_touch = self.mj_data.sensor('lh_lfdistal_touch').data.reshape((3, 8))
        lh_lfdistal_touch = lh_lfdistal_touch[[1, 2, 0]]

        lh_thproximal_touch = self.mj_data.sensor('lh_thproximal_touch').data.reshape((3, 8))
        lh_thproximal_touch = lh_thproximal_touch[[1, 2, 0]]

        lh_thmiddle_touch = self.mj_data.sensor('lh_thmiddle_touch').data.reshape((3, 8))
        lh_thmiddle_touch = lh_thmiddle_touch[[1, 2, 0]]

        lh_thdistal_touch = self.mj_data.sensor('lh_thdistal_touch').data.reshape((3, 8))
        lh_thdistal_touch = lh_thdistal_touch[[1, 2, 0]]

        tactiles = np.concatenate((lh_palm_touch, lh_ffproximal_touch, lh_ffmiddle_touch, lh_ffdistal_touch,
                                   lh_mfproximal_touch, lh_mfmiddle_touch, lh_mfdistal_touch,
                                   lh_rfproximal_touch, lh_rfmiddle_touch, lh_rfdistal_touch,
                                   lh_lfmetacarpal_touch, lh_lfproximal_touch, lh_lfmiddle_touch, lh_lfdistal_touch,
                                   lh_thproximal_touch, lh_thmiddle_touch, lh_thdistal_touch), axis=1)

        if self.use_symlog:
            tactiles = np.sign(tactiles) * np.log(1 + np.abs(tactiles))
        observation['tactile'] = tactiles
        
        return observation