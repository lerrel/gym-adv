import numpy as np
from gym import utils, spaces
from gym.envs.adversarial.mujoco import mujoco_env

class InvertedPendulumEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, 'inverted_pendulum.xml', 2)
        ## Adversarial setup
        self._adv_f_bname = b'pole' #Byte String name of body on which the adversary force will be applied
        bnames = self.model.body_names
        self._adv_bindex = bnames.index(self._adv_f_bname) #Index of the body on which the adversary force will be applied
        adv_max_force = 5.
        high_adv = np.ones(2)*adv_max_force
        low_adv = -high_adv
        self.adv_action_space = spaces.Box(low_adv, high_adv)
        self.pro_action_space = self.action_space

    def _adv_to_xfrc(self, adv_act):
        new_xfrc = self.model.data.xfrc_applied*0.0
        new_xfrc[self._adv_bindex] = np.array([adv_act[0], 0., adv_act[1], 0., 0., 0.])
        self.model.data.xfrc_applied = new_xfrc

    def sample_action(self):
        class act(object):
            def __init__(self,pro=None,adv=None):
                self.pro=pro
                self.adv=adv
        sa = act(self.pro_action_space.sample(), self.adv_action_space.sample())
        return sa

    def _step(self, action):
        if hasattr(action, '__dict__'):
            self._adv_to_xfrc(action.adv)
            a = action.pro
        else:
            a = action

        reward = 1.0
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        notdone = np.isfinite(ob).all() and (np.abs(ob[1]) <= .2)
        done = not notdone
        return ob, reward, done, {}

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-0.01, high=0.01)
        qvel = self.init_qvel + self.np_random.uniform(size=self.model.nv, low=-0.01, high=0.01)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def reset_model_zero(self):
        qpos = self.init_qpos
        qvel = self.init_qvel
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([self.model.data.qpos, self.model.data.qvel]).ravel()

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid=0
        v.cam.distance = v.model.stat.extent
