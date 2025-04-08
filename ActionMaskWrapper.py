import gymnasium as gym


class ActionMaskWrapper(gym.Wrapper):
    def __init__(self, env):
        super(ActionMaskWrapper, self).__init__(env)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.action_mask = info.get("action_mask", [1] * self.action_space.n)
        print("ActionMaskWrapper - reset:")
        print(f"Obs: {obs}, Action Mask: {self.action_mask}")
        return obs, info


    def step(self, action):
        print(f"Action received by ActionMaskWrapper: {action}")
        obs, reward, done, truncated, info = self.env.step(action)
        self.action_mask = info.get("action_mask", [1] * self.action_space.n)
        print("ActionMaskWrapper - step:")
        print(f"Obs: {obs}, Action Mask: {self.action_mask}, Reward: {reward}, Done: {done}")
        return obs, reward, done, truncated, info


    def get_action_mask(self):
        """
        Getter dla obecnej maski akcji
        """
        return self.action_mask
