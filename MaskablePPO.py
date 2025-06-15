import numpy as np

try:
    from stable_baselines3 import PPO
except Exception:  # pragma: no cover - stable_baselines3 may be missing
    PPO = object  # type: ignore


class MaskablePPO(PPO):  # type: ignore[misc]
    """Simple PPO variant that avoids invalid actions using an action mask."""

    def predict(self, observation, state=None, episode_start=None, deterministic=False, action_mask=None):
        action, state = super().predict(observation, state, episode_start, deterministic)

        if action_mask is None:
            env = getattr(self, "env", None)
            if env is not None:
                if hasattr(env, "get_action_mask"):
                    action_mask = env.get_action_mask()
                elif hasattr(env, "envs"):
                    # VecEnv
                    action_mask = np.array([e.get_action_mask() for e in env.envs])

        if action_mask is not None:
            action = self._apply_mask(action, action_mask)
        return action, state

    @staticmethod
    def _apply_mask(action, mask):
        mask = np.array(mask)
        if mask.ndim == 2:
            for i, act in enumerate(np.atleast_1d(action)):
                if mask[i, act] == 0:
                    valid = np.flatnonzero(mask[i])
                    action[i] = np.random.choice(valid)
        else:
            if mask[action] == 0:
                valid = np.flatnonzero(mask)
                action = np.random.choice(valid)
        return action
