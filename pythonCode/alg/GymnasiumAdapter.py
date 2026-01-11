import gymnasium
from gymnasium import spaces

class GymnasiumAdapter(gymnasium.Env):
    """
    Adattatore universale da Vecchio Gym (Unity) a Nuovo Gymnasium (SB3)
    """
    def __init__(self, old_env):
        self.env = old_env
        # self.observation_space = old_env.observation_space
        self.observation_space = gymnasium.spaces.Box(
            low=old_env.observation_space.low,
            high=old_env.observation_space.high,
            shape=old_env.observation_space.shape,
            dtype=old_env.observation_space.dtype
        )
        # self.action_space = old_env.action_space
        self.action_space = spaces.Discrete(old_env.action_space.n)
        self.metadata = getattr(old_env, "metadata", {})
        self.render_mode = getattr(old_env, "render_mode", None)

    def reset(self, seed=None, options=None):
        # 1. Gestione del Seed (Gymnasium lo passa qui, Gym no)
        if seed is not None:
            # Se il vecchio env ha il metodo seed, usalo. Altrimenti ignora.
            if hasattr(self.env, "seed"):
                self.env.seed(seed)
        
        # 2. Chiamata al vecchio reset
        obs = self.env.reset()
        
        # 3. Adattamento Output (da 1 a 2 valori)
        # Se obs è già una tupla (obs, info), va bene. Altrimenti aggiungi {}
        if isinstance(obs, tuple) and len(obs) == 2:
            return obs
        return obs, {}

    def step(self, action):
        # 1. Chiamata al vecchio step
        step_result = self.env.step(action)
        
        # 2. Adattamento Output (da 4 a 5 valori)
        if len(step_result) == 4:
            obs, reward, done, info = step_result
            # In Gymnasium, "done" è diviso in "terminated" (finito logico) e "truncated" (limite tempo)
            # Per sicurezza, impostiamo terminated=done e truncated=False
            return obs, reward, done, False, info
            
        return step_result

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()

    # Passa tutte le altre chiamate (es. override_reward) all'ambiente originale
    def __getattr__(self, name):
        return getattr(self.env, name)