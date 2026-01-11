from stable_baselines3.common.callbacks import BaseCallback

class ColregsMetricsCallback(BaseCallback):
    """
    Legge le info alla fine dell'episodio e le forza dentro il logger di WandB/SB3
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # Controlla se qualche environment ha finito l'episodio
        for i, done in enumerate(self.locals['dones']):
            if done:
                # Recupera il dizionario info dell'ultimo step
                info = self.locals['infos'][i]
                
                # Se troviamo le nostre metriche, le registriamo
                if 'colregs_penalty' in info:
                    self.logger.record('rollout/colregs_penalty', info['colregs_penalty'])
                
                if 'colregs_violation' in info:
                    self.logger.record('rollout/colregs_violation', info['colregs_violation'])
        return True