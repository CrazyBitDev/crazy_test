from env.robotic_navigation import RoboticNavigation
from alg.DDQN import DDQN
import time, sys, argparse
import tensorflow as tf
import config
physical_devices = tf.config.list_physical_devices('GPU') 
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)
import traceback

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, DummyVecEnv
from stable_baselines3.common.monitor import Monitor

import wandb
from wandb.integration.sb3 import WandbCallback # Importa il callback di WandB

# from alg.GymnasiumAdapter import GymnasiumAdapter
# from alg.ColregsMetricsCallback import ColregsMetricsCallback

def train(env, args, static=True, colregs=False, emergency_state=False):

	# assert emergency_state only when colregs is enabled
	assert not emergency_state or colregs, "Emergency state can only be enabled when COLREGs are enforced."

	name = f"PPO_{'static' if static else 'dynamic'}{'_colregs' if colregs else ''}_run_{int(time.time())}"

	run = wandb.init(
		project=args.wandb_project_name,
		entity=args.wandb_entity,
		config=vars(args),
		name=name,
		sync_tensorboard=True,
		monitor_gym=True,
		save_code=True,

		# resume="must",
		# id="s053j02f",
	)

	wandb_callback = WandbCallback(
		model_save_path=f"models/{run.id}",
		verbose=2,
	)

	eval_env = DummyVecEnv([lambda: generate_environment(editor_build=False, env_type="testing", render=False, worker_id_offset=1, static=static, colregs=colregs, emergency_state=emergency_state)])

	eval_callback = EvalCallback(eval_env, best_model_save_path=f'./logs_best_model/',
                                     log_path=f'./logs_best_model/',
									 eval_freq=25_000,
									 n_eval_episodes=20,
                                     deterministic=True, render=False)

	# colregs_metrics_callback = ColregsMetricsCallback()

	# Execution of the training loop
	try: 

		policy_kwargs = dict(net_arch=[256, 256])
		
		model = PPO(
			"MlpPolicy", env,
			gamma=args.gamma,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            learning_rate=args.lr,
			policy_kwargs=policy_kwargs,
			ent_coef=0.01,
			tensorboard_log=f"runs/{run.id}"
		)

		# model = PPO.load(f"./models/{run.id}/model.zip",
		# 	env=env,
		# 	tensorboard_log=f"runs/{run.id}",
		# )
		# model.ent_coef = 0.01

		model.learn(
			total_timesteps=500_000,
			callback=[wandb_callback, eval_callback],
			progress_bar=True,
			reset_num_timesteps=False
		)
		model.save(f"BoatSimulator_PPO_{'static' if static else 'dynamic'}{'_colregs' if colregs else ''}_{int(time.time())}")
		run.finish()

		# algo = DDQN(env, args)
		# algo.loop(args)
	
	# Listener for errors and print of the eventual error message
	except Exception as e: 
		print(e)
		print(traceback.format_exc())

	# In any case, close the Unity3D environment
	finally:
		env.close()

def generate_environment(editor_build, env_type, render=False, worker_id_offset=0, static=True, colregs=False, emergency_state=False):

	worker_id = int(round(time.time() % 1, 4)*10000) + worker_id_offset
	env = RoboticNavigation( editor_build=editor_build, worker_id=worker_id, env_type=env_type, render=render, static=static, colregs=colregs, emergency_state=emergency_state )
	# env = GymnasiumAdapter(env)
	# env = Monitor(env, info_keywords=("colregs_penalty", "colregs_violation"))
	return env

# Call the main function
if __name__ == "__main__":

	# Default parameters
	args = config.parse_args()
	# seed = None implies random seed
	editor_build = False
	env_type = "training"
	render = False
	static = False
	colregs = True
	emergency_state = False

	print( "Mobile Robotics Lecture on ML-agents and DDQN! \n")
	env = generate_environment(editor_build, env_type, render=render, static=static, colregs=colregs, emergency_state=emergency_state)
	train(env, args, static=static, colregs=colregs, emergency_state=emergency_state)


