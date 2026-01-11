import gym; gym.logger.set_level(40)
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper
from mlagents_envs.base_env import TerminalSteps, TerminalStep
# from gym_unity.envs import UnityToGymWrapper
import numpy as np
import os, time
import math


class RoboticNavigation( gym.Env ):
	
	"""
	A class that implements a wrapper between the Unity Engine environment of and a custom Gym environment.
	
	The main motivations for this wrapper are:

		1) Fix the sate
			originally the LiDAR scans arrive with a size of 2 * number_of_scan, beacuse for each direction Unity returns two values, the 
			first one is a float that represent the distance from the first obstacle, nomralized between [0, 1]. The second one is a flag integer [0, 1]
			which indicates if there is an obstacle in the range of the corresponing scan. To avoid a strong correlation between the sensors input of the network, 
			we removed the flag value. This is also to increase the explainability of the state value (useful also for the properties).

		2) Change the reward
			this wrapper allows us to change the reward function without modifying the Unity3D project.
	"""

	def __init__(
			self, step_limit=500, worker_id=0, editor_build=False,
			env_type="training", random_seed=0, render=False,
			static=True, colregs=False, emergency_state=False 
		):

		"""
		Constructor of the class.

		Parameters
		----------
			rendered : bool
				flag to run the envornoment in rendered mode, currently unused (default: False)
		"""

		self.DEBUG = False
		self.DEBUG_REWARD = False

		# If the env_path is given as input override the environment search
		if not editor_build:

			# Sanity check for the 'env_type' option
			assert_message = "Invalid env_type, options [None, training, render]"
			assert (env_type in [None, "training", "render", "gym", "testing"]), assert_message

			# Detect the platform (linux) and load the corresponding environment path
			# # without any specification using the default one.
			if env_type == "training": env_path = "env/linux_training/BoatSim"
			if env_type == "render": env_path = "env/linux_render/SafeRobotics"
			if env_type == "gym": env_path = "env/linux_gym/SafeRobotics"
			if env_type == "testing": env_path = "env/linux_training/BoatSim"
			# if env_type == "testing": env_path = "env/linux_testing/SafeRobotics"

			# If on windows override with the default parameters for it
			if os.name == "nt": "env/windows_training/SafeRobotics"

			additional_args = [
				"--headless" if not render else ""
			]

		# For the editor build force the path to None and the worker id to 0, 
		# assigned values for the editor build.
		else: 
			env_path = None
			worker_id = 0
			print("Start the environment please!")

			additional_args = []

			if not render:
				additional_args.append( "--no-camera" )

		self.static = static
		self.enable_colregs = colregs
		self.enable_emergency_state = emergency_state

		if self.static:
			additional_args.append("--static-obstacles")

		# Load the Unity Environment
		unity_env = UnityEnvironment(env_path, worker_id=worker_id , seed=random_seed, no_graphics=not render, additional_args=additional_args )

		# Convert the Unity Environment in a OpenAI Gym Environment, setting some flag 
		# according with the current setup (only one branch in output and no vision observation)
		self.env = UnityToGymWrapper( unity_env, flatten_branched=True )

		# Override the action space of the wrapper
		#self.action_space = self.env.action_space

		self.action_space = gym.spaces.Discrete( (7 * 7) ) # total actions: 49 (0 to 48)

		self.observation_space = self.env.observation_space

		accelerations = [-10, -5.0, -2.0, 0, 2.0, 5.0, 10.0]
		# turning_rates = [-30, -15, -5, 0, 5, 15, 30]
		# turning_rates = [-60, -30, -10, 0, 10, 30, 60]
		turning_rates = [-90, -45, -20, 0, 20, 45, 90]
		self.actions = [
			np.array([acc, turn]) for acc in accelerations for turn in turning_rates
		]

		if self.DEBUG:
			print("Action Space: ", self.action_space)
			print("Observation Space: ", self.observation_space)
			print("Total actions: ", len(self.actions))

		# Initialize the counter for the maximum step counter
		self.step_counter = 0
		self.colregs_reward_total = 0
		self.max_steps = step_limit
		self.sensing_distance = 10.0
		self.dhull = 2.0
		self.norm_dhull = self.dhull / self.sensing_distance
		self.dmin_ahead = 0.5
		self.kp_a = 1.0
		self.kp_omega = 45.0

		# Coefficients for the reward function

		# self.coefficient_time = -0.5
		# self.coefficient_area = -5.0
		# self.coefficient_goal = 75.0
		# self.coefficient_stopped = -5.0
		# self.coefficient_collision = -50.0
		# self.coefficient_emergency = -0.25

		self.coefficient_time = -5.0
		self.coefficient_area = -250.0
		self.coefficient_goal = 100.0
		self.coefficient_stopped = -25.0
		self.coefficient_collision = -500.0
		self.coefficient_emergency = -0.25

		# Coefficients for the COLREGs reward
		# self.alpha = 75.0

		self.alpha = 5.0 if self.enable_colregs else 0.0
		self.gamma_phi_dyn = 2.0
		self.zeta_obs_d = 0.8

		# self.gamma_starboard = 0.007
		# self.gamma_port = 0.009
		# self.gamma_stern = 0.01
		self.gamma_starboard = 0.007
		self.gamma_port = 0.009
		self.gamma_stern = 0.01

		self.gamma_starboard_pos = 0.004
		self.gamma_starboard_neg = 0.05
		self.gamma_port_pos = 0.007
		self.gamma_port_neg = 0.005
		self.gamma_stern_pos = 0.007
		self.gamma_stern_neg = 0.005

		# Coefficients for the goal reaching reward
		self.coefficient_reach = 100.0

		# Coefficients for the velocity penalty
		self.velocity_high_threshold = 0.8
		self.velocity_low_threshold = 0.2
		self.coefficient_velocity = -6.0

		# Coefficient for the deviation penalty
		self.coefficient_deviate = -1.0 if self.enable_colregs else -2.0
		self.coefficient_deviate = -2.0

		self.sector_threshold = 112.5 * (np.pi / 180.0)

	def reset( self ):

		"""
		Override of the reset function of OpenAI Gym

		Returns
		----------
			state : list
				a list of the observation, with scan_number + 2 elements, the first part contains the information
				about the ldiar sensor and the second angle and distance in respect to the target. All the values
				are normalized between [0, 1]
		"""

		# Reset the counter for the maximum step counter
		self.step_counter = 0
		self.colregs_reward_total = 0

		# Override the state to return with the fixed state, as described in the constructor
		state = self.env.reset()

		self.old_state = state
		self.old_goal_distance = state[4]

		# Call the function that fix the state according with our setup
		#state = self.fix_state( state )

		#
		return state

	def predict_collision(self, sego, sobs, t_pred=5.0, dt=0.5, safety_radius=0.1):
		"""
		Predicts future intersection between ego and obstacle occupancy.
		sego, sobs: dicts with {pos: np.array([x,y]), heading, vel}
		"""
		for t in np.arange(0, t_pred, dt):
			# Predict ego future position (keep course)
			ego_future = sego["pos"] + sego["vel"] * t * np.array([
				math.cos(sego["heading"]), math.sin(sego["heading"])
			])

			# Predict obstacle future position (constant velocity)
			obs_future = sobs["pos"] + sobs["vel"] * t * np.array([
				math.cos(sobs["heading"]), math.sin(sobs["heading"])
			])

			# Compute distance
			d = np.linalg.norm(ego_future - obs_future)
			if d <= safety_radius:
				return True  # Emergency — predicted collision
		return False

	def is_emergency(self, emergency_state):
		if emergency_state:
			# the agent is in emergency state
			# check if can quit
			sector_distances = [self.old_state[10 + i*3] for i in range(4)]
			sector_distance_rates = [self.old_state[12 + i*3] for i in range(4)]

			safe_distance = 0.6 #  60% of max distance

			can_quit = True
			for i, (distance, distance_rate) in enumerate(zip(sector_distances, sector_distance_rates)):
				if i == 3: # behind
					continue # skip check
				if distance < safe_distance or distance_rate < 0:
					can_quit = False
					break

			if can_quit:
				emergency_state = False

		else:
			# the agent is NOT in emergency state
			# check if must enter in emergency state
			
			must_enter = False

			sego = {"pos": np.array([0.0, 0.0]), "vel": self.old_state[0], "heading": self.old_state[1] * math.pi}

			safety_radius = self.sensing_distance * 0.5

			for i in range(4):
				dist = self.old_state[10 + i*3]
				if dist >= 1:
					continue  # ignore far obstacles, too far!
				dist *= self.sensing_distance
				angle = self.old_state[11 + i*3] * math.pi  # [-π, π]
				dist_rate = self.old_state[12 + i*3]
				obs_heading = self.old_state[1] * math.pi + angle
				obs_speed = self.old_state[0] + dist_rate
				sobs = {"pos": np.array([
					dist * math.cos(angle), dist * math.sin(angle)
				]), "vel": obs_speed, "heading": obs_heading}

				if self.predict_collision(sego, sobs, t_pred=2.0, dt=0.5, safety_radius=safety_radius):
					must_enter = True
					break

			if must_enter:
				emergency_state = True

		return emergency_state

	def emergency_mode(self):
		front_dist, left_dist, right_dist, behind_dist = [self.old_state[10 + i*3] for i in range(4)]
		mode = None
		# check if obstacle is mostly behind or on sides (non-front)
		if behind_dist < min(front_dist, left_dist, right_dist) or min(left_dist, right_dist) < front_dist:
			mode = 'stern'
		else:
			# closest front obstacle
			if front_dist > self.dmin_ahead:
				mode = 'base'
			else:
				mode = 'ahead'
		
		return mode


	def step(self, action, emergency_mode=None):

		"""
		Override of the step function of OpenAI Gym

		Parameters
		----------
			action : int
				integer that represent the action that the agent must performs

		Returns
		----------
			state : list
				a list of the observation, with scan_number + 2 elements, the first part contains the information
				about the ldiar sensor and the second angle and distance in respect to the target. All the values
				are normalized between [0, 1]
			reward : float
				a single value that represent the value of the reward function from the tuple (state, action)
			done : bool
				flag that indicates if the current state is terminal
			state : dict
				a dictionary with some additional information, currently empty
		"""

		continuous_action = None

		# if action == 49:
		# 	# emergency state, no base mode, custom handling
			
		# 	if emergency_mode == 'stern':
		# 		continuous_action = np.array([2.0, 0.0])
		# 	#elif emergency_mode == 'ahead':
		# 	else:
				
		# 		front_indices = [0, 1, 2] # front, left and right sectors
    
		# 		# Compute sector distances and angles
		# 		sector_distances = np.array([self.old_state[10 + i*3] * self.sensing_distance for i in front_indices])
		# 		sector_angles = np.array([self.old_state[11 + i*3] * np.pi for i in front_indices])  # [-π, π]

		# 		# Find closest front obstacle
		# 		min_dist = np.min(sector_distances)
		# 		min_idx = front_indices[np.argmin(sector_distances)]
		# 		angle_to_obstacle = sector_angles[min_idx]

		# 		a = self.kp_a * min_dist / self.sensing_distance
		# 		# Turn away from obstacle
		# 		omega = -self.kp_omega * np.sign(angle_to_obstacle) 
		# 		continuous_action = np.array([a, omega])
		# else:
		# 	# normal state, using action returned by NN
		# 	continuous_action = self.actions[action]

		continuous_action = self.actions[action]

		state, reward, done, _ = self.env.step( 
			continuous_action
		)

		# Initialize the empty dictionary
		info = {}

		# Increase the step counter
		self.step_counter += 1

		state[5] = 1 - (self.step_counter / self.max_steps)
		state[22] = 1.0 if self.step_counter >= self.max_steps else 0.0 

		# Computing all the info from the environment 
		# info["goal_reached"] = (reward == 1)
		info["goal_reached"] = (state[26] == 1)
		info["collision"] = (state[25] == 1)
		info["not_moving"] = (state[24] == 1)
		info["out_of_bounds"] = (state[23] == 1)
		# info["cost"] = (state[-1] == 1)
		info["cost"] = (state[26] == 1)
		# info["time_out"] = (self.step_counter >= 300)
		info["time_out"] = (state[22] == 1)

		info['is_success'] = info["goal_reached"]

		# Call the function that fix the state according with our setup
		#state = self.fix_state( state )

		# Overrride the Done function, now from the environment we recived 'done'
		# only for the timeout
		# done = done or info["goal_reached"] or info["collision"] or info["not_moving"] or info['out_of_bounds'] or info["time_out"]
		done = done or info["goal_reached"] or info["collision"] or info['out_of_bounds'] or info["time_out"]

		self.old_state = state

		reward, r_colregs = self.override_reward( state, reward, action, done, emergency_mode )

		self.colregs_reward_total += r_colregs
		info['colregs_penalty'] = self.colregs_reward_total
		info['colregs_violation'] = (self.colregs_reward_total < -0.01)
		
		if self.DEBUG:
			print("vel: {:.2f}, orient: {:.2f}, accel: {:.2f}, turn: {:.2f}".format(
				state[0], state[1], state[2], state[3]
			))
			print("d_goal: {:.2f}, step_left: {:.2f}, goal_orient: {:.2f}".format(
				state[4], state[5], state[6]
			))
			print("long deviation: {:.2f}, lat deviation: {:.2f}, far dist: {:.2f}".format(
				state[7], state[8], state[9]
			))
			print("sectors (dist, angle, rate):")
			for i in range(4):
				print("  sector {}: ({:.2f}, {:.2f}, {:.2f})".format(
					i, state[10 + i * 3], state[11 + i * 3], state[12 + i * 3]
				))
			print("max step reach: {:1d}, out_of_bounds: {:1d}, not_moving: {:1d}, collision: {:1d}, goal_reached: {:1d}".format(
				int(state[22]), int(state[23]), int(state[24]), int(state[25]), int(state[26])
			))
			print("-----------------------------------------------------")

		return state, reward, done, info
	
	def zeta_x(self, phi):
		"""
		zeta_x function is a COLREGs parameter that depends on the angle to the obstacle.

		
		:param phi: Angle to obstacle in radians
		"""
		if phi > 0 and phi < self.sector_threshold: # if obstacle on the right (0°, 112.5°)
			return self.gamma_starboard
		elif phi < 0 and phi > -self.sector_threshold: # if obstacle on the left (-112.5°, 0°)
			return self.gamma_port
		else: # if obstacle is behind
			return self.gamma_stern
	
	def zeta_v(self, phi, v_obs_phi):
		"""
		zeta_v function is a COLREGs parameter that depends on the angle to the obstacle and its relative velocity.
		
		:param phi: Angle to obstacle in radians
		:param v_obs_phi: Relative velocity to obstacle along the line of sight
		"""
		if phi > 0 and phi < self.sector_threshold: # if obstacle on the right (0°, 112.5°)
			if v_obs_phi >= 0: # checking if the obstacle is moving away or approaching
				return self.gamma_starboard_pos
			else:
				return self.gamma_starboard_neg
		elif phi < 0 and phi > -self.sector_threshold: # if obstacle on the left (-112.5°, 0°)
			if v_obs_phi >= 0:  # checking if the obstacle is moving away or approaching
				return self.gamma_port_pos
			else:
				return self.gamma_port_neg
		else: # if obstacle is behind
			if v_obs_phi >= 0:  # checking if the obstacle is moving away or approaching
				return self.gamma_stern_pos
			else: 
				return self.gamma_stern_neg

	def colregs_reward( self, phi, v_obs_phi, d_obs ):
		"""
		COLREGs-based reward component.
		
		:param phi: Angle to obstacle in radians
		:param v_obs_phi: Relative velocity to obstacle along the line of sight
		:param d_obs: Distance to obstacle
		"""

		# The weighting term reduces the penalty for obstacles at large angles
		weighting_term = 1.0 / ( 1.0 + np.exp(self.gamma_phi_dyn * abs(phi)) )

		# The raw penalty is an exponential function of the adjusted distance and relative velocity
		# multiplied by the distance to obstacle
		# Each area (starboard, port and stern) has its own scaling factors through zeta_x and zeta_v
		# zeta_v depends also on the sign of v_obs_phi (approaching or moving away)
		# The distance d_obs is used to scale the penalty (closer obstacles yield higher penalties)
		# The final penalty is negative, as it is a cost to be minimized
		# Note: np.clip is used to avoid overflow in the exponential calculation
		raw_penalty = -self.alpha * np.exp(
			np.clip(
				(self.zeta_v(phi, v_obs_phi) * v_obs_phi - self.zeta_x(phi)) * d_obs,
				-5, 5
			)
		)

		return weighting_term * raw_penalty

	def override_reward( self, state, reward, action, done, emergency_mode = None ):

		r_sparse = self.coefficient_time * state[22] + self.coefficient_area * state[23] + self.coefficient_goal * state[26] \
			+ self.coefficient_stopped * state[24] + self.coefficient_collision * state[25] \
			+ self.coefficient_emergency * (0.0 if emergency_mode == None else 1.0)

		r_colregs = 0
		base_idx = 10
		num_sectors = 4
		for i in range(num_sectors):
			idx = base_idx + i * 3
			d_obs = state[idx]        # distance
			if d_obs >= 1.0:
				continue
			phi = state[idx + 1] * np.pi # relative angle, in [-pi, pi]
			v_obs_phi = state[idx + 2]  # relative velocity (distance_rate)
			v_obs_phi = v_obs_phi * self.sensing_distance  # denormalize
			d_obs = d_obs * self.sensing_distance  # denormalize
			r_colregs += self.colregs_reward(phi, v_obs_phi, d_obs)
		#r_colregs /= num_sectors

		r_goal = self.coefficient_reach * (
			self.old_goal_distance - state[4]
		)
		self.old_goal_distance = state[4]
		
		# if self.static:
		# 	r_facing = 0.5 * (1.0 - 2.0 * abs(state[6]))
		# 	r_goal += r_facing

		r_velocity = 0
		if state[0] > self.velocity_high_threshold:
			r_velocity = self.coefficient_velocity * (state[0] - self.velocity_high_threshold)
		elif state[0] < self.velocity_low_threshold:
			r_velocity = self.coefficient_velocity * (self.velocity_low_threshold - state[0])

		r_deviate = self.coefficient_deviate * min(
			abs(state[8]), self.norm_dhull
		)

		if self.DEBUG_REWARD:
			print("R_sparse: {:.4f}, R_colregs: {:.4f}, R_goal: {:.4f}, R_velocity: {:.4f}, R_deviate: {:.4f}".format(
				r_sparse, r_colregs, r_goal, r_velocity, r_deviate
			))

		reward = r_sparse + r_colregs + r_goal + r_velocity + r_deviate

		return reward, r_colregs
	
	# Override the "close" function
	def close( self ): self.env.close()


	# Override the "render" function
	def render( self ):	pass

