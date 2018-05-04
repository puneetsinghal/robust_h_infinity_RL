from math import *
import os
import argparse
from copy import copy

import sys
# sys.path.insert(0, '../paper_implementation/scripts')
import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from time import sleep
import pickle
from IPython import embed
from scipy import integrate as SCI_INT
import h5py
import shutil

try:
	from StringIO import StringIO
except ImportError:
	from io import StringIO

# from helper_functions import *
from Network import Network
from Logger import Logger
from robot import RTAC as RTAC
from robot import LinearSystems as LinearSystems
from robot import Acrobot as Acrobot
from robot import PlanarRR as PlanarRR

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.log_device_placement = False
config.allow_soft_placement = True
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 1
SESS = tf.Session(config=config)

def make_log_dir(log_parent_dir):
	import datetime, os
	current_timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
	return current_timestamp

def generateInputFunction(tspan):
	T = tspan.size
	I1 = np.identity(T)
	KK = np.linspace(1,20,20,True).reshape(1,20)

	control_generator = I1[:,0].reshape(T,1)*KK
	control_generator = np.hstack((control_generator, -I1[:,0].reshape(T,1)*KK))
	control_generator = np.hstack((control_generator, np.random.uniform(-1,1, [T,1])))
	control_generator = np.hstack((control_generator, np.sin(np.pi*tspan.reshape(T,1)*KK)))
	control_generator = np.hstack((control_generator, np.cos(np.pi*tspan.reshape(T,1)*KK)))
	control_generator = np.hstack((control_generator, np.exp(-np.pi*tspan.reshape(T,1)*KK)))

	return control_generator

def train(robot, log_dir, dropout_prob, numIterations=3000):
	
	U, W, X = robot.generateSystemData(control_generator)
	T = robot.T
	M = robot.M

	X_t = np.zeros([M*(T-1), robot.DIM])
	X_tPlus = np.zeros([M*(T-1), robot.DIM])
	U_t = np.zeros([M*(T-1), 1, robot.numInputs])
	U_tPlus = np.zeros([M*(T-1), 1, robot.numInputs])
	W_t = np.zeros([M*(T-1), 1, robot.disturbanceSize])
	W_tPlus = np.zeros([M*(T-1), 1, robot.disturbanceSize])

	for k in range(M):
		X_t[k*(T-1):(k+1)*(T-1), :] = X[k*T:(k+1)*T-1, :]
		X_tPlus[k*(T-1):(k+1)*(T-1), :] = X[k*T+1:(k+1)*T, :]
		U_t[k*(T-1):(k+1)*(T-1), 0, :] = U[k*T:(k+1)*T-1, :]
		U_tPlus[k*(T-1):(k+1)*(T-1), 0, :] = U[k*T+1:(k+1)*T, :]
		W_t[k*(T-1):(k+1)*(T-1), 0, :] = W[k*T:(k+1)*T-1, :]
		W_tPlus[k*(T-1):(k+1)*(T-1), 0, :] = W[k*T+1:(k+1)*T, :]

	G_X_t = np.zeros([M*(T-1), robot.DIM, robot.numInputs])
	G_X_tPlus = np.zeros([M*(T-1), robot.DIM, robot.numInputs])
	K_X_t = np.zeros([M*(T-1), robot.DIM, robot.disturbanceSize])
	K_X_tPlus = np.zeros([M*(T-1), robot.DIM, robot.disturbanceSize])

	for j in range(M*(T-1)):
		# embed()
		G_X_t[j, :, :] = robot.g_function(X_t[j,:])
		G_X_tPlus[j, :, :] = robot.g_function(X_tPlus[j,:])
		K_X_t[j, :, :] = robot.k_function(X_t[j,:])
		K_X_tPlus[j, :, :] = robot.k_function(X_tPlus[j,:])

	# Initialize the variables (i.e. assign their default value)
	init = tf.global_variables_initializer()

	# Add ops to save and restore all the variables.
	saver = tf.train.Saver(max_to_keep=1)

	SESS.run(init)
	# save_path = saver.save(SESS, './')

	shutil.copyfile('main.py', (log_dir + '/main.py'))  
	shutil.copyfile('Network.py', (log_dir + '/Network.py'))  
	shutil.copyfile('Logger.py', (log_dir + '/Logger.py'))  
	shutil.copyfile('robot.py', (log_dir + '/robot.py'))  

	logger 		= Logger('./' + log_dir + '/train_log/')
	filename 	= './' + log_dir + '/model_NN_pickleData'
	modelName 	= './' + log_dir + '/model'

	# feed_dict = {nn.X_t:X_t, nn.X_tPlus:X_tPlus, nn.U_t:U_t, nn.U_tPlus:U_tPlus, nn.W_t:W_t, nn.W_tPlus:W_tPlus,
	# 			nn.G_X_t:G_X_t, nn.G_X_tPlus:G_X_tPlus, nn.K_X_t:K_X_t, nn.K_X_tPlus:K_X_tPlus}
	for step in range(1, numIterations):
		batchIndex = np.random.choice(M*(T-1), 10000)
		feed_dict = {nn.X_t:X_t[batchIndex], nn.X_tPlus:X_tPlus[batchIndex], nn.U_t:U_t[batchIndex], 
						nn.U_tPlus:U_tPlus[batchIndex], nn.W_t:W_t[batchIndex], nn.W_tPlus:W_tPlus[batchIndex],
						nn.G_X_t:G_X_t[batchIndex], nn.G_X_tPlus:G_X_tPlus[batchIndex], nn.K_X_t:K_X_t[batchIndex], 
						nn.K_X_tPlus:K_X_tPlus[batchIndex], nn.dropout_prob:dropout_prob}
		_, residual_error = SESS.run([nn.apply_grads, nn.residual_error], feed_dict=feed_dict)
		average_error = np.sum(residual_error)/(T*M)
		logger.log_scalar(tag='cost',value=average_error, step=step)
		if(step%10 == 0):
			print("the error at step {} is: {}". format(step, average_error))

		if(step%50 == 0 or step == 1):
			save_path = saver.save(SESS, modelName, write_meta_graph=True)
			print("testing after step: {}".format(step))
			test(robot, SESS, log_dir, False)

	save_path = saver.save(SESS, modelName, write_meta_graph=True)

def test(robot, SESS, log_dir, init_flag = True):
	# x0     = np.random.uniform(-1,1,robot.DIM)
	x0 = np.array([np.random.uniform(-1,1,1), np.random.uniform(-1,1,1), 0, 0])
	newT = 100
	dt = 0.033

	totalPoints = int(newT/dt)

	x = np.zeros([totalPoints,robot.DIM])
	x[0,:] = x0
	robot.sess = SESS

	filename 	= './' + log_dir + '/model_NN_pickleData'
	modelName 	= './' + log_dir + '/model'

	if init_flag:
		init = tf.global_variables_initializer()
		saver = tf.train.Saver(max_to_keep=1)

		SESS.run(init)

		saver = tf.train.import_meta_graph(modelName + '.meta')
		saver.restore(SESS, tf.train.latest_checkpoint(log_dir))
		print("Model restored.")

	r = SCI_INT.ode(robot.dynamics_test).set_integrator("dopri5")#, nsteps=1000) 
	r.set_initial_value(x0, 0)
	for i in range(1, totalPoints):
		# embed()
		x[i, :] = r.integrate(r.t+dt)    # get one more value, add it to the array
		if not r.successful():
			raise RuntimeError("Could not integrate")

	t = np.linspace(0, newT, totalPoints)
	uHistory = robot.findControl(x)

	plt.figure()
	plt.rc('text', usetex=True)
	plt.rc('font', family='serif')
	plt.subplot(2, 1, 1)
	x1,=plt.plot(t,x[:,0],'b',label='x1')
	x2,=plt.plot(t,x[:,1],'r',label='x2')
	x3,=plt.plot(t,x[:,2],'m',label='x3')
	if(robot.DIM==4):
		x4,=plt.plot(t,x[:,3],'k',label='x4')
		plt.legend([x1,x2,x3,x4], ['$x_1$','$x_2$','$x_3$', '$x_4$'])
	else:
		plt.legend([x1,x2,x3], ['$x_1$','$x_2$','$x_3$'])
	
	plt.xlabel('Time [s]', fontsize=15)
	plt.ylabel('States', fontsize=15)
	plt.title('$H_{\infty}$ stabilization of the Acrobot problem', fontsize=15)
	plt.tick_params(labelsize=15)

	plt.grid(True)
	plt.subplot(2, 1, 2)
	plt.plot(t,uHistory,'-b')
	plt.xlabel('Time [s]', fontsize=15)
	plt.ylabel('u', fontsize=15)
	plt.title('Controls', fontsize=15)
	plt.tick_params(labelsize=15)
	plt.grid(True)

	plt.show()

	# plt.figure(1)
	# plt.plot(t,x[:,0],'-b','linewidth',1.8)
	# plt.plot(t,x[:,1],'-r','linewidth',1.8)
	# plt.plot(t,x[:,2],'-m','linewidth',1.8)
	# if(robot.DIM==4):
	# 	plt.plot(t,x[:,3],'-k','linewidth',1.8)
	
	# embed()
	# plt.figure(2)
	# plt.plot(t, uHistory[:,0],'-b','linewidth',1.8)
	# plt.show()

def get_stats_test(robot, SESS, log_dir, init_flag = True):
	max_Counts  = 100
	newT = 100
	dt = 0.033
	totalPoints = int(newT/dt)
	robot.sess = SESS
	filename 	= './' + log_dir + '/model_NN_pickleData'
	modelName 	= './' + log_dir + '/model'

	if init_flag:
		init = tf.global_variables_initializer()
		saver = tf.train.Saver(max_to_keep=1)
		SESS.run(init)
		saver = tf.train.import_meta_graph(modelName + '.meta')
		saver.restore(SESS, tf.train.latest_checkpoint(log_dir))
		print("Model restored.")

	r = SCI_INT.ode(robot.dynamics_test).set_integrator("dopri5") 

	CX = np.zeros(max_Counts);
	for count in range(max_Counts):
		x0     = (np.random.rand(robot.DIM)-np.array([0.5,0.5,0.5,0.5]))
		x = np.zeros([totalPoints,robot.DIM])
		x[0,:] = x0
		
		r.set_initial_value(x0, 0)
		for i in range(1, totalPoints):
			# embed()
			x[i, :] = r.integrate(r.t+dt)    # get one more value, add it to the array
			if not r.successful():
				raise RuntimeError("Could not integrate")
		
		X_end =  np.asarray(np.linalg.norm(np.mean(x[-30:,:], axis=0)))
		CX[count] = X_end
		print("testing round: {} done with error: {}".format(count, CX[count,0]))

	pickle.dump(CX,open('error', 'wb'))
	plt.figure(1)
	plt.plot(CX,'-b')
	plt.xlabel('trials', fontsize=15)
    plt.ylabel('Error', fontsize=15)
    plt.title('Average error at end of 100 seconds', fontsize=15)
    plt.tick_params(labelsize=15)
	plt.show()
	mean = np.mean(CX)
	std  = np.std(CX)
	return mean, std

if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--mode', type=str, default='test')
	parser.add_argument('--model', type=str, default=None)
	parser.add_argument('--robot', type=str, default="linear") # linear/ RTAC
	# parser.add_argument('--data', type=str, default=None)

	args = parser.parse_args()

	if(args.robot == 'linear'):
		params = {}
		params['hiddenSize'] = 8
		params['dt'] = 0.01
		params['learningRate'] = 1e-3
		params['numIterations'] = 3000
		params['gamma'] = 6
		params['numState'] = 3
		params['action_size'] = 1
		params['disturbance_size'] = 1
		params['robot'] = args.robot
		params['dropout_prob'] = 1.0
		
		tspan = np.arange(0, 60, params['dt']) 

		T = tspan.size
		u = np.zeros(T)
		w = np.zeros(T)

		nn = Network(params)
		robot = LinearSystems(params, tspan, u, w, nn)
		robot.M  = 20
		robot.T = T
		robot.dt = params['dt']

	elif(args.robot == 'RTAC'):
		params = {}
		params['hiddenSize'] = 8
		params['dt'] = 0.033
		params['learningRate'] = 1e-3
		params['numIterations'] = 10000
		params['gamma'] = 6
		params['numState'] = 4
		params['action_size'] = 1
		params['disturbance_size'] = 1
		params['robot'] = args.robot
		params['dropout_prob'] = 0.9

		tspan = np.arange(0, 1, params['dt']) 

		T = tspan.size
		u = np.zeros(T)
		w = np.zeros(T)

		nn = Network(params)
		robot = RTAC(params, tspan, u, w, nn)
		robot.M  = 100
		robot.T = T
		robot.dt = params['dt']

	elif(args.robot == 'acrobot'):
		params = {}
		params['hiddenSize'] = 8
		params['dt'] = 0.033
		params['learningRate'] = 1e-3
		params['numIterations'] = 10000
		params['gamma'] = 6
		params['numState'] = 4
		params['action_size'] = 1
		params['disturbance_size'] = 1
		params['robot'] = args.robot
		params['dropout_prob'] = 0.9

		params['m1'] = 1.
		params['m2'] = 1.
		params['l1'] = 0.5
		params['l2'] = 0.5 
		params['g'] = -9.81
		params['I1'] = 0.1
		params['I2'] = 0.1

		tspan = np.arange(0, 1, params['dt']) 

		T = tspan.size
		u = np.zeros(T)
		w = np.zeros(T)

		nn = Network(params)
		robot = Acrobot(params, tspan, u, w, nn)
		robot.M  = 300
		robot.T = T
		robot.dt = params['dt']
	
	elif(args.robot == 'planarRR'):
		params = {}
		params['hiddenSize'] = 8
		params['dt'] = 0.033
		params['learningRate'] = 1e-3
		params['numIterations'] = 10000
		params['gamma'] = 6
		params['numState'] = 4
		params['action_size'] = 2
		params['disturbance_size'] = 2
		params['robot'] = args.robot
		params['dropout_prob'] = 0.9

		params['m1'] = 1.
		params['m2'] = 1.
		params['l1'] = 0.5
		params['l2'] = 0.5 
		params['g'] = -9.81
		params['I1'] = 0.1
		params['I2'] = 0.1

		tspan = np.arange(0, 0.1, params['dt']) 

		T = tspan.size
		u = np.zeros([2,T])
		w = np.zeros([2,T])

		nn = Network(params)
		robot = PlanarRR(params, tspan, u, w, nn)
		robot.M  = 300
		robot.T = T
		robot.dt = params['dt']

	#     L = sigmaL(np.zeros(DIM)).size
	if(args.mode == 'train'):
		log_dir 	= make_log_dir('')
		if not os.path.exists(log_dir):
			os.makedirs(log_dir)
		
		control_generator = generateInputFunction(tspan)
		train(robot, log_dir, params['dropout_prob'], params['numIterations'])
		test(robot, SESS, log_dir)

	if(args.mode == 'test'):
		test(robot, SESS, args.model)

	if(args.mode=='full-test'):
		test(robot, SESS, args.model)
		print(get_stats_test(robot, SESS, args.model, False))
	embed()

	# j = legend('$x_1$','$x_2$','$x_3$','$x_4$');
	# set(j,'interpreter','latex','fontsize',28)
	# grid on
	# xlabel('Time [s]','interpreter','latex','fontsize',28);
	# ylabel('States','interpreter','latex','fontsize',28);
	# title('$H_{\infty}$ stabilization of RTAC-nonlinear Problem','interpreter','latex','fontsize',32)