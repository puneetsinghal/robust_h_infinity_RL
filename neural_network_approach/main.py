from math import *
import os
import argparse
from copy import copy

import sys
sys.path.insert(0, '../paper_implementation/scripts')
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

from helper_functions import *

try:
	from StringIO import StringIO
except ImportError:
	from io import StringIO

from Network import Network
from Logger import Logger
from robot import RTAC as RTAC

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.log_device_placement = False
config.allow_soft_placement = True
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 1
SESS = tf.Session(config=config)

M  = 20
dt = 0.033
DIM = 4

tspan = np.arange(0,60,dt) 

T = tspan.size
u = np.zeros(T)
w = np.zeros(T)

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
	control_generator = np.hstack((control_generator, np.random.rand(T,1)))
	control_generator = np.hstack((control_generator, np.sin(np.pi*tspan.reshape(T,1)*KK)))
	control_generator = np.hstack((control_generator, np.cos(np.pi*tspan.reshape(T,1)*KK)))
	control_generator = np.hstack((control_generator, np.exp(-np.pi*tspan.reshape(T,1)*KK)))

	return control_generator

def train(robot, log_dir):
	
	U, W, X = robot.generateSystemData(control_generator, dt, T, M)

	X_t = np.zeros([M*(T-1), DIM])
	X_tPlus = np.zeros([M*(T-1), DIM])
	U_t = np.zeros([M*(T-1), 1, 1])
	U_tPlus = np.zeros([M*(T-1), 1, 1])
	W_t = np.zeros([M*(T-1), 1, 1])
	W_tPlus = np.zeros([M*(T-1), 1, 1])

	for k in range(M):
		X_t[k*(T-1):(k+1)*(T-1), :] = X[k*T:(k+1)*T-1, :]
		X_tPlus[k*(T-1):(k+1)*(T-1), :] = X[k*T+1:(k+1)*T, :]
		U_t[k*(T-1):(k+1)*(T-1), 0, :] = U[k*T:(k+1)*T-1, :]
		U_tPlus[k*(T-1):(k+1)*(T-1), 0, :] = U[k*T+1:(k+1)*T, :]
		W_t[k*(T-1):(k+1)*(T-1), 0, :] = W[k*T:(k+1)*T-1, :]
		W_tPlus[k*(T-1):(k+1)*(T-1), 0, :] = W[k*T+1:(k+1)*T, :]

	G_X_t = np.zeros([M*(T-1), DIM, 1])
	G_X_tPlus = np.zeros([M*(T-1), DIM, 1])
	K_X_t = np.zeros([M*(T-1), DIM, 1])
	K_X_tPlus = np.zeros([M*(T-1), DIM, 1])

	for j in range(M*(T-1)):
		G_X_t[j, :, :] = g_function(X_t[j,:], DIM)
		G_X_tPlus[j, :, :] = g_function(X_tPlus[j,:], DIM)
		K_X_t[j, :, :] = k_function(X_t[j,:], DIM)
		K_X_tPlus[j, :, :] = k_function(X_tPlus[j,:], DIM)

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

		
	feed_dict = {nn.X_t:X_t, nn.X_tPlus:X_tPlus, nn.U_t:U_t, nn.U_tPlus:U_tPlus, nn.W_t:W_t, nn.W_tPlus:W_tPlus,
				nn.G_X_t:G_X_t, nn.G_X_tPlus:G_X_tPlus, nn.K_X_t:K_X_t, nn.K_X_tPlus:K_X_tPlus}

	for step in range(1, 3000):
		_, residual_error = SESS.run([nn.apply_grads, nn.residual_error], feed_dict=feed_dict)
		average_error = np.sum(residual_error)/(T*M)
		logger.log_scalar(tag='cost',value=average_error, step=step)
		if(step%10 == 0):
			print("the error at step {} is: {}". format(step, average_error))

		if(step%10 == 0 or step == 1):
			save_path = saver.save(SESS, modelName, write_meta_graph=True)
			print("testing after step: {}".format(step))
			test(robot, SESS, log_dir, True)

	save_path = saver.save(SESS, modelName, write_meta_graph=True)

def test(robot, SESS, log_dir, avoidInit = False):
	x0     = np.random.rand(DIM)

	newT = 100
	dt = 0.033

	totalPoints = int(newT/dt)

	x = np.zeros([totalPoints,4])
	x[0,:] = x0
	robot.sess = SESS

	filename 	= './' + log_dir + '/model_NN_pickleData'
	modelName 	= './' + log_dir + '/model'

	if not avoidInit:
		init = tf.global_variables_initializer()
		saver = tf.train.Saver(max_to_keep=1)

		SESS.run(init)

		saver = tf.train.import_meta_graph(modelName + '.meta')
		saver.restore(SESS, tf.train.latest_checkpoint(log_dir))
		print("Model restored.")

	r = SCI_INT.ode(robot.dynamics).set_integrator("dopri5") 
	r.set_initial_value(x0, 0)
	for i in range(1, totalPoints):
		# embed()
		x[i, :] = r.integrate(r.t+dt)    # get one more value, add it to the array
		if not r.successful():
			raise RuntimeError("Could not integrate")

	t = np.linspace(0, newT, totalPoints)

	plt.figure(1)
	plt.plot(t,x[:,0],'-b','linewidth',1.8)
	plt.plot(t,x[:,1],'-r','linewidth',1.8)
	plt.plot(t,x[:,2],'-m','linewidth',1.8)
	plt.plot(t,x[:,3],'-k','linewidth',1.8)
	plt.show()

if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--mode', type=str, default='test')
	parser.add_argument('--model', type=str, default=None)
	# parser.add_argument('--data', type=str, default=None)

	args = parser.parse_args()

	params = {}
	params['hiddenSize'] = 8
	params['dt'] = 0.033
	params['learningRate'] = 1e-2
	params['gamma'] = 6
	params['numState'] = DIM
	params['action_size'] = 1
	params['disturbance_size'] = 1
	
	
	nn = Network(params)
	robot = RTAC(DIM, tspan, u, w, nn)
	#     L = sigmaL(np.zeros(DIM)).size
	if(args.mode == 'train'):
		log_dir 	= make_log_dir('')
		if not os.path.exists(log_dir):
			os.makedirs(log_dir)
		
		control_generator = generateInputFunction(tspan)
		train(robot, log_dir)
		test(robot, SESS, log_dir)
	if(args.mode == 'test'):
		test(robot, SESS, args.model)

	embed()

	# j = legend('$x_1$','$x_2$','$x_3$','$x_4$');
	# set(j,'interpreter','latex','fontsize',28)
	# grid on
	# xlabel('Time [s]','interpreter','latex','fontsize',28);
	# ylabel('States','interpreter','latex','fontsize',28);
	# title('$H_{\infty}$ stabilization of RTAC-nonlinear Problem','interpreter','latex','fontsize',32)