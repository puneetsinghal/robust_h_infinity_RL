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

from helper_functions import *

try:
	from StringIO import StringIO
except ImportError:
	from io import StringIO

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
	
class Logger(object):
	"""Logging in tensorboard without tensorflow ops."""

	def __init__(self, log_dir):
		"""Creates a summary writer logging to log_dir."""
		self.writer = tf.summary.FileWriter(log_dir)

	def log_scalar(self, tag, value, step):
		"""Log a scalar variable.
		Parameter
		----------
		tag : basestring
			Name of the scalar
		value
		step : int
			training iteration
		"""
		summary = tf.Summary(value=[tf.Summary.Value(tag=tag,
													 simple_value=value)])
		self.writer.add_summary(summary, step)


class network(object):
	def __init__(self, params):
		# tf Graph input
		self.X_t = tf.placeholder(shape=[None, params['numState']], dtype=tf.float64, name="X_t")
		self.X_tPlus = tf.placeholder(shape=[None, params['numState']], dtype=tf.float64, name="X_tPlus")
		self.U_t = tf.placeholder(shape=[None, 1, params['action_size']], dtype=tf.float64, name="U_t")
		self.U_tPlus = tf.placeholder(shape=[None, 1, params['action_size']], dtype=tf.float64, name="U_tPlus")
		self.W_t = tf.placeholder(shape=[None, 1, params['disturbance_size']], dtype=tf.float64, name="W_t")
		self.W_tPlus = tf.placeholder(shape=[None, 1, params['disturbance_size']], dtype=tf.float64, name="W_tPlus")
#         self.Y = tf.placeholder(shape=[None, params['numOutput']], dtype=tf.float64, name="Y")
		
		self.G_X_t = tf.placeholder(shape=[None, params['numState'], params['action_size']], dtype=tf.float64, name="g_x")
		self.G_X_tPlus = tf.placeholder(shape=[None, params['numState'], params['action_size']], dtype=tf.float64, name="g_x")
		self.K_X_t = tf.placeholder(shape=[None, params['numState'], params['disturbance_size']], dtype=tf.float64, name="k_x")
		self.K_X_tPlus = tf.placeholder(shape=[None, params['numState'], params['disturbance_size']], dtype=tf.float64, name="k_x")
	
		self.gamma = params['gamma']
		self.dt = params['dt']

		self.generateNetwork(params)

	def generateNetwork(self, params):
		
		self.input = tf.concat([self.X_t, self.X_tPlus],0)
#         print("shape of final input: {}".format(self.input.shape))
		
		self.layer = self.input
		self.layer = layers.fully_connected(inputs=self.layer, 
											num_outputs=params['hiddenSize'], activation_fn=tf.nn.sigmoid)
		self.layer = layers.fully_connected(inputs=self.layer, 
											num_outputs=params['hiddenSize'], activation_fn=tf.nn.sigmoid)
		self.layer = layers.fully_connected(inputs=self.layer,
											num_outputs=params['hiddenSize'], activation_fn=tf.nn.sigmoid)
		self.output = layers.fully_connected(inputs=self.layer, 
											 num_outputs=1, activation_fn=None)
		
		self.value_t, self.value_tPlus = tf.split(self.output, num_or_size_splits=2, axis=0)
#         print("shape of value_t: {}".format(self.value_t.shape))
#         print("shape of value_tPlus: {}".format(self.value_tPlus.shape))
		
		
		self.value_grad_t = tf.gradients(self.value_t, self.X_t)[0]
		self.value_grad_t = tf.expand_dims(self.value_grad_t, 1)
#         print("shape of value_grad_t: {}".format(self.value_grad_t.shape))
		
		self.value_grad_tPlus = tf.gradients(self.value_tPlus, self.X_tPlus)[0]
		self.value_grad_tPlus = tf.expand_dims(self.value_grad_tPlus, 1)
		
		# Define loss and optimizer
		self.calculateResidualError()
#         self.trainer = tf.train.AdamOptimizer(learning_rate=params['learningRate'])
		
		local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
		
#         try:
#             self.batchsize = self.residual_error.shape[0].value
#             self.gradient_vector = []
#             for i in range(self.batchsize):
#                 grad = tf.gradients(self.residual_error[i,0], local_vars)
#                 self.gradient_vector.append([self.residual_error[i,0]*g, v] for g,v in grad])
#             self.gradient_residual_error = tf.reduce_mean(self.gradient_vector)
#         except:
#             print("error")
#             self.gradient_residual_error = tf.gradients(self.residual_error[0,0], local_vars)

		self.batch_gradients = []
		self.batchsize = self.residual_error.shape[0].value
		if(self.batchsize == None):
			self.batchsize = 1
			
#         for i in range(self.batchsize):
#             instance_grads_and_vars = tf.gradients(self.residual_error[i,0], local_vars)
# #             instance_gradients = [self.residual_error[i,0]*grad/self.batchsize for grad, variable in instance_grads_and_vars]
#             self.batch_gradients.append(instance_grads_and_vars)
		trainer = tf.contrib.opt.NadamOptimizer(learning_rate=params['learningRate'])
	
		self.gradients = tf.gradients(self.residual_error, local_vars)
#         self.gradients = tf.multiply(self.residual_error, self.gradients)
#         self.gradients *= self.residual_error
		self.var_norms = tf.global_norm(local_vars)
		grads, self.grad_norms = tf.clip_by_global_norm(self.gradients, 300)
		self.apply_grads = trainer.apply_gradients(zip(grads, local_vars))
		
#         self.gradient_residual_error = tf.gradients(self.residual_error, local_vars)
#         print("shape of gradient_residual_error: {}".format(self.gradient_residual_error))
#         self.weighted_average = tf.reduce_mean(np.multiply(self.residual_error,self.gradient_residual_error),0)
		
#         self.trainer = tf.contrib.opt.NadamOptimizer(learning_rate=params['learningRate'])
#         print(tf.reduce_sum(self.batch_gradients))
#         print(zip(tf.reduce_sum(self.batch_gradients), local_vars))
#         self.apply_grads = self.trainer.apply_gradients(zip(tf.reduce_sum(self.batch_gradients), local_vars))

	def calculateResidualError(self):

		self.actor_t = -0.5*tf.matmul(self.value_grad_t, self.G_X_t)
		self.actor_tPlus = -0.5*tf.matmul(self.value_grad_tPlus, self.G_X_tPlus)
		
		firstTerm = -2.0*tf.reduce_sum(tf.multiply(self.actor_t, (self.U_t - self.actor_t)), 2)
		firstTerm += -2.0*tf.reduce_sum(tf.multiply(self.actor_tPlus, (self.U_tPlus - self.actor_tPlus)), 2)
		firstTerm *= self.dt
		firstTerm /= 2.0
		
		disturbance_t = 1/(2.0*self.gamma**2)*tf.matmul(self.value_grad_t, self.K_X_t)
		disturbance_tPlus = 1/(2.0*self.gamma**2)*tf.matmul(self.value_grad_tPlus, self.K_X_tPlus)
	
		secondTerm = (2.0*self.gamma**2)*tf.reduce_sum(tf.multiply(disturbance_t, (self.W_t - disturbance_t)), 2)
		secondTerm += (2.0*self.gamma**2)*tf.reduce_sum(tf.multiply(disturbance_tPlus, (self.W_tPlus - disturbance_tPlus)), 2)
		secondTerm *= self.dt
		secondTerm /= 2.0
		
		thirdTerm = self.value_t - self.value_tPlus
		
		h_t = tf.expand_dims(sqrt(0.1)*self.X_t,1)
		h_tPlus = tf.expand_dims(sqrt(0.1)*self.X_tPlus,1)
		
		fourthTerm = tf.reduce_sum(tf.multiply(h_t, h_t), 2)
		fourthTerm += tf.reduce_sum(tf.multiply(h_tPlus, h_tPlus), 2)
		fourthTerm *= self.dt
		fourthTerm /= 2.0
		
		fifthTerm = tf.reduce_sum(tf.multiply(self.actor_t, self.actor_t),2)
		fifthTerm += tf.reduce_sum(tf.multiply(self.actor_tPlus, self.actor_tPlus),2)
		fifthTerm *= self.dt
		fifthTerm /= 2.0
		
		sixthTerm = tf.reduce_sum(tf.multiply(disturbance_t, disturbance_t),2)
		sixthTerm += tf.reduce_sum(tf.multiply(disturbance_tPlus, disturbance_tPlus),2)
		sixthTerm *= self.dt
		sixthTerm /= 2.0
		sixthTerm *= self.gamma**2
		
		self.residual_error = firstTerm + secondTerm + thirdTerm - (fourthTerm + fifthTerm - sixthTerm)
		self.residual_error = tf.square(self.residual_error)

def dynamicsknown(t, x):
	# embed()
	ut = np.interp(t, tspan, u)
	wt = np.interp(t, tspan, w)

	epsilon = 0.2
	x1 = x[0]
	x2 = x[1]
	x3 = x[2]
	x4 = x[3]
	# embed()

	D = 1. - (epsilon*np.cos(x3))**2
	f = np.array([x2, (-x1+epsilon*x4**2*np.sin(x3))/D, x4, epsilon*np.cos(x3)*(x1 - epsilon*x4**2*np.sin(x3))/D])
	g = np.array([0, -epsilon*np.cos(x3)/D, 0, 1/D])
	k = np.array([0, 1/D, 0, -epsilon*np.cos(x3)/D])
	# embed()

	# ut   = -0.5*g.T*JsigmaL(x).T*theta_current

	xdot = f + g*ut + k*wt
	return xdot

def generateSystemData(control_generator, T, M, DIM):
	U = np.zeros([T*M,1])
	W = np.zeros([T*M,1])
	X = np.zeros([T*M,DIM])
	x = np.zeros([T, DIM])
	# embed()

	SM = np.random.randint(0, 101, M)
	for k in range(M):

		x0 = np.random.rand(DIM)

		j = SM[k]
		
		u      = 0.1*control_generator[:,j]
		w      = 0.2*np.random.rand(T)
		x[0,:] = x0

		r = SCI_INT.ode(dynamicsknown).set_integrator("dopri5") 
		r.set_initial_value(x0, 0)
		for i in range(1, T):
			# embed()
			x[i, :] = r.integrate(r.t+dt)    # get one more value, add it to the array
			if not r.successful():
				raise RuntimeError("Could not integrate")
		# embed()
		U[k*T:(k+1)*T,0] = u
		W[k*T:(k+1)*T,0] = w
		X[k*T:(k+1)*T,:] = x

		print(j)

	file = h5py.File('systemData3.h5', 'w') 
	file.create_dataset('U', data=U)
	file.create_dataset('W', data=W)
	file.create_dataset('X', data=X)
	file.close()

	return U, W, X

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

def dynamics(t, x):
	
	x1 = x[0]
	x2 = x[1]
	x3 = x[2]
	x4 = x[3]

	epsilon = 0.2 ; 
	D = 1. - (epsilon*np.cos(x3))**2
	f = np.array([x2, (-x1+epsilon*x4**2*np.sin(x3))/D, x4, epsilon*np.cos(x3)*(x1 - epsilon*x4**2*np.sin(x3))/D])
	g = np.array([0, -epsilon*np.cos(x3)/D, 0, 1/D])
	k = np.array([0, 1/D, 0, -epsilon*np.cos(x3)/D])
	
	grad = SESS.run(nn.value_grad_t,feed_dict={nn.X_t:x.reshape(1,DIM), nn.X_tPlus:x.reshape(1,DIM)})
	u   = -0.5*np.matmul(g.reshape(1,4), grad.reshape(DIM,1))
	d   = 0.0*np.exp(-0.1*t)*np.sin(t)
#     print(u)
	xdot = f + g*u + k*d
#     print(xdot)
	return xdot


if __name__=='__main__':
	params = {}
	params['hiddenSize'] = 16
	params['dt'] = 0.033
	params['learningRate'] = 1e-2
	params['gamma'] = 6
	params['numState'] = DIM
	params['action_size'] = 1
	params['disturbance_size'] = 1
	
	nn = network(params)
	#     L = sigmaL(np.zeros(DIM)).size

	control_generator = generateInputFunction(tspan)

	U, W, X = generateSystemData(control_generator, T, M, DIM)

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
	save_path = saver.save(SESS, './')
	
	log_dir 	= make_log_dir('')
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

		if(step%100 == 0 or step == 1):
			save_path = saver.save(SESS, modelName, write_meta_graph=True)

	save_path = saver.save(SESS, modelName, write_meta_graph=True)
	x0     = np.random.rand(DIM)

	newT = 100
	dt = 0.033

	totalPoints = int(newT/dt)

	x = np.zeros([totalPoints,4])
	x[0,:] = x0

	r = SCI_INT.ode(dynamics).set_integrator("dopri5") 
	r.set_initial_value(x0, 0)
	for i in range(1, totalPoints):
		# embed()
		x[i, :] = r.integrate(r.t+dt)    # get one more value, add it to the array
		if not r.successful():
			raise RuntimeError("Could not integrate")

	t = np.linspace(0,newT,totalPoints)

	plt.figure(1)
	plt.plot(t,x[:,0],'-b','linewidth',1.8)
	plt.plot(t,x[:,1],'-r','linewidth',1.8)
	plt.plot(t,x[:,2],'-m','linewidth',1.8)
	plt.plot(t,x[:,3],'-k','linewidth',1.8)
	plt.show()
	embed()

	# j = legend('$x_1$','$x_2$','$x_3$','$x_4$');
	# set(j,'interpreter','latex','fontsize',28)
	# grid on
	# xlabel('Time [s]','interpreter','latex','fontsize',28);
	# ylabel('States','interpreter','latex','fontsize',28);
	# title('$H_{\infty}$ stabilization of RTAC-nonlinear Problem','interpreter','latex','fontsize',32)