import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np
from math import sqrt

class Network(object):
	def __init__(self, params):
		# tf Graph input
		self.X_t = tf.placeholder(shape=[None, params['numState']], dtype=tf.float64, name="X_t")
		self.X_tPlus = tf.placeholder(shape=[None, params['numState']], dtype=tf.float64, name="X_tPlus")
		self.U_t = tf.placeholder(shape=[None, 1, params['action_size']], dtype=tf.float64, name="U_t")
		self.U_tPlus = tf.placeholder(shape=[None, 1, params['action_size']], dtype=tf.float64, name="U_tPlus")
		self.W_t = tf.placeholder(shape=[None, 1, params['disturbance_size']], dtype=tf.float64, name="W_t")
		self.W_tPlus = tf.placeholder(shape=[None, 1, params['disturbance_size']], dtype=tf.float64, name="W_tPlus")
		
		self.G_X_t = tf.placeholder(shape=[None, params['numState'], params['action_size']], dtype=tf.float64, name="g_x")
		self.G_X_tPlus = tf.placeholder(shape=[None, params['numState'], params['action_size']], dtype=tf.float64, name="g_x")
		self.K_X_t = tf.placeholder(shape=[None, params['numState'], params['disturbance_size']], dtype=tf.float64, name="k_x")
		self.K_X_tPlus = tf.placeholder(shape=[None, params['numState'], params['disturbance_size']], dtype=tf.float64, name="k_x")
		
		self.dropout_prob = tf.placeholder(shape=[], dtype=tf.float64)
		self.gamma = params['gamma']
		self.dt = params['dt']
		self.hiddenSize = params['hiddenSize']
		self.generateNetwork(params)

	def generateNetwork(self, params):

		if(params['robot']=='linear'):
			self.createLayers_Linear()

		elif(params['robot']=='RTAC'):
			self.createLayers_RTAC()

		elif(params['robot']=='acrobot'):
			self.createLayers_Acrobot()

		elif(params['robot']=='planarRR'):
			self.createLayers_PlanarRR()

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
			
		# self.trainer = tf.train.AdamOptimizer(learning_rate=params['learningRate'])
		# self.optimizer = self.trainer.minimize(self.residual_error)
		
		local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

		# self.batch_gradients = []
		# self.batchsize = self.residual_error.shape[0].value
		# if(self.batchsize == None):
		# 	self.batchsize = 1
		# trainer = tf.contrib.opt.NadamOptimizer(learning_rate=params['learningRate'])
		# self.gradients = tf.gradients(self.residual_error, local_vars)
		# self.var_norms = tf.global_norm(local_vars)
		# grads, self.grad_norms = tf.clip_by_global_norm(self.gradients, 300)
		# grads = self.gradients
		# self.apply_grads = trainer.apply_gradients(zip(grads, local_vars))

		# opt = GradientDescentOptimizer(learning_rate=params['learningRate'])
		# grads_and_vars = opt.compute_gradients(self.residual_error, local_vars)
		# weighted_grads_and_vars = [[gv[0], gv[1]] for gv in grads_and_vars]

		self.trainer = tf.train.AdamOptimizer(learning_rate=params['learningRate'], beta1=0.9, beta2=0.999)
		self.gradients = self.trainer.compute_gradients(self.residual_error, local_vars)
		# # grads = self.gradients
		self.apply_grads = self.trainer.apply_gradients(self.gradients)

	
	def createLayers_Linear(self):
		self.input = tf.concat([self.X_t, self.X_tPlus],0)
		self.layer = self.input
		self.layer = layers.fully_connected(inputs=self.layer, 
											num_outputs=self.hiddenSize, activation_fn=tf.nn.sigmoid)
		self.layer = layers.fully_connected(inputs=self.layer, 
											num_outputs=self.hiddenSize, activation_fn=tf.nn.sigmoid)
		self.layer = layers.fully_connected(inputs=self.layer,
											num_outputs=self.hiddenSize, activation_fn=tf.nn.sigmoid)
		self.output = layers.fully_connected(inputs=self.layer, 
											 num_outputs=1, activation_fn=None)

	def createLayers_RTAC(self):
		self.input = tf.concat([self.X_t, self.X_tPlus],0)
		self.layer = self.input
		self.layer = layers.fully_connected(inputs=self.layer, 
											num_outputs=self.hiddenSize, activation_fn=tf.nn.sigmoid)
		self.layer = layers.fully_connected(inputs=self.layer, 
											num_outputs=self.hiddenSize, activation_fn=tf.nn.sigmoid)
		self.layer = layers.fully_connected(inputs=self.layer,
											num_outputs=self.hiddenSize, activation_fn=tf.nn.sigmoid)
		# self.layer = layers.fully_connected(inputs=self.layer,
		# 									num_outputs=self.hiddenSize, activation_fn=tf.nn.sigmoid)
		# self.layer = layers.fully_connected(inputs=self.layer,
		# 									num_outputs=self.hiddenSize, activation_fn=tf.nn.sigmoid)
		# self.layer = layers.dropout(self.layer, self.dropout_prob)
		self.output = layers.fully_connected(inputs=self.layer, 
											 num_outputs=1, activation_fn=None)

	def createLayers_Acrobot(self):
		self.input = tf.concat([self.X_t, self.X_tPlus], 0)
		# angles, velocities = tf.split(self.input, num_or_size_splits=2, axis=1)
		# self.sines = tf.sin(angles)
		# self.cosines = tf.cos(angles)
		# self.input = tf.angle(tf.complex(self.cosines, self.sines))
		# self.input = tf.concat([self.input, velocities], 0)
		self.layer = self.input
		self.layer = layers.fully_connected(inputs=self.layer, 
											num_outputs=self.hiddenSize, activation_fn=tf.nn.sigmoid)
		self.layer = layers.fully_connected(inputs=self.layer, 
											num_outputs=self.hiddenSize, activation_fn=tf.nn.sigmoid)
		self.layer = layers.fully_connected(inputs=self.layer,
											num_outputs=self.hiddenSize, activation_fn=tf.nn.sigmoid)
		self.output = layers.fully_connected(inputs=self.layer, 
											 num_outputs=1, activation_fn=None)

	def createLayers_PlanarRR(self):
		self.input = tf.concat([self.X_t, self.X_tPlus],0)
		self.layer = self.input
		self.layer = layers.fully_connected(inputs=self.layer, 
											num_outputs=self.hiddenSize, activation_fn=tf.nn.sigmoid)
		self.layer = layers.fully_connected(inputs=self.layer, 
											num_outputs=self.hiddenSize, activation_fn=tf.nn.sigmoid)
		self.layer = layers.fully_connected(inputs=self.layer,
											num_outputs=self.hiddenSize, activation_fn=tf.nn.sigmoid)
		self.layer = layers.dropout(self.layer, self.dropout_prob)
		self.output = layers.fully_connected(inputs=self.layer, 
											 num_outputs=1, activation_fn=None)

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

	def train(self):
		pass