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
#         self.Y = tf.placeholder(shape=[None, params['numOutput']], dtype=tf.float64, name="Y")
		
		self.G_X_t = tf.placeholder(shape=[None, params['numState'], params['action_size']], dtype=tf.float64, name="g_x")
		self.G_X_tPlus = tf.placeholder(shape=[None, params['numState'], params['action_size']], dtype=tf.float64, name="g_x")
		self.K_X_t = tf.placeholder(shape=[None, params['numState'], params['disturbance_size']], dtype=tf.float64, name="k_x")
		self.K_X_tPlus = tf.placeholder(shape=[None, params['numState'], params['disturbance_size']], dtype=tf.float64, name="k_x")
	
		self.gamma = params['gamma']
		self.dt = params['dt']
		self.hiddenSize = params['hiddenSize']
		self.generateNetwork(params)

	def generateNetwork(self, params):

		if(params['robot']=='linear'):
			self.createLayers_Linear()

		elif(params['robot']=='RTAC'):
			self.createLayers_RTAC()

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