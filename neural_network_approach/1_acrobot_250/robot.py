import numpy as np
from scipy import integrate as SCI_INT
import h5py
from IPython import embed
from math import sin, cos

class RTAC(object):
	def __init__(self, params, tspan, u, w, nn):
		self.DIM = params['numState']
		self.numInputs = params['action_size']
		self.tspan = tspan
		self.u = u
		self.w = w
		self.nn = nn

	def dynamicsTrain(self, t, x):
		# embed()
		ut = np.interp(t, self.tspan, self.u)
		wt = np.interp(t, self.tspan, self.w)

		epsilon = 0.2
		x1, x2, x3, x4 = x
		# embed()

		D = 1.0 - (epsilon*cos(x3))**2
		f = np.array([x2, (-x1+epsilon*(x4**2)*sin(x3))/D, x4, epsilon*cos(x3)*(x1 - epsilon*(x4**2)*sin(x3))/D])
		g = np.array([0, -epsilon*cos(x3)/D, 0, 1/D])
		k = np.array([0, 1/D, 0, -epsilon*cos(x3)/D])
		# embed()

		# ut   = -0.5*g.T*JsigmaL(x).T*theta_current
		# embed()
		xdot = f + g*ut + k*wt
		return xdot

	def generateSystemData(self, control_generator):
		T = self.T
		M = self.M

		U = np.zeros([T*M,1])
		W = np.zeros([T*M,1])
		X = np.zeros([T*M, self.DIM])
		x = np.zeros([T, self.DIM])
		
		SM = np.random.randint(0, 101, M)
		for k in range(M):

			x0 = np.random.uniform(-1,1,self.DIM)

			j = SM[k]
			
			u      = 0.1 * control_generator[:,j]
			w      = 0.2 * np.random.uniform(-1, 1, T)
			x[0,:] = x0

			r = SCI_INT.ode(self.dynamicsTrain).set_integrator("dopri5") 
			r.set_initial_value(x0, 0)
			for i in range(1, T):
				# embed()
				x[i, :] = r.integrate(r.t+self.dt)    # get one more value, add it to the array
				if not r.successful():
					raise RuntimeError("Could not integrate")
			# embed()
			U[k*T:(k+1)*T,0] = u
			W[k*T:(k+1)*T,0] = w
			X[k*T:(k+1)*T,:] = x

		file = h5py.File('RTAC_systemData3.h5', 'w') 
		file.create_dataset('U', data=U)
		file.create_dataset('W', data=W)
		file.create_dataset('X', data=X)
		file.close()

		print("Data generation completed")
		return U, W, X

	def dynamics_test(self, t, x):

		x1, x2, x3, x4 = x

		epsilon = 0.2 
		D = 1.0 - (epsilon*cos(x3))**2
		f = np.array([x2, (-x1+epsilon*(x4**2)*sin(x3))/D, x4, epsilon*cos(x3)*(x1 - epsilon*(x4**2)*sin(x3))/D])
		g = np.array([0, -epsilon*cos(x3)/D, 0, 1/D])
		k = np.array([0, 1/D, 0, -epsilon*cos(x3)/D])

		feed_dict={self.nn.X_t:x.reshape(1,self.DIM), self.nn.X_tPlus:x.reshape(1,self.DIM), self.nn.dropout_prob:1.0}

		grad = self.sess.run(self.nn.value_grad_t,feed_dict=feed_dict)
		u   = -0.5*np.matmul(g.reshape(1,4), grad.reshape(self.DIM,1))[0]
		w   = 0.1*np.exp(-0.1*t)*sin(t)
		# w = 1e-3*np.random.uniform(-1, 1)
		# print(w1, w)
	#     print(u)
		# embed()
		xdot = f + g*u + k*w
	#     print(xdot)
		return xdot

	def k_function(self, x):
		x1, x2, x3, x4 = x

		epsilon = 0.2  
		D = 1-(epsilon*cos(x3))**2

		k = np.array([0, 1/D, 0, -epsilon*cos(x3)/D]).reshape(self.DIM,1)
		return k

	def h_function(self, x):

		z = np.sqrt(0.1)*x

		h = np.linalg.norm(z)
		return h

	def g_function(self, x):
		x1, x2, x3, x4 = x

		epsilon = 0.2  
		D = 1-(epsilon*cos(x3))**2
		g = np.array([0, -epsilon*cos(x3)/D, 0, 1/D]).reshape(self.DIM,1)
		return g


class LinearSystems(RTAC):
	def __init__(self, params, tspan, u, w, nn):
		self.DIM = params['numState']
		self.numInputs = params['action_size']
		self.tspan = tspan
		self.u = u
		self.w = w
		self.nn = nn

	def dynamicsTrain(self, t, x):
		# embed()
		ut = np.interp(t, self.tspan, self.u)
		wt = np.interp(t, self.tspan, self.w)

		x1, x2, x3 = x
		# embed()

		f = np.array([-1.01887*x1 + 0.90506*x2 -0.00215*x3 , 0.8225*x1 - 1.07741*x2 - 0.17555*x3, -x3])
		g = np.array([0,0,1])
		k = np.array([1,0,0])
		# embed()

		# ut   = -0.5*g.T*JsigmaL(x).T*theta_current

		xdot = f + g*ut + k*wt
		return xdot

	def generateSystemData(self, control_generator):
		T= self.T
		M = self.M

		U = np.zeros([T*M,1])
		W = np.zeros([T*M,1])
		X = np.zeros([T*M, self.DIM])
		x = np.zeros([T, self.DIM])
		# embed()

		SM = np.random.randint(0, 101, M)
		for k in range(M):

			x0 = np.random.uniform(-1, 1, self.DIM)

			j = SM[k]
			
			u      = 0.1 * control_generator[:,j]
			w      = 0.2 * np.random.uniform(-1, 1, T)
			x[0,:] = x0

			r = SCI_INT.ode(self.dynamicsTrain).set_integrator("dopri5") 
			r.set_initial_value(x0, 0)
			for i in range(1, T):
				# embed()
				x[i, :] = r.integrate(r.t+self.dt)    # get one more value, add it to the array
				if not r.successful():
					raise RuntimeError("Could not integrate")
			# embed()
			U[k*T:(k+1)*T,0] = u
			W[k*T:(k+1)*T,0] = w
			X[k*T:(k+1)*T,:] = x

		file = h5py.File('LS_systemData3.h5', 'w') 
		file.create_dataset('U', data=U)
		file.create_dataset('W', data=W)
		file.create_dataset('X', data=X)
		file.close()

		print("Data generation completed")
		return U, W, X

	def dynamics_test(self, t, x):
	
		x1, x2, x3 = x

		f = np.array([-1.01887*x1 + 0.90506*x2 -0.00215*x3 , 0.8225*x1 - 1.07741*x2 - 0.17555*x3, -x3])
		g = np.array([0, 0, 1])
		k = np.array([1, 0, 0])
		
		grad = self.sess.run(self.nn.value_grad_t,feed_dict={self.nn.X_t:x.reshape(1,self.DIM), self.nn.X_tPlus:x.reshape(1,self.DIM)})
		u   = -0.5*np.matmul(g.reshape(1,self.DIM), grad.reshape(self.DIM,1))
		# d   = 4*np.exp(-0.1*t)*sin(t)
		d   = 0.4*np.exp(-0*t)*sin(t)
	#     print(u)
		xdot = f + g*u + k*d
	#     print(xdot)
		return xdot

	def k_function(self, x):
		x1, x2, x3 = x

		k = np.array([1,0,0]).reshape(self.DIM,1)
		return k

	def h_function(self, x):

		z = np.sqrt(0.1)*x

		h = np.linalg.norm(z)
		return h

	def g_function(self, x):
		x1, x2, x3 = x

		g = np.array([0,0,1]).reshape(self.DIM,1)
		return g

class Acrobot(object):
	def __init__(self, params, tspan, u, w, nn):
		self.DIM = params['numState']
		self.numInputs = params['action_size']
		self.disturbanceSize = params['disturbance_size']
		self.tspan = tspan
		self.u = u
		self.w = w
		self.nn = nn
		self.m1 = params['m1']
		self.m2 = params['m2']
		self.l1 = params['l1']
		self.l2 = params['l2']
		self.g = params['g']
		self.I1 = params['I1']
		self.I2 = params['I2']

	def dynamicsTrain(self, t, x):

		theta_1, theta_2, dtheta_1, dtheta_2 = x
		ut = np.interp(t, self.tspan, self.u)
		wt = np.interp(t, self.tspan, self.w)

		sine_2 = sin(theta_2)
		cosine_2 = cos(theta_2)

		M = np.zeros((2,2))
		M[0,0] = self.m1*(self.l1/2)**2 + self.m2*(self.l1**2 + (self.l2/2)**2 + 2*self.l1*(self.l2/2)*cosine_2) + self.I1 + self.I2
		M[0,1] = self.m2*((self.l2/2)**2 + self.l1*(self.l2/2)*cosine_2) + self.I2
		M[1,0] = M[0,1]
		M[1,1] = self.m2*(self.l2/2)**2 + self.I2

		Cor = np.zeros((2,2))
		Cor[0,0] = -2*self.m2* self.l1*(self.l2/2)*sine_2*dtheta_2
		Cor[0,1] = -self.m2*self.l1*(self.l2/2)*sine_2*dtheta_2
		Cor[1,0] = self.m2*self.l1*(self.l2/2)*sine_2*dtheta_1

		G_vector = np.zeros((2,1))
		G_vector[0,0] = (self.m1*(self.l1/2) + self.m2*self.l1)*self.g*np.cos(np.pi/2+theta_1) + self.m2* (self.l2/2)*self.g*np.cos(np.pi/2+theta_1 + theta_2)
		G_vector[1,0] = self.m2*(self.l2/2)*self.g*np.cos(np.pi/2 + theta_1 + theta_2)

		B = np.array([0, 1]).reshape((2,1))
		# B = np.identity(2)
		M_inv = np.linalg.pinv(M)

		f = np.array([dtheta_1, dtheta_2])
		f = np.hstack((f, np.matmul(M_inv, - np.matmul(Cor,np.array([dtheta_1, dtheta_2]).reshape((2,1))) - G_vector).T[0]))

		g = np.array([0, 0])
		g = np.hstack((g, np.matmul(M_inv, B).T[0]))

		k = np.array([0, 0, 1, 1])
		# print(np.linalg.pinv(M))
		# print(B*u)
		# print(np.matmul(Cor,np.array([dtheta_1, dtheta_2]).reshape((2,1))))
		# print(G)
		# print((B*u - np.matmul(Cor, np.array([dtheta_1, dtheta_2]).reshape((2,1))) - G))
		# print(np.matmul(np.linalg.pinv(M),(B*u - np.matmul(Cor, np.array([dtheta_1, dtheta_2])))))
		# embed()
		xdot = f + g*ut + k*wt
		return xdot

	def generateSystemData(self, control_generator):
		T= self.T
		M = self.M

		U = np.zeros([T*M,1])
		W = np.zeros([T*M,1])
		X = np.zeros([T*M, self.DIM])
		x = np.zeros([T, self.DIM])
		# embed()

		SM = np.random.randint(0, 101, M)
		for k in range(M):

			x0 = np.random.uniform(-1, 1, self.DIM)

			j = SM[k]
			
			u      = 1.0 * control_generator[:,j]
			w      = 0.2 * np.random.uniform(-1,1,T)
			x[0,:] = x0

			r = SCI_INT.ode(self.dynamicsTrain).set_integrator("dopri5") 
			r.set_initial_value(x0, 0)
			for i in range(1, T):
				# embed()
				x[i, :] = r.integrate(r.t+self.dt)    # get one more value, add it to the array
				if not r.successful():
					raise RuntimeError("Could not integrate")
			# embed()
			U[k*T:(k+1)*T,0] = u
			W[k*T:(k+1)*T,0] = w
			X[k*T:(k+1)*T,:] = x
			print(k)
		file = h5py.File('LS_systemData3.h5', 'w') 
		file.create_dataset('U', data=U)
		file.create_dataset('W', data=W)
		file.create_dataset('X', data=X)
		file.close()

		print("Data generation completed")
		return U, W, X

	def dynamics_test(self, t, x):
	
		theta_1, theta_2, dtheta_1, dtheta_2 = x

		sine_2 = sin(theta_2)
		cosine_2 = cos(theta_2)

		M = np.zeros((2,2))
		M[0,0] = self.m1*(self.l1/2)**2 + self.m2*(self.l1**2 + (self.l2/2)**2 + 2*self.l1*(self.l2/2)*cosine_2) + self.I1 + self.I2
		M[0,1] = self.m2*((self.l2/2)**2 + self.l1*(self.l2/2)*cosine_2) + self.I2
		M[1,0] = M[0,1]
		M[1,1] = self.m2*(self.l2/2)**2 + self.I2

		Cor = np.zeros((2,2))
		Cor[0,0] = -2*self.m2* self.l1*(self.l2/2)*sine_2*dtheta_2
		Cor[0,1] = -self.m2*self.l1*(self.l2/2)*sine_2*dtheta_2
		Cor[1,0] = self.m2*self.l1*(self.l2/2)*sine_2*dtheta_1

		G_vector = np.zeros((2,1))
		G_vector[0,0] = (self.m1*(self.l1/2) + self.m2*self.l1)*self.g*np.cos(np.pi/2+theta_1) + self.m2* (self.l2/2)*self.g*np.cos(np.pi/2+theta_1 + theta_2)
		G_vector[1,0] = self.m2*(self.l2/2)*self.g*np.cos(np.pi/2 + theta_1 + theta_2)

		B = np.array([0, 1]).reshape((2,1))
		# B = np.identity(2)
		M_inv = np.linalg.pinv(M)

		f = np.array([dtheta_1, dtheta_2])
		f = np.hstack((f, np.matmul(M_inv, - np.matmul(Cor,np.array([dtheta_1, dtheta_2]).reshape((2,1))) - G_vector).T[0]))
		# embed()
		g = np.array([0, 0])
		g = np.hstack((g, np.matmul(M_inv, B).T[0]))

		k = np.array([0, 0, 1, 1])
		
		grad = self.sess.run(self.nn.value_grad_t,feed_dict={self.nn.X_t:x.reshape(1,self.DIM), self.nn.X_tPlus:x.reshape(1,self.DIM)})
		u   = -0.5*np.matmul(g.reshape(self.numInputs,self.DIM), grad.reshape(self.DIM,1))
		# d   = 4*np.exp(-0.1*t)*sin(t)
		d   = 0.4*np.exp(-0.1*t)*sin(t)
	#     print(u)
		# embed()
		xdot = f + g*u + k*d
	#     print(xdot)
		return xdot

	def k_function(self, x):
		# x1, x2, x3, x4 = x

		k = np.array([0, 0, 1, 1])
		return k.reshape(self.DIM, self.numInputs)

	def h_function(self, x):

		z = np.sqrt(0.1)*x

		h = np.linalg.norm(z)
		return h

	def g_function(self, x):
		theta_1, theta_2, dtheta_1, dtheta_2 = x

		sine_2 = sin(theta_2)
		cosine_2 = cos(theta_2)

		M = np.zeros((2,2))
		M[0,0] = self.m1*(self.l1/2)**2 + self.m2*(self.l1**2 + (self.l2/2)**2 + 2*self.l1*(self.l2/2)*cosine_2) + self.I1 + self.I2
		M[0,1] = self.m2*((self.l2/2)**2 + self.l1*(self.l2/2)*cosine_2) + self.I2
		M[1,0] = M[0,1]
		M[1,1] = self.m2*(self.l2/2)**2 + self.I2

		Cor = np.zeros((2,2))
		Cor[0,0] = -2*self.m2* self.l1*(self.l2/2)*sine_2*dtheta_2
		Cor[0,1] = -self.m2*self.l1*(self.l2/2)*sine_2*dtheta_2
		Cor[1,0] = self.m2*self.l1*(self.l2/2)*sine_2*dtheta_1

		G_vector = np.zeros((2,1))
		G_vector[0,0] = (self.m1*(self.l1/2) + self.m2*self.l1)*self.g*np.cos(np.pi/2+theta_1) + self.m2* (self.l2/2)*self.g*np.cos(np.pi/2+theta_1 + theta_2)
		G_vector[1,0] = self.m2*(self.l2/2)*self.g*np.cos(np.pi/2 + theta_1 + theta_2)

		B = np.array([0, 1]).reshape((2,1))
		# B = np.identity(2)
		M_inv = np.linalg.pinv(M)

		g = np.array([0, 0])
		g = np.hstack((g, np.matmul(M_inv, B).T[0]))

		return g.reshape(self.DIM, self.numInputs)

	def findControl(self, xHistory):
		uHistory = np.zeros([xHistory.shape[0], self.numInputs])
		for i in range(xHistory.shape[0]):
			x = xHistory[i,:]
			g = self.g_function(x)
			grad = self.sess.run(self.nn.value_grad_t,feed_dict={self.nn.X_t:x.reshape(1,self.DIM), self.nn.X_tPlus:x.reshape(1,self.DIM)})
			uHistory[i,:] = -0.5*np.matmul(g.reshape(self.numInputs,self.DIM), grad.reshape(self.DIM,1))
		return uHistory
		
class PlanarRR(object):
	def __init__(self, params, tspan, u, w, nn):
		self.DIM = params['numState']
		self.numInputs = params['action_size']
		self.disturbanceSize = params['disturbance_size']
		self.tspan = tspan
		self.u = u
		self.w = w
		self.nn = nn
		self.m1 = params['m1']
		self.m2 = params['m2']
		self.l1 = params['l1']
		self.l2 = params['l2']
		self.g = 0*params['g']
		self.I1 = params['I1']
		self.I2 = params['I2']

	def dynamicsTrain(self, t, x):

		theta_1, theta_2, dtheta_1, dtheta_2 = x
		# embed()
		ut = np.array([np.interp(t, self.tspan, self.u[0]), np.interp(t, self.tspan, self.u[1])])
		wt = np.array([np.interp(t, self.tspan, self.w[0]), np.interp(t, self.tspan, self.w[1])])

		sine_2 = sin(theta_2)
		cosine_2 = cos(theta_2)

		M = np.zeros((2,2))
		M[0,0] = self.m1*(self.l1/2)**2 + self.m2*(self.l1**2 + (self.l2/2)**2 + 2*self.l1*(self.l2/2)*cosine_2) + self.I1 + self.I2
		M[0,1] = self.m2*((self.l2/2)**2 + self.l1*(self.l2/2)*cosine_2) + self.I2
		M[1,0] = M[0,1]
		M[1,1] = self.m2*(self.l2/2)**2 + self.I2

		Cor = np.zeros((2,2))
		Cor[0,0] = -2*self.m2* self.l1*(self.l2/2)*sine_2*dtheta_2
		Cor[0,1] = -self.m2*self.l1*(self.l2/2)*sine_2*dtheta_2
		Cor[1,0] = self.m2*self.l1*(self.l2/2)*sine_2*dtheta_1

		G_vector = np.zeros((2,1))
		G_vector[0,0] = (self.m1*(self.l1/2) + self.m2*self.l1)*self.g*np.cos(np.pi/2+theta_1) + self.m2* (self.l2/2)*self.g*np.cos(np.pi/2+theta_1 + theta_2)
		G_vector[1,0] = self.m2*(self.l2/2)*self.g*np.cos(np.pi/2 + theta_1 + theta_2)

		# B = np.array([0, 1]).reshape((2,1))
		B = np.identity(2)
		M_inv = np.linalg.pinv(M)

		f = np.array([dtheta_1, dtheta_2])
		f = np.hstack((f, np.matmul(M_inv, - np.matmul(Cor,np.array([dtheta_1, dtheta_2]).reshape((2,1))) - G_vector).T[0]))

		# embed()
		g = np.zeros([2,2])
		g = np.hstack((g, np.matmul(M_inv, B)))

		k = np.array([[0, 0, 1, 0],[0,0,0,1]])
		# print(np.linalg.pinv(M))
		# print(B*u)
		# print(np.matmul(Cor,np.array([dtheta_1, dtheta_2]).reshape((2,1))))
		# print(G)
		# print((B*u - np.matmul(Cor, np.array([dtheta_1, dtheta_2]).reshape((2,1))) - G))
		# print(np.matmul(np.linalg.pinv(M),(B*u - np.matmul(Cor, np.array([dtheta_1, dtheta_2])))))
		# embed()
		xdot = f + np.matmul(g.T,ut) + np.matmul(k.T, wt)
		return xdot

	def generateSystemData(self, control_generator):
		T= self.T
		M = self.M

		U = np.zeros([T*M, self.numInputs])
		W = np.zeros([T*M, self.disturbanceSize])
		X = np.zeros([T*M, self.DIM])
		x = np.zeros([T, self.DIM])
		# embed()

		SM = np.random.randint(0, 101, [self.numInputs, M])
		for k in range(M):

			x0 = np.random.uniform(-1, 1, self.DIM)

			j = SM[:,k]
			
			u      = 1.0 * np.vstack((0.1 * control_generator[:,j[0]], 0.1 * control_generator[:,j[1]]))
			w      = 0.2 * np.random.uniform(-1, 1, [self.disturbanceSize, T])
			x[0,:] = x0

			r = SCI_INT.ode(self.dynamicsTrain).set_integrator("dopri5") 
			r.set_initial_value(x0, 0)
			for i in range(1, T):
				# embed()
				x[i, :] = r.integrate(r.t+self.dt)    # get one more value, add it to the array
				if not r.successful():
					raise RuntimeError("Could not integrate")
			# embed()
			U[k*T:(k+1)*T,:] = u.T
			W[k*T:(k+1)*T,:] = w.T
			X[k*T:(k+1)*T,:] = x
			print(k)
		file = h5py.File('LS_systemData3.h5', 'w') 
		file.create_dataset('U', data=U)
		file.create_dataset('W', data=W)
		file.create_dataset('X', data=X)
		file.close()

		print("Data generation completed")
		return U, W, X

	def dynamics_test(self, t, x):
	
		theta_1, theta_2, dtheta_1, dtheta_2 = x

		sine_2 = sin(theta_2)
		cosine_2 = cos(theta_2)

		M = np.zeros((2,2))
		M[0,0] = self.m1*(self.l1/2)**2 + self.m2*(self.l1**2 + (self.l2/2)**2 + 2*self.l1*(self.l2/2)*cosine_2) + self.I1 + self.I2
		M[0,1] = self.m2*((self.l2/2)**2 + self.l1*(self.l2/2)*cosine_2) + self.I2
		M[1,0] = M[0,1]
		M[1,1] = self.m2*(self.l2/2)**2 + self.I2

		Cor = np.zeros((2,2))
		Cor[0,0] = -2*self.m2* self.l1*(self.l2/2)*sine_2*dtheta_2
		Cor[0,1] = -self.m2*self.l1*(self.l2/2)*sine_2*dtheta_2
		Cor[1,0] = self.m2*self.l1*(self.l2/2)*sine_2*dtheta_1

		G_vector = np.zeros((2,1))
		G_vector[0,0] = (self.m1*(self.l1/2) + self.m2*self.l1)*self.g*np.cos(np.pi/2+theta_1) + self.m2* (self.l2/2)*self.g*np.cos(np.pi/2+theta_1 + theta_2)
		G_vector[1,0] = self.m2*(self.l2/2)*self.g*np.cos(np.pi/2 + theta_1 + theta_2)

		# B = np.array([0, 1]).reshape((2,1))
		B = np.identity(2)
		M_inv = np.linalg.pinv(M)

		f = np.array([dtheta_1, dtheta_2])
		f = np.hstack((f, np.matmul(M_inv, - np.matmul(Cor,np.array([dtheta_1, dtheta_2]).reshape((2,1))) - G_vector).T[0]))

		g = np.zeros([2,2])
		g = np.hstack((g, np.matmul(M_inv, B)))

		k = np.array([[0, 0, 1, 0],[0,0,0,1]])
		feed_dict={self.nn.X_t:x.reshape(1,self.DIM), self.nn.X_tPlus:x.reshape(1,self.DIM), self.nn.dropout_prob:1.0}

		grad = self.sess.run(self.nn.value_grad_t,feed_dict=feed_dict)
		ut   = -0.5*np.matmul(g, grad.reshape(self.DIM,1)).reshape(self.numInputs,)
		# d   = 4*np.exp(-0.1*t)*sin(t)
		wt   = 0.4*np.exp(-0*t)*sin(t)*np.ones(2)
	#     print(u)
		# embed()
		xdot = f + np.matmul(g.T,ut) + np.matmul(k.T, wt)
	#     print(xdot)
		# embed()
		return xdot

	def k_function(self, x):
		# x1, x2, x3, x4 = x

		k = np.array([[0, 0, 1, 0],[0,0,0,1]])
		return k.T

	def h_function(self, x):

		z = np.sqrt(0.1)*x

		h = np.linalg.norm(z)
		return h

	def g_function(self, x):
		theta_1, theta_2, dtheta_1, dtheta_2 = x

		sine_2 = sin(theta_2)
		cosine_2 = cos(theta_2)

		M = np.zeros((2,2))
		M[0,0] = self.m1*(self.l1/2)**2 + self.m2*(self.l1**2 + (self.l2/2)**2 + 2*self.l1*(self.l2/2)*cosine_2) + self.I1 + self.I2
		M[0,1] = self.m2*((self.l2/2)**2 + self.l1*(self.l2/2)*cosine_2) + self.I2
		M[1,0] = M[0,1]
		M[1,1] = self.m2*(self.l2/2)**2 + self.I2

		Cor = np.zeros((2,2))
		Cor[0,0] = -2*self.m2* self.l1*(self.l2/2)*sine_2*dtheta_2
		Cor[0,1] = -self.m2*self.l1*(self.l2/2)*sine_2*dtheta_2
		Cor[1,0] = self.m2*self.l1*(self.l2/2)*sine_2*dtheta_1

		G_vector = np.zeros((2,1))
		G_vector[0,0] = (self.m1*(self.l1/2) + self.m2*self.l1)*self.g*np.cos(np.pi/2+theta_1) + self.m2* (self.l2/2)*self.g*np.cos(np.pi/2+theta_1 + theta_2)
		G_vector[1,0] = self.m2*(self.l2/2)*self.g*np.cos(np.pi/2 + theta_1 + theta_2)

		# B = np.array([0, 1]).reshape((2,1))
		B = np.identity(2)
		M_inv = np.linalg.pinv(M)

		f = np.array([dtheta_1, dtheta_2])
		f = np.hstack((f, np.matmul(M_inv, - np.matmul(Cor,np.array([dtheta_1, dtheta_2]).reshape((2,1))) - G_vector).T[0]))

		g = np.zeros([2,2])
		g = np.hstack((g, np.matmul(M_inv, B)))

		return g.T