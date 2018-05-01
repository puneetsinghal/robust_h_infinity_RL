import numpy as np
from scipy import integrate as SCI_INT
import h5py

class RTAC(object):
	def __init__(self, DIM, tspan, u, w, nn):
		self.DIM = DIM
		self.tspan = tspan
		self.u = u
		self.w = w
		self.nn = nn


	def dynamicsTrain(self, t, x):
		# embed()
		ut = np.interp(t, self.tspan, self.u)
		wt = np.interp(t, self.tspan, self.w)

		x1 = x[0]
		x2 = x[1]
		x3 = x[2]
		# embed()

		f = np.array([-1.01887*x1 + 0.90506*x2 -0.00215*x3 , 0.8225*x1 - 1.07741*x2 - 0.17555*x3, -x3])
		g = np.array([0,0,1])
		k = np.array([1,0,0])
		# embed()

		# ut   = -0.5*g.T*JsigmaL(x).T*theta_current

		xdot = f + g*ut + k*wt
		return xdot

	def generateSystemData(self, control_generator, dt, T, M):
		U = np.zeros([T*M,1])
		W = np.zeros([T*M,1])
		X = np.zeros([T*M, self.DIM])
		x = np.zeros([T, self.DIM])
		# embed()

		SM = np.random.randint(0, 101, M)
		for k in range(M):

			x0 = np.random.rand(self.DIM)

			j = SM[k]
			
			u      = 0.1*control_generator[:,j]
			w      = 0.2*np.random.rand(T)
			x[0,:] = x0

			r = SCI_INT.ode(self.dynamicsTrain).set_integrator("dopri5") 
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

		file = h5py.File('systemData3.h5', 'w') 
		file.create_dataset('U', data=U)
		file.create_dataset('W', data=W)
		file.create_dataset('X', data=X)
		file.close()

		print("Data generation completed")
		return U, W, X

	def dynamics(self, t, x):
	
		x1 = x[0]
		x2 = x[1]
		x3 = x[2]

		f = np.array([-1.01887*x1 + 0.90506*x2 -0.00215*x3 , 0.8225*x1 - 1.07741*x2 - 0.17555*x3, -x3])
		g = np.array([0,0,1])
		k = np.array([1,0,0])
		
		grad = self.sess.run(self.nn.value_grad_t,feed_dict={self.nn.X_t:x.reshape(1,self.DIM), self.nn.X_tPlus:x.reshape(1,self.DIM)})
		u   = -0.5*np.matmul(g.reshape(1,3), grad.reshape(self.DIM,1))
		d   = 4*np.exp(-0.1*t)*np.sin(t)
	#     print(u)
		xdot = f + g*u + k*d
	#     print(xdot)
		return xdot