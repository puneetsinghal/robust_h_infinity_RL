import numpy as np
import h5py
import math
from scipy import integrate as SCI_INT
from IPython import embed
from time import sleep
from helper_functions import *

M  = 20
dt = 0.033
DIM = 4

tspan = np.arange(0,60,dt) 

T = tspan.size

u = np.zeros(T)
w = np.zeros(T)

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

	SM = np.random.randint(0, 80, M)
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
			x[i, :] = r.integrate(r.t+dt)	 # get one more value, add it to the array
			if not r.successful():
				raise RuntimeError("Could not integrate")
		# embed()
		U[k*T:(k+1)*T,0] = u
		W[k*T:(k+1)*T,0] = w
		X[k*T:(k+1)*T,:] = x

		print(j)

	systemDataFile = h5py.File('systemData3.h5', 'w') 
	systemDataFile.create_dataset('U', data=U)
	systemDataFile.create_dataset('W', data=W)
	systemDataFile.create_dataset('X', data=X)
	systemDataFile.close()

	return U, W, X

def generateRhoMatrices(U, W, X):
	T = U.shape[0]
	xtry  = np.array([0,0,0,0])
	NN = sigmaL(xtry)
	L = NN.size

	rho_delphi		= np.zeros([T-1,L])   # T x L
	rho_gdelphi		= np.zeros([T-1,L,L]) # T x L x L
	rho_kdelphi		= np.zeros([T-1,L,L]) # T x L x L
	rho_uphi		= np.zeros([T-1,L])   # T x L
	rho_wphi		= np.zeros([T-1,L])   # T x L
	rho_h			= np.zeros([T-1,1])   # T x 1

	for t in range(T-1):
		# sleep(0.001)
		print(t*100//T)

		xt      =  X[t,:]
		xt1     =  X[t+1,:]
		phit    =  sigmaL(xt) 
		phit1   =  sigmaL(xt1)
		Jt      =  JsigmaL(xt, DIM)
		Jt1     =  JsigmaL(xt1, DIM)
		gt      =  g_function(xt, DIM)
		gt1     =  g_function(xt1, DIM)
		kt      =  k_function(xt, DIM)
		kt1     =  k_function(xt1, DIM)
		ht      =  h_function(xt, DIM)
		ht1     =  h_function(xt1, DIM)
		ut      =  U[t]
		ut1     =  U[t+1]
		wt      =  W[t]
		wt1     =  W[t+1]
		# embed()
		rho_delphi[t,:]   	= phit-phit1											# T x L
		rho_gdelphi[t,:,:]	= (np.matmul(np.matmul(Jt, gt*gt.T), Jt.T) + np.matmul(np.matmul(Jt1, gt1*gt1.T), Jt1.T))*dt/2		# T x L x L
		rho_kdelphi[t,:,:]	= (np.matmul(np.matmul(Jt, kt*kt.T), Jt.T) + np.matmul(np.matmul(Jt1, kt1*kt1.T), Jt1.T))*dt/2		# T x L x L
		rho_uphi[t,:]     	= (ut*np.matmul(gt.T, Jt.T) + ut1*np.matmul(gt1.T, Jt1.T))*dt/2					# T x L
		rho_wphi[t,:]     	= (wt*np.matmul(kt.T, Jt.T) + wt1*np.matmul(kt1.T, Jt1.T))*dt/2					# T x L
		rho_h[t,:]        	= ((ht*ht)+(ht1*ht1))*dt/2								# T x 1

	systemDataFile = h5py.File('matrices_data3.h5', 'w') 
	systemDataFile.create_dataset('rho_delphi', data=rho_delphi)
	systemDataFile.create_dataset('rho_gdelphi', data=rho_gdelphi)
	systemDataFile.create_dataset('rho_kdelphi', data=rho_kdelphi)
	systemDataFile.create_dataset('rho_uphi', data=rho_uphi)
	systemDataFile.create_dataset('rho_wphi', data=rho_wphi)
	systemDataFile.create_dataset('rho_h', data=rho_h)

	systemDataFile.close()

	return rho_delphi, rho_gdelphi, rho_kdelphi, rho_uphi, rho_wphi, rho_h
	
if __name__ == '__main__':
	L = sigmaL(np.zeros(DIM)).size

	control_generator = generateInputFunction(T)

	U, W, X = generateSystemData(control_generator, T, M, DIM)

	rho_delphi, rho_gdelphi, rho_kdelphi, rho_uphi, rho_wphi, rho_h = generateRhoMatrices(U, W, X)

