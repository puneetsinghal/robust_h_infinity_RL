import numpy as np

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

def sigmaL(x):

	x1 = x[0]
	x2 = x[1]
	x3 = x[2]
	x4 = x[3]

	t2 = x1**2
	t3 = x2**2
	t4 = x3**2
	t5 = x4**2
	
	sigL = np.array([t2**2, t2*x1*x2, t2*x1*x3, t2*x1*x4, t2*t3, t2*x2*x3, t2*x2*x4, t2*t4, t2*x3*x4, t2*t5,
				t2, t3*x1*x2, t3*x1*x3, t3*x1*x4, t4*x1*x2, x1*x2*x3*x4, t5*x1*x2, x1*x2, t4*x1*x3,
				t4*x1*x4, t5*x1*x3, x1*x3, t5*x1*x4, x1*x4, t3**2, t3*x2*x3, t3*x2*x4, t3*t4, t3*x3*x4,
				t3*t5, t3, t4*x2*x3, t4*x2*x4, t5*x2*x3, x2*x3, t5*x2*x4, x2*x4, t4**2, t4*x3*x4, t4*t5,
				t4, t5*x3*x4, x3*x4, t5**2, t5])
	return sigL

def k_function(x, DIM):
	x1 = x[0]
	x2 = x[1]
	x3 = x[2]


	k = np.array([1,0,0]).reshape(DIM,1)
	return k

def h_function(x, DIM):

	z = np.sqrt(0.1)*x

	h = np.linalg.norm(z)
	return h

def g_function(x, DIM):
	x1 = x[0]
	x2 = x[1]
	x3 = x[2]

	g = np.array([0,0,1]).reshape(DIM,1)
	return g

def JsigmaL(x, DIM):
	x1 = x[0]
	x2 = x[1]
	x3 = x[2]
	x4 = x[3]

	t2 = x1**2
	t3 = t2*x2
	t4 = t2*x4
	t5 = t2*x3
	t6 = x2**2
	t7 = x1*x2*x3*2.0
	t8 = x1*x2*x4*2.0
	t9 = t6*x1
	t10 = x3**2
	t11 = x4**2
	t12 = x1*x3*x4*2.0
	t13 = t10*x1
	t14 = t11*x1
	t15 = t6*x4
	t16 = t6*x3
	t17 = t10*x4
	t18 = x2*x3*x4*2.0
	t19 = t10*x2
	t20 = t11*x3
	t21 = t11*x2
	JsigmaL = np.array([t2*x1*4.0, t2*x2*3.0, t2*x3*3.0, t2*x4*3.0, t6*x1*2.0, t7, t8, t10*x1*2.0, t12, 
						t11*x1*2.0, x1*2.0, t6*x2, t16, t15, t19, x2*x3*x4, t21, x2,
						t10*x3, t17, t20, x3, t11*x4, x4, 0.0, 0.0, 0.0, 
						0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
						0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
						0.0, t2*x1, 0.0, 0.0, t2*x2*2.0, t5, t4, 0.0, 0.0, 
						0.0, 0.0, t6*x1*3.0, t7, t8, t13, x1*x3*x4, t14, x1, 
						0.0, 0.0, 0.0, 0.0, 0.0, 0.0, t6*x2*4.0, t6*x3*3.0, t6*x4*3.0, 
						t10*x2*2.0, t18, t11*x2*2.0, x2*2.0, t10*x3, t17, t20, x3, t11*x4, 
						x4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
						0.0, 0.0, t2*x1, 0.0, 0.0, t3, 0.0, t2*x3*2.0, t4, 
						0.0, 0.0, 0.0, t9, 0.0, t7, x1*x2*x4, 0.0, 0.0, 
						t10*x1*3.0, t12, t14, x1, 0.0, 0.0, 0.0, t6*x2, 0.0, 
						t6*x3*2.0, t15, 0.0, 0.0, t10*x2*3.0, t18, t21, x2, 0.0, 
						0.0, t10*x3*4.0, t10*x4*3.0, t11*x3*2.0, x3*2.0, t11*x4, x4, 0.0, 0.0, 
						0.0, 0.0, 0.0, t2*x1, 0.0, 0.0, t3, 0.0, t5, 
						t2*x4*2.0, 0.0, 0.0, 0.0, t9, 0.0, x1*x2*x3, t8, 0.0, 
						0.0, t13, t12, 0.0, t11*x1*3.0, x1, 0.0, 0.0, t6*x2, 
						0.0, t16, t6*x4*2.0, 0.0, 0.0, t19, t18, 0.0, t11*x2*3.0, 
						x2, 0.0, t10*x3, t10*x4*2.0, 0.0, t11*x3*3.0, x3, t11*x4*4.0, x4*2.0]).reshape(45, DIM)
	return JsigmaL