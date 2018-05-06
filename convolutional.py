import numpy as np

def conv_single_step(a_slice_prev, W, b):
 
	s = np.multiply(a_slice_prev,W)
    	Z = np.sum(s)
    	Z = Z+b

    	return Z


def conv_forward(A_prev, W, b, hparameters):
 
	(m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

        (f, f, n_C_prev, n_C) = W.shape
        stride = hparameters["stride"]
        pad = hparameters["pad"]

        n_H = int((n_H_prev - f + 2 * pad)/stride) + 1
        n_W = int((n_W_prev - f + 2 * pad)/stride) + 1

        Z = np.zeros((m, n_H, n_W, n_C))

        A_prev_pad = zero_pad(A_prev, pad)

        for i in range(m):                               
		a_prev_pad = A_prev_pad[i,:,:,:]             
		for h in range(n_H):                       		    
			for w in range(n_W):                      		        
				for c in range(n_C):                   

		           
		                	vert_start = stride * h
		                	vert_end = vert_start + f
		                	horiz_start = stride * w
		                	horiz_end = horiz_start + f

		                	a_slice_prev = a_prev_pad[vert_start:vert_end,horiz_start:horiz_end,:]

			                Z[i, h, w, c] = conv_single_step(a_slice_prev,W[:,:,:,c], b[:,:,:,c])				

	assert(Z.shape == (m, n_H, n_W, n_C))    
	cache = (A_prev, W, b, hparameters)	    
	return Z, cache

def fully_connected(A_prev, W, b):
	A_new = np.reshape(A_prev, (A_prev.shape[0], -1))
	Z = A_new.dot(W) + b
	return Z

def relu(A_prev):
	return np.maximum(0, A_prev)

np.random.seed(1)
#A_prev = 32*32 image
W = np.random.randn(1,3,3,4)
b = np.random.randn(1,1,1,4)
hparameters = {"pad":0, "stride":2}

Z, cache_conv = conv_forward(A_prev, W, b, hparameters)
Y = relu(Z)

W1 = np.random.randn(1064,10)
b1 = np.random.randn(1,10)

K = fully_connected(Y, W1, b1)


	
	
