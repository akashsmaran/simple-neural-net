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

def conv_backward(dZ, cache):
	(A_prev, W, b, hparameters) = cache

    
	(m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

    	(f, f, n_C_prev, n_C) = W.shape

    	stride = hparameters["stride"]
    	pad = hparameters["pad"]

    	(m, n_H, n_W, n_C) = dZ.shape

    	dA_prev = np.zeros((m, n_H_prev, n_W_prev, n_C_prev))                           
    	dW = np.zeros((f, f, n_C_prev, n_C))
    	db = np.zeros((1, 1, 1, n_C))

    	A_prev_pad = zero_pad(A_prev, pad)
    	dA_prev_pad = zero_pad(dA_prev, pad)

    	for i in range(m):
        	a_prev_pad = A_prev_pad[i,:,:,:]
        	da_prev_pad = dA_prev_pad[i,:,:,:]
        	for h in range(n_H):                   
            		for w in range(n_W):               
                		for c in range(n_C):           
                    			vert_start = stride * h
                    			vert_end = vert_start + f
                    			horiz_start = stride * w
                    			horiz_end = horiz_start + f
                    			a_slice = A_prev_pad[i, vert_start:vert_end, horiz_start:horiz_end, :]

                    			da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += W[:,:,:,c] * dZ[i, h, w, c]
                    			dW[:,:,:,c] += a_slice * dZ[i, h, w, c]
                    			db[:,:,:,c] += dZ[i, h, w, c]

        	dA_prev[i, :, :, :] = da_prev_pad[pad:-pad, pad:-pad, :]

    	assert(dA_prev.shape == (m, n_H_prev, n_W_prev, n_C_prev))

    	return dA_prev, dW, db

def relu_backward(A_next):
	return 1. * ( x > 0)

def mse(X, Y):
	return (1. / 2. * X.shape[0]) * ((X - Y) ** 2.)

def mse_deriv(result, expected):
	return (X - Y)/ X.shape[0]

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

final_loss = mse_deriv(K, expected_result)  # expected result of size 1 X 10

# for fcl

grad1 = K.T.dot(final_loss)
W1 -= grad1

# for relu

grad2 = relu_backward(Y) * grad1
grad3, grad4, grad5 = conv_backward(grad1, cache)
W -= grad3/grad4
b -= grad3/grad5  
