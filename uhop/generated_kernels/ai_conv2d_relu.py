import numpy as np


def generated_conv2d_relu(a, b):
    # Get dimensions
    n, h, w, c = a.shape
    kh, kw, _, co = b.shape
    
    # Calculate output dimensions
    oh = h - kh + 1
    ow = w - kw + 1
    
    # Initialize output
    output = np.zeros((n, oh, ow, co))
    
    # Perform convolution
    for i in range(oh):
        for j in range(ow):
            for k in range(co):
                output[:, i, j, k] = np.sum(a[:, i:i+kh, j:j+kw, :] * b[:, :, :, k], axis=(1, 2, 3))
    
    # Apply ReLU activation
    output = np.maximum(0, output)
    
    return output