import numpy as np

def receptive_field(n_layers, n_dilation_depth, factor=2):
    kernel_size = 2
    dilation = lambda x: factor ** x
    dilations = [dilation(i % n_dilation_depth) for i in range(1, n_layers+1)]
    return (kernel_size - 1) * sum(dilations) + 1
    #return n_stack * (2 ** n_dilation_depth) - (n_stack -1 )