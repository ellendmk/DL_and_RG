# Adapted by Ellen de Mello Koch
# Code has been adapted from from https://github.com/MelkoCollective/ICTP-SAIFR-MLPhysics

from __future__ import print_function
import tensorflow as tf
import itertools as it
from rbm import RBM
import numpy as np
import math
import os
import time

start_time = time.time()

# Input parameters:
#Input parameters:
Lv           = 10     #linear size of the visible lattice
Lh           = 9      #linear size of the hidden lattice
num_visible  = Lv*Lv  #number of visible nodes
num_hidden   = Lh*Lh  #number of hidden nodes
T            = 2.269  #a temperature for which there are MC configurations stored in data_ising2d/MC_results_solutions

num_visible         = Lv*Lv     #number of visible nodes
num_hidden          = Lh*Lh     #number of hidden nodes
nsteps              = 50000     #number of training steps
learning_rate       = 0.001     #the learning rate will start at this value and decay exponentially
batch_size          = 100       #batch size
num_gibbs           = 1         #number of Gibbs iterations (steps of contrastive divergence)
num_samples         = 1         #number of chains in PCD


### Function to save weights and biases to a parameter file ###
def save_parameters(sess, rbm_num,rbm,epoch=-1):
    weights, visible_bias, hidden_bias = sess.run([rbm.weights, rbm.visible_bias, rbm.hidden_bias])
    parameter_dir = 'data_ising2d/RBM_parameters_tanh81_'+str(rbm_num)+'_'+str(Lv)
    if not(os.path.isdir(parameter_dir)):
      os.mkdir(parameter_dir)
    
    if epoch==-1:
        parameter_file_path =  '%s/parameters_nH%d_L%d' %(parameter_dir,num_hidden,Lv)
        parameter_file_path += '_nsteps'+str(nsteps)+'_T' +str(T)
        np.savez_compressed(parameter_file_path, weights=weights, visible_bias=visible_bias, hidden_bias=hidden_bias)
        return
    parameter_file_path =  '%s/parameters_nH%d_L%d_%d' %(parameter_dir,num_hidden,Lv,epoch)
    parameter_file_path += '_nsteps'+str(nsteps)+'_T' + str(T)
    np.savez_compressed(parameter_file_path, weights=weights, visible_bias=visible_bias, hidden_bias=hidden_bias)

class Placeholders(object):
    pass

class Ops(object):
    pass



ising_dir = 'data_ising2d/'
if not(os.path.isdir(ising_dir)):
  os.mkdir(ising_dir)

parameter_dir = 'data_ising2d/RBM_parameters_Lv_'+str(Lv)+'/'
if not(os.path.isdir(parameter_dir)):
  os.mkdir(parameter_dir)

weights      = None  #weights
visible_bias = None  #visible bias
hidden_bias  = None  #hidden bias

str1="lr:"+str(learning_rate)+'\nnsteps:'+str(nsteps)+'\nbsize:'+str(batch_size)+'\ngibbs:'+str(num_gibbs)+'\nnum_samples:'+str(num_samples)

f=open(parameter_dir+'/params.txt','w')
f.write(str1)
f.close()

# Load the MC configuration training data:
trainFileName = 'path_to_data'
xtrain        = np.load(trainFileName)
ept           = np.random.permutation(xtrain) # random permutation of training data
iterations_per_epoch = xtrain.shape[0] / batch_size

# Initialize the RBM class
rbm = RBM(num_hidden=num_hidden, num_visible=num_visible, weights=weights, visible_bias=visible_bias,hidden_bias=hidden_bias, num_samples=num_samples)

# Initialize operations and placeholders classes
ops          = Ops()
placeholders = Placeholders()
placeholders.visible_samples = tf.placeholder(tf.float32, shape=(None, num_visible), name='v') # placeholder for training data

total_iterations = 0 # starts at zero
ops.global_step  = tf.Variable(total_iterations, name='global_step_count', trainable=False)

cost      = rbm.neg_log_likelihood_grad(placeholders.visible_samples, num_gibbs=num_gibbs)
optimizer = tf.train.AdamOptimizer(learning_rate, epsilon=1e-2)
ops.lr    = learning_rate
ops.train = optimizer.minimize(cost, global_step=ops.global_step)
ops.init  = tf.group(tf.initialize_all_variables(), tf.initialize_local_variables())

with tf.Session() as sess:
  sess.run(ops.init)

  bcount      = 0  #counter
  epochs_done = 1  #epochs counter
  for i in range(0,nsteps):
    if bcount*batch_size+ batch_size>=xtrain.shape[0]:
      bcount = 0
      ept    = np.random.permutation(xtrain)

    batch     =  ept[ bcount*batch_size: bcount*batch_size + batch_size,:]
    bcount    += 1
    feed_dict =  {placeholders.visible_samples: batch}

    _, num_steps = sess.run([ops.train, ops.global_step], feed_dict=feed_dict)

    if i % 1000 == 0:
      print ('Epoch = %d' % epochs_done)
    epochs_done += 1
  save_parameters(sess,RBM_num, rbm)
print("--- %s seconds ---" % (time.time() - start_time))


