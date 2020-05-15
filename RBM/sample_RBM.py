# Adapted by Ellen de Mello Koch
# Code has been adapted from from https://github.com/MelkoCollective/ICTP-SAIFR-MLPhysics


## Samples outputs and flows from trained RBM ##

from __future__ import print_function
import tensorflow as tf
from rbm import RBM as RBM
import numpy as np
import os

import time

#Input parameters:
Lv           = 10     #linear size of the visible lattice
Lh           = 10     #linear size of the hidden lattice
num_visible  = Lv*Lv  #number of visible nodes
num_hidden   = Lh*Lh  #number of hidden nodes


def sampleOutput(data_path, batch_size):
    #Sampling parameters:
    num_samples  = 1  # how many independent chains will be sampled
    gibb_updates = 1    # how many gibbs updates per call to the gibbs sampler
    nbins        = 1  # number of calls to the RBM sampler
    nsteps       = 10000
    samplesData  = np.load(data_path)
    nbins        = int(samplesData.shape[0]/batch_size)
   
    spins_dir = 'RBM_observables_Lv'+str(Lv)+'/'
    if not(os.path.isdir(spins_dir)):
      os.mkdir(spins_dir)

    #Initialize the RBM for each temperature in T_list:
    rbms           = []
    rbm_samples    = []
    
    output_file_path = spins_dir+"output_"+str(nsteps)+".npy"

    #Read in the trained RBM parameters:
    path_to_params =  'RBM_parameters/parameters_nH%d_Lv%d' %(num_hidden,Lv)
    path_to_params += '_nsteps'+str(nsteps)+'_T'+str(2.269)+'.npz'
    params         =  np.load(path_to_params)
    weights        =  params['weights']
    visible_bias   =  params['visible_bias']
    hidden_bias    =  params['hidden_bias']
    hidden_bias    =  np.reshape(hidden_bias,(hidden_bias.shape[0],1))
    visible_bias   =  np.reshape(visible_bias,(visible_bias.shape[0],1))


    rbm = RBM(
      num_hidden = num_hidden, num_visible = num_visible,
      weights = weights, visible_bias = visible_bias, hidden_bias = hidden_bias,
      num_samples = num_samples
    )

    # Initialize tensorflow
    init = tf.group(tf.initialize_all_variables(), tf.initialize_local_variables())

    # Store spins
    N       = num_visible
    samples = []

    sess    = tf.Session()
    sess.run(init)

    rbm_samples = sess.run(rbm.stochastic_maximum_likelihood_startT(num_iterations = 1, start_config = samplesData, 1))

    np.save(output_file_path, rbm_samples)

def sampleFlows(data_path):
    #Sampling parameters:
    num_samples  = 1    # how many independent chains will be sampled
    gibb_updates = 1    # how many gibbs updates per call to the gibbs sampler
    nbins        = 1    # number of calls to the RBM sampler
    nsteps       = 10000

    samplesData=np.load(data_path)
    
    # Collect flows from length 1 up to 30
    for gibb_updates in range(1,30):
      start_time = time.time()
      print("Gibbs updates = "+str(gibb_updates))
      
      spins_dir = 'RBM_observables_Lv'+str(Lv)+'/'
      if not(os.path.isdir(spins_dir)):
        os.mkdir(spins_dir)

      #Initialize the RBM for each temperature in T_list:
      rbms           = []
      rbm_samples    = []
     
      spin_file_path=spins_dir+"spins"+str(gibb_updates)+"_"+str(nsteps)+".npy"

      #Read in the trained RBM parameters:
      path_to_params =  'RBM_parameters/parameters_nH%d_Lv%d' %(num_hidden,Lv)
      path_to_params += '_nsteps'+str(nsteps)+'_T'+str(2.269)+'.npz'
      params         =  np.load(path_to_params)
      weights        =  params['weights']
      visible_bias   =  params['visible_bias']
      hidden_bias    =  params['hidden_bias']
      hidden_bias    =  np.reshape(hidden_bias,(hidden_bias.shape[0],1))
      visible_bias   =  np.reshape(visible_bias,(visible_bias.shape[0],1))
      
      # Initialize RBM class

      rbms = RBM(
        num_hidden = num_hidden, num_visible = num_visible,
        weights = weights, visible_bias = visible_bias, hidden_bias = hidden_bias,
        num_samples = num_samples
      )
      rbm_samples = rbms.stochastic_maximum_likelihood_startT(gibb_updates,samplesData)

      # Initialize tensorflow
      init = tf.group(tf.initialize_all_variables(), tf.initialize_local_variables())

      # Store samples:
      N = num_visible
      samples = []

      with tf.Session() as sess:
        sess.run(init)
        spins = np.zeros((rbm_samples.get_shape().as_list()[0], rbm_samples.get_shape().as_list()[1]))

        for i in range(nbins):
          for k in range(0, rbm_samples.get_shape().as_list()[0], 10000):
            samples = (np.array(sess.run(rbm_samples[k : k + 10000])))
            spins[k : k + 10000, : ] = (np.asarray((samples)).copy())

        np.save(spin_file_path,spins)

        end_time = time.time()

        print("Time: " + str(end_time - start_time))


sampleOutput(data_path = "path_to_data", batch_size = 100)
sampleFlows(data_path = "path_to_data")