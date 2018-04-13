'''
Created on Mar 29, 2018

@author: pangzhanzhong
'''
import theano
from theano import tensor as T
import lasagne
from lasagne.layers import *
from lasagne.objectives import *
from lasagne.nonlinearities import *
from lasagne.updates import *
from lasagne.utils import *
import numpy as np
import cPickle as pickle
import gzip

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import os
import sys

# --------------

def author_net(args={}):
    l_in = InputLayer((None, 1, 28, 28))
    l_conv = Conv2DLayer(l_in, num_filters=128, filter_size=7, nonlinearity=softplus)
    l_conv = Pool2DLayer(l_conv, 2, mode='average_inc_pad')
    l_conv = DenseLayer(l_conv, args["h"], nonlinearity=softplus)
    for layer in get_all_layers(l_conv)[::-1]:
        if isinstance(layer, InputLayer):
            break
        l_conv = InverseLayer(l_conv, layer)

    for layer in get_all_layers(l_conv):
        print(layer, layer.output_shape)
    print(count_params(layer))

    l_out = l_conv
    
    return l_out

# -----------------------------------
def author_net1(args={}):
    l_in = InputLayer((None, 274))
    l_conv = DenseLayer(l_in, 10, nonlinearity=softplus)
    l_conv = DenseLayer(l_conv, 2, nonlinearity=softplus)
    for layer in get_all_layers(l_conv)[::-1]:
        if isinstance(layer, InputLayer):
            break
        l_conv = InverseLayer(l_conv, layer)

    for layer in get_all_layers(l_conv):
        print(layer, layer.output_shape)
    print(count_params(layer))

    l_out = l_conv
    
    return l_out

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
srng = RandomStreams(123)

def get_net(net_cfg, args={"lambda":0.5}):
    l_out = net_cfg(args)

    X = T.matrix('X')
    X_noise = X + srng.normal(X.shape, std=1.)
    b_prime = theano.shared( np.zeros( (274) ) )
    net_out = get_output(l_out, X)
    net_out_noise = get_output(l_out, X_noise)
    energy = args["lambda"]*((X-b_prime)**2).sum() - net_out.sum()
    energy_noise = args["lambda"]*((X_noise-b_prime)**2).sum() - net_out_noise.sum()
    # reconstruction
    fx = X - T.grad(energy, X)
    fx_noise = X_noise - T.grad(energy_noise, X_noise)
    loss = ((X-fx_noise)**2).sum(axis=[1]).mean()

    
    
    params = get_all_params(l_out, trainable=True)
    params += [b_prime]
    lr = theano.shared(floatX(args["learning_rate"]))
    #updates = nesterov_momentum(loss, params, learning_rate=lr, momentum=0.9)
    updates = adadelta(loss, params, learning_rate=lr)
    #updates = rmsprop(loss, params, learning_rate=lr)
    train_fn = theano.function([X], [loss,energy], updates=updates)
    energy_fn = theano.function([X], energy)
    out_fn = theano.function([X], fx)
    loss_fn = theano.function([X], loss)
    
    return {
        "train_fn": train_fn,
        "energy_fn": energy_fn,
        "out_fn": out_fn,
        "lr": lr,
        "b_prime": b_prime,
        "l_out": l_out,
        "loss_fn": loss_fn
    }

def iterate(X_train, bs=32):
    b = 0
    while True:
        if b*bs >= X_train.shape[0]:
            break
        yield X_train[b*bs:(b+1)*bs]
        b += 1

def train(cfg, data, num_epochs, out_file, sched={}, batch_size=100):
    train_fn = cfg["train_fn"]
    energy_fn = cfg["energy_fn"]
    b_prime = cfg["b_prime"]
    out_fn = cfg["out_fn"]
    print("ok")
    X_train_nothing, X_train_anomaly, X_valid_nothing, X_valid_anomaly = data
    idxs = [x for x in range(0, X_train_nothing.shape[0])]
    #train_losses = []
    lr = cfg["lr"]
    print("ok")
    with open(out_file, "wb") as f:
        f.write("epoch,loss,avg_base_energy,avg_anom_energy\n")
        for epoch in range(0, num_epochs):
#             print(epoch)
            if epoch+1 in sched:
                lr.set_value( floatX(sched[epoch+1]) )
                sys.stderr.write("changing learning rate to: %f\n" % sched[epoch+1])
            np.random.shuffle(idxs)
            X_train_nothing = X_train_nothing[idxs]

            losses = []
            energies = []

            for X_batch in iterate(X_train_nothing, bs=batch_size):
                this_loss, this_energy = train_fn(X_batch)
                losses.append(this_loss)
                energies.append(this_energy)

            anomaly_energies = []
            for X_batch in iterate(X_train_anomaly, bs=batch_size):
                this_energy = energy_fn(X_batch)
                anomaly_energies.append(this_energy)

            #valid_losses = []
            #for X_batch in iterate(X_valid_nothing, bs=batch_size):
            #    this_loss, _ = 

                
            print(epoch+1, np.mean(losses), np.mean(energies), np.mean(anomaly_energies))
            f.write("%i,%f,%f,%f\n" % (epoch+1, np.mean(losses), np.mean(energies), np.mean(anomaly_energies)))
            
            if epoch > 100 and epoch%100 ==0:
               with open("%s_" %out_file+str(epoch)+'.model', "wb") as f1:
                   pickle.dump([ get_all_param_values(cfg["l_out"]), b_prime.get_value()], f1, pickle.HIGHEST_PROTOCOL)


            
      
def get_energies(cfg, data, batch_size=1):
    energy_fn = cfg["energy_fn"]
    tot = []
    for dataset in data:
        energies = []
        for X_single in iterate(dataset, bs=batch_size):
            energies.append( float(energy_fn(X_single)) )
        tot.append( energies )
    return tot

def get_loss(cfg, data, batch_size=1):
    loss_fn = cfg["loss_fn"]
    tot = []
    for dataset in data:
        losses = []
        for X_single in iterate(dataset, bs=batch_size):
            losses.append( float(loss_fn(X_single)) )
        tot.append( losses )
    return tot


def save_array(arr, filename, header):
    with open(filename,"wb") as f:
        f.write("%s\n" % header)
        for elem in arr:
            f.write("%f\n" % elem)

Fla =0
        
if __name__ == "__main__":
#     data = three_vs_seven()
#     X_train_three, X_train_seven, X_valid_three, X_valid_seven = data
    pkl_file = open('./arr.pkl', 'rb')
    data = pickle.load(pkl_file) 
    train_set,all_set, test_set = data['train'][0],data['all'][0],data['test'][0]
      
    train_set = train_set.astype(np.float32)
    all_set = all_set.astype(np.float32)
    test_set = test_set.astype(np.float32)

    prefix = "three_vs_seven/author_net_512h.txt"
    lamb=0.5
    if Fla ==1:      
       X_train_three, X_train_seven, X_valid_three, X_valid_seven = \
         train_set, test_set[113:], test_set[:113],test_set[113:]
       cfg = get_net(author_net1, {"learning_rate": 0.1, "lambda":lamb, "h":512})
       data = (X_train_three, X_train_seven, X_valid_three, X_valid_seven)
       train(cfg, data, num_epochs=1000, out_file=prefix, batch_size=113)
    else:
    #collect the energies
       data = (train_set,all_set, test_set)
       ep = 300
       cfg = get_net(author_net1, {"learning_rate": 0.1, "lambda":lamb, "h":512})
       with open("%s_" %prefix +str(ep)+'.model') as g:
           model = pickle.load(g)
           set_all_param_values(cfg["l_out"], model[0])
           cfg["b_prime"].set_value(model[1])
       train, all, test = get_energies(cfg, data, batch_size=1)
#        save_array(train, "%s.train_e.csv" % prefix, "train")
       save_array(all, "%s.all_e.csv" % prefix, "all")
       save_array(test, "%s.test_e" % prefix+str(ep)+".csv", "test")
       
       train, all, test = get_loss(cfg, data, batch_size=1)
#        save_array(train, "%s.train_r.csv" % prefix, "train")
       save_array(all, "%s.all_r.csv" % prefix, "all")
       save_array(test, "%s.test_r" % prefix+str(ep)+".csv", "test")

