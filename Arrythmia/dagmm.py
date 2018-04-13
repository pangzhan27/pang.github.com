import tensorflow as tf
import numpy as np
from pandas import read_csv
from Dataset_1D import DataSet_1D
from dagmm import DAGMM
import math
import os
from matplotlib import pyplot as plt
import logging

TRAIN = 0
lr = 1e-4
batch_size = 128
epochs = 10000
output_path = './model_Arrhythmia/'
#output_path1 = './model_Thyroid/1/'
out_gmm = ''#-40000
model_name = 'epoch{0}'.format(epochs)


'''logging'''

logging.basicConfig(level=logging.DEBUG,format='[%(levelname)s] %(asctime)s [%(filename)s:%(lineno)d] %(message)s',
                filename='none.log', filemode='a')
logger = logging.getLogger(__name__)
logger.info("*************************************************************************************************")
if TRAIN == 1:
    logger.warning("learning rate = {:.5f} ".format(lr))
    logger.warning("batch size = {} ".format(batch_size))
    logger.warning("model = {} ".format(output_path+model_name))
    logger.info("*****************************************************")

'''data read'''
def read_data():
    norm_file = './data/Arrhythmia/norm.csv'
    anorm_file = './data/Arrhythmia/anorm.csv'
    norm = read_csv(norm_file, header=0, index_col=0)
    anorm = read_csv(anorm_file, header=0, index_col=0)
    norm_data = norm.values;
    anorm_data = anorm.values

    total_num = len(norm_data) + len(anorm_data)
    perm = np.arange(len(norm_data))
    np.random.seed(12345678)
    np.random.shuffle(perm)
    norm_data = norm_data[perm]

    threshold = int(total_num/2)
    train_x = norm_data[:int(threshold)]
    train_y = np.zeros([int(threshold),1])
    test1_x = norm_data[threshold:]
    test1_y = np.zeros([len(norm_data) - threshold, 1])
    test_x = np.row_stack((test1_x, anorm_data))
    test_y = np.row_stack((test1_y, np.ones([len(anorm_data), 1])))

    max = train_x.max(axis=0)
    min = train_x.min(axis=0)
    xx = len(train_x)
    _max = np.tile(max, (xx, 1))
    _min = np.tile(min, (xx, 1))
    diff = _max - _min
    train_xs = (train_x - _min) / (diff + 1e-6)

    xx = len(test_x)
    _max = np.tile(max, (xx, 1))
    _min = np.tile(min, (xx, 1))
    diff = _max - _min
    test_xs = (test_x - _min) / (diff + 1e-6)


    train = DataSet_1D(train_xs, train_y)
    test = DataSet_1D(test_xs, test_y)
    return train, test
'''training'''
def train(data):
    model = DAGMM(ae_layers=[274, 10, 2],es_layers=[4,10,2],lr = lr)
    saver = tf.train.Saver(max_to_keep=10)
    saver1 =  tf.train.Saver()
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    with tf.Session() as sess:
       sess.run(tf.global_variables_initializer())
       ckpt = tf.train.get_checkpoint_state('./model_Arrhythmia/')
       # if ckpt and ckpt.model_checkpoint_path:
       #      saver.restore(sess, ckpt.model_checkpoint_path)
       step = 0
       best = 100.0
       bestiter = 0
       ave_loss = 0.0
       outputs = [model.opt, model.loss, model.recon_loss, model.ez_loss, model.p_loss]
       #outputs = [model.opt, model.loss, model.recon_loss, model.ez_loss, model.p_loss, model.center_loss]
       while True:
           batch, _, epoch = data.next_batch( batch_size )
           _, loss, recon_loss, ez_loss, p_loss = sess.run(outputs, feed_dict={model.inputs: batch,model.keep_prob: 0.5})
           #_, loss, recon_loss, ez_loss, p_loss, center_loss = sess.run(outputs,feed_dict={model.inputs: batch, model.keep_prob: 0.5})
           step += 1
           ave_loss += loss
           if (step+1) % 100 == 0:
               print('Epoch: %d, Step: %d, aver: %g, loss: %g, recon: %g, ez: %g, p: %g ' % (epoch, step+1, ave_loss/100, loss, recon_loss, ez_loss, p_loss))
               logger.info('Epoch: {}, Step: {}, aver: {:.5f}, loss: {:.5f}, recon: {:.5f}, ez: {:.5f}, p: {:.5f}'.format(
                   epoch, step+1, ave_loss/100, loss, recon_loss, ez_loss, p_loss))
               # print('Epoch: %d, Step: %d, aver: %g, loss: %g, recon: %g, ez: %g, p: %g, cl: %g '
               #       % (epoch, step + 1, ave_loss / 100, loss, recon_loss, ez_loss, p_loss, center_loss))
               # logger.info('Epoch: {}, Step: {}, aver: {:.5f}, loss: {:.5f}, recon: {:.5f}, ez: {:.5f}, p: {:.5f}, cl: {:.5f}'.format(
               #         epoch, step + 1, ave_loss / 100, loss, recon_loss, ez_loss, p_loss, center_loss))
               if (step+1)  % 1000 ==0 and (step+1) >4000:
                   best = ave_loss/100
                   bestiter = step
                   phi, mu, sigma = sess.run((model.phi, model.mu, model.sigma), feed_dict={model.inputs: batch, model.keep_prob: 0.5})
                   saver.save(sess, os.path.join(output_path, model_name),global_step=step)
                   np.save(os.path.join(output_path, model_name)+str(step+1)+'_mu.npy', mu)
                   np.save(os.path.join(output_path, model_name)+str(step+1)+'_phi.npy', phi)
                   np.save(os.path.join(output_path, model_name)+str(step+1)+'_sigma.npy', sigma)

               if epoch > epochs:
                   break
               # if (epoch+2) % 10000 ==0 :
               #     phi, mu, sigma = sess.run((model.phi, model.mu, model.sigma), feed_dict={model.inputs: batch, model.keep_prob: 0.5})
               #     saver1.save(sess, os.path.join(output_path1, model_name), global_step=epoch+2)
               #     np.save(os.path.join(output_path1, model_name) + '-{}_mu.npy'.format(epoch+2), mu)
               #     np.save(os.path.join(output_path1, model_name) + '-{}_phi.npy'.format(epoch+2), phi)
               #     np.save(os.path.join(output_path1, model_name) + '-{}_sigma.npy'.format(epoch+2), sigma)
               ave_loss = 0
'''testing'''
sss = 10000
def energy_cal(z,k_comp=4):
    try:
       mu = np.load(os.path.join(output_path, model_name)+str(sss)+'_mu.npy')
       phi = np.load(os.path.join(output_path, model_name)+str(sss)+'_phi.npy')
       sigma = np.load(os.path.join(output_path, model_name)+str(sss)+'_sigma.npy')

    except:
       print("model need to be trained first!!!")
       exit()
    EZ = None
    for k in range(k_comp):
        mu_k = mu[k]
        sig = sigma[k]
        phi_k = phi[k]
        temp = np.array(z - mu_k)
        up = -0.5 * np.dot(np.dot(temp, np.linalg.inv(sig)), np.transpose(temp))
        a = np.diag(up)
        ez = phi_k * np.exp(np.diag(up)) / (np.sqrt((2 * math.pi) ** k_comp * np.linalg.det(sig)))

        if EZ is None:
            EZ = ez
        else:
            EZ += ez
    return -np.log(EZ + 1e-8)

def test(train_set,test_set):
    model = DAGMM(ae_layers=[274, 10, 2],es_layers=[4,10,2], forward_only=True)
    with tf.Session() as sess:

       sess.run(tf.global_variables_initializer())
       saver = tf.train.Saver()
       ckpt = tf.train.get_checkpoint_state(output_path )
       if ckpt and ckpt.model_checkpoint_path:
           saver.restore(sess, ckpt.model_checkpoint_path)

       '''threshold calculating'''
       new_x0 = sess.run(model.z,feed_dict={model.inputs: train_set.inputs})
       new_x1 = sess.run(model.z,feed_dict={model.inputs: test_set.inputs})
       new_x = np.row_stack((new_x0, new_x1))
       EZ = energy_cal(new_x,k_comp=2)
       EZ1 = energy_cal(new_x1,k_comp=2)
       EZ = np.sort(EZ)
       total_len = len(train_set.inputs)+ len(test_set.inputs)
       threshold_num = int(0.85 *total_len)
       threshold = EZ[threshold_num-1]

       a = test_set.labels
       EZ1.shape = -1,1
       db = np.column_stack((EZ1,a))
       np.save('./roc.npy',db)
       '''Predicting'''

       pred = (EZ1 > threshold)
       pred = pred.astype(int)

       temp = pred - a
       n_n2a = (temp == -1).sum()
       n_a2n = (temp == 1).sum()
       n_a = test_set.labels.sum()
       n_n = len(test_set.labels) - n_a

       b = np.multiply(pred, a)
       TP = np.sum(b)
       precision = TP / (TP + n_a2n)
       recall = TP / (TP + n_n2a)
       F1 = 2 * TP / (2 * TP + n_n2a + n_a2n)

       plotDataset = [[], []]
       count = len(EZ)
       for i in range(count):
           plotDataset[0].append(float(EZ[i]))
           plotDataset[1].append((i + 1) / count)
       plt.figure()
       plt.plot(plotDataset[0], plotDataset[1], '-', linewidth=2)
       plt.ylabel("Percentage")
       plt.xlabel("Energy")
       plt.show()

if __name__=='__main__':
    train_set, test_set = read_data()
    if TRAIN == 1:
        train(train_set)
    else:
        test(train_set, test_set)