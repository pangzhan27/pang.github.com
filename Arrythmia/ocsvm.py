import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn import svm
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve

def read_data():
    pkl_file = open('./data/arr.pkl', 'rb')
    data = pickle.load(pkl_file)
    all_set, all_label = data['all'][0], data['all'][1]
    all_label =  np.array((np.ones(386)).tolist() + (-1*np.ones(66)).tolist())
    return all_set, all_label

if __name__ == "__main__":
    data, label = read_data()
    train_data, test_data, train_target, test_target = data[:226, :], data[226:, :], label[:226], label[226:]

    nu = 0.1
    print("The calculated values of nu is:", nu)

    model = svm.OneClassSVM(nu=nu, kernel='rbf', gamma=1e-4,tol= 1e-5)
    model.fit(train_data)

    values_preds = model.predict(train_data)
    values_targs = train_target

    values_preds = model.predict(test_data)
    values_targs = test_target

    temp = values_preds - values_targs
    n_a2n = (temp == -2).sum()
    n_n2a = (temp == 2).sum()
    # print(n_n2a, n_a2n)

    TP = 66 - n_n2a
    precision = TP / (TP + n_a2n)
    recall = TP / (TP + n_n2a)
    F1 = 2 * TP / (2 * TP + n_n2a + n_a2n)
    print(precision, recall, F1)


