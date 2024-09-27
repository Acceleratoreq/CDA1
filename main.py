import numpy as np
import scipy.sparse as sp
import tensorflow as tf
import gc
import random
from clac_metric import cv_model_evaluate
from utils import *
from model import GCNModel
from opt import Optimizer
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
def PredictScore(train_circ_dis_matrix, circ_matrix, dis_matrix, seed, epochs, emb_dim, dp, lr,  adjdp):
    np.random.seed(seed)
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    adj = constructHNet(train_circ_dis_matrix, circ_matrix, dis_matrix)
    adj = sp.csr_matrix(adj)
    association_nam = train_circ_dis_matrix.sum()
    X = constructNet(train_circ_dis_matrix)
    features = sparse_to_tuple(sp.csr_matrix(X))
    num_features = features[2][1]
    features_nonzero = features[1].shape[0]
    adj_orig = train_circ_dis_matrix.copy()
    adj_orig = sparse_to_tuple(sp.csr_matrix(adj_orig))
    adj_norm = preprocess_graph(adj)
    adj_nonzero = adj_norm[1].shape[0]
    placeholders = {
        'features': tf.sparse_placeholder(tf.float32),
        'adj': tf.sparse_placeholder(tf.float32),
        'adj_orig': tf.sparse_placeholder(tf.float32),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'adjdp': tf.placeholder_with_default(0., shape=())
    }
    model = GCNModel(placeholders, num_features, emb_dim,
                     features_nonzero, adj_nonzero, train_circ_dis_matrix.shape[0])
    with tf.name_scope('optimizer'):
        opt = Optimizer(
            preds=model.reconstructions,
            labels=tf.reshape(tf.sparse_tensor_to_dense(
                placeholders['adj_orig'], validate_indices=False), [-1]),
            model=model,
            lr=lr, num_u=train_circ_dis_matrix.shape[0], num_v=train_circ_dis_matrix.shape[1], association_nam=association_nam)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for epoch in range(epochs):
        feed_dict = dict()
        feed_dict.update({placeholders['features']: features})
        feed_dict.update({placeholders['adj']: adj_norm})
        feed_dict.update({placeholders['adj_orig']: adj_orig})
        feed_dict.update({placeholders['dropout']: dp})
        feed_dict.update({placeholders['adjdp']: adjdp})
        _, avg_cost = sess.run([opt.opt_op, opt.cost], feed_dict=feed_dict)
        if epoch % 100 == 0:
            feed_dict.update({placeholders['dropout']: 0})
            feed_dict.update({placeholders['adjdp']: 0})
            res = sess.run(model.reconstructions, feed_dict=feed_dict)
            print("Epoch:", '%04d' % (epoch + 1),
                  "train_loss=", "{:.5f}".format(avg_cost))
    print('Optimization Finished!')
    feed_dict.update({placeholders['dropout']: 0})
    feed_dict.update({placeholders['adjdp']: 0})
    res = sess.run(model.reconstructions, feed_dict=feed_dict)
    res = res.reshape(train_circ_dis_matrix.shape)
    np.savetxt('predicted_matrix.csv', res, delimiter=',')
    sess.close()
    return res



def cross_validation_experiment(circ_dis_matrix, circ_matrix, dis_matrix, seed, epochs, emb_dim, dp, lr, adjdp):
    index_matrix = np.mat(np.where(circ_dis_matrix == 1))
    association_nam = index_matrix.shape[1]
    random_index = index_matrix.T.tolist()
    random.seed(seed)
    random.shuffle(random_index)
    k_folds = 5
    CV_size = int(association_nam / k_folds)
    temp = np.array(random_index[:association_nam - association_nam % k_folds]).reshape(k_folds, CV_size, -1).tolist()
    temp[k_folds - 1] = temp[k_folds - 1] + random_index[association_nam - association_nam % k_folds:]
    random_index = temp
    metric = np.zeros((1, 8))
    print("seed=%d, evaluating circ-disease...." % (seed))
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    plt.figure()
    for k in range(k_folds):
        print("------this is %dth cross validation------" % (k + 1))
        train_matrix = np.matrix(circ_dis_matrix, copy=True)
        train_matrix[tuple(np.array(random_index[k]).T)] = 0
        circ_len = circ_dis_matrix.shape[0]
        dis_len = circ_dis_matrix.shape[1]
        circ_disease_res = PredictScore(train_matrix, circ_matrix, dis_matrix, seed, epochs, emb_dim, dp, lr, adjdp)
        predict_y_proba = circ_disease_res.reshape(circ_len, dis_len)
        metric_tmp = cv_model_evaluate(circ_dis_matrix, predict_y_proba, train_matrix)
        print(metric_tmp)
        metric += metric_tmp
        fpr, tpr, _ = roc_curve(circ_dis_matrix.flatten(), predict_y_proba.flatten())
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=2, alpha=0.3, label='ROC fold %d (AUC = %0.4f)' % (k, roc_auc))
        del train_matrix
        gc.collect()
    print(metric / k_folds)
    metric = np.array(metric / k_folds)
    plt.plot([0, 1], [0, 1], linestyle='--', lw=1, color='r', alpha=0.8)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xticks([0.0, 0.1,0.2,0.3, 0.4,0.5, 0.6,0.7, 0.8,0.9, 1.0])  # 自定义 x 轴刻度
    plt.yticks([0.0, 0.1,0.2,0.3, 0.4,0.5, 0.6,0.7, 0.8,0.9, 1.0])  # 自定义 y 轴刻度
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()
    return metric
if __name__ == "__main__":
    circ_sim = np.loadtxt('../data/integrated_circ_sim.csv', delimiter=',')
    dis_sim = np.loadtxt('../data/integrated_dise_sim.csv', delimiter=',')
    circ_dis_matrix = np.loadtxt('../data/Association Matrixss.csv', delimiter=',')
    epoch = 3000
    emb_dim = 64
    lr = 0.01
    adjdp = 0.6
    dp = 0.4
    simw = 6
    result = np.zeros((1, 8), float)
    average_result = np.zeros((1, 8), float)
    circle_time = 1
    for i in range(circle_time):
        result += cross_validation_experiment(
            circ_dis_matrix, circ_sim*simw, dis_sim*simw, i, epoch, emb_dim, dp, lr, adjdp)
    average_result = result / circle_time
    np.savetxt('average_result.csv', average_result, delimiter=',')
    print(average_result)
