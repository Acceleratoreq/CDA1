import tensorflow as tf
from utils import *

class GraphConvolution():
    """Basic graph convolution layer for undirected graph without edge labels."""

    def __init__(self, input_dim, output_dim, adj, name, dropout=0., act=tf.nn.relu):
        self.name = name
        self.vars = {}
        self.issparse = False
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(
                input_dim, output_dim, name='weights')
        self.dropout = dropout
        self.adj = adj
        self.act = act

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            x = inputs
            x = tf.nn.dropout(x, 1-self.dropout)
            x = tf.matmul(x, self.vars['weights'])
            x = tf.sparse_tensor_dense_matmul(self.adj, x)
            outputs = self.act(x)
        return outputs


class GraphConvolutionSparse():
    """Graph convolution layer for sparse inputs."""

    def __init__(self, input_dim, output_dim, adj, features_nonzero, name, dropout=0., act=tf.nn.relu):
        self.name = name
        self.vars = {}
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(
                input_dim, output_dim, name='weights')
        self.dropout = dropout
        self.adj = adj
        self.act = act
        self.issparse = True
        self.features_nonzero = features_nonzero

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            x = inputs
            x = dropout_sparse(x, 1-self.dropout, self.features_nonzero)
            x = tf.sparse_tensor_dense_matmul(x, self.vars['weights'])
            x = tf.sparse_tensor_dense_matmul(self.adj, x)
            outputs = self.act(x)
        return outputs
class MLP():
    """Multi-layer perceptron (MLP) class."""

    def __init__(self, input_dim, hidden_dims, output_dim, name, dropout=0., act=tf.nn.relu):
        self.name = name
        self.vars = {}
        with tf.variable_scope(self.name + '_vars'):
            # Define MLP layers based on hidden_dims
            self.hidden_layers = []
            for i, h_dim in enumerate(hidden_dims):
                if i == 0:
                    input_dim = input_dim
                else:
                    input_dim = hidden_dims[i-1]
                self.hidden_layers.append(tf.keras.layers.Dense(h_dim, activation=act, name=f'hidden_layer{i+1}'))
            self.output_layer = tf.keras.layers.Dense(output_dim, activation=None, name='output_layer')
        self.dropout = dropout
        self.act = act

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            x = inputs
            for layer in self.hidden_layers:
                x = tf.nn.dropout(x, 1 - self.dropout)
                x = layer(x)
            x = self.output_layer(x)
            outputs = self.act(x)
        return outputs


class InnerProductDecoder():
    """Decoder model layer for link prediction."""

    def __init__(self, input_dim, name, num_r, dropout=0., act=tf.nn.sigmoid):
        self.name = name
        self.vars = {}
        self.issparse = False
        self.dropout = dropout
        self.act = act
        self.num_r = num_r
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(
                input_dim, input_dim, name='weights')

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            inputs = tf.nn.dropout(inputs, 1-self.dropout)
            R = inputs[0:self.num_r, :]
            D = inputs[self.num_r:, :]
            R = tf.matmul(R, self.vars['weights'])
            D = tf.transpose(D)
            x = tf.matmul(R, D)
            x = tf.reshape(x, [-1])
            outputs = self.act(x)
        return outputs

class DotProductDecoder():
    """Decoder model layer for link prediction."""

    def __init__(self, input_dim, name, num_r, dropout=0., act=tf.nn.sigmoid):
        self.name = name
        self.vars = {}
        self.issparse = False
        self.dropout = dropout
        self.act = act
        self.num_r = num_r

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            inputs = tf.nn.dropout(inputs, 1 - self.dropout)
            R = inputs[0:self.num_r, :]
            D = inputs[self.num_r:, :]
            R = tf.matmul(R, tf.transpose(D))
            x = tf.reshape(R, [-1])
            outputs = self.act(x)
        return outputs

class BilinearDecoder():
    def __init__(self, input_dim, name, num_r, rate=0., act=tf.nn.sigmoid):
        self.name = name
        self.vars = {}
        self.num_r = num_r  # 设置 num_r 属性
        self.rate = rate
        self.act = act
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(input_dim, input_dim, name='weights')

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            inputs = tf.nn.dropout(inputs, rate=self.rate)
            R = inputs[:self.num_r, :]  # 使用 self.num_r
            D = inputs[self.num_r:, :]
            R = tf.matmul(R, self.vars['weights'])
            D = tf.transpose(D)
            x = tf.matmul(R, D)
            x = tf.reshape(x, [-1])
            outputs = self.act(x)
        return outputs
