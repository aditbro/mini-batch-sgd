import random
import math
import arff
import numpy as np

class Node():
    def init_weight(self, nb_input=0):
        self.nb_input = nb_input
        self.weights = [random.random() * 0.0001 for x in range(nb_input)]
        self.bias = 1
        self.bias_delta = 0
        self.prev_bias_delta = 0
        self.prev_delta = [0 for x in range(nb_input)]
        self.current_delta = [0 for x in range(nb_input)]

class Layer():
    def __init__(self, nb_nodes=1, learning_rate=0, momentum=0):
        self.nb_nodes = nb_nodes
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.nodes = [Node() for x in range(nb_nodes)]
        self.input = []
        self.output = [0 for x in range(nb_nodes)]
        self.nb_input = 0
        
    def init_weight(self):
        for node in self.nodes:
            node.init_weight(nb_input=self.nb_input)
            
    def calculate_output(self):
        for i in range(len(self.nodes)):
            total = sum([self.input[x] * self.nodes[i].weights[x] for x in range(len(self.input))])
            total += self.nodes[i].bias
            self.output[i] = self.sigmoid(total)
            
    def sigmoid(self, n):
#         try:
            return 1/(1 + np.exp(-n))
#         except Exception as e:
#             print(n)
#             if n > 2000: return 1.0
#             if n < -2000: return 0
    
    def calculate_output_delta(self, target=0):
        for i in range(len(self.nodes)):
            node = self.nodes[i]
            self.dk = self.output[i] * (1 - self.output[i]) * (target - self.output[i])
            
            for j in range(len(node.weights)):
                delta = node.current_delta[j] + (self.learning_rate * self.dk * self.input[j])
                node.prev_delta[j] = node.current_delta[j]
                node.current_delta[j] = delta
                
            delta = node.bias_delta + (self.learning_rate * self.dk)
            node.prev_bias_delta = node.bias_delta
            node.bias_delta = delta
                
    def calculate_delta(self, next_layer=None):
        for i in range(len(self.nodes)):
            node = self.nodes[i]
            sum_delta = self.calculate_sum_delta(idx=i, next_layer=next_layer)
            self.dk = self.output[i] * (1 - self.output[i]) * sum_delta
            
            for j in range(len(node.weights)):
                delta = node.current_delta[j] + (self.learning_rate * self.dk * self.input[j])
                node.prev_delta[j] = node.current_delta[j]
                node.current_delta[j] = delta
                
            delta = node.bias_delta + (self.learning_rate * self.dk)
            
    def calculate_sum_delta(self, idx=0, next_layer=None):
        return sum([node.weights[idx] * next_layer.dk for node in next_layer.nodes])
            
        
    def update_weight(self):
        for node in self.nodes:
            node.bias = node.bias + node.bias_delta + (self.momentum * node.prev_bias_delta)
            node.prev_bias_delta = node.bias_delta
            for i in range(len(node.weights)):
                node.weights[i] = node.weights[i] + node.current_delta[i] + (self.momentum * node.prev_delta[i])
                node.prev_delta[i] = node.current_delta[i]

class MiniBatchSGDClassifier():
    def __init__(self, batch_size=1, learning_rate=0.1, momentum=0.1, nb_epoch=1):
        self.layers = []
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.nb_epoch = nb_epoch
        self.batches = []
        
    def add_layer(self, nb_nodes=1):
        self.layers.append(Layer(nb_nodes=nb_nodes, learning_rate=self.learning_rate, momentum=self.momentum))
        
    def set_training_data(self, x=[], y=[]):
        self.x = x
        self.y = y
        
        for i in range(math.ceil(len(x)/self.batch_size)):
            current_batch = []
            offset = i * self.batch_size

            for j in range(self.batch_size):
                if(offset + j >= len(x)):
                    pass
                else:
                    current_batch.append({
                        'x': self.x[offset + j],
                        'y': self.y[offset + j]
                    })
            
            self.batches.append(current_batch)
        
    def fit(self):
        self.init_weight()
        for _ in range(self.nb_epoch):
            for batch in self.batches:
                for data in batch:
                    self.feed_forward(data)
                    self.backward_propagate(data)
                self.update_weight()
            
    
    def feed_forward(self, data):
        for i in range(len(self.layers)):
            if i == 0:
                self.layers[0].input = data['x']
            else:
                self.layers[i].input = self.layers[i-1].output
            self.layers[i].calculate_output()
            
    def backward_propagate(self, data):
        for i in range(0, len(self.layers)):
            idx = len(self.layers) - i - 1
            
            if idx == len(self.layers) - 1 :
                self.layers[idx].calculate_output_delta(target=data['y'])
            else:
                self.layers[idx].calculate_delta(next_layer=self.layers[idx+1])
                
    def update_weight(self):
        for layer in self.layers:
            layer.update_weight()
    
    def init_weight(self):
        for i in range(len(self.layers)):
            if i == 0:
                self.layers[0].nb_input = len(self.batches[0][0]['x'])
            else :
                self.layers[i].nb_input = self.layers[i-1].nb_nodes
            
            self.layers[i].init_weight()
            
    def predict(self, data):
        self.feed_forward(data)
        return self.layers[-1].output