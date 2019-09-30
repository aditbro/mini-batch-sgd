import random
import math
import arff
import numpy as np
from classifier import MiniBatchSGDClassifier

data = arff.load(open('weather.arff', 'r'))
outlook_dict = {'rainy': 50, 'overcast': 75, 'sunny': 100}
windy_dict = {'TRUE': 50, 'FALSE': 100}
play_dict = {'yes': 1, 'no': 0}

for i in range(len(data['data'])):
    data['data'][i][0] = outlook_dict[data['data'][i][0]]
    data['data'][i][3] = windy_dict[data['data'][i][3]]
    data['data'][i][4] = play_dict[data['data'][i][4]]

random.shuffle(data['data'])
x = [x[0:4] for x in data['data'][:8]]
y = [x[4] for x in data['data'][:8]]
x1 = [x[0:4] for x in data['data'][8:]]
y1 = [x[4] for x in data['data'][8:]]

sgd = MiniBatchSGDClassifier(batch_size=1, learning_rate=10e-5, momentum=0.5, nb_epoch=1000)
sgd.set_training_data(x=x, y=y)
sgd.add_layer(nb_nodes=10)
sgd.add_layer(nb_nodes=1)
sgd.fit()

total_correct = 0

for i in range(len(x1)):
    result = sgd.predict(data={'x': x1[i]})
    print('result: {}, target: {}'.format(result,y1[i]))
    if(abs(y1[i] - result[0]) < 0.5):
        total_correct += 1
        
print(total_correct/len(x1) * 100, '%')
