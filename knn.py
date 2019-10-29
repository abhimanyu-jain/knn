import math
from collections import Counter

class KNeighborsClassifier:

    def __init__(self, k):
        self.k = k

    def get_mode(self, my_list):
    	counter = Counter(my_list)
    	max_count = max(counter.values())
    	mode = [k for k,v in counter.items() if v == max_count]
    	return mode[0]

    def euclidean(self, x, y):
    	distance = math.sqrt(sum([(a - b) ** 2 for a, b in zip(x, y)]))
    	return distance

    def fit(self, X_train, y_train):
    	self.x_train = X_train
    	self.y_train = y_train

    def predict(self, X_test):
    	#Iterate over X_test
    	y_test = []
    	index = 0
    	for x in X_test:
    		h = []
    		for i in range(len(self.x_train)):
    			#Calculate distance between x and a
    			a = self.x_train[i]
    			b = self.y_train[i]
    			dist = self.euclidean(x, a)
    			h.append([dist, b])
    		
    		sorted_d = sorted(h, key=lambda x: x[0])
    		#Get top k closest labels.
    		top_5 = sorted_d[0:self.k]
    		values = []
    		for j in range(len(top_5)):
    			values.append(top_5[j][1])
    		y_test.append(self.get_mode(values))
    	return y_test

    def score(self, y_pred_test, y_test):
    	correct = 0
    	wrong = 0
    	for i in range(len(y_pred_test)):
    		if(y_pred_test[i] == y_test[i]):
    			correct+=1
    		else:
    			wrong+=1
    	score = correct/(correct + wrong)*100
    	return score
