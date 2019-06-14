#coding:utf-8   

from keras.models import load_model
# backend.set_image_dim_ordering('tf')
import math
import pickle
import sys

all_txt=" "
train_txt=" "
test_txt=" "
img_path=' /'
model  = load_model(' ')

new_test=test_txt

X_test = pickle.load( open(sys.argv[1], 'rb'))
test_label = pickle.load( open(sys.argv[2], 'rb'))

result = model.predict(X_test)

#---------------------------------------true_low_data--------------------------------

res_f = open(' ',' ')
test_content = open(new_test).readlines()

j_0_0 = 0
j_0_1 = 0
j_0_2 = 0
j_1_0 = 0
j_1_1 = 0
j_1_2 = 0
j_2_0 = 0
j_2_1 = 0
j_2_2 = 0

for i,pred in enumerate(list(result)):

	sample = test_content[i].strip('\n').split('\t')[0]
	label = test_content[i].strip('\n').split('\t')[1]
	res = list(pred)
	label_pred = res.index(max(res))
	print(sample+'\t'+label + "\t"+str(label_pred))
	res_f.write(test_content[i].strip('\n')+'\t'+ str(label_pred) +'\n')

	if label=='0' and label_pred == 0:
		j_0_0+=1
	elif label=='0' and label_pred == 1:
		j_0_1+=1
	elif label=='0' and label_pred == 2:
		j_0_2+=1
	elif label=='1' and label_pred == 0:
		j_1_0+=1
	elif label=='1' and label_pred == 1:
		j_1_1+=1
	elif label=='1' and label_pred == 2:
		j_1_2+=1
	elif label=='2' and label_pred == 0:
		j_2_0+=1
	elif label=='2' and label_pred == 1:
		j_2_1+=1
	elif label=='2' and label_pred == 2:
		j_2_2+=1
res_f.close()

print("j_0_0 = " + str(j_0_0))
print("j_0_1 = " + str(j_0_1))
print("j_0_2 = " + str(j_0_2))
print("j_1_0 = " + str(j_1_0))
print("j_1_1 = " + str(j_1_1))
print("j_1_2 = " + str(j_1_2))
print("j_2_0 = " + str(j_2_0))
print("j_2_1 = " + str(j_2_1))
print("j_2_2 = " + str(j_2_2))

