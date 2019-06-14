#coding:utf-8  

import sys
import os
import random
def generate_list(file_dir,all_png_list):
	count0 =os.popen(" ls "+file_dir+"*.0.*.png |wc -l " ).readlines()
	count1 =os.popen(" ls "+file_dir+"*.1.*.png |wc -l " ).readlines()
	count2 =os.popen(" ls "+file_dir+"*.2.*.png |wc -l " ).readlines()
	c0 = int(count0[0].strip('\n'))
	c1 = int(count1[0].strip('\n'))
	c2 = int(count2[0].strip('\n'))

	if not os.path.exists(all_png_list):
		f = open(all_png_list,'w')
		for file in os.listdir(file_dir):
			tmp = file.split('.')
			f.write(tmp[0]+'.'+tmp[1]+'\t'+tmp[1]+'\n')
		f.close()
	else:
		print("统计总数文件已存在，名为： "+all_png_list)
	return c0,c1,c2


def divide_data(all_txt,train_file,test_file,train_num_each_label,reset):
	
	if reset=='1' or ( not (os.path.exists(train_file) and os.path.exists(test_file) ) ):
		for i in [train_file,test_file]:
			if os.path.exists(i):
				os.remove(i)
		fa=open(train_file,'w')
		fb=open(test_file,'w')
		label_count =[0,0,0]
		all_line = open(all_txt,'r').readlines()
		random.shuffle(all_line)
		for i in all_line:
			label_now = int(i.strip().split('\t')[-1])
			if label_count[label_now] < train_num_each_label:
				fa.write(i)
				label_count[label_now]+=1
			else:
				fb.write(i)
		print ("division done---------------------------------------------------------------")
		fa.close()
		fb.close()
	elif reset=='0' and os.path.exists(train_file) and os.path.exists(test_file) :
		print("训练集、测试集的left,right 四个文件都已存在，不再做划分操作")

def main():
	
	img_path=sys.argv[1] 
	all_txt=sys.argv[2] 
	train_txt=sys.argv[3] 
	test_txt=sys.argv[4] 
	reset=sys.argv[5]
	c0,c1,c2 = generate_list(img_path,all_txt)
	print("三种图像的个数分别为：")
	print(c0,c1,c2)

	train_num_each_label = int(min(c0,c1,c2)*4/5)
	divide_data(all_txt,train_txt,test_txt,train_num_each_label,reset)
	
if __name__ == '__main__':
	main()
