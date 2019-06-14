#coding:utf-8  

from div_data import divide_data
from div_data import generate_list
import pycuda.autoinit
import pycuda.driver as cuda
from timeit import default_timer as timer
from pycuda.compiler import SourceModule
from pycuda.driver import DeviceAllocation
from pycuda.driver import Stream
from pycuda.driver import Context
import pickle
import sys
import numpy as np
from PIL import Image
import time

def get_RGB(img_path,width,height):
	img=Image.open(img_path)
	# plt.plot(img)
	pixel=img.getpixel((width,height))
	return float((pixel[0]-128)/128),float((pixel[1]-128)/128),float((pixel[2]-128)/128)


def normalization(img_path):
	img=Image.open(img_path)
	img_array = np.array(img,dtype=np.float32)
	mod = SourceModule("""
		__global__ void normalization(float *img)
		{
		    int idx = blockIdx.x * blockDim.x + threadIdx.x;
		    img[idx*3] = (img[idx*3]-128.0)/128.0;
		    img[idx*3+1] = (img[idx*3+1]-128.0)/128.0;
		    img[idx*3+2] = (img[idx*3+2]-128.0)/128.0;

		    __syncthreads();
		}
		""")
	func = mod.get_function("normalization")
	func( cuda.InOut(img_array), block=( 100, 1, 1 ), grid=( 100,1 ) )
	# DeviceAllocation.free(img_array)
	# print(Context.get_api_version())
	# Stream.synchronize()
	return img_array


def loaddata():
	image_label_list = sys.argv[1]
	file_dir = sys.argv[2]
	print ("Begin normalizing the images-----")
	num = len( open(image_label_list).readlines() )
	print("Loading "+str(num)+" images...")
	X_data=np.zeros((num,100,100,3),dtype=np.float32)
	labels=[]
	sample=-1
	for  img_label in open(image_label_list): 
		sample=sample+1
		print (sample)
		if sample<num:
			img_label=img_label.strip('\n')
			img=img_label.split('\t')[0]
			label=int(img_label.split('\t')[1])
			# if label>1:
			# 	label=1
			img_path=file_dir+img
			labels.append(label)
			# GPU代码
			X_data[sample,:,:,:] = normalization(img_path)
			# CPU代码
			# for height in range(100):
			# 	for width in range(100):
			# 		X_data[sample,height,width,:]=get_RGB(img_path,width,height)

	print ("done----------X_train---X_label-----------------")


	xdata_file = open(sys.argv[3], 'wb')
	xlabel_file = open(sys.argv[4], 'wb')
	pickle.dump(X_data, xdata_file)
	pickle.dump(labels, xlabel_file)

def main():
	all_txt=""
	train_txt=""
	test_txt=""
	img_path=''

	if sys.argv[5]=='train':
		c0,c1,c2 = generate_list(img_path,all_txt)
		print("三种图像的个数分别为：")
		print(c0,c1,c2)

		train_num_each_label = int(min(c0,c1,c2)*4/5)
		divide_data(all_txt,train_txt,test_txt,train_num_each_label)
	
	t0=time.time()
	loaddata()
	t1=time.time()
	print ("loading data uses time: "+str(t1-t0))
if __name__ == '__main__':
	main()
