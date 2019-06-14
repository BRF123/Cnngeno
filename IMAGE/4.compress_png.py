#coding:utf-8

from PIL import Image
import numpy as np
from numpy import *  
import os
import sys
from timeit import default_timer as timer

def pooling(target_size, path, img_name):
    
    img=Image.open(path+img_name)
    pixel_array = np.array(img)
    
    prior_wid, prior_hei = img.size[0],img.size[1]
    width,height = target_size[0], target_size[1]
    
    window = int(prior_wid/width)
    newim = Image.new("RGB", (width, height), (255, 255, 255))

    for h in range(prior_hei):
        count=0
        new_colomn = -1
        box = []
        for w in range(prior_wid):
            count+=1
            if count == window:
                new_colomn += 1
                
                count_white=0
                for i in box:
                    if sum(i) < sum(newim.getpixel((new_colomn, h))):
                        newim.putpixel((new_colomn, h), (i[0], i[1], i[2]))
                    if i[0]==255 and i[1]==255 and i[2]==255:
                        count_white+=1
                if count_white > window/2:
                    newim.putpixel((new_colomn, h), (255, 255, 255))

                count = 0
                box = []
            elif count < window:
                box.append(pixel_array[h][w])

    newim.save(''+img_name.strip('.png')+'.cpu.png', "PNG")

def main():
    img_path = sys.argv[1]
    target_size = (,)
    start = timer()
    for file in os.listdir(img_path):
        if '.png' in file:
            pooling(target_size, img_path, file)
    print("Finally, cpu run time %f seconds " % (timer() - start ))

if __name__ == '__main__':
    main()
