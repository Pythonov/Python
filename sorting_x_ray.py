# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 15:16:54 2021

@author: Pimonov EI
"""

import os, shutil

original_dataset_dir = 'C:\code\svertok'
base_dir = 'C:\code\svertok\cats_and_dogs_small'
os.mkdir(base_dir) 

train_dir = os.path.join(base_dir, 'train')
os.mkdir(train_dir)
validation_dir = os.path.join(base_dir, 'validation')
os.mkdir(validation_dir)
test_dir = os.path.join(base_dir, 'test')
os.mkdir(test_dir)

train_cats_dir = os.path.join(train_dir,'cats')
os.mkdir(train_cats_dir)

train_dogs_dir = os.path.join(train_dir, 'dogs')
os.mkdir(train_dogs_dir)

validation_cats_dir = os.path.join(validation_dir, 'cats')
os.mkdir(validation_cats_dir)

validation_dogs_dir = os.path.join(validation_dir, 'dogs')
os.mkdir(validation_dogs_dir)

test_cats_dir = os.path.join(test_dir, 'cats')
os.mkdir(test_cats_dir)

test_dogs_dir = os.path.join(test_dir, 'dogs')
os.mkdir(test_dogs_dir)

   
for j in range(1,800):
    k = '0'*(5 - len(str(j))) + str(j)
    fnames = ['dogs_{}.jpg'.format(k)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(train_dogs_dir, fname)
        shutil.copyfile(src,dst)
    
    fnames = ['cats_{}.jpg'.format(k)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(train_cats_dir, fname)
        shutil.copyfile(src,dst)
        

for j in range(800,900):
    k = '0'*(5 - len(str(j))) + str(j)
    fnames = ['dogs_{}.jpg'.format(k)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(validation_dogs_dir, fname)
        shutil.copyfile(src,dst)
    
    fnames = ['cats_{}.jpg'.format(k)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(validation_cats_dir, fname)
        shutil.copyfile(src,dst)
 
        
for j in range(900,1000):
    k = '0'*(5 - len(str(j))) + str(j)
    fnames = ['dogs_{}.jpg'.format(k)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(test_dogs_dir, fname)
        shutil.copyfile(src,dst)
    
    fnames = ['cats_{}.jpg'.format(k)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(test_cats_dir, fname)
        shutil.copyfile(src,dst)






