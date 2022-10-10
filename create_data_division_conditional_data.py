# this data subdivision is based on the data division explained in BASE_DIR/read_me.txt

import os
import random

BASE_DIR = '/home/wizard/buckets/bsp-ai-science-public-datasets/stylegan-beautification-various-levels'

if __name__ == '__main__':
    train_files = []
    val_files = []
    test_files = []
    possible_files = [f'{i:05d}.jpg' for i in range(0,20000)]
    for i, f in enumerate(possible_files):
        if i < 19450:
            train_files.append(f+' 0')
        elif i < 19950:
            val_files.append(f+' 0')
        else:
            test_files.append(f+' 0')
    possible_files = [f'{i:05d}.jpg' for i in range(20000,40000)]
    for i, f in enumerate(possible_files):
        if i < 19450:
            train_files.append(f+' 1')
        elif i < 19950:
            val_files.append(f+' 1')
        else:
            test_files.append(f+' 1')
    
    possible_files = [f'{i:05d}.jpg' for i in range(40000,60000)]
    for i, f in enumerate(possible_files):
        if i < 19450:
            train_files.append(f+' 2')
        elif i < 19950:
            val_files.append(f+' 2')
        else:
            test_files.append(f+' 2')
    random.shuffle(test_files)
    random.shuffle(val_files)
    random.shuffle(train_files)

    with open(os.path.join(BASE_DIR, 'train.txt'), 'w') as f:
        for i in train_files:
            print(i, file=f)
    with open(os.path.join(BASE_DIR, 'val.txt'), 'w') as f:
        for i in val_files:
            print(i, file=f)
    with open(os.path.join(BASE_DIR, 'test.txt'), 'w') as f:
        for i in test_files:
            print(i, file=f)