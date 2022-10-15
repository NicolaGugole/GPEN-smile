import os
import random
TRAIN = 'train_non_shuffled.txt'

if __name__ =='__main__':
    train_files = [r for r in open(TRAIN)]
    random.shuffle(train_files)
    print(train_files)
    with open('train.txt', 'a') as f:
        for s in train_files:
            print(s[:-1], file=f)