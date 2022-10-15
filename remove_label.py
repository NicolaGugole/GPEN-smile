TRAIN = 'train_0.txt'
VAL = 'val_0.txt'
TEST = 'test_0.txt'

if __name__ == '__main__':
    with open('train_un.txt', 'a') as f:
        files = [r[:-3] for r in open(TRAIN)]
        for fi in files:
            print(fi, file=f)
    
    with open('val_un.txt', 'a') as f:
        files = [r[:-3] for r in open(VAL)]
        for fi in files:
            print(fi, file=f)
    
    with open('test_un.txt', 'a') as f:
        files = [r[:-3] for r in open(TEST)]
        for fi in files:
            print(fi, file=f)