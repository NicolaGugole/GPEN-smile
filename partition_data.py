import os

PATH = '/home/wizard/buckets/bsp-ai-science-public-datasets/stylegan-beautification-various-levels/input'

if __name__ == '__main__':
    num = 20000
    for j in [0, num, 2 * num]:
        for i,f in enumerate(os.listdir(PATH)[j:j+num]):
            if i < 18950:
                with open('train.txt', 'a') as outfile:
                    print(f'{i+j:05d}.jpg {j // num}', file=outfile)
            elif i < 19000:
                with open('val.txt', 'a') as outfile:
                    print(f'{i+j:05d}.jpg {j // num}', file=outfile)
            else:
                with open('test.txt', 'a') as outfile:
                    print(f'{i+j:05d}.jpg {j // num}', file=outfile)