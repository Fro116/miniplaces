import numpy as np

vals = np.zeros([10000, 100])
    
def init(filename):
    with open(filename, "r") as f:
        row = 0
        col = 0    
        for line in f:
            for val in line.split():
                vals[row, col] = val
                col += 1
                if col == 100:
                    col = 0
                    row += 1

def accuracy():
    correct = np.zeros([10000], dtype = np.int)
    guesses = np.zeros([10000,100], dtype = np.int)
    paths = []
    with open("../../data/val.txt", "r") as f:
        i = 0
        for line in f:
            path, lab =line.rstrip().split(' ')
            correct[i] =  int(lab)
            paths.append(path)
            i += 1
    for row in range(10000):
        k = 5
        ind = np.argpartition(vals[row,:], -k)[-k:]
        guesses[row, ind] = 1
    
    missed = np.zeros(100)
    acc = 0    
    for row in range(10000):
        if guesses[row, correct[row]] == 1:
            acc += 1
        else:
            print(paths[row] + " " + str(correct[row]));
            missed[correct[row]] += 1
    print(sorted(missed))
    print(acc)

def process():
    paths = []
    with open("../../data/test.txt", "r") as f:
        for line in f:
            path, lab =line.rstrip().split(' ')
            paths.append(path)
    for row in range(10000):
        k = 10
        ind = np.argpartition(vals[row,:], -k)[-k:]
        line = paths[row]
        for i in range(k):
            line = line + " " + str(ind[k-1-i])
        print(line)


init("tmp4")
accuracy()
#process()