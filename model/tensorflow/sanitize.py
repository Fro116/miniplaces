import numpy as np

vals = np.zeros([10000, 100])
    
def init(filename, scale):
    with open(filename, "r") as f:
        row = 0
        col = 0    
        for line in f:
            for val in line.split():
                vals[row, col] += float(val)*scale
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
            #print(paths[row] + " " + str(correct[row]));
            missed[correct[row]] += 1
    #print(sorted(missed))
    return acc

def process():
    paths = []
    with open("../../data/test.txt", "r") as f:
        for line in f:
            path, lab =line.rstrip().split(' ')
            paths.append(path)
    for row in range(10000):
        k = 5
        ind = np.argpartition(vals[row,:], -k)[-k:]
        ind = sorted(ind, key = lambda x: vals[row, x], reverse = True)
        line = paths[row]
        for i in range(k):
            line = line + " " + str(ind[i])
        print(line)

p = 0.2     
init("tmp2", 1)
init("tmp4", 1)
#init("tmp6", p)
#init("tmp8", p)        
print(accuracy())
#process()
