from Neural import *
import random

def writeIdentical(filename):
    #writes a data file for input = output
    data = []
    for i in range(50000):
        if i % 500 == 0:
            print(i)
        inputMatrix = [random.random(),random.random(),random.random(),random.random()]
        data.append((inputMatrix,inputMatrix))
    for i in range(10000):
        if i % 500 == 0:
            print(i)
        inputMatrix = [random.randrange(0,2),random.randrange(0,2),random.randrange(0,2),random.randrange(0,2)]
        data.append((inputMatrix,inputMatrix))
    random.shuffle(data)
    open(filename,"w").writelines([":".join([str(y) for y in x]) + "\n" for x in data])

def getDataFromFile(filename):
    data = open(filename).readlines()
    data = [(eval(line.split(":")[0]),eval(line.split(":")[1])) for line in data]
    return data

n = Neural(4,4,2,8,[1,2,3,4])

data = getDataFromFile("data.txt")
print("Data loaded")

n.train(data,100,1000,0.5, 50)
print("\n===Test===\n\n")
print("[1,0,0,0]:",n.calc([1,0,0,0]))
print("[0,1,0,0]:",n.calc([0,1,0,0]))
print("[1,1,0,0]:",n.calc([1,1,0,0]))
print("[1,1,1,1]:",n.calc([1,1,1,1]))

print("\n\n====Weights=====\n\n")
print(n.getWeights())
print("\n\n====Biases=====\n\n")
print(n.getBiases())
