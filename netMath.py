import math

def sigmoid(x):
    return 1/(1+(math.e**(-x)))

#takes two 1D arrays a and b and returns the dot product
def dot(a,b):
    assert(len(a)==len(b))
    return sum([a[x]*b[x] for x in range(len(a))])

if __name__ == "__main__":
    #unit testing / misc
    a = [4,6,3,1]
    b = [2,1,4,5]
    assert(dot(a,b) == 8+6+12+5)
    assert(dot([1,0],[0,1])==0)
