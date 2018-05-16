import math

def sigmoid(x):
    return 1/(1+(math.e**(-x)))

#takes two 1D arrays a and b and returns the dot product
def dot(a,b):
    assert(len(a)==len(b))
    return sum([a[x]*b[x] for x in range(len(a))])

#takes two 2D matricies and returns sum
def add2D(a,b):
    assert(len(a)==len(b))
    #needs more asserts
    return [[a[x][y]+b[x][y] for y in range(len(a[x]))] for x in range(len(a))]

def add3D(a,b):
    assert(len(a)==len(b))
    #needs more asserts
    return [[[a[x][y][z]+b[x][y][z] for z in range(len(a[x][y]))] for y in range(len(a[x]))] for x in range(len(a))]

if __name__ == "__main__":
    #unit testing / misc
    a = [4,6,3,1]
    b = [2,1,4,5]
    assert(dot(a,b) == 8+6+12+5)
    assert(dot([1,0],[0,1])==0)

    a = [[1,2,3],[4,5,6]]
    b = [[10,11,12],[6,5,3]]
    assert(add2D(a,b) == [[11,13,15],[10,10,9]])

    a = [[[1,2,3],[3,4]],[[10,9],[8,7]]]
    b = [[[5,3,1],[9,0]],[[-5,2],[4,-5]]]
    c = [[[6,5,4],[12,4]],[[5,11],[12,2]]]
    assert(add3D(a,b)==c)
