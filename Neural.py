import netMath
import random
#import numpy

class Neural(object):
    def __init__(self, numInputs, numOutputs, numLayers, numNodes):
        #numLayers is the number of *hidden* layers
        self.__numInputs = numInputs
        self.__numOutputs = numOutputs
        self.__numLayers = numLayers
        self.__numNodes = numNodes
        self.initNodes()
        self.randomiseWeights()
        self.randomiseBiases()

    def initNodes(self):
        self.__nodes = []
        #self.__nodes is structured as [layer][number (from top if visualising)]
        for layer in range(self.__numLayers+2):
            self.__nodes.append([])
            numNodes = None
            if layer == 0:
                numNodes = self.__numInputs
            elif layer == self.__numLayers+1:
                numNodes = self.__numOutputs
            else:
                numNodes = self.__numNodes
            self.__nodes[layer] = [0]*numNodes
                

    def randomiseWeights(self):
        self.__weights = [[]]*(self.__numLayers+1)
        #self.__weights is formatted as [left layer][leftNodeIndex][rightNodeIndex]
        for layer in range(self.__numLayers+1):
            numLeftNodes = self.__numInputs if layer == 0 else self.__numNodes
            numRightNodes = self.__numOutputs if layer == self.__numLayers else self.__numNodes
            self.__weights[layer] = [[[]]*numRightNodes]*numLeftNodes
            for l in range(numLeftNodes):
                for r in range(numRightNodes):
                    self.__weights[layer][l][r] = random.randrange(-5,5)

    def randomiseBiases(self):
        self.__biases = [[]]*(self.__numLayers+2)
        for layer in range(self.__numLayers+2):
            numNodes = None
            if layer == 0:
                numNodes = self.__numInputs
            elif layer == self.__numLayers+1:
                numNodes = self.__numOutputs
            else:
                numNodes = self.__numNodes
            self.__biases[layer] = [random.randrange(-5,5) for i in range(numNodes)]
        
if __name__ == "__main__":
    n = Neural(2,12,3,8)
                
