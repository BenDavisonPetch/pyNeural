import netMath
import random

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
        #self.__weights is formatted as [right layer][rightNodeIndex][leftNodeIndex]
        for layer in range(1,self.__numLayers+2):
            numLeftNodes = self.__numInputs if layer == 1 else self.__numNodes
            numRightNodes = self.__numOutputs if layer == self.__numLayers+1 else self.__numNodes
            self.__weights[layer] = [[[]]*numLeftNodes]*numRightNodes
            for r in range(numRightNodes):
                for l in range(numLeftNodes):
                    self.__weights[layer][r][l] = random.randrange(-5,5)

    def randomiseBiases(self):
        self.__biases = [[]]*(self.__numLayers+2)
        #self.__biases is formatted [layer][nodeIndex]
        for layer in range(self.__numLayers+2):
            numNodes = None
            if layer == 0:
                numNodes = self.__numInputs
            elif layer == self.__numLayers+1:
                numNodes = self.__numOutputs
            else:
                numNodes = self.__numNodes
            self.__biases[layer] = [random.randrange(-5,5) for i in range(numNodes)]

    def getNumNodesForLayer(layer):
        if layer == 0:
            return self.__numInputs
        elif layer == self.__numLayers+1:
            return self.__numOutputs
        else:
            return self.__numNodes

    def calc(self,inputMatrix):
        assert(len(inputMatrix)==self.__numInputs)
        
    
if __name__ == "__main__":
    n = Neural(2,12,3,8)
                
