import netMath
import random

class Neural(object):
    def __init__(self, numInputs, numOutputs, numLayers, numNodes, outputMap):
        #numLayers is the number of *hidden* layers
        self.__numInputs = numInputs
        self.__numOutputs = numOutputs
        self.__numLayers = numLayers
        self.__numNodes = numNodes
        self.initNodes()
        self.randomiseWeights()
        self.randomiseBiases()
        assert(self.__numInputs == len(outputMap))
        self.__outputMap = outputMap

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
        self.__weights = [[]]*(self.__numLayers+2)
        #self.__weights is formatted as [right layer][rightNodeIndex][leftNodeIndex]
        #note that __weights[0] is empty
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

    def getNumNodesForLayer(self, layer):
        assert(layer>=0 and layer<=self.__numLayers+1)
        if layer == 0:
            return self.__numInputs
        elif layer == self.__numLayers+1:
            return self.__numOutputs
        else:
            return self.__numNodes

    def calc(self,inputMatrix):
        assert(len(inputMatrix)==self.__numInputs)
        #set input nodes
        self.__nodes[0] = inputMatrix

        #calculate activation for each node
        for layer in range(1,self.__numLayers+2):
            numNodes = self.getNumNodesForLayer(layer)
            for node in range(numNodes):
                self.__nodes[layer][node] = netMath.sigmoid(netMath.dot(self.__nodes[layer-1],self.__weights[layer][node])+self.__biases[layer][node])
        #return output matrix
        return self.__nodes[self.__numLayers+1]

    def predict(self,inputMatrix):
        currentMax = -float('inf')
        maxIndex = -1
        output = self.calc(inputMatrix)
        for i in range(self.__numOutputs):
            if output[i] > currentMax:
                maxIndex = i
                currentMax = output[i]
        assert(maxIndex != -1)
        return self.__outputMap[maxIndex]
        
    
if __name__ == "__main__":
    n = Neural(5,5,3,8,[x for x in range(5)])
    out = n.predict([0.4,0.1,0.3,0.8,1.0])
    print(out)
                
