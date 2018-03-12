# Daniil Koshelyuk exercise on AI: Neural Network led by WelchLabs.

################################## Imports #####################################

import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

############################# General Functions ################################

def sigmoid(value):
    return 1 / (1 + np.exp(-value))
def sigmoidPrime(value):
    return np.exp(-value)/((1 + np.exp(-value))**2)

###################### NeuralNetwork Class Structures ##########################

# Neural Network Object.
class neuralNetwork():
    ################### Initiation ###################
    # Define network.
    def __init__(self, inputLayerSize=2, outputLayerSize=1, hiddenLayerSizes=[3], regularizationFactor=None):
        if (not(isinstance(hiddenLayerSizes, list))):
            hiddenLayerSizes = [hiddenLayerSizes]
        if (regularizationFactor == None):
            regularizationFactor = 1e-4
        self.regularizationFactor = regularizationFactor
        self.layerSizes = []
        self.layerSizes.append(inputLayerSize)
        self.layerSizes.extend(hiddenLayerSizes)
        self.layerSizes.append(outputLayerSize)
        # Potentially self-decision to add more layers or increase Sizes of hidden layers.

        # Declare initial random weights.
        self.weights = []
        for i in range(0, len(self.layerSizes) - 1):
            weightSizeStart = self.layerSizes[i]
            weightSizeEnd = self.layerSizes[i+1]
            self.weights.append(np.random.randn(weightSizeStart, weightSizeEnd))

    ############### Forward Propagation ##############
    # Popogate inputs through network.
    def forward(self, value):
        self.summationResults = []
        self.activatedResults = []
        self.activatedResults.append(value)
        for i in range(0, len(self.layerSizes) - 1):
            self.summationResults.append(np.dot(self.activatedResults[i], self.weights[i]))
            self.activatedResults.append(self.activation(self.summationResults[i]))
        return self.activatedResults[-1]
    # Activate results of summations.
    def activation(self, value, activationType=0):
        if (activationType == 0): # Sigmoid Activation Version.
            return sigmoid(value)
        else: # Just in case.
            return value

    ############## Backward Propagation ##############
    # Cost function to assess the performance.
    def cost(self, examples, answers, costEvaluationType=0):
        if (costEvaluationType == 0): # half of the square of the difference regularized.
            self.estimations = self.forward(examples)
            weights = 0
            for weight in self.weights:
                weights += np.sum(weight**2)
            cost = 0.5*sum((answers - self.estimations)**2) / examples.shape[0] + (self.regularizationFactor/2) * weights
        elif (costEvaluationType == 1): # half of the square of the difference.
            self.estimations = self.forward(examples)
            cost = 0.5*sum((answers - self.estimations)**2)
        else: # Just in case.
            self.estimations = self.forward(examples)
            cost = sum(answers - self.estimations)
        return cost
    # Derivative of cost function with respect to weights.
    def costPrime(self, examples, answers, activationType=0):
        delta = []
        exampleCosts = []
        weightDirivatives = []
        self.estimations = self.forward(examples)
        for i in reversed(range(1, len(self.layerSizes))):
            if (activationType == 0): # Sigmoid Activation Version.
                activationDelta = sigmoidPrime(self.summationResults[i-1])
            else: # Just in case.
                activationDelta = self.summationResults[i-1]
            if (i == len(self.layerSizes) - 1):
                preactivationDelta = self.estimations - answers
                delta.insert(0, np.multiply(preactivationDelta, activationDelta))
            else:
                preactivationDelta = np.dot(delta[0], self.weights[i].T)
                delta.insert(0, preactivationDelta * activationDelta)
            exampleCosts.insert(0, self.activatedResults[i-1].T)
            weightDirivatives.insert(0, np.dot(exampleCosts[0], delta[0]))
        return weightDirivatives
    # Derivative of cost function with respect to weights.
    def costPrimeRegularized(self, examples, answers, activationType=0):
        delta = []
        exampleCosts = []
        weightDirivatives = []
        self.estimations = self.forward(examples)
        for i in reversed(range(1, len(self.layerSizes))):
            if (activationType == 0): # Sigmoid Activation Version.
                activationDelta = sigmoidPrime(self.summationResults[i-1])
            else: # Just in case.
                activationDelta = self.summationResults[i-1]
            if (i == len(self.layerSizes) - 1):
                preactivationDelta = self.estimations - answers
            else:
                preactivationDelta = np.dot(delta[0], self.weights[i].T)
            delta.insert(0, np.multiply(preactivationDelta, activationDelta))
            exampleCosts.insert(0, self.activatedResults[i-1].T)
            weightDirivatives.insert(0, np.dot(exampleCosts[0], delta[0]) / examples.shape[0] + self.regularizationFactor * self.weights[i-1])
        return weightDirivatives
    # Derivative of cost function with respect to weights.
    def costDerivatives(self, examples, answers, costEvaluationType=0, activationType=0):
        if (costEvaluationType == 0): # Batch Gradient Descent Regularized.
            return self.costPrimeRegularized(examples, answers, activationType)
        elif (costEvaluationType == 1): # Batch Gradient Descent.
            return self.costPrime(examples, answers, activationType)
        else: # Just in case.
            return

    ################# Vactor Shaping #################
    # Reshape Weights to form a vector.
    def getWeightVector(self):
        vector = []
        for weight in self.weights:
            vector.extend(weight.ravel())
        return np.array(vector)
    # Set weights using vector shaped data.
    def setWeightVector(self, vector):
        start = 0
        for i in range(1, len(self.layerSizes)):
            end = start + self.layerSizes[i-1] * self.layerSizes[i]
            self.weights[i-1] = np.reshape(vector[start:end], (self.layerSizes[i-1], self.layerSizes[i]))
            start = end
    # Calculate derivatives and reshape to match Vector notation.
    def gradientVector(self, examples, answers, costEvaluationType=0, activationType=0):
        derivatives = self.costDerivatives(examples, answers, costEvaluationType, activationType)
        vector = []
        for derivativeLayer in derivatives:
            vector.extend(derivativeLayer.ravel())
        return np.array(vector)

    ################# Error Checking #################
    # Compute numerical gradient of multidimensional cost function.
    def numericalGradientVector(self, examples, answers, costEvaluationType=0):
        initialWeightVector = self.getWeightVector()
        numericalGradientVector = np.zeros(initialWeightVector.shape)
        peturbation = np.zeros(initialWeightVector.shape)
        # For each of the weights calculate numerical gradient of minimal change.
        for i in range(0, len(initialWeightVector)):
            peturbation[i] = epsilon
            self.setWeightVector(initialWeightVector + peturbation)
            loss1 = self.cost(examples, answers, costEvaluationType)
            self.setWeightVector(initialWeightVector - peturbation)
            loss2 = self.cost(examples, answers, costEvaluationType)
            # Calulate specific weight gradient and restore peturbanction vector.
            numericalGradientVector[i] = (loss1 - loss2) / (2 * epsilon)
            peturbation[i] = 0
        # Restore state
        self.setWeightVector(initialWeightVector)
        return numericalGradientVector
    # Compute and compare cost function prime corrections and numerical gradient.
    # Typical results should be on the order of 10^-8
    def weightCorrectionEfectiveness(self, examples, answers, costEvaluationType=0, activationType=0):
        gradientVector          = self.gradientVector(examples, answers, costEvaluationType, activationType)
        numericalGradientVector = self.numericalGradientVector(examples, answers, costEvaluationType)
        return np.linalg.norm(gradientVector - numericalGradientVector) / np.linalg.norm(gradientVector + numericalGradientVector)

# Neural Network Trainer Object.
class neuralNetworkTrainer():
    ################### Initiation ###################
    # Define network.
    def __init__(self, neuralNetwork, ceiling=200, comments=True):
        self.neuralNetwork = neuralNetwork
        self.ceiling = ceiling
        self.comments = comments

    ############### Training Functions ###############
    # Call back method.
    def callbackFunction(self, vector):
        self.neuralNetwork.setWeightVector(vector)
        self.costs.append(self.neuralNetwork.cost(self.examples, self.answers))
        if (self.testData):
            self.testCosts.append(self.neuralNetwork.cost(self.testExamples, self.testAnswers))

    # Prepare function for training method.
    def costWrapper(self, vector, examples, answers):
        self.neuralNetwork.setWeightVector(vector)
        cost        = self.neuralNetwork.cost(examples, answers)
        gradient    = self.neuralNetwork.gradientVector(examples, answers)
        return cost, gradient

    # Actually train the network.
    def train(self, examples, answers, testExamples=None, testAnswers=None):
        if (isinstance(testExamples, np.ndarray) & isinstance(testAnswers, np.ndarray)):
            self.testData = True
        else:
            self.testData = False
        self.examples     = examples
        self.answers      = answers
        self.testExamples = testExamples
        self.testAnswers  = testAnswers
        self.costs        = []
        self.testCosts    = []
        initialWeightVector = self.neuralNetwork.getWeightVector()
        options = {'maxiter': self.ceiling, 'disp': self.comments}
        _res = optimize.minimize(self.costWrapper, initialWeightVector,
                                jac=True, method='BFGS', options=options,
                                args=(self.examples, self.answers),
                                callback=self.callbackFunction)
        self.neuralNetwork.setWeightVector(_res.x)
        self.optimizationResults = _res

    # Plot Learning Curve.
    def plotLearning(self):
        plt.xlabel("Iterations")
        plt.ylabel("Cost")
        plt.title("Cost through iterations")
        plt.grid(True)
        plt.plot(self.costs)
        if (self.testData):
            plt.plot(self.testCosts)
        plt.show()
