# Daniil Koshelyuk exercise on AI: Decision trees led by WelchLabs.

################################## Imports #####################################

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from NeuralNetwork import *

################################ Display #######################################
# Display estimations over entire data space.
def displayDataSpace(neuralNetwork):
    hoursStudy = np.linspace(0, hoursMaxStudy, 100)
    hoursSleep = np.linspace(0, hoursMaxSleep, 100)
    hoursStudyNorm = hoursStudy / hoursMaxStudy
    hoursSleepNorm = hoursSleep / hoursMaxSleep
    a, b = np.meshgrid(hoursStudyNorm, hoursSleepNorm)
    inputs = np.zeros((a.size, 2))
    inputs[:, 0] = a.ravel()
    inputs[:, 1] = b.ravel()
    outputs = neuralNetwork.forward(inputs)
    xSpace = np.dot(hoursStudy.reshape(100,1), np.ones((1, 100))).T
    ySpace = np.dot(hoursSleep.reshape(100,1), np.ones((1, 100)))
    # 2D contour Plot
    contours = plt.contour(xSpace, ySpace, scoreMax*outputs.reshape(100, 100))
    plt.clabel(contours, inline=1, fonsize=5)
    plt.xlabel('Hours Study')
    plt.ylabel('Hours Sleep')
    plt.show()
    # 3D Data-Space Plot
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(hoursMaxStudy*examples[:,0], hoursMaxSleep*examples[:,1], scoreMax*answers, c='k', alpha=1, s=30)
    surf = ax.plot_surface(xSpace, ySpace, scoreMax*outputs.reshape(100,100), cmap=cm.jet, alpha=.5)
    ax.set_xlabel('Hours Study')
    ax.set_ylabel('Hours Sleep')
    ax.set_zlabel('Test Score')
    plt.show()
# Display input data.
def displayData(examples, answers):
    figure = plt.figure(0, (8, 3))

    plt.subplot(1,2,1)
    plt.scatter(examples[:,0], answers)
    plt.grid(True)
    plt.xlabel('Hours Studying')
    plt.ylabel('Test Score')

    plt.subplot(1,2,2)
    plt.scatter(examples[:,1], answers)
    plt.grid(True)
    plt.xlabel('Hours Sleeping')
    plt.ylabel('Test Score')

    plt.show()

################################################################################
################################ Active Zone ###################################
################################################################################

# Constants
epsilon          = 1e-4

# Inputs
# examples    = np.array(([3,5], [5,1], [10,2]), dtype=float)
# answers     = np.array(([75], [82], [93]), dtype=float)
# Normalization
hoursMaxStudy    = 16
hoursMaxSleep    = 8
scoreMax         = 100
# examples    /= np.amax(examples, axis=0)
# for example in examples:
#     example[0] /= hoursMaxStudy
#     example[1] /= hoursMaxSleep
# answers     /= scoreMax
# Network initiation
neuralNetwork = neuralNetwork(inputLayerSize=2, outputLayerSize=1, hiddenLayerSizes=[3])

# Test1: Random Estimations.
# estimations = neuralNetwork.forward(examples)
# print("Estimations:\n", estimations)
# print("Answers:\n", answers)

# Test 2: Costs.
# cost0 = neuralNetwork.cost(examples, answers, 1)
# cost1 = neuralNetwork.cost(examples, answers)
# print("Cost Simple:\n", cost0)
# print("Cost with squares:\n", cost1)

# Test 3: Derivatives.
# weightDirivatives = neuralNetwork.costDerivatives(examples, answers)
# print("Weights' Dirivatives:")
# j = 0
# for weightDerivative in weightDirivatives:
#     print("\tLayer :" + str(j) + "\n", weightDerivative)
#     j += 1
# scalar = 3
# for i in range(0, len(neuralNetwork.layerSizes)-1):
#     neuralNetwork.weights[i] -= scalar * weightDirivatives[i]
# cost2 = neuralNetwork.cost(examples, answers)
# for i in range(0, len(neuralNetwork.layerSizes)-1):
#     neuralNetwork.weights[i] += 2 * scalar * weightDirivatives[i]
# cost3 = neuralNetwork.cost(examples, answers)
# print("Cost no corrections:\n", cost1)
# print("Cost - correction:\n", cost2)
# print("Cost + correction:\n", cost3)

# Test 4: Numerical gradient.
# gradientVector          = neuralNetwork.gradientVector(examples, answers)
# numericalGradientVector = neuralNetwork.numericalGradientVector(examples, answers)
# print("Gradient Vector:\n", gradientVector)
# print("Numerical Gradient Vector:\n", numericalGradientVector)
# print("Norm:\n", neuralNetwork.weightCorrectionEfectiveness(examples, answers))

# Test 5: Training.
trainer = neuralNetworkTrainer(neuralNetwork)
# trainer.train(examples, answers)
# trainer.plotLearning()

# estimations = neuralNetwork.forward(examples)
# print("Estimations:\n", estimations)
# print("Answers:\n", answers)

# test = np.array(([9,5]), dtype=float)
# print("Test Data:\n", test)
# estimations = neuralNetwork.forward(test)
# print("Test Estimations:\n", estimations)
# displayDataSpace(neuralNetwork)

# Test 6: Overfitting
examples     = np.array(([3,5], [5,1], [10,2], [6,1.5]), dtype=float)
answers      = np.array(([75], [82], [93], [70]), dtype=float)
for example in examples:
    example[0] /= hoursMaxStudy
    example[1] /= hoursMaxSleep
answers      /= scoreMax
# displayData(examples, answers)
testExamples = np.array(([4,5.5], [4.5,1], [9,2.5], [6,2]), dtype=float)
testAnswers  = np.array(([70], [89], [85], [75]), dtype=float)
for example in testExamples:
    example[0] /= hoursMaxStudy
    example[1] /= hoursMaxSleep
testAnswers  /= scoreMax

# 1. Exadurated example.
# trainer.train(examples, answers)
# trainer.plotLearning()
# displayDataSpace(neuralNetwork)

# 2. Show the point of diversion.
# trainer.train(examples, answers, testExamples, testAnswers)
# trainer.plotLearning()

# 3. Check if the gradients are correct.
# gradientVector          = neuralNetwork.gradientVector(examples, answers)
# numericalGradientVector = neuralNetwork.numericalGradientVector(examples, answers)
# print("Gradient Vector:\n", gradientVector)
# print("Numerical Gradient Vector:\n", numericalGradientVector)
# print("Norm:\n", neuralNetwork.weightCorrectionEfectiveness(examples, answers))

# 4. Train the network.
trainer.train(examples, answers, testExamples, testAnswers)
trainer.plotLearning()
displayDataSpace(neuralNetwork)
