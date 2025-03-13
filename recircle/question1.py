import numpy as np
import matplotlib.pyplot as plt


""" We will use (i) Linear regression for classification followed
by pocket algorithm.

Prompt: Use your classification algorithm of choice to find the best
separator using the training data only. For each example, use the two
features you computed in HW2 to as the inputs; and the output is +1
if the handwritten digit is 1 and -1 if the handwritten digit is 5.
Once you have found a separator using your classification algorithm:
"""
def load_data(file_path):
    trainingCSV = open(file_path, "r")
    trainingCSVLines = trainingCSV.readlines()

    onesAndFives = []
    for line in trainingCSVLines:
        if (int(line[0]) == 1 or int(line[0]) == 5):
            onesAndFives.append([float(value.strip()) for value in line.split(",")[:-1]])

    # print(onesAndFives)
    label1 = onesAndFives[0][0]
    img1 = onesAndFives[0][1:]
    img1 = np.array(img1).reshape(16, 16)

    label2 = onesAndFives[1][0]
    img2 = onesAndFives[1][1:]
    img2 = np.array(img2).reshape(16, 16)

    point_matrix = []

    for value in onesAndFives:
        data = np.array(np.zeros(3))
        data[0] = value[0]
        data[1] = np.mean(value[1:257])
        data[2] = np.mean(np.array(value[1:257] * np.array(value[1:257])))
        point_matrix.append(data)
    return np.array(point_matrix)

trainingCSV = load_data("/Users/jhulendrabhattarai/Desktop/ZipDigits.train.csv")
testingCSV = load_data("/Users/jhulendrabhattarai/Desktop/ZipDigits.test.csv")
#print(trainingCSV)

# Convert labels to -1 and 1
trainingCSV[:, 0] = [1 if x == 1 else -1 for x in trainingCSV[:, 0]]
testingCSV[:, 0] = [1 if x == 1 else -1 for x in testingCSV[:, 0]]


# We will use same functions that were implemented in HW2:
def PLA(x_array, yValues, weights=None, maxIterations = 10):
    # initialize all weights to 0, weights is a matrix of dimension 3
    if weights is None:
        weights = np.zeros(x_array.shape[1])
    # Run until the weights no longer updated (converge)
    for _ in range(maxIterations):
        errors = 0
        for index in range(x_array.shape[0]):
            outputFromWeights = np.dot(weights.T, x_array[index])
            predictedYValue = 1 if outputFromWeights >= 0 else -1
            if (predictedYValue != yValues[index]):
                errors += 1
                weights += yValues[index] * x_array[index]
        if (errors == 0):
            break
    return weights / weights[-1]


def linearRegression(xValues, yValues):
    xPseudoInverse = np.matmul(np.linalg.pinv(np.matmul(xValues.T, xValues)), xValues.T)
    weights = np.matmul(xPseudoInverse, yValues)
    return weights


def mean_square_error(predictions, true_values):  # E_in and E_out
    return 1 / len(predictions) * np.sum((predictions - true_values) ** 2)


#############################################################################################################################

# Pocket Algorithm
w_avg = linearRegression(trainingCSV[:, 1:], trainingCSV[:, 0])  # Initialize w_avg after running linear regression
w_t = np.copy(w_avg)  # set for w_0
for t in range(1):  # max iteration-random large number #trainingCSV.shape[0]

    # run PLA for each w(t+1) update
    w_t1 = PLA(trainingCSV[:, 1:], trainingCSV[:, 0], w_t)

    # Get E_in for each w(t+1) vs w_avg
    w_t1_predict = np.matmul(trainingCSV[:, 1:], w_t1)
    w_t1_err = mean_square_error(w_t1_predict, trainingCSV[:, 0])  # mean square error between predictions and true values

    w_avg_predict = np.matmul(trainingCSV[:, 1:], w_avg)
    w_avg_err = mean_square_error(w_avg_predict, trainingCSV[:, 0])

    # If E_in of w(t+1) lower than that of w_avg, replace w_avg
    if w_t1_err < w_avg_err:
        w_avg = np.copy(w_t1)
    w_t = w_t1

"""(a) Give separate plot of the training data (ZipDigits.train) and test data
(ZipDigits.test) which display the data points using two features you computed in
HW2, together with the separator
"""
train_predict = np.matmul(trainingCSV[:, 1:], w_avg)
train_predict = [1 if v > 0 else -1 for v in train_predict]
test_predict = np.matmul(testingCSV[:, 1:], w_avg)
test_predict = [1 if v > 0 else -1 for v in test_predict]
# Train_set plot
for i in range(len(trainingCSV)):
    items = trainingCSV[i]
    if (train_predict[i] > 0):
        plt.scatter(items[2], items[1], color='red', marker="x", label='+1')
    else:
        plt.scatter(items[2], items[1], color='blue', marker="o", label='-1')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())
pltxs = np.linspace(0.6, 1.0, len(train_predict))
pltys = - ( w_avg[1] * pltxs) / w_avg[0]
plt.plot(pltxs, pltys, '-', color='green', label='separator')
plt.xlabel("Squared Mean")
plt.ylabel("Mean")
plt.show()


# Test_set plot
for i in range(len(testingCSV)):
    items = testingCSV[i]
    if (test_predict[i] > 0):
        plt.scatter(items[2], items[1], color='red', marker="x", label='+1')
    else:
        plt.scatter(items[2], items[1], color='blue', marker="o", label='-1')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())
pltxs = np.linspace(0.6, 1.0, len(test_predict))
pltys = - ( w_avg[1] * pltxs) / w_avg[0]
plt.plot(pltxs, pltys, '-', color='green', label='separator')
plt.xlabel("Squared Mean")
plt.ylabel("Mean")
plt.show()



""" (b) Compute E_in on your training data (ZipDigits.train) and E_test, the error
of your separator on the test data (ZipDigits.test).
"""
E_in = mean_square_error(train_predict, trainingCSV[:, 0])  # in-sample error (train-set error)
E_test = mean_square_error(test_predict, testingCSV[:, 0])

print("E_in of train data: ", E_in)
print("E_test of test data: ", E_test)

"""(c) Obtain a bound on the true out-of-sample error (E_out). You should get
two bounds, one based on E_in and another based on E_test. Use a tolerance of
theta = 0.05. Which is the better bound?
"""
E_out_lower = E_test
theta = 0.05
N = len(trainingCSV)
M = len(testingCSV)
error_bar = np.sqrt(0.05 / N * np.log(2 * M / theta))
E_out_upper = E_in + error_bar
print("Bounds of E_out: [{},{}]".format(E_out_lower, E_out_upper))
# the upper bound is better because we don't know the entire (infinite) data to interpolate, so we want to keep the bound higher

"""(d) Repeat parts (a)-(c) using a 3-rd order polynomial transform

Answer:     First we need to transform both train and test set
            Original a, b
            Each set has a 3-deg polynomial features of 1, a, b, a^2, b^2, a^3, b^3, a*b
"""
# a1 + a2*x + a3*y + a4*x^2 + a5*y^2 + a6*x^3 + a7*y^3 + a8*x*y
transformed_train = np.zeros(
    (trainingCSV.shape[0], 8))  # empty array to store new 3rd-deg trainset; each column is a new feature
transformed_train[:, 0] = 1. #intercept a1
transformed_train[:, 1] = trainingCSV[:, 1] #a2
transformed_train[:, 2] = trainingCSV[:, 2] #a3
transformed_train[:, 3] = trainingCSV[:, 1] ** 2
transformed_train[:, 4] = trainingCSV[:, 2] ** 2
transformed_train[:, 5] = trainingCSV[:, 1] ** 3
transformed_train[:, 6] = trainingCSV[:, 2] ** 3
transformed_train[:, 7] = trainingCSV[:, 1] * trainingCSV[:, 2] #a8

transformed_test = np.zeros((testingCSV.shape[0], 8))
transformed_test[:, 0] = 1.
transformed_test[:, 1] = testingCSV[:, 1]
transformed_test[:, 2] = testingCSV[:, 2]
transformed_test[:, 3] = testingCSV[:, 1] ** 2
transformed_test[:, 4] = testingCSV[:, 2] ** 2
transformed_test[:, 5] = testingCSV[:, 1] ** 3
transformed_test[:, 6] = testingCSV[:, 2] ** 3
transformed_test[:, 7] = testingCSV[:, 1] * testingCSV[:, 2]

# Run Pocket Algorithm for transformed train set: use transformed features, keep labels as the same
w_avg = linearRegression(transformed_train[:, :], trainingCSV[:, 0])
w_t = w_avg
for t in range(1): #transformed_train.shape[0]
    w_t1 = PLA(transformed_train[:, :], trainingCSV[:, 0], w_t)

    w_t1_predict = np.matmul(transformed_train[:, :], w_t1)
    w_t1_err = mean_square_error(w_t1_predict, trainingCSV[:, 0])

    w_avg_predict = np.matmul(transformed_train[:, :], w_avg)
    w_avg_err = mean_square_error(w_avg_predict, trainingCSV[:, 0])

    if w_t1_err < w_avg_err:
        w_avg = w_t1
    w_t = w_t1


# plot train and test set
train_predict = np.matmul(transformed_train[:, :], w_avg)
train_predict = [1 if v >0 else -1 for v in train_predict]
test_predict = np.matmul(transformed_test[:, :], w_avg)

# train plot
for i in range(len(trainingCSV)):
    items = trainingCSV[i]
    if (train_predict[i] < 0):
        plt.scatter(items[2], items[1], color='red', marker="x", label='+1')
    else:
        plt.scatter(items[2], items[1], color='blue', marker="o", label='-1')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())
plt.xlabel("Squared Mean")
plt.ylabel("Mean")
plt.show()


# test plot
for i in range(len(testingCSV)):
    items = testingCSV[i]
    if (test_predict[i] < 0):
        plt.scatter(items[2], items[1], color='red', marker="x", label='+1')
    else:
        plt.scatter(items[2], items[1], color='blue', marker="o", label='-1')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())
plt.xlabel("Squared Mean")
plt.ylabel("Mean")
plt.show()

# Report E_in and E_out
E_in = mean_square_error(train_predict, trainingCSV[:, 0])
E_test = mean_square_error(test_predict, testingCSV[:, 0])
print("E_in of 3rd-order polynomial train data: ", E_in)
print("E_test of 3rd-order polynomial test data: ", E_test)
# Bounds for E_out
E_out_lower = E_test
theta = 0.05
N = len(trainingCSV)
M = len(testingCSV)
error_bar = np.sqrt(0.5 / N * np.log(2 * M / theta))
E_out_upper = E_in + error_bar
print("Bounds of 3rd-order polynomial E_out: [{},{}]".format(E_out_lower, E_out_upper))

"""(e) Which model would you deliver to the USPS, the linear model with the 3-rd order polynomial
transform or the one without? Explain?

Answer: We definitely want to use the 3rd order polynomial transform for the USPS because
the 3rd order polynomial transform separates 1s and 5s better comparing to the linear model.
In other words, we can visualize the overlapping points better.
"""
