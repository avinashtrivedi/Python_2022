import matplotlib.pyplot as plt
import numpy as np


trainingCSV = open("/Users/jhulendrabhattarai/Desktop/train.csv", "r")
trainingCSVLines = trainingCSV.readlines()

onesAndFives = []
for line in trainingCSVLines:
    if (int(line[0]) == 1 or int(line[0]) == 5):
        onesAndFives.append([float(value.strip()) for value in line.split(",")[:-1]])

# print(onesAndFives)
label1 = onesAndFives[0][0]
img1 = onesAndFives[0][1:]
img1 = np.array(img1).reshape(16,16)

label2 = onesAndFives[1][0]
img2 = onesAndFives[1][1:]
img2 = np.array(img2).reshape(16,16)

point_matrix = []

for value in onesAndFives:
    data = np.array(np.zeros(3))
    data[0] = value[0]
    data[1] = np.mean(value[1:257])
    data[2] = np.mean(np.array(value[1:257] * np.array(value[1:257])))
    point_matrix.append(data)


for items in point_matrix:
    if(items[0] == 1):
        plt.scatter(items[2], items[1], color = 'red', marker="x", label='1')
    else:
        plt.scatter(items[2], items[1], color = 'blue', marker="o",label='5')

#plt.xlabel("Squared Mean")
#plt.ylabel("Mean")
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())

# plt.imshow(img1, interpolation='nearest')
# plt.title(label1)
# plt.show()
# plt.imshow(img2, interpolation='nearest')
# plt.title(label2)
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())
plt.show()
