from random import shuffle
import sys
import xlrd
import numpy as np
import matplotlib
"""
Workaround to solve bug using matplotlib. See:
https://stackoverflow.com/questions/49367013/pipenv-install-matplotlib
"""
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

class LinearRegression:
  def __init__(self, data, labels):
    self.data = data
    self.labels = labels
    self.partition_by_label()

  # Split data into 2 index groups
  def partition_by_label(self):
    one, zero = [], []
    for idx, val in enumerate(self.labels):
      one.append(idx) if val == 1 else zero.append(idx)
    self.idx_one, self.idx_zero = one, zero

  # shuffle and determine training set
  def prepare(self, training_sample_size):
    shuffle(self.idx_one)
    shuffle(self.idx_zero)
    self.training_set = self.idx_one[:training_sample_size] + self.idx_zero[:training_sample_size]
    self.test_set = self.idx_one[training_sample_size:] + self.idx_zero[training_sample_size:]

  # calculate the parameters of current training set
  def get_parameter_model(self):
    # Prepend 1 as x0, which is required to calculate w0
    samples = list(map(lambda idx : [1] + self.data[idx], self.training_set))
    labels = list(map(lambda idx : self.labels[idx], self.training_set))

    x = np.array(samples)
    x_transpose = x.T
    t = np.array(labels).reshape(len(labels), 1)
    x_dot_inv = np.linalg.inv(x_transpose.dot(x))
    w = x_dot_inv.dot(x_transpose).dot(t)
    return w

  # get the accumulated correct prediction
  def get_correct_prediction(self):
    w = self.get_parameter_model()
    correct = 0

    for idx in self.test_set:
      data = [1] + self.data[idx] # Prepend 1 as x0 to get multiplied by w0
      x = np.array(data).reshape(len(data), 1)
      prediction = w.T.dot(x)
      res = 1 if np.asscalar(prediction) >= 0.5 else 0
      correct += int(res == self.labels[idx])

    return correct

def parse_raw_data_from_file(path):
    xlsx = xlrd.open_workbook(path)
    sheet = xlsx.sheet_by_index(0)

    data = []
    labels = []

    # Exclude the 1st row since we only need raw data
    rows = sheet.nrows - 1
    # Partition the raw data into data and labels
    for row_idx in range(rows):
      row = sheet.row_values(row_idx + 1)
      data.append(row[:-1])
      labels.append(row[-1])

    return data, labels

def to_accuracy_by_samples(data, labels, experiments):
  linear_regression = LinearRegression(data, labels)

  def calcAccuracy(training_sample_size):
    correct_res = 0
    for _ in range(experiments):
      linear_regression.prepare(training_sample_size)
      correct_res += linear_regression.get_correct_prediction()
    return correct_res / len(linear_regression.test_set * experiments)

  return calcAccuracy

"""
Render the line chart. `matplotlib` conflicts with pipenv 😞. Check:
https://matplotlib.org/faq/osx_framework.html
"""
def plot_accuracy_line(accuracies, training_sample_size):
  # Take n from group 0 and 1, so multiply by 2
  training_sample_size = list(map(lambda x : x * 2, training_sample_size))

  plt.plot(training_sample_size, list(map(lambda x : x * 100, accuracies)), 'r-o')
  # Zoom in
  plt.axis([0, max(training_sample_size), 70, 80])
  # plt.axis([0, max(training_sample_size), 0, 100])
  plt.xlabel('Training Sample Size (2n)')
  plt.ylabel('Accuracy (%)')
  plt.show()

if __name__ == '__main__':
    file_path = sys.argv[1]
    experiments = int(sys.argv[2])
    training_sample_size = list(map(lambda x : int(x), sys.argv[3].split(',')))

    # Parse data
    data, labels = parse_raw_data_from_file(file_path)

    # Get mapped accuracy
    toAccuracy = to_accuracy_by_samples(data, labels, experiments)
    accuracies = list(map(toAccuracy, training_sample_size))

    print("Accuracy:", accuracies)

    # Plot the accuracy with nice line chart
    plot_accuracy_line(accuracies, training_sample_size)