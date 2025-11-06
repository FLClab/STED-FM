
import pandas
import random
import os

from stedfm.DEFAULTS import BASE_PATH

data = pandas.read_csv(os.path.join(BASE_PATH, "evaluation-data", "hpa-labels", "train.csv"))
print(len(data))

random.seed(42)
training_indices = random.sample(range(len(data)), int(0.8*len(data)))

print(len(training_indices))

validation_testing_indices = list(set(range(len(data))) - set(training_indices))
random.shuffle(validation_testing_indices)
validation_indices = validation_testing_indices[:int(0.5 * len(validation_testing_indices))]
testing_indices = validation_testing_indices[int(0.5 * len(validation_testing_indices)):]

print(len(validation_indices), len(testing_indices))

training_data = data.iloc[training_indices, :]
validation_data = data.iloc[validation_indices, :]
testing_data = data.iloc[testing_indices, :]
print(training_data.shape, validation_data.shape, testing_data.shape)

training_data.to_csv(os.path.join(BASE_PATH, "evaluation-data", "hpa-labels", "training-samples.csv"))
validation_data.to_csv(os.path.join(BASE_PATH, "evaluation-data", "hpa-labels", "validation-samples.csv"))
testing_data.to_csv(os.path.join(BASE_PATH, "evaluation-data", "hpa-labels", "testing-samples.csv"))