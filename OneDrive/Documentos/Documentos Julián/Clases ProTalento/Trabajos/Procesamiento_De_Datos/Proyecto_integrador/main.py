from datasets import load_dataset
import numpy as np

dataset = load_dataset("mstz/heart_failure")


data = dataset["train"]
arr_age = np.array(data['age'])
print('The average age in dataset is: {:.2f}'
      .format(arr_age.mean()))
