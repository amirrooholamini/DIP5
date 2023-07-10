import numpy as np

# First array
array1 = np.array([[0, 0, 0],
                   [0, 2222, 0],
                   [0, 0, 0]])

# Second array
array2 = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

# Combine the arrays based on the condition
combined_array = np.where(array1 == 0, array2, array1)

# Print the combined array
print(combined_array)