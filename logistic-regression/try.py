import numpy as np

# Create a sample 3D array (2x3x4)
J_logistic_loss = np.random.rand(2, 3, 4)
print(J_logistic_loss)

# First mean operation: across the first dimension (axis 0)
mean_axis_0 = np.mean(J_logistic_loss[:,:,:], axis = 0)
print(mean_axis_0)

# Second mean operation: across the second dimension (axis 1)
mean_axis_1 = np.mean(mean_axis_0, axis = 1)

print("Original array shape:", J_logistic_loss.shape)
print("After first mean (shape):", mean_axis_0.shape)
print("Final result (shape):", mean_axis_1.shape)
print("Final result:", mean_axis_1)