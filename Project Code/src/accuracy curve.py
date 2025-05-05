import matplotlib.pyplot as plt

epochs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
train_accuracies = [0.60, 0.65, 0.70, 0.75, 0.78, 0.80, 0.82, 0.85, 0.88, 0.89]
val_accuracies = [0.57, 0.64, 0.675, 0.73, 0.76, 0.79, 0.81, 0.83, 0.85, 0.86]

# Plot the accuracy curve
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_accuracies, label='Training Accuracy', color='blue', marker='o')
plt.plot(epochs, val_accuracies, label='Validation Accuracy', color='green', marker='x')

# Adding labels and title
plt.title('Model Accuracy Curve')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Display the plot
plt.grid(True)
plt.show()
