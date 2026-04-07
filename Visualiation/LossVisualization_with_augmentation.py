import pandas as pd
import matplotlib.pyplot as plt


# Read the loss data
loss_data = pd.read_csv('../training_with_augmentation/loss.csv')

#Plot the training and validation loss
#plt.figure(figsize=(10, 6))
plt.plot(loss_data['epoch'], loss_data['train_loss'], label='Training Loss')
plt.plot(loss_data['epoch'], loss_data['val_loss'], label='Validation Loss')
plt.axline((0, 2.3), slope=0, color='red', label = 'Baseline')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.legend()
plt.grid(True)
plt.savefig('loss_plot.png')

plt.savefig('augmentation_training.svg')
plt.show()