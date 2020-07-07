import matplotlib.pyplot as plt
import numpy as np
import csv

train_progress_file = './tnet_checkpoints/train_loss_and_acc.txt'
val_progress_file = './tnet_checkpoints/val_loss_and_acc.txt'

def plot_training_progress ():
    train_losses = []
    train_acc = []
    val_losses = []
    val_acc = []

    with open (train_progress_file, newline='') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=' ')
        for row in csv_reader:
            train_losses.append(float(row[0]))
            train_acc.append(float(row[1]))
    
    with open (val_progress_file, newline='') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=' ')
        for row in csv_reader:
            val_losses.append(float(row[0]))
            val_acc.append(float(row[1]))
    
    plt.subplot(1, 2, 1)
    plt.plot(np.arange(len(train_losses)), train_losses)
    plt.plot(np.arange(len(train_acc)), train_acc)
    plt.xlabel('Epoch')
    plt.ylabel('Loss/Accuracy')
    plt.title('Training loss and accuracy vs. Epoch')
    plt.legend(['Loss', 'Accuracy'])

    plt.subplot(1, 2, 2)
    plt.plot(np.arange(len(val_losses)), val_losses)
    plt.plot(np.arange(len(val_acc)), val_acc)
    plt.xlabel('Epoch')
    plt.ylabel('Loss/Accuracy')
    plt.title('Validation loss and accuracy vs. Epoch')
    plt.legend(['Loss', 'Accuracy'])

    plt.show()
    

if __name__ == '__main__':
    plot_training_progress()