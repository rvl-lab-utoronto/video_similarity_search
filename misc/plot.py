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
            train_losses.append(float(row[3]))
            train_acc.append(float(row[4]))

    with open (val_progress_file, newline='') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=' ')
        for row in csv_reader:
            val_losses.append(float(row[2]))
            val_acc.append(float(row[3]))

    plt.subplot(1, 2, 1)
    plt.plot(np.arange(len(train_losses)), train_losses)
    plt.plot(np.arange(len(val_losses)), val_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss vs. Epoch')
    plt.legend(['Training', 'Validation'])

    plt.subplot(1, 2, 2)
    plt.plot(np.arange(len(train_acc)), train_acc)
    plt.plot(np.arange(len(val_acc)), val_acc)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy vs. Epoch')
    plt.legend(['Training', 'Validation'])
    plt.savefig('train_val_loss.png')
    # plt.show()


if __name__ == '__main__':
    plot_training_progress()
