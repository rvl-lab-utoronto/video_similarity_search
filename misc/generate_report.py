import csv
import os
import io
import pickle
import argparse
import gspread
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from httplib2 import Http
from oauth2client.service_account import ServiceAccountCredentials

#from upload_gdrive import GoogleDriveUploader#upload_file_to_gdrive, SCOPES


SOURCE_CODE_DIR = os.path.dirname(os.path.abspath(__file__))

train_progress_file = './tnet_checkpoints/train_loss_and_acc.txt'
val_progress_file = './tnet_checkpoints/val_loss_and_acc.txt'
global_retrieval_file = './tnet_checkpoints/global_retrieval_acc.txt'
nmi_progress_file = './tnet_checkpoints/NMIs.txt'
ami_progress_file = './tnet_checkpoints/AMIs.txt'

def parse():
    parser = argparse.ArgumentParser("Video Similarity Search Training Script")
    parser.add_argument(
        '--name',
        type=str, action='store',
        default=None,
        help='used for plot name and google worksheet name'
    )
    parser.add_argument(
        '--result_dir',
        type=str, action='store',
        default=None,
        help='result directory'
    )
    parser.add_argument(
        '-p', '--plot', action='store_true', help='generate_plots'
    )
    return parser.parse_args()


def parse_file(result_dir, f_type='train'):
    epoch = []
    losses = []
    acc = []
    top1_acc = []
    top5_acc = []
    runtime = []
    nmis = []
    amis = []
    assert f_type in ['train', 'val', 'global_retrieval', 'nmi', 'ami'], "f_type:{} is not recognized".format(f_type)
    processed_epoch = []

    if f_type == 'train':
        with open (os.path.join(result_dir, train_progress_file), newline='') as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=' ')
            for row in csv_reader:
                cur_epoch = float(row[0].replace('epoch:', '').replace(',',''))
                if cur_epoch in processed_epoch:
                    continue
                epoch.append(cur_epoch)
                processed_epoch.append(cur_epoch)
                losses.append(float(row[2]))
                # acc.append(float(row[3]))
                runtime.append(float(row[1].replace('runtime:', '').replace(',','')))
    elif f_type=='val':
        with open (os.path.join(result_dir, val_progress_file), newline='') as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=' ')
            for row in csv_reader:
                cur_epoch = float(row[0].replace('epoch:', '').replace(',',''))
                if cur_epoch in processed_epoch:
                    continue
                processed_epoch.append(cur_epoch)
                losses.append(float(row[1]))
                acc.append(float(row[2]))
                top1_acc.append(float(row[3]))
                top5_acc.append(float(row[4]))
    elif f_type=='nmi':
        with open (os.path.join(result_dir, nmi_progress_file), newline='') as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=' ')
            for row in csv_reader:
                cur_epoch = float(row[0].replace('epoch:', '').replace(',',''))
                if cur_epoch in processed_epoch:
                    continue
                processed_epoch.append(cur_epoch)
                nmis.append(float(row[1]))
    elif f_type=='ami':
        with open (os.path.join(result_dir, ami_progress_file), newline='') as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=' ')
            for row in csv_reader:
                cur_epoch = float(row[0].replace('epoch:', '').replace(',',''))
                if cur_epoch in processed_epoch:
                    continue
                processed_epoch.append(cur_epoch)
                amis.append(float(row[1]))
    else:
        with open (os.path.join(result_dir, global_retrieval_file), newline='') as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=' ')
            for row in csv_reader:
                cur_epoch = float(row[0].replace('epoch:', '').replace(',',''))
                if cur_epoch in processed_epoch:
                    continue
                epoch.append(cur_epoch)
                processed_epoch.append(cur_epoch)
                top1_acc.append(float(row[1]))
                top5_acc.append(float(row[2]))

    return epoch, runtime, losses, acc, top1_acc, top5_acc, nmis, amis


def plot_training_progress(result_dir, name, show_plot=False, service=None):
    _, _, train_losses, _, _, _, _, _  = parse_file(result_dir, 'train') 
    _, _, val_losses, val_acc, top1_acc, top5_acc, _, _ = parse_file(result_dir, 'val')
    top1_5_epoch, _, _, _, global_top1_acc, global_top5_acc, _, _ = parse_file(result_dir, 'global_retrieval')

    num_plots = 3

    if (os.path.exists(os.path.join(result_dir, nmi_progress_file))):
        _, _, _, _, _, _, nmis, _ = parse_file(result_dir, 'nmi')
        _, _, _, _, _, _, _, amis = parse_file(result_dir, 'ami')
        num_plots += 2

    # print(top1_5_epoch)
    f = plt.figure(figsize=(22,6))
    #f = plt.figure()

    #font = {'size':20}
    #plt.rc('font', **font)

    ax1 =  plt.subplot(1, num_plots, 1)
    ax1.plot(np.arange(len(train_losses)), train_losses)
    ax1.plot(np.arange(len(val_losses)), val_losses)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('Training Curve')
    ax1.legend(['Training', 'Validation'])

    ax2 = plt.subplot(1, num_plots, 2)
    #ax2.plot(np.arange(len(train_acc)), train_acc)
    ax2.plot(np.arange(len(val_acc)), val_acc)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Val Triplet Acc vs. Epoch')
    #ax2.legend(['Training', 'Validation'])
    plt.grid()

    ax3 = plt.subplot(1, num_plots, 3)
    # ax3.plot(np.arange(len(top1_acc)), top1_acc)
    # ax3.plot(np.arange(len(top5_acc)), top5_acc)
    ax3.plot(top1_5_epoch, global_top1_acc)
    ax3.plot(top1_5_epoch, global_top5_acc)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Top-k Retrieval Accuracy (%)')
    ax3.set_title('Top-1/5 Retrieval Accuracy')
    ax3.legend(['Top-1', 'Top-5'])
    plt.grid()

    if (os.path.exists(os.path.join(result_dir, nmi_progress_file))):

        cluster_interval = round(len(train_losses) / len(nmis))

        ax4 = plt.subplot(1, num_plots, 4)
        ax4.plot(cluster_interval*np.arange(len(nmis)), nmis)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('NMI - Cluster Assign. / Labels')
        ax4.set_title('Clustering Quality')

        ax5 = plt.subplot(1, num_plots, 5)
        ax5.plot(cluster_interval*np.arange(len(amis)), amis)
        ax5.set_xlabel('Epoch')
        ax5.set_ylabel('Cluster Assignment vs True Label AMI')
        ax5.set_title('AMI vs. Epoch')

    plt.grid()

    plot_name = '{}_train_val_loss.png'.format(name)
    f.savefig(plot_name, bbox_inches ='tight')
    print('plots saved to:{}'.format(plot_name))
    if service:
        service.upload_file_to_gdrive(plot_name, 'evaluate')
        print('plot uploaded to google drive folder under /evaluate')

    if (show_plot):
        plt.show()


if __name__ == '__main__':
    args = parse()

    if not args.name:
        name=input("Please specify a worksheet name with --name (e.g. ResNet18_K, SlowFast_U): ")
    else:
        name = args.name

    result_dir = args.result_dir
    if not args.result_dir:
        result_dir=input("Please specify the results directory: ")

    # gs_report(result_dir, name)
    if args.plot:
        # gdrive_service = GoogleDriveUploader()
        gdrive_service=None
        plot_training_progress(result_dir, name, show_plot=True, service=gdrive_service)
