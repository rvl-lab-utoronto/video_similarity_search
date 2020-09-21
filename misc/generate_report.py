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

from upload_gdrive import GoogleDriveUploader#upload_file_to_gdrive, SCOPES


SOURCE_CODE_DIR = os.path.dirname(os.path.abspath(__file__))

train_progress_file = './tnet_checkpoints/train_loss_and_acc.txt'
val_progress_file = './tnet_checkpoints/val_loss_and_acc.txt'


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
    assert f_type in ['train', 'val'], "f_type:{} is not recognized".format(f_type)

    if f_type == 'train':
        with open (os.path.join(result_dir, train_progress_file), newline='') as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=' ')
            for row in csv_reader:
                epoch.append(float(row[0].replace('epoch:', '').replace(',','')))
                losses.append(float(row[2]))
                # acc.append(float(row[3]))
                runtime.append(float(row[1].replace('runtime:', '').replace(',','')))
    else:
        with open (os.path.join(result_dir, val_progress_file), newline='') as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=' ')
            for row in csv_reader:
                losses.append(float(row[1]))
                acc.append(float(row[2]))
                top1_acc.append(float(row[3]))
                top5_acc.append(float(row[4]))
    return epoch, runtime, losses, acc, top1_acc, top5_acc


def plot_training_progress(result_dir, name, show_plot=False, service=None):
    _, _, train_losses, _, _, _ = parse_file(result_dir, 'train')
    _, _, val_losses, val_acc, top1_acc, top5_acc = parse_file(result_dir, 'val')

    f = plt.figure(figsize=(9,4))
    ax1 =  plt.subplot(1, 3, 1)
    ax1.plot(np.arange(len(train_losses)), train_losses)
    ax1.plot(np.arange(len(val_losses)), val_losses)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss vs. Epoch')
    ax1.legend(['Training', 'Validation'])

    ax2 = plt.subplot(1, 3, 2)
    #ax2.plot(np.arange(len(train_acc)), train_acc)
    ax2.plot(np.arange(len(val_acc)), val_acc)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Validation Triplet Accuracy vs. Epoch')
    #ax2.legend(['Training', 'Validation'])

    ax3 = plt.subplot(1, 3, 3)
    ax3.plot(np.arange(len(top1_acc)), top1_acc)
    ax3.plot(np.arange(len(top5_acc)), top5_acc)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Accuracy (%)')
    ax3.set_title('Validation Top 1/5 Retrieval Accuracy vs. Epoch')
    ax3.legend(['Top1', 'Top5'])

    plot_name = '{}_train_val_loss.png'.format(name)
    f.savefig(plot_name)

    service.upload_file_to_gdrive(plot_name, 'evaluate')
    print('plots saved to:{}, and uploaded to google drive folder under /evaluate'.format(plot_name))

    if (show_plot):
        plt.show()

def write_to_google_sheet(result_dir, client, worksheet_name):
    epoch, runtime, train_losses, train_acc = parse_file(result_dir, 'train')
    _, _, val_losses, val_acc = parse_file(result_dir, 'val')
    sh = client.open('training_results')

    try:
        worksheet = sh.worksheet(worksheet_name)
    except Exception as e:
        print('creating worksheet: {}'.format(worksheet_name))
        worksheet = sh.add_worksheet(title=worksheet_name, rows=1000, cols=20)

    df = pd.DataFrame()
    df['epoch'] = epoch
    df['train_loss'] = train_losses
    # df['train_acc'] = train_acc
    df['val_losses'] = val_losses
    df['val_acc'] = val_acc
    df['runtime'] = runtime
    worksheet.update([df.columns.values.tolist()] + df.values.tolist())

def gs_report(result_dir, name):
    scope = ['https://spreadsheets.google.com/feeds',
            'https://www.googleapis.com/auth/drive']
    creds = ServiceAccountCredentials.from_json_keyfile_name(os.path.join(SOURCE_CODE_DIR, 'gs_credentials.json'), scope)
    client = gspread.authorize(creds)
    write_to_google_sheet(result_dir, client, name)
    print('updated to worksheet:{}'.format(name))


if __name__ == '__main__':
    args = parse()

    if not args.name:
        name=input("Please specify a worksheet name with --name (e.g. ResNet18_K, SlowFast_U): ")
    else:
        name = args.name

    result_dir = args.result_dir
    if not args.result_dir:
        result_dir=input("Please specify the results directory: ")

    #gs_report(result_dir, name)
    if args.plot:
        gdrive_service = GoogleDriveUploader()
        plot_training_progress(result_dir, name, show_plot=True, service=gdrive_service)
