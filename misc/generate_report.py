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
from upload_gdrive import upload_file_to_gdrive, SCOPES


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
    runtime = []
    assert f_type in ['train', 'val'], "f_type:{} is not recognized".format(f_type)

    if f_type == 'train':
        with open (os.path.join(result_dir, train_progress_file), newline='') as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=' ')
            for row in csv_reader:
                epoch.append(float(row[0].replace('epoch:', '').replace(',','')))
                losses.append(float(row[3]))
                acc.append(float(row[4]))
                runtime.append(float(row[1].replace('runtime:', '').replace(',','')))
    else:
        with open (os.path.join(result_dir, val_progress_file), newline='') as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=' ')
            for row in csv_reader:
                losses.append(float(row[2]))
                acc.append(float(row[3]))
    return epoch, runtime, losses, acc


def plot_training_progress(result_dir, name, show_plot=False):
    _, _, train_losses, train_acc = parse_file(result_dir, 'train')
    _, _, val_losses, val_acc = parse_file(result_dir, 'val')

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
    plot_name = '{}_train_val_loss.png'.format(name)
    plt.savefig(plot_name)
    upload_file_to_gdrive(plot_name, 'evaluate')
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
    df['train_acc'] = train_acc
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

    gs_report(result_dir, name)
    if args.plot:
        plot_training_progress(result_dir, name, show_plot=True)
