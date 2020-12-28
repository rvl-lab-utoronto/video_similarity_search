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
    _, _, train_losses, _, _, _, _, _ = parse_file(result_dir, 'train')
    _, _, val_losses, val_acc, top1_acc, top5_acc, _, _ = parse_file(result_dir, 'val')
    top1_5_epoch, _, _, _, global_top1_acc, global_top5_acc, _, _ = parse_file(result_dir, 'global_retrieval')

    num_plots = 3

    if (os.path.exists(os.path.join(result_dir, nmi_progress_file))):
        _, _, _, _, _, _, nmis, _ = parse_file(result_dir, 'nmi')
        _, _, _, _, _, _, _, amis = parse_file(result_dir, 'ami')
        num_plots += 2

    # print(top1_5_epoch)
    f = plt.figure(figsize=(18,5))
    ax1 =  plt.subplot(1, num_plots, 1)
    ax1.plot(np.arange(len(train_losses)), train_losses)
    # ax1.plot(np.arange(len(val_losses)), val_losses)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Train/Val Loss vs. Epoch')
    ax1.legend(['Training', 'Validation'])

    ax2 = plt.subplot(1, num_plots, 2)
    #ax2.plot(np.arange(len(train_acc)), train_acc)
    ax2.plot(np.arange(len(val_acc)), val_acc)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Val Triplet Acc vs. Epoch')
    #ax2.legend(['Training', 'Validation'])

    ax3 = plt.subplot(1, num_plots, 3)
    # ax3.plot(np.arange(len(top1_acc)), top1_acc)
    # ax3.plot(np.arange(len(top5_acc)), top5_acc)
    ax3.plot(top1_5_epoch, global_top1_acc)
    ax3.plot(top1_5_epoch, global_top5_acc)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Accuracy (%)')
    ax3.set_title('Val Top 1/5 Retrieval Acc vs. Epoch')
    ax3.legend(['Top1', 'Top5'])

    if (os.path.exists(os.path.join(result_dir, nmi_progress_file))):
        ax4 = plt.subplot(1, num_plots, 4)
        ax4.plot(np.arange(len(nmis)), nmis)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Cluster Assignment vs True Label NMI')
        ax4.set_title('NMI vs. Epoch')

        ax5 = plt.subplot(1, num_plots, 5)
        ax5.plot(np.arange(len(amis)), amis)
        ax5.set_xlabel('Epoch')
        ax5.set_ylabel('Cluster Assignment vs True Label AMI')
        ax5.set_title('AMI vs. Epoch')

    plot_name = '{}_train_val_loss.png'.format(name)
    f.savefig(plot_name)

    service.upload_file_to_gdrive(plot_name, 'evaluate')
    print('plots saved to:{}, and uploaded to google drive folder under /evaluate'.format(plot_name))

    if (show_plot):
        plt.show()

def write_to_google_sheet(result_dir, client, worksheet_name):
    epoch, runtime, train_losses, _, _, _, _ = parse_file(result_dir, 'train')
    _, _, val_losses, val_acc, top1_acc, top5_acc, _ = parse_file(result_dir, 'val')
    top1_5_epoch, _, _, _, global_top1_acc, global_top5_acc, _ = parse_file(result_dir, 'global_retrieval')


    # best_idx = np.argmax(np.array(top1_acc))
    # print('best epoch:{}, triplet accuracy:{}, val_top1 accuracy:{}, val_top5 accuracy:{}'.format(epoch[best_idx],
    #                     val_acc[best_idx], top1_acc[best_idx], top5_acc[best_idx]))

    # best_idx = np.argmax(np.array(global_top1_acc))
    # print('best epoch:{}, global top1 acc:{}, global top5 acc:{}'.format(top1_5_epoch[best_idx], global_top1_acc[best_idx], global_top5_acc[best_idx]))

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
    # df['val_losses'] = val_losses
    # df['val_acc'] = val_acc
    # df['runtime'] = runtime
    # df['top1_acc'] = top1_acc
    # df['top5_acc'] = top5_acc
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
        gdrive_service = GoogleDriveUploader()
        plot_training_progress(result_dir, name, show_plot=True, service=gdrive_service)
