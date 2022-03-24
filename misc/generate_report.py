import csv
import os
import io
import pickle
import glob
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
# nmi_progress_file = './tnet_checkpoints/NMIs.txt'
# ami_progress_file = './tnet_checkpoints/AMIs.txt'

nmi_progress_file = './tnet_checkpoints/NMIs*.txt'
ami_progress_file = './tnet_checkpoints/AMIs*.txt'


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


def parse_file(result_dir, f_type='train', filename=None):
    epoch = []
    losses = []
    acc = []
    top1_acc = []
    top5_acc = []
    runtime = []
    nmis = []
    amis = []
    fp = []
    fn = []

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
                fp.append(float(row[3]))
                fn.append(float(row[4]))

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
        with open (os.path.join(result_dir, filename), newline='') as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=' ')
            for row in csv_reader:
                cur_epoch = float(row[0].replace('epoch:', '').replace(',',''))

                if len(processed_epoch) == 0 and cur_epoch != 0:
                    cluster_interval = 5  # TODO: don't hardcode
                    for i in range(0,int(cur_epoch),cluster_interval):
                        nmis.append(0.0)

                if cur_epoch in processed_epoch:
                    continue
                processed_epoch.append(cur_epoch)
                nmis.append(float(row[1]))
    
    elif f_type=='ami':
        with open (os.path.join(result_dir, filename), newline='') as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=' ')
            for row in csv_reader:
                cur_epoch = float(row[0].replace('epoch:', '').replace(',',''))

                if len(processed_epoch) == 0 and cur_epoch != 0:
                    cluster_interval = 5  # TODO: don't hardcode
                    for i in range(0,int(cur_epoch),cluster_interval):
                        amis.append(0.0)

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

    return epoch, runtime, losses, acc, top1_acc, top5_acc, nmis, amis, fp, fn


def plot_training_progress(result_dir, name, show_plot=False, service=None):
    _, _, train_losses, _, _, _, _, _, fp, fn  = parse_file(result_dir, 'train') 
    _, _, val_losses, val_acc, top1_acc, top5_acc, _, _, _, _ = parse_file(result_dir, 'val')
    top1_5_epoch, _, _, _, global_top1_acc, global_top5_acc, _, _, _, _ = parse_file(result_dir, 'global_retrieval')

    num_plots = 3

    NMI_files = glob.glob(nmi_progress_file)
    AMI_files = glob.glob(ami_progress_file)
    assert len(NMI_files) == len(AMI_files), "check if we have the same number of NMI and AMI files"

    if len(NMI_files) > 1:
        NMI_files.sort()
        AMI_files.sort()
        print(NMI_files)
        print(AMI_files)
        partitions = [os.path.basename(nmi_f).replace("NMIs_p", '').replace('.txt', '') for nmi_f in NMI_files]

    else:
        partitions = [0]
    
    num_plots += 2


    NMIs, AMIs = [], []
    for i in range(len(NMI_files)):
    # if (os.path.exists(os.path.join(result_dir, nmi_progress_file))):
        _, _, _, _, _, _, nmis, _, _, _ = parse_file(result_dir, 'nmi', filename=NMI_files[i])
        _, _, _, _, _, _, _, amis, _, _ = parse_file(result_dir, 'ami', filename=AMI_files[i])
        NMIs.append(nmis)
        AMIs.append(amis)

    if len(fp) > 0:
        num_plots += 1

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
    plt.grid()

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

    cur_plot_idx = 4
    if len(NMI_files) > 0:
        ax4 = plt.subplot(1, num_plots, 4)
        cluster_interval = round(len(train_losses) / len(NMIs[0]))

        for nmis in NMIs:
            ax4.plot(cluster_interval*np.arange(len(nmis)), nmis)
        ax4.legend(partitions)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('NMI - Cluster Assign. / Labels')
        ax4.set_title('Clustering Quality')
        plt.grid()

        ax5 = plt.subplot(1, num_plots, 5)
        for amis in AMIs:
            ax5.plot(cluster_interval*np.arange(len(amis)), amis)
        ax5.legend(partitions)
        ax5.set_xlabel('Epoch')
        ax5.set_ylabel('Cluster Assignment vs True Label AMI')
        ax5.set_title('AMI vs. Epoch')

        cur_plot_idx += 2
    plt.grid()

    if len(fp) > 0:
        # print(len(fp), fp)
                
        # if len(partitions) > 1:
        bs=input("Please specify the batch size: ")
        bs = float(bs)
        pos_replace = input("Please specify the positvie sampling rate: ")
        pos_replace = float(pos_replace)

        ##update fp & fn
        fp = np.array(fp)/(bs*(1 - pos_replace))
        fn = np.array(fn)/(bs*(1 - pos_replace))

        print(fp)
        ax6 = plt.subplot(1, num_plots, cur_plot_idx)
        ax6.plot(np.arange(len(fp)), fp)
        ax6.plot(np.arange(len(fn)), fn)
        ax6.set_xlabel('Epoch')
        ax6.set_ylabel('Num')
        ax6.set_title('FP, FN v.s. Epochs\n(batch_size=8, pos_sample=0.2)')
        ax6.legend(['False Positive', 'False Negative'])  

        plt.grid()

    plot_name = '{}_train_val_loss.png'.format(name)
    f.savefig(plot_name, bbox_inches ='tight')
    print('plots saved to:{}'.format(plot_name))
    if service:
        service.upload_file_to_gdrive(plot_name, 'evaluate')
        print('plot uploaded to google drive folder under /evaluate')

    if (show_plot):
        plt.show()

def write_to_google_sheet(result_dir, client, worksheet_name):
    epoch, runtime, train_losses, _, _, _, _, _ = parse_file(result_dir, 'train')
    _, _, val_losses, val_acc, top1_acc, top5_acc, _, _ = parse_file(result_dir, 'val')
    top1_5_epoch, _, _, _, global_top1_acc, global_top5_acc, _, _ = parse_file(result_dir, 'global_retrieval')


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

    # gs_report(result_dir, name)
    if args.plot:
        # gdrive_service = GoogleDriveUploader()
        gdrive_service=None
        plot_training_progress(result_dir, name, show_plot=True, service=gdrive_service)
