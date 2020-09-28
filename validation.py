import torch
from models.model_utils import multipathway_input, AverageMeter, accuracy
import misc.distributed_helper as du_helper
from evaluate import get_distance_matrix, get_closest_data_mat, get_topk_acc

#
# def topk_retrieval_validation(train_loader, test_loader, model, train_data, val_data, cfg):
#     top1_acc, top5_acc = k_nearest_embeddings(model, train_loader, test_loader, train_data, val_data, cfg, plot=False)
#
#

def validate(val_loader, tripletnet, criterion, epoch, cfg, cuda, device, is_master_proc=True):
    metric = cfg.VAL.METRIC
    if is_master_proc:
        print('=> validating with metric: {} and batch_size: {}'.format(metric, cfg.VAL.BATCH_SIZE))

    losses = AverageMeter()
    accs = AverageMeter()
    embeddings = []
    labels = []
    top1_accs = AverageMeter()
    top5_accs = AverageMeter()

    world_size = du_helper.get_world_size()

    tripletnet.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            (anchor, positive, negative) = inputs
            (anchor_target, positive_target, negative_target) = targets

            batch_size = anchor.size(0)

            if cfg.MODEL.ARCH == 'slowfast':
                anchor = multipathway_input(anchor, cfg)
                positive = multipathway_input(positive, cfg)
                negative = multipathway_input(negative, cfg)
                if cuda:
                    for i in range(len(anchor)):
                        anchor[i], positive[i], negative[i] = anchor[i].to(device), positive[i].to(device), negative[i].to(device)
            else:
                if cuda:
                    anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

            dista, distb, embedded_x, embedded_y, embedded_z = tripletnet(anchor, positive, negative)

            target = torch.FloatTensor(dista.size()).fill_(-1)
            if cuda:
                target = target.to(device)
                anchor_target = anchor_target.to(device)

            # ==> Triplet loss
            loss = criterion(dista, distb, target)
            acc = accuracy(dista.detach(), distb.detach())      # measure accuracy
            accs.update(acc.item(), batch_size)                 # record loss and accuracy
            losses.update(loss.item(), batch_size)

            if metric == 'global':
                if cfg.NUM_GPUS > 1:
                    embedded_x, anchor_target = du_helper.all_gather([embedded_x, anchor_target])
                embeddings.append(embedded_x.detach().cpu())
                labels.append(anchor_target.detach().cpu())

            elif metric == 'local_batch':
                embeddings = torch.cat((embedded_x.detach().cpu(), embedded_y.detach().cpu()), dim=0)
                labels = torch.cat((anchor_target.detach().cpu(), positive_target.detach().cpu()), dim=0)
                distance_matrix = get_distance_matrix(embeddings, dist_metric=cfg.LOSS.DIST_METRIC)
                top1_acc, top5_acc = get_topk_acc(distance_matrix, labels.tolist())
                top1_accs.update(top1_acc)
                top5_accs.update(top5_acc)

            else:
                print('Metric type:{} is not implemented'.format(metric))

            batch_size_world = batch_size * world_size
            if ((batch_idx + 1) * world_size) % cfg.VAL.LOG_INTERVAL == 0:
                if (is_master_proc):
                    msg = 'Val Epoch: {} [{}/{} | {:.1f}%]\t'\
                          'Loss: {:.4f} ({:.4f}) \t'\
                          'Triplet Acc: {:.2f}% ({:.2f}%)'.format(
                              epoch, (batch_idx+1)*batch_size_world,
                              len(val_loader.dataset), ((batch_idx+1)*100.*batch_size_world/len(val_loader.dataset)),
                              losses.val, losses.avg,
                              accs.val*100., accs.avg*100.)

                    if metric == 'local_batch':
                        msg += '\t'
                        msg +=  'Top1 Acc: {:.2f}% ({:.2f}%) \t'\
                                'Top5 Acc: {:.2f}% ({:.2f}%)'.format(
                                top1_accs.val*100., top1_accs.avg*100.,
                                top5_accs.val*100., top5_accs.avg*100.)
                    print(msg)

    if cfg.NUM_GPUS > 1: # ==> triplet loss
        acc_sum = torch.tensor([accs.sum], dtype=torch.float32, device=device)
        acc_count = torch.tensor([accs.count], dtype=torch.float32, device=device)

        losses_sum = torch.tensor([losses.sum], dtype=torch.float32, device=device)
        losses_count = torch.tensor([losses.count], dtype=torch.float32, device=device)

        acc_sum, losses_sum, acc_count, losses_count = du_helper.all_reduce([acc_sum, losses_sum, acc_count, losses_count], avg=False)

        accs.avg = acc_sum.item() / acc_count.item()
        losses.avg = losses_sum.item() / losses_count.item()

        if metric == 'local_batch':
            top1_acc_sum = torch.tensor([top1_accs.sum], dtype=torch.float32, device=device)
            top1_acc_count = torch.tensor([top1_accs.count], dtype=torch.float32, device=device)
            top5_acc_sum = torch.tensor([top5_accs.sum], dtype=torch.float32, device=device)
            top5_acc_count = torch.tensor([top5_accs.count], dtype=torch.float32, device=device)

            top1_acc_sum, top5_acc_sum, top1_acc_count, top5_acc_count = du_helper.all_reduce([top1_acc_sum, top5_acc_sum, top1_acc_count, top5_acc_count], avg=False)
            top1_accs.avg = top1_acc_sum.item() / top1_acc_count.item()
            top5_accs.avg = top5_acc_sum.item() / top5_acc_count.item()


    if metric == 'global':
        # Top 1/5 Acc
        if (is_master_proc):
            embeddings = torch.cat(embeddings, dim=0)
            labels = torch.cat(labels, dim=0).tolist()
            print('embeddings size', embeddings.size())
            print('labels size', len(labels))
            distance_matrix = get_distance_matrix(embeddings, dist_metric=cfg.LOSS.DIST_METRIC)
            top1_acc, top5_acc = get_topk_acc(distance_matrix, labels)
            print('top1_acc', top1_acc, 'top5_acc', top5_acc)
            top1_accs.update(top1_acc)
            top5_accs.update(top5_acc)

            top1_accs.avg = top1_accs.val
            top5_accs.avg = top5_accs.val

    if (is_master_proc):
        # Log
        msg = '\nTest set: Average loss: {:.4f}, Triplet Accuracy: {:.2f}%'.format(losses.avg, accs.avg*100.)
        to_write = 'epoch:{} {:.4f} {:.2f}'.format(epoch, losses.avg, accs.avg*100.)
        if metric == 'global' or metric == 'local_batch':
            msg += ', '
            msg += 'Top1 Acc: {:.2f}% ({:.2f}%) \t'\
                   'Top5 Acc: {:.2f}% ({:.2f}%)'.format(100.*top1_accs.val, 100.*top1_accs.avg,
                                                        100.*top5_accs.val, 100.*top5_accs.avg)
            to_write += ' {:.2f} {:.2f}'.format(100.*top1_accs.avg, 100.*top5_accs.avg)

        to_write += '\n'
        print(msg)
        with open('{}/tnet_checkpoints/val_loss_and_acc.txt'.format(cfg.OUTPUT_PATH), "a") as val_file:
            val_file.write(to_write)

    return accs.avg
