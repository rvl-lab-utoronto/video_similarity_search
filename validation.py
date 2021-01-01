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

    losses = AverageMeter()
    accs = AverageMeter()
    embeddings = []
    labels = []
    top1_accs = AverageMeter()
    top5_accs = AverageMeter()

    world_size = du_helper.get_world_size()

    tripletnet.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets, idx) in enumerate(val_loader):
            (anchor, positive, negative) = inputs
            (anchor_target, positive_target, negative_target) = targets
            batch_size = torch.tensor(anchor.size(0)).to(device)

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
            embedded_x = embedded_x.flatten(1)
            embedded_y = embedded_y.flatten(1)
            embedded_z = embedded_z.flatten(1)

            target = torch.FloatTensor(dista.size()).fill_(-1)
            if cuda:
                target = target.to(device)
                anchor_target = anchor_target.to(device)

            # ==> Triplet loss
            loss = criterion(dista, distb, target)
            acc = accuracy(dista.detach(), distb.detach())      # measure accuracy

            if metric == 'global':
                if cfg.NUM_GPUS > 1:
                    embedded_x, anchor_target = du_helper.all_gather([embedded_x, anchor_target])
                embeddings.append(embedded_x.detach().cpu())
                labels.append(anchor_target.detach().cpu())
            elif metric == 'local_batch':
                embeddings = torch.cat((embedded_x.detach().cpu(), embedded_y.detach().cpu()), dim=0)
                labels = torch.cat((anchor_target.detach().cpu(), positive_target.detach().cpu()), dim=0)
                distance_matrix = get_distance_matrix(embeddings, dist_metric=cfg.LOSS.DIST_METRIC)
                topk_acc = get_topk_acc(distance_matrix, labels.tolist())
                top1_acc = torch.tensor(topk_acc[0]).to(device)
                top5_acc = torch.tensor(topk_acc[1]).to(device)

            else:
                print('Metric type:{} is not implemented'.format(metric))

            if cfg.NUM_GPUS > 1:
                [loss, acc] = du_helper.all_reduce([loss, acc], avg=True)
                [batch_size_world] = du_helper.all_reduce([batch_size], avg=False)

                if metric == 'local_batch':
                    [top1_acc, top5_acc] = du_helper.all_reduce([top1_acc, top5_acc], avg=True)
            else:
                batch_size_world = batch_size

            batch_size_world = batch_size_world.item()

            # Update running loss and accuracy
            accs.update(acc.item(), batch_size_world)
            losses.update(loss.item(), batch_size_world)
            if metric == 'local_batch':
                top1_accs.update(top1_acc)
                top5_accs.update(top5_acc)

            if ((batch_idx + 1) * world_size) % cfg.VAL.LOG_INTERVAL == 0:
                if (is_master_proc):
                    msg = 'Val Epoch: {} [{}/{} | {:.1f}%]\t'\
                          'Loss: {:.4f} ({:.4f}) \t'\
                          'Triplet Acc: {:.2f}% ({:.2f}%)'.format(
                              epoch, losses.count,
                              len(val_loader.dataset), (losses.count*100./len(val_loader.dataset)),
                              losses.val, losses.avg,
                              accs.val*100., accs.avg*100.)

                    if metric == 'local_batch':
                        msg += '\t'
                        msg +=  'Top1 Acc: {:.2f}% ({:.2f}%) \t'\
                                'Top5 Acc: {:.2f}% ({:.2f}%)'.format(
                                top1_accs.val*100., top1_accs.avg*100.,
                                top5_accs.val*100., top5_accs.avg*100.)
                    print(msg)

    if metric == 'global':
        # Top 1/5 Acc
        if (is_master_proc):
            embeddings = torch.cat(embeddings, dim=0)
            labels = torch.cat(labels, dim=0).tolist()
            distance_matrix = get_distance_matrix(embeddings, dist_metric=cfg.LOSS.DIST_METRIC)
            topk_acc = get_topk_acc(distance_matrix, labels)
            top1_accs.update(topk_acc[0])
            top5_accs.update(topk_acc[1])

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
