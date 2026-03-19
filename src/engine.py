import math
import sys
from typing import Iterable, Optional
import torch
from timm.data import Mixup
from timm.utils import accuracy
import util.misc as misc
import util.lr_sched as lr_sched
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from util.misc import NativeScalerWithGradNormCount as NativeScaler
import copy
from collections import OrderedDict
import torch.nn.functional as F

def weight_delivery(model, epoch, momentum_schedule):
    with torch.no_grad():
        m = momentum_schedule[epoch]  # momentum parameter
        student = model
        all_keys = list(student.state_dict().keys())
        new_dict = OrderedDict()
        for key in all_keys:
            if key.startswith('backbone.'):
                new_dict[key] = student.state_dict()[key]
            elif key.startswith('mmae_core_block.levels.') and not 'norm.' in key:
                new_dict[key] = student.state_dict()[key]
            else:
                new_dict[key] = student.state_dict()[key]
    return new_dict


def EMA_process(model, model_teacher, epoch, momentum_schedule):
    student_crop = copy.deepcopy(model_teacher)
    m = momentum_schedule[epoch]
    with torch.no_grad():
        new_dict = weight_delivery(model, epoch, momentum_schedule)
        student_crop.load_state_dict(new_dict, strict=False)
        for param_q, param_k in zip(student_crop.parameters(), model_teacher.parameters()):
            param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

def forward_learning_loss(loss_pred, mask, loss_target, relative=False):
    if relative:

        labels_positive = loss_target.unsqueeze(1) > loss_target.unsqueeze(2)
        labels_negative = loss_target.unsqueeze(1) < loss_target.unsqueeze(2)
        labels_valid = labels_positive + labels_negative
        loss_matrix = loss_pred.unsqueeze(1) - loss_pred.unsqueeze(2)
        loss = - labels_positive.int() * torch.log(torch.sigmoid(loss_matrix) + 1e-6) \
               - labels_negative.int() * torch.log(1 - torch.sigmoid(loss_matrix) + 1e-6)

        return loss.sum() / labels_valid.sum()

    else:
        mean = loss_target.mean(dim=1, keepdim=True)
        var = loss_target.var(dim=1, keepdim=True)
        loss_target = (loss_target - mean) / (var + 1.e-6) ** .5
        loss = (loss_pred - loss_target) ** 2
        loss = loss.mean()
        return loss



def pretrain_one_epoch(model: torch.nn.Module, model_teacher: torch.nn.Module,
                       data_loader: Iterable, optimizer: torch.optim.Optimizer,
                       device: torch.device, epoch: int, loss_scaler, amp_autocast,
                       log_writer=None,
                       model_without_ddp=None,
                       momentum_schedule=None, ema_op=None, is_ema=None,
                       args=None, loss_pred_s=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('steps', misc.SmoothedValue(window_size=1, fmt='{value:.0f}'))
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    steps_of_one_epoch = len(data_loader)

    for data_iter_step, data in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        steps = steps_of_one_epoch * epoch + data_iter_step
        metric_logger.update(steps=int(steps))


        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
        samples, _ = data
        samples = samples.to(device, non_blocking=True)

        with amp_autocast():
            with torch.no_grad():
                x_out_teacher, pre_pred_unmix_x_rec_tea, loss_pred_t = model_teacher(samples)
            loss, loss_matrix, _, _, unmix_x_rec, pre_pred_unmix_x_rec_stu, loss_pred_stu, mask_ratio = model(samples, mask_ratio=args.mask_ratio, gene_mask=None, gene_ids_restore=None, loss_pred=loss_pred_s)
            loss_pred_s = loss_pred_stu

            mean = pre_pred_unmix_x_rec_tea.mean(dim=-1, keepdim=True)
            var = pre_pred_unmix_x_rec_tea.var(dim=-1, keepdim=True)
            pre_pred_unmix_x_rec_tea = (pre_pred_unmix_x_rec_tea - mean) / (var + 1.e-6)**.5

        loss_2 = 1 - F.cosine_similarity(pre_pred_unmix_x_rec_tea, pre_pred_unmix_x_rec_stu).mean()
        if args.learning_loss:
            loss_learn = forward_learning_loss(loss_pred_stu, mask=None, loss_target=loss_matrix, relative=True)
            loss_learn_value = loss_learn.item() * 0.1
            loss = loss + 0.1 * loss_learn

        loss += loss_2
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        if isinstance(loss_scaler, NativeScaler):
            loss /= accum_iter
            loss_scaler(loss, optimizer, parameters=model.parameters(),
                        update_grad=(data_iter_step + 1) % accum_iter == 0)
            if (data_iter_step + 1) % accum_iter == 0:
                optimizer.zero_grad()
        else:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss2=loss_2.item())
        if args.learning_loss:
            metric_logger.update(loss_learn=loss_learn_value)



        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)
        if args.output_dir and steps % args.save_steps_freq == 0 and epoch > 0:
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch, name='step' + str(steps))

        EMA_process(model, model_teacher, epoch, momentum_schedule)


    # if not 'per_ite' in ema_op:
    #     if is_ema is True:
    #         #Process EMA for the teacher every epoch.
    #         print('ema')
    #         EMA_process(model, model_teacher, epoch, momentum_schedule)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, loss_pred_s


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, amp_autocast, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    pred_all = []
    target_all = []

    for data_iter_step, data in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples, targets = data
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with amp_autocast():
            outputs = model(samples)
            loss = criterion(outputs, targets)

        _, pred = outputs.topk(1, 1, True, True)
        pred = pred.t()
        pred_all.extend(pred[0].cpu())
        target_all.extend(targets.cpu())

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        if isinstance(loss_scaler, NativeScaler):
            loss /= accum_iter
            loss_scaler(loss, optimizer, clip_grad=max_norm,
                        parameters=model.parameters(), create_graph=False,
                        update_grad=(data_iter_step + 1) % accum_iter == 0)
            if (data_iter_step + 1) % accum_iter == 0:
                optimizer.zero_grad()
        else:
            loss.backward()
            if max_norm is not None and max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}




@torch.no_grad()
def evaluate(data_loader, model, device, if_stat=False, visualize=True, save_vis_path="attn_vis.png"):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    pred_all = []
    target_all = []

    for batch in metric_logger.log_every(data_loader, 10, header):
        images, target = batch
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        torch.cuda.synchronize()

        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        _, pred = output.topk(1, 1, True, True)
        pred = pred.t()

        pred_all.extend(pred[0].cpu())
        target_all.extend(target.cpu())

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        torch.cuda.synchronize()

        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item() / 100, n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item() / 100, n=batch_size)


    macro = precision_recall_fscore_support(target_all, pred_all, average='weighted')
    cm = confusion_matrix(target_all, pred_all)

    # compute acc, precision, recall, f1 for each class
    acc = accuracy_score(target_all, pred_all)
    pre_per_class, rec_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(target_all,
                                                                                                    pred_all,
                                                                                                    average=None)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.4f} Acc@5 {top5.global_avg:.4f} loss {losses.global_avg:.4f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
    print(
        '* Pre {macro_pre:.4f} Rec {macro_rec:.4f} F1 {macro_f1:.4f}'
        .format(macro_pre=macro[0], macro_rec=macro[1],
                macro_f1=macro[2]))

    test_state = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    test_state['weighted_pre'] = macro[0]
    test_state['weighted_rec'] = macro[1]
    test_state['weighted_f1'] = macro[2]
    test_state['cm'] = cm
    test_state['acc'] = acc
    test_state['pre_per_class'] = pre_per_class
    test_state['rec_per_class'] = rec_per_class
    test_state['f1_per_class'] = f1_per_class
    test_state['support_per_class'] = support_per_class

    return test_state


