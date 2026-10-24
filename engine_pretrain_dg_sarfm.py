import math
import sys
from typing import Iterable

import torch
from torch import nn

import util.misc as misc
import util.lr_sched as lr_sched


def get_tau(step, total_steps, base_tau=0.996):
    return 1 - (1 - base_tau) * (math.cos(math.pi * step / total_steps) + 1) / 2


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
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

    for data_iter_step, (samples, label) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
            
        current_step = epoch * len(data_loader) + data_iter_step
        tau = get_tau(current_step, args.epochs * len(data_loader), base_tau=args.momentum)

        samples[0] = [samples.to(device, non_blocking=True) for samples in samples[0]]
        samples[1] = samples[1].to(device, non_blocking=True)
        samples[2] = samples[2].to(device, non_blocking=True)
        samples[3] = samples[3].to(device, non_blocking=True)
        # print(torch.cuda.memory_allocated() / 1024 ** 2)
        criterion = nn.CrossEntropyLoss().cuda()
        with torch.cuda.amp.autocast():
            loss = model(input_img=samples[0], mode='loss', output_img=samples[1], contrast_img=samples[2], ious=samples[3], m=tau)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=1.0, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

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


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
