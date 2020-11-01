import torch
import numpy as np

from .models import Detector, save_model
from .utils import load_detection_data
from . import dense_transforms
import torch.utils.tensorboard as tb

# def accuracy(img, label):
#     return (img.max(1)[1] == label).float().mean()

def train(args):
    from os import path
    model = Detector()
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    """
    Your code here, modify your HW3 code
    """
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = Detector().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
    loss = torch.nn.BCEWithLogitsLoss()
    num_epoch = 250
    
    train_data = load_detection_data('dense_data/train')
    valid_data = load_detection_data('dense_data/valid')

    global_step = 0
    best_vacc = 0
    best_vloss = 1
    for epoch in range(num_epoch):
        print(epoch)
        model.train()
        acc_vals = []
        loss_vals = []
        for img, label, ec in train_data:
            # img = dense_transforms.ToHeatmap(img)
            # label = dense_transforms.ToHeatmap(label)
            img, label = img.to(device), label.to(device)

            logit = model(img)
            label = label
            #Maybe float?
            loss_val = loss(logit, label)
            loss_vals.append(loss_val.item())
            # acc_val = accuracy(logit, label)

            if train_logger is not None:
                train_logger.add_scalar('loss', loss_val, global_step)
            # acc_vals.append(acc_val)

            optimizer.zero_grad()
            loss_val.backward()
            # print(loss_val.item())
            optimizer.step()
            global_step += 1
        avg_loss = sum(loss_vals)/len(loss_vals)
        if(avg_loss < best_vloss):
            print("saving!")
            best_vloss = avg_loss
            save_model(model)

        # avg_acc = sum(acc_vals) / len(acc_vals)
        avg_acc = 0
        

        if train_logger:
            train_logger.add_scalar('accuracy', avg_acc, global_step)

        model.eval()
        acc_vals = []



        # for img, label, ec in valid_data:
        #     img, label = img.to(device), label.to(device)
        #     # acc_vals.append(accuracy(model(img), label.long()).detach().cpu().numpy())
        # # avg_vacc = sum(acc_vals) /  len(acc_vals)
        # avg_vacc = 0
        # if(avg_vacc > best_vacc):
        #     print(global_step, "accuracy", avg_vacc)
        #     best_vacc = avg_vacc
        #     save_model(model)

        # if valid_logger:
        #     valid_logger.add_scalar('accuracy', avg_vacc, global_step)

        # if valid_logger is None or train_logger is None:
        #     print('epoch %-3d \t acc = %0.3f \t val acc = %0.3f' % (epoch, avg_acc, avg_vacc))

    # save_model(model)


def log(logger, imgs, gt_det, det, global_step):
    """
    logger: train_logger/valid_logger
    imgs: image tensor from data loader
    gt_det: ground-truth object-center maps
    det: predicted object-center heatmaps
    global_step: iteration
    """
    logger.add_images('image', imgs[:16], global_step)
    logger.add_images('label', gt_det[:16], global_step)
    logger.add_images('pred', torch.sigmoid(det[:16]), global_step)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here

    args = parser.parse_args()
    train(args)
