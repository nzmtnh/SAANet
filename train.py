import torch
from torch.autograd import Variable
import os, argparse
from datetime import datetime
from model.Network import SAANet
from utils.data import get_loader
from utils.func import label_edge_prediction, clip_gradient, adjust_lr
import pytorch_iou
import pytorch_fm

torch.cuda.set_device(0)

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=60, help='epoch number')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--batchsize', type=int, default=8, help='training batch size')
parser.add_argument('--trainsize', type=int, default=352,
                    help='training dataset size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=30, help='every n epochs decay learning rate')
opt = parser.parse_args()

print('Learning Rate: {}'.format(opt.lr))

model = SAANet()
model.cuda()

params = model.parameters()
optimizer = torch.optim.Adam(params, opt.lr)

image_root = './datasets/EORSSD/train/Image-train/'
gt_root = './datasets/EORSSD/train/GT-train/'

train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
total_step = len(train_loader)

CE = torch.nn.BCEWithLogitsLoss()
IOU = pytorch_iou.IOU(size_average=True)
floss = pytorch_fm.FLoss()


def train(train_loader, model, optimizer, epoch):
    model.train()

    for i, pack in enumerate(train_loader, start=1):
        optimizer.zero_grad()
        images, gts = pack
        images = Variable(images).cuda()
        gts = Variable(gts).cuda()

        edges = label_edge_prediction(gts)

        s1, s1_sig, s2, s2_sig, s3, s3_sig, s4, s4_sig, x1_edge, x2_edge, x3_edge, x4_edge = model(images)

        loss1 = CE(s1, gts) + IOU(s1_sig, gts) + floss(s1_sig, gts) + CE(x1_edge, edges)
        loss2 = CE(s2, gts) + IOU(s2_sig, gts) + floss(s2_sig, gts) + CE(x2_edge, edges)
        loss3 = CE(s3, gts) + IOU(s3_sig, gts) + floss(s3_sig, gts) + CE(x3_edge, edges)
        loss4 = CE(s4, gts) + IOU(s4_sig, gts) + floss(s4_sig, gts) + CE(x4_edge, edges)

        loss = loss1 + loss2 + loss3 + loss4

        loss.backward()

        clip_gradient(optimizer, opt.clip)

        optimizer.step()

        if i % 20 == 0 or i == total_step:
            print(
                '{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Learning Rate: {}, Loss: {:.4f}, Loss1: {:.4f}, Loss_edge: {:.4f}'.
                format(datetime.now(), epoch, opt.epoch, i, total_step,
                       opt.lr * opt.decay_rate ** (epoch // opt.decay_epoch), loss.item(), loss1.item(),
                       loss2.item()))

    # Save model every epoch
    save_path = 'models/EORSSD/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if (epoch + 1) >= 5:
        torch.save(model.state_dict(), save_path + 'SAANet_EORSSD.pth' + '.%d' % epoch)


# Main loop
print("Let's go!")
if __name__ == '__main__':
    for epoch in range(1, opt.epoch):
        adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        train(train_loader, model, optimizer, epoch)