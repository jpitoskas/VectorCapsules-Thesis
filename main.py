
import argparse
import torch
# from __future__ import print_function
#
# import torch
# import torch.nn as nn
import torch.nn.functional as F
# import math
#

# from torch.autograd import Variable
#
from tqdm import tqdm
# import os
#
# import argparse
import torch.optim as optim
from torch.optim import lr_scheduler
# from torchvision import datasets, transforms
from torch.autograd import Variable
import logging
#
# import logging

from dataloaders import load_dataset
from model import CapsNet, ReconstructionNet, CapsNetWithReconstruction, MarginLoss


if __name__ == '__main__':

    # import argparse
    # import torch.optim as optim
    # from torchvision import datasets, transforms
    # from torch.autograd import Variable

    logging.basicConfig(level=logging.INFO, format='%(message)s')

    # Training settings
    parser = argparse.ArgumentParser(description='CapsNet with MNIST')
    parser.add_argument('--dataset', default='mnist')
    parser.add_argument('--batch_size', type=int, default=1, metavar='N')
    # parser.add_argument('--batch-size', type=int, default=128, metavar='N',
    #                     help='input batch size for training (default: 64)')
    # parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
    #                     help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--routing_iterations', type=int, default=3)
    parser.add_argument('--with_reconstruction', action='store_true', default=True)
    # parser.add_argument('--n_epochs', type=int, default=300)
    # parser.add_argument('--weight_decay', type=float, default=1e-6)
    # parser.add_argument('--routing_iter', type=int, default=3)
    # parser.add_argument('--padding', type=int, default=4)
    parser.add_argument('--brightness', type=float, default=0)
    parser.add_argument('--contrast', type=float, default=0)
    # parser.add_argument('--patience', default=1e+4)
    parser.add_argument('--crop_dim', type=int, default=32)
    # parser.add_argument('--arch', nargs='+', type=int, default=[64,16,16,16,5]) # architecture n caps
    parser.add_argument('--num_workers', type=int, default=1)
    # parser.add_argument('--load_checkpoint_dir', default='../experiments')
    # parser.add_argument('--test_affnist', dest='test_affNIST', action='store_true')
    # parser.add_argument('--routing', default='vb', help='routing algorithm (vb, naive)')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)


    train_loader, valid_loader, test_loader = load_dataset(args)







    model = CapsNet(args)

    if args.with_reconstruction:
        reconstruction_model = ReconstructionNet(args, model.digitCaps.output_dim)
        reconstruction_alpha = 0.0005
        model = CapsNetWithReconstruction(model, reconstruction_model)

    if args.cuda:
        model.cuda()


    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, patience=15, min_lr=1e-6)


    loss_fn = MarginLoss(0.9, 0.1, 0.5)


    def train(epoch):
        model.train()
        train_loss = 0
        correct = 0
        print(f"Train Epoch:{epoch}:")
        for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target, requires_grad=False)
            optimizer.zero_grad()
            if args.with_reconstruction:
                output, probs = model(data, target)
                reconstruction_loss = F.mse_loss(output, data.view(-1, output.size(1)))
                margin_loss = loss_fn(probs, target)
                loss = reconstruction_alpha * reconstruction_loss + margin_loss
                train_loss += loss
            else:
                output, probs = model(data)
                loss = loss_fn(probs, target)
                train_loss += loss
            pred = probs.data.max(1, keepdim=True)[1]  # get the index of the max probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

            loss.backward()
            optimizer.step()

        train_loss /= len(train_loader.dataset)
        print('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
            train_loss, correct, len(train_loader.dataset),
            100. * correct / len(train_loader.dataset)))
            # if batch_idx % args.log_interval == 0:
        # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #     epoch, batch_idx * len(data), len(train_loader.dataset),
        #            100. * batch_idx / len(train_loader), loss.item()))

    def test(split="Validation"):
        # split = "Validation" or "Test"

        model.eval()
        test_loss = 0
        correct = 0

        loader = valid_loader if split=="Validation" else test_loader

        with torch.no_grad():
            for data, target in loader:
                if args.cuda:
                    data, target = data.cuda(), target.cuda()

                if args.with_reconstruction:
                    output, probs = model(data, target)
                    reconstruction_loss = F.mse_loss(output, data.view(-1, output.size(1)), size_average=False).item()
                    test_loss += loss_fn(probs, target, size_average=False).item()
                    test_loss += reconstruction_alpha * reconstruction_loss
                else:
                    output, probs = model(data)
                    test_loss += loss_fn(probs, target, size_average=False).item()

                pred = probs.data.max(1, keepdim=True)[1]  # get the index of the max probability
                correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        test_loss /= len(loader.dataset)
        print('{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            split, test_loss, correct, len(loader.dataset),
            100. * correct / len(loader.dataset)))
        return test_loss


    logging.info('\nTrain: {} - Validation: {} - Test: {}'.format(
        len(train_loader.dataset), len(valid_loader.dataset),
        len(test_loader.dataset)))

    for epoch in range(1, args.epochs + 1):
        train(epoch)
        valid_loss = test("Validation")
        scheduler.step(valid_loss)
        test_loss = test("Test")
        # torch.save(model.state_dict(),
        #            '{:03d}_model_dict_{}routing_reconstruction{}.pth'.format(epoch, args.routing_iterations,
        #                                                                          args.with_reconstruction))
