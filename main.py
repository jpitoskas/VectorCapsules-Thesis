
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
import os

import matplotlib.pyplot as plt

from dataloaders import load_dataset
from model import CapsNet, ReconstructionNet, CapsNetWithReconstruction, MarginLoss
import warnings
import csv

warnings.filterwarnings("ignore")


if __name__ == '__main__':

    # import argparse
    # import torch.optim as optim
    # from torchvision import datasets, transforms
    # from torch.autograd import Variable


    experiments_dir = os.path.join(os.getcwd(),'experiments')
    model_dir_prefix = "Model_"


    if os.path.exists(experiments_dir):
        model_dirs = [model_dir if os.path.isdir(os.path.join(experiments_dir, model_dir)) else None for model_dir in os.listdir(experiments_dir)]
        model_dirs = list(filter(None, model_dirs))
        ids = [int(dd.replace(model_dir_prefix,"")) if (model_dir_prefix) in dd and dd.replace(model_dir_prefix,"").isnumeric() else None for dd in model_dirs]
        ids = list(filter(None, ids))
        new_id = str(max(ids) + 1) if ids else "1"
    else:
        os.mkdir(experiments_dir)
        new_id = "1"

    new_model_dir = os.path.join(experiments_dir, model_dir_prefix + new_id)
    print(new_model_dir)
    os.mkdir(new_model_dir)


    # reset root logger
    [logging.root.removeHandler(handler) for handler in logging.root.handlers[:]]
    # info logger for saving command line outputs during training
    logging.basicConfig(level=logging.INFO, format='%(message)s',
         handlers=[logging.FileHandler(os.path.join(new_model_dir, f'trainlogs_{new_id}.log')),
            logging.StreamHandler()])


    # Training settings
    parser = argparse.ArgumentParser(description='Vector CapsNet')
    parser.add_argument('--dataset', default='mnist', choices=['mnist', 'smallnorb', 'fashionmnist', 'svhn', 'cifar10'],
                        help='dataset (mnist, smallnorb, fashionmnist, svhn, cifar10)')
    parser.add_argument('--batch_size', type=int, default=1, metavar='N')
    # parser.add_argument('--batch-size', type=int, default=128, metavar='N',
    #                     help='input batch size for training (default: 64)')
    # parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
    #                     help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--scheduler', default=None, choices=['plateau', 'exponential'], help='scheduler (plateau, exponential)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # parser.add_argument('--log-interval', type=int, default=10, metavar='N',
    #                     help='how many batches to wait before logging training status')
    parser.add_argument('--with_reconstruction', action='store_true', default=True)
    # parser.add_argument('--n_epochs', type=int, default=300)
    # parser.add_argument('--weight_decay', type=float, default=1e-6)
    # parser.add_argument('--routing_iter', type=int, default=3)
    parser.add_argument('--padding', type=int, default=4)
    parser.add_argument('--brightness', type=float, default=0)
    parser.add_argument('--contrast', type=float, default=0)
    # parser.add_argument('--patience', default=1e+4)
    parser.add_argument('--crop_dim', type=int, default=32)
    # parser.add_argument('--arch', nargs='+', type=int, default=[64,16,16,16,5]) # architecture n caps
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--extra_conv', action='store_true', default=False)
    # parser.add_argument('--load_checkpoint_dir', default='../experiments')
    parser.add_argument('--test_affnist', dest='test_affNIST', action='store_true')
    # parser.add_argument('--routing', default='vb', help='routing algorithm (vb, naive)')
    parser.add_argument('--routing_module', default='WeightedAverageRouting', choices=['AgreementRouting', 'WeightedAverageRouting', 'DropoutWeightedAverageRouting', 'SubsetRouting', 'RansacRouting'],
                        help='Routing algorithm (AgreementRouting, WeightedAverageRouting, DropoutWeightedAverageRouting, SubsetRouting, RansacRouting)')
    parser.add_argument('--routing_iterations', type=int, default=3, help='if AgreementRouting is chosen')
    parser.add_argument('--dropout_probability', type=float, default=0.2, help='if DropoutWeightedAverageRouting is chosen')
    parser.add_argument('--n_hypotheses', type=int, default=10, help='if RansacRouting is chosen')
    parser.add_argument('--subset_fraction', type=float, default=0.8, help='if SubsetRouting or RansacRouting is chosen')
    args = parser.parse_args()



    device = torch.device('cuda:0' if not args.no_cuda and torch.cuda.is_available() else 'cpu')
    args.cuda = not args.no_cuda and torch.cuda.is_available()




    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)



    train_loader, valid_loader, test_loader = load_dataset(args)



    logging.info('\n'.join(f'{k}: {v}' for k, v in vars(args).items()))


    model = CapsNet(args)


    if torch.cuda.device_count() > 1:
        logging.info('\nUsing', torch.cuda.device_count(), 'GPU(s).\n')
        model = nn.DataParallel(model)

    model.to(device)


    if args.with_reconstruction:
        reconstruction_model = ReconstructionNet(args, model.digitCaps.output_dim)
        reconstruction_alpha = 0.0005
        model = CapsNetWithReconstruction(model, reconstruction_model)

    model.to(device)

    # if args.cuda:
    #     model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if args.scheduler == "plateau":
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, patience=15, min_lr=1e-6)
    elif args.scheduler == "exponential":
        scheduler = lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.96)


    loss_fn = MarginLoss(0.9, 0.1, 0.5)


    def train(epoch):
        model.train()
        train_loss = 0
        correct = 0
        logging.info("\n===================================================================\n")
        logging.info(f"Train Epoch:{epoch}:\n")
        for batch_idx, (data, target) in enumerate(tqdm(train_loader)):

            # total_memory, used_memory, free_memory = map(int, os.popen('free -t -m').readlines()[-1].split()[1:])
            # print("RAM memory % used:", round((used_memory/total_memory) * 100, 2))

            # import psutil
            # print('RAM memory % used:', psutil.virtual_memory()[2])

            # exit()
            # if args.cuda:
            #     data, target = data.cuda(), target.cuda()
            data, target = data.to(device), target.to(device)
            data, target = Variable(data), Variable(target, requires_grad=False)
            optimizer.zero_grad()
            if args.with_reconstruction:
                output, probs = model(data, target)
                # reconstruction_loss = F.mse_loss(output, data.view(-1, output.size(1))).item()
                reconstruction_loss = F.mse_loss(output, data.view(-1, output.size(1)))
                # print(output.shape, data.view(-1, output.size(1)).shape)
                # dif = output - data.view(-1, output.size(1))
                # dif = torch.square(dif)
                # print(torch.mean(dif, dim=1).mean())
                # print(reconstruction_loss)
                # exit()
                margin_loss = loss_fn(probs, target)
                loss = reconstruction_alpha * reconstruction_loss + margin_loss
                train_loss += loss * data.size(0)
                # print(loss)
            else:
                output, probs = model(data)
                loss = loss_fn(probs, target)
                train_loss += loss * data.size(0)
            pred = probs.data.max(1, keepdim=True)[1]  # get the index of the max probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

            loss.backward()
            optimizer.step()

        train_loss /= len(train_loader.dataset)
        train_accuracy = 100. * correct / len(train_loader.dataset)
        logging.info('Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
            train_loss, correct, len(train_loader.dataset),
            train_accuracy))

        return train_loss.item(), train_accuracy.item()
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
                # if args.cuda:
                #     data, target = data.cuda(), target.cuda()
                data, target = data.to(device), target.to(device)

                if args.with_reconstruction:
                    output, probs = model(data, target)
                    # reconstruction_loss = F.mse_loss(output, data.view(-1, output.size(1)), size_average=True).item()
                    # margin_loss = loss_fn(probs, target, size_average=True).item()
                    reconstruction_loss = F.mse_loss(output, data.view(-1, output.size(1)), size_average=True)
                    margin_loss = loss_fn(probs, target, size_average=True)
                    loss = reconstruction_alpha * reconstruction_loss + margin_loss
                    test_loss += loss * data.size(0)
                    # print(loss)
                    # print(len(loader))
                    # print(data.shape)
                    # print(output.shape, data.view(-1, output.size(1)).shape)
                    # dif = output - data.view(-1, output.size(1))
                    # dif = torch.square(dif)
                    # print('\n', torch.mean(dif, dim=1).mean())
                    # print('\n', reconstruction_loss)
                    # print('\n', margin_loss)
                    # print()
                    # exit()

                else:
                    output, probs = model(data)
                    margin_loss += loss_fn(probs, target, size_average=False)
                    # margin_loss += loss_fn(probs, target, size_average=False).item()
                    loss = margin_loss
                    test_loss += loss * data.size(0)

                pred = probs.data.max(1, keepdim=True)[1]  # get the index of the max probability
                correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        test_loss /= len(loader.dataset)
        test_accuracy = 100. * correct / len(loader.dataset)
        logging.info('{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
            split, test_loss, correct, len(loader.dataset),
            test_accuracy))
        return test_loss.item(), test_accuracy.item()


    logging.info('\n===================================================================\n')

    logging.info('Train: {} - Validation: {} - Test: {}'.format(
        len(train_loader.dataset), len(valid_loader.dataset),
        len(test_loader.dataset)))

    train_losses = []
    train_accuracies = []
    valid_losses = []
    valid_accuracies = []
    test_losses = []
    test_accuracies = []
    for epoch in range(1, args.epochs + 1):

        train_loss, train_accuracy = train(epoch)
        valid_loss, valid_accuracy = test("Validation")
        if args.scheduler:
            scheduler.step(valid_loss)
        test_loss, test_accuracy = test("Test")

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        valid_losses.append(valid_loss)
        valid_accuracies.append(valid_accuracy)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)


    logging.info('\n===================================================================\n')
    logging.info(f'- Best Train Accuracy is {round(max(train_accuracies), 2)}% (i.e. {round(100.0-round(max(train_accuracies), 2), 2)}% Train Error) on Epoch {train_accuracies.index(max(train_accuracies))+1}.')
    logging.info(f'- Best Validation Accuracy is {round(max(valid_accuracies), 2)}% (i.e. {round(100.0-round(max(valid_accuracies), 2), 2)}% Validation Error) on Epoch {valid_accuracies.index(max(valid_accuracies))+1}.')
    logging.info(f'- Best Test Accuracy is {round(max(test_accuracies), 2)}% (i.e. {round(100.0-round(max(test_accuracies), 2), 2)}% Test Error) on Epoch {test_accuracies.index(max(test_accuracies))+1}.')

    logging.info('\n===================================================================\n')
    logging.info(f'- Best Train Loss is {round(min(train_losses), 4)} on Epoch {train_losses.index(min(train_losses))+1}.')
    logging.info(f'- Best Validation Loss is {round(min(valid_losses), 4)} on Epoch {valid_losses.index(min(valid_losses))+1}.')
    logging.info(f'- Best Test Loss is {round(min(test_losses), 4)} on Epoch {test_losses.index(min(test_losses))+1}.')

    logging.info('\n===================================================================\n')


    accuracies = [train_accuracies, valid_accuracies, test_accuracies]
    losses = [train_losses, valid_losses, test_losses]
    with open(os.path.join(new_model_dir, "accuracies.csv"), "w") as f1, open(os.path.join(new_model_dir, "losses.csv"), "w") as f2:
        wr1 = csv.writer(f1)
        wr2 = csv.writer(f2)
        wr1.writerows(accuracies)
        wr2.writerows(losses)


    plt.figure(figsize=(10,7))
    plt.title("Training and Validation Accuracy")
    plt.plot(train_accuracies, label="train")
    plt.plot(valid_accuracies, label="val")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(new_model_dir, f'accuracy_{new_id}.png'))


    plt.figure(figsize=(10,7))
    plt.title("Training and Validation Loss")
    plt.plot(train_losses, label="train")
    plt.plot(valid_losses, label="val")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(new_model_dir, f'loss_{new_id}.png'))


    # reset root logger
    [logging.root.removeHandler(handler) for handler in logging.root.handlers[:]]

        # torch.save(model.state_dict(),
        #            '{:03d}_model_dict_{}routing_reconstruction{}.pth'.format(epoch, args.routing_iterations,
        #                                                                          args.with_reconstruction))
