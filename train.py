import logging
from tqdm import tqdm
from torch.autograd import Variable
import torch.nn.functional as F

def train(epoch, model, dataloaders, optimizer, loss_fn, args):

    model.train()
    train_loss = 0
    correct = 0

    train_loader, _, _ = dataloaders
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
