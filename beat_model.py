import argparse
import logging
import random

import build_dataset

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

logger = logging.getLogger('ddr')
logger.setLevel(logging.INFO)

log_handler = logging.StreamHandler()
log_formatter = logging.Formatter(fmt="%(asctime)s %(levelname)s: %(message)s",
                                  datefmt='%Y-%m-%d %H:%M:%S')
log_handler.setFormatter(log_formatter)
logger.addHandler(log_handler)

def set_random_seed(seed, cuda):
    """
    Set a random seed on all of the things which might need it.
    torch, np, python random, and torch.cuda
    """
    if seed is None:
        seed = random.randint(0, 1000000000)

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)
    return seed

def parse_args():
    """
    Add arguments for building the classifier.
    Parses command line args and returns the result.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', default=None, type=int, help='Random seed for model')

    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to run training')

    parser.add_argument('--weight_decay', default=0.0001, type=float, help='Weight decay (eg, l2 reg) to use in the optimizer')
    parser.add_argument('--lr', default=0.001, type=float, help='Learning rate to use in the optimizer')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum to use in the optimizer')

    parser.add_argument('--batch_size', default=50, type=int, help='Batch size when training')

    parser.add_argument('--cuda', action='store_true', help='Use CUDA for training/testing', default=torch.cuda.is_available())
    parser.add_argument('--cpu', action='store_false', help='Ignore CUDA.', dest='cuda')

    parser.add_argument('--base_songs_dir', default='c:/Users/horat/Documents/DDR/Songs', help='Where to find the folders with the songs')
    parser.add_argument('--folders_file', default='folders.txt', help='A file with a list of the subdirectories containing usable songs')

    parser.add_argument('--train_samples', default=20000, type=int, help='How many samples to extract from the training files')
    parser.add_argument('--dev_samples',   default= 1000, type=int, help='How many samples to extract from the dev files')
    parser.add_argument('--test_samples',  default= 2000, type=int, help='How many samples to extract from the test files')

    parser.add_argument('--train_size', default=0.7, type=float, help='How many files to use for training')
    parser.add_argument('--dev_size',   default=0.1, type=float, help='How many files to use for dev')
    parser.add_argument('--test_size',  default=0.2, type=float, help='How many files to use for test')

    return parser.parse_args()


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        # Input will be 2 channels (L & R), 1x8000 audio
        # This layer will take it to 200x1x159
        # TODO: make some of these sizes, especially the channels, into parameters
        self.base_layer = nn.Conv2d(in_channels=2,
                                    out_channels=200,
                                    stride=(1, 50),
                                    kernel_size=(1, 100))

        # This will make it 200x1x45
        self.middle_layer = nn.Conv2d(in_channels=200,
                                      out_channels=200,
                                      stride=(1, 3),
                                      kernel_size=(1, 27))

        # output layer: should be 1x1x2 after this
        self.output_layer = nn.Conv2d(in_channels=200,
                                      out_channels=2,
                                      kernel_size=(1, 45))

        # TODO: make the dropout factor a parameter as well
        self.dropout = nn.Dropout(0.5)


    def forward(self, inputs):
        base = self.dropout(F.relu(self.base_layer(inputs)))
        middle = self.dropout(F.relu(self.middle_layer(base)))
        out = self.output_layer(middle)
        out = out.squeeze()
        return out

def score_dataset(model, args, dataset, labels):
    model.eval()

    num_data = dataset.shape[0]

    correct = 0
    batch_size=args.batch_size
    for batch_start in range(0, num_data, batch_size):
        batch_end = min(batch_start + batch_size, num_data)
        batch = dataset[batch_start:batch_end, :, :, :]
        correct_labels = labels[batch_start:batch_end]

        output = model(batch)
        # TODO: there is probably a convenient vector math way of doing this
        for i in range(batch_end - batch_start):
            predicted = torch.argmax(output[i])
            predicted_label = predicted.item()
            if predicted_label == correct_labels[i]:
                correct = correct + 1

    return correct
    
def train_model(model, args, train_set, train_labels, dev_set, dev_labels):
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)

    loss_function = nn.CrossEntropyLoss()

    batch_size=args.batch_size

    num_train = train_set.shape[0]
    num_dev = dev_set.shape[0]

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        epoch_loss = 0.0

        batch_starts = list(range(0, num_train, batch_size))
        random.shuffle(batch_starts)

        for batch_num, batch_start in enumerate(batch_starts):
            batch_end = min(batch_start + batch_size, num_train)
            batch = train_set[batch_start:batch_end, :, :, :]
            labels = train_labels[batch_start:batch_end]

            # zero the parameter gradients
            optimizer.zero_grad()

            outputs = model(batch)
            batch_loss = loss_function(outputs, labels)
            batch_loss.backward()
            optimizer.step()

            running_loss += batch_loss.item()
            if ((batch_num + 1) * batch_size) % 2000 < batch_size: # print every 2000 items
                logger.info('[%d, %5d] average loss: %.3f' %
                            (epoch + 1, ((batch_num + 1) * batch_size), running_loss / 2000))
                epoch_loss += running_loss
                running_loss = 0.0
        # Add any leftover loss to the epoch_loss
        epoch_loss += running_loss

        correct = score_dataset(model, args, dev_set, dev_labels)
        logger.info("Finished epoch %d.  Correct: %d of %d.  Total loss: %f" %
                    ((epoch + 1), correct, num_dev, epoch_loss))

    
def main():
    args = parse_args()
    
    # It should be noted that the dataset will frequently change.
    # That might make it impossible to reproduce a dataset
    seed = set_random_seed(args.seed, args.cuda)
    logger.info("Using random seed: %d" % seed)

    useful_simfiles, simfile_map = build_dataset.collect_simfiles(args.base_songs_dir, args.folders_file)

    # we should log the test files so we know which files were
    # untouched when trying to check whether a model can identify a
    # correct BPM
    train_files, dev_files, test_files = build_dataset.split_dataset(useful_simfiles, args.train_size, args.dev_size, args.test_size)

    train_set, train_labels = build_dataset.extract_samples(train_files, simfile_map, args.train_samples, args.cuda)
    dev_set, dev_labels = build_dataset.extract_samples(dev_files, simfile_map, args.dev_samples, args.cuda)
    test_set, test_labels = build_dataset.extract_samples(test_files, simfile_map, args.test_samples, args.cuda)

    model = SimpleCNN()

    if args.cuda:
        model.cuda()

    train_model(model, args, train_set, train_labels, dev_set, dev_labels)

    # TODO: need to be able to save & load a model
    # stanza's models have some good examples of this

if __name__ == '__main__':
    main()

