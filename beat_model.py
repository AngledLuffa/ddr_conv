import random

import build_dataset

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        # Input will be 2 channels (L & R), 1x8000 audio
        # This layer will take it to 200x1x159
        # TODO: make some of these sizes, especially the channels, into parameters
        # TODO: put things on the GPU when relevant
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

def train_model(model, train_set, train_labels, dev_set, dev_labels):
    # TODO: make some of these args (and possibly use other loss / optimizer as args)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9,
                          weight_decay=0.0001)

    loss_function = nn.CrossEntropyLoss()

    # TODO: make the batch size a parameter
    batch_size=50

    num_train = train_set.shape[0]

    # TODO: make the number of epochs a parameter
    for epoch in range(20):
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
                print('[%d, %5d] average loss: %.3f' %
                      (epoch + 1, ((batch_num + 1) * batch_size), running_loss / 2000))
                epoch_loss += running_loss
                running_loss = 0.0
        # Add any leftover loss to the epoch_loss
        epoch_loss += running_loss

        # TODO: run the dev set
        
        print("Finished epoch %d.  Total loss: %f" %
              ((epoch + 1), epoch_loss))

    
if __name__ == '__main__':
    # TODO: add command line args and make the seed a possible arg
    # although it should be noted that the dataset will frequently change.
    # that might make it impossible to reproduce a dataset
    random.seed(10000)

    # make these sizes arguments as well
    train_size = 0.7
    dev_size = 0.1
    test_size = 0.2

    # also should be parameters
    train_samples = 10000
    dev_samples = 1000
    test_samples = 2000
    
    useful_simfiles, simfile_map = build_dataset.collect_simfiles()

    # we should log the test files so we know which files were
    # untouched when trying to check whether a model can identify a
    # correct BPM
    train_files, dev_files, test_files = build_dataset.split_dataset(useful_simfiles, train_size, dev_size, test_size)

    # TODO: put these tensors and the model on the GPU if available
    train_set, train_labels = build_dataset.extract_samples(train_files, simfile_map, train_samples)
    dev_set, dev_labels = build_dataset.extract_samples(dev_files, simfile_map, dev_samples)
    test_set, test_labels = build_dataset.extract_samples(test_files, simfile_map, test_samples)

    model = SimpleCNN()
    train_model(model, train_set, train_labels, dev_set, dev_labels)

