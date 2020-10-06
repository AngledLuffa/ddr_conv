import random

import build_dataset

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

    train_set, train_labels = build_dataset.extract_samples(train_files, simfile_map, train_samples)
    dev_set, dev_labels = build_dataset.extract_samples(dev_files, simfile_map, dev_samples)
    test_set, test_labels = build_dataset.extract_samples(test_files, simfile_map, test_samples)

    # TODO: now we need to build a model

