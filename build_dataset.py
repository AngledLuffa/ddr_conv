"""
plan to build a dataset:

get the directories where songs exist
randomly select a subset of the songs
randomly select beats and non-beats
  - beats: downbeats only
  - non-beats: a variety of 8ths, 16ths, 12ths, and random offsets

notes on how to build the dataset:
  split into train,dev,test by song title
  this way the tool is more likely to generalize by song
"""

import os
import random

import torch
import torchaudio

import simfile

extensions = [".ssc", ".sm", ".dwi"]
known_extensions = [".sm"]

def find_simfiles(paths, already_searched=None):
    """
    Finds all the simfiles under the given paths.

    Does so recursively, so you can give it a folder with a bunch of
    songs in it or even give it the top level directory.

    paths: a list of string paths to search
    already_searched: a set of paths to ignore

    return: a list of simfiles
    """
    if isinstance(paths, str):
        paths = [paths]
    if already_searched is None:
        # this is useful in case there are symlinks
        already_searched = set()
    simfiles = []
    for path in paths:
        simfiles.extend([f.path for f in os.scandir(path)
                         if os.path.splitext(f.path)[1] in extensions])
    subdirs = []
    for path in paths:
        subdirs.extend([f.path for f in os.scandir(path) if f.is_dir()])
    for subdir in subdirs:
        subdir = os.path.abspath(subdir)
        if subdir in already_searched:
            continue
        already_searched.add(subdir)
        simfiles.extend(find_simfiles(subdir, already_searched))
    return simfiles

def find_duplicates(simfiles):
    """
    Looks for simfile directories with multiple simfiles.

    Duplicates are annoying.

    Does so by looking at the path and ignoring the simfile name
    itself.  The assumption is that earlier steps found all
    .dwi/.sm/.ssc files in those paths
    """
    found_duplicate = False
    known_directories = {}
    for filename in simfiles:
        directory, filename = os.path.split(filename)
        if directory in known_directories:
            print("Found two simfiles in %s: %s and %s" %
                  (directory, known_directories[directory], filename))
            found_duplicate = True
        known_directories[directory] = filename
    if found_duplicate:
        raise RuntimeError("Please eliminate any duplicates before proceeding")

def get_candidate_directories(folders_file):
    # TODO: make this a bit more general
    return [x.strip() for x in open(folders_file).readlines()]

def print_unknown_types(simfiles):
    """
    Print any simfiles which had an unknown extension
    """
    print("Unknown: \n%s" % "\n  ".join(f for f in simfiles
                                        if os.path.splitext(f)[1] not in known_extensions))
    

def is_useful(sim):
    # currently we don't fully trust any simfile with multiple BPMs
    # TODO: bpms with harmonics of a base BPM should be fine
    if len(sim.bpms) > 1:
        return False
    # Currently Windows torchaudio is not compatible with mp3.  Either
    # needs sox or an update to soundfile which includes mp3 support
    # (such a thing is apparently planned)
    if os.path.splitext(sim.music)[1] == '.mp3':
        return False
    return True

def filter_known_types(simfiles):
    simfiles = [f for f in simfiles
                if os.path.splitext(f)[1] in known_extensions]
    return simfiles

def collect_simfiles():
    simfile_root_directories = get_candidate_directories("folders.txt")
    print(simfile_root_directories)
    known_simfiles = find_simfiles(simfile_root_directories)
    print("%d simfiles found" % len(known_simfiles))

    # verify that no directories have 2 simfiles
    find_duplicates(known_simfiles)

    # currently some simfile types cannot be read
    print_unknown_types(known_simfiles)
    known_simfiles = filter_known_types(known_simfiles)
    print("%d .sm simfiles found" % len(known_simfiles))

    simfile_map = {}
    for filename in known_simfiles:
        simfile_map[filename] = simfile.read_simfile(filename)

    # filter to only have simfiles the beat detection model wants
    useful_simfiles = [x for x in known_simfiles
                       if is_useful(simfile_map[x])]

    print("%d simfiles to be used in the dataset" % len(useful_simfiles))

    return useful_simfiles, simfile_map


def split_dataset(useful_simfiles, train_size, dev_size, test_size):
    """
    Splits the given list into train, dev, test

    Randomly shuffles and then splits the list proportionally based on
    train_size, dev_size, and test_size
    """
    total_size = train_size + dev_size + test_size

    train_cutoff = int(len(useful_simfiles) * train_size / total_size)
    dev_cutoff = int(len(useful_simfiles) * (train_size + dev_size) / total_size)
    dataset = list(useful_simfiles)
    random.shuffle(dataset)
    train = dataset[:train_cutoff]
    dev = dataset[train_cutoff:dev_cutoff]
    test = dataset[dev_cutoff:]

    return train, dev, test

def load_normalized(music_file):
    """
    Normalize in a couple ways: convert to 44100 samples and make
    something fake stereo if needed
    """
    audio, baud = torchaudio.load(music_file)
    print("Audio shape: {}".format(audio.shape))
    if baud != 44100:
        print("Converting %s from %d to 44100" % (music_file, baud))
        transform = torchaudio.transforms.Resample(baud, 44100)
        audio = transform(audio)
        print("New shape: {}".format(audio.shape))

    if audio.shape[0] == 1:
        # It would appear either the reader is always reading things
        # in stereo even if they are mono files, or none of the
        # simfiles I have are mono, because this was never printed
        print("WARNING: Simfile %s is mono, not stereo" % music_file)
        audio = torch.cat((audio, audio), 0)

    return audio

def featurize(audio):
    """
    TODO: try various other things, such as MelSpectrogram
    """
    transform = torchaudio.transforms.AmplitudeToDB()
    return transform(audio)

def extract_samples(dataset_files, simfile_map, num_samples):
    samples = []
    for file_idx, filename in enumerate(dataset_files):
        print(filename)

        music_file = simfile_map[filename].music
        sim = simfile_map[filename]
        audio = load_normalized(music_file)
        audio = featurize(audio)
        print("Featurized shape: {}".format(audio.shape))

        start_sample = len(samples)
        if file_idx == len(dataset_files) - 1:
            end_sample = num_samples
        else:
            end_sample = int(num_samples * (file_idx + 1) / len(simfile_map))
        # TODO
        samples.extend([1234] * (start_sample - end_sample))
    return samples

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
    
    useful_simfiles, simfile_map = collect_simfiles()

    # we should log the test files so we know which files were
    # untouched when trying to check whether a model can identify a
    # correct BPM
    train_files, dev_files, test_files = split_dataset(useful_simfiles, train_size, dev_size, test_size)

    train_set = extract_samples(train_files, simfile_map, train_samples)
    dev_set = extract_samples(dev_files, simfile_map, dev_samples)
    test_set = extract_samples(test_files, simfile_map, test_samples)
