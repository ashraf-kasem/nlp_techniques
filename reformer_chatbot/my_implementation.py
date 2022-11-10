import json
import os
import random
import shutil
import numpy as np
from termcolor import colored
import trax
from trax import layers as tl
from trax.supervised import training


# help function to load a JSON file
def load_json(directory, file):
    with open(f'{directory}/{file}') as file:
        db = json.load(file)
    return db


def get_conversation(file, data_db):
    """
    Args:
        file (string): filename of the dialogue file saved as json
        data_db (dict): dialogue database

    Returns:
        string: A string containing the 'text' fields of  data[file]['log'][x]
    """
    # initialize empty string
    result = ''
    # get length of file's log list
    len_msg_log = len(data_db[file]['log'])
    # set the delimiter strings
    delimiter_1 = ' Person 1: '
    delimiter_2 = ' Person 2: '
    # loop over the file's log list
    for i in range(len_msg_log):
        # get i'th element of file log list
        cur_log = data_db[file]['log'][i]
        # check if i is even
        if i % 2 == 0:
            # append the 1st delimiter string
            result += delimiter_1
        else:
            # append the 2nd delimiter string
            result += delimiter_2
        # append the message text from the log
        result += cur_log['text']
    return result


def print_conversation(conversation):
    delimiter_1 = 'Person 1: '
    delimiter_2 = 'Person 2: '

    split_list_d1 = conversation.split(delimiter_1)

    for sublist in split_list_d1[1:]:
        split_list_d2 = sublist.split(delimiter_2)
        print(colored(f'Person 1: {split_list_d2[0]}', 'red'))

        if len(split_list_d2) > 1:
            print(colored(f'Person 2: {split_list_d2[1]}', 'green'))

# generator function to yield a tuple of (input, target) for training
def stream(data):
    # loop over the entire data
    while True:
        # get a random element
        d = random.choice(data)
        # yield a tuple pair of identical values
        # (i.e. our inputs to the model will also be our targets during training)
        yield d, d

# function build a Reformer model
def ReformerLM(vocab_size=33000,
               n_layers=2,
               mode='train',
               attention_type=tl.SelfAttention):
    # initialize an instance of Trax's ReformerLM class
    model = tl.Serial(
        trax.models.reformer.ReformerLM(
            # set vocab size
            vocab_size=vocab_size,
            # set number of layers
            n_layers=n_layers,
            # set mode
            mode=mode,
            # set attention type
            attention_type=attention_type
        )
        , tl.LogSoftmax()
    )
    return model


def training_loop(ReformerLM, train_gen, eval_gen, output_dir="./model/"):
    """
    Args:
        ReformerLM:  the Reformer language model you are building
        train_gen (generator): train data generator.
        eval_gen (generator): Validation generator.
        output_dir (string): Path to save the model output. Defaults to './model/'.

    Returns:
        trax.supervised.training.Loop: Training loop for the model.
    """
    # use the warmup_and_rsqrt_decay learning rate schedule
    lr_schedule = trax.lr.warmup_and_rsqrt_decay(
        n_warmup_steps=1000, max_value=0.01)
    # define the train task
    train_task = training.TrainTask(
        # labeled data
        train_gen,
        # loss layer
        loss_layer=tl.CrossEntropyLoss(),
        # optimizer
        optimizer=trax.optimizers.Adam(0.01),
        # lr_schedule
        lr_schedule=lr_schedule,
        # n_steps
        n_steps_per_checkpoint=1
    )
    # define the eval task
    eval_task = training.EvalTask(
        # labeled data
        eval_gen,
        # metrics
        metrics=[tl.CrossEntropyLoss(), tl.Accuracy()]
    )
    loop = training.Loop(ReformerLM(mode='train'),
                         train_task,
                         eval_tasks=[eval_task],
                         output_dir=output_dir)
    return loop


if __name__ == "__main__":
    # filename of the MultiWOZ dialogue dataset
    DATA_FILE = 'data.json'
    # data directory
    DATA_DIR = './data'
    # dictionary where we will load the dialogue dataset
    DIALOGUE_DB = {}
    # vocabulary filename
    VOCAB_FILE = 'en_32k.subword'
    # vocabulary file directory
    VOCAB_DIR = 'data/vocabs'

    # load the dialogue data set into our dictionary
    DIALOGUE_DB = load_json(DATA_DIR, DATA_FILE)

    # show an example of the dataset
    # the keys are the file names
    all_files = DIALOGUE_DB.keys()
    # show a conversation from the dataset
    # take a random file name
    file = random.choice(list(all_files))
    print_conversation(get_conversation(file, DIALOGUE_DB))
    print()

    # preprocessing

    # initialize empty list
    untokenized_data = []
    # loop over all files
    for file in all_files:
        # this is the graded function you coded
        # returns a string delimited by Person 1 and Person 2
        result = get_conversation(file, DIALOGUE_DB)
        # append to the list
        untokenized_data.append(result)

    # shuffle the list we generated above
    random.shuffle(untokenized_data)
    # define a cutoff (5% of the total length for this assignment)
    # convert to int because we will use it as a list index
    cut_off = int(len(untokenized_data) * .05)
    # slice the list. the last elements after the cut_off value will be the eval set. the rest is for training.
    train_data, eval_data = untokenized_data[:-cut_off], untokenized_data[-cut_off:]
    print(f'number of conversations in the data set: {len(untokenized_data)}')
    print(f'number of conversations in train set: {len(train_data)}')
    print(f'number of conversations in eval set: {len(eval_data)}', end='\n\n')

    # tokenize the data
    # first we build a data_pipeline which basically a bunch of functions that we should apply
    # to our data before we feed it to the model
    # trax allows us to use combinators to generate our data pipeline
    data_pipeline = trax.data.Serial(
        # randomize the stream
        trax.data.Shuffle(),
        # tokenize the data
        trax.data.Tokenize(vocab_dir=VOCAB_DIR,
                           vocab_file=VOCAB_FILE),
        # filter too long sequences
        trax.data.FilterByLength(2048),
        # bucket by length
        trax.data.BucketByLength(boundaries=[128, 256, 512, 1024],
                                 batch_sizes=[16, 8, 4, 2, 1]),
        # add loss weights but do not add it to the padding tokens (i.e. 0)
        trax.data.AddLossWeights(id_to_mask=0)
    )

    # apply the data pipeline to our train and eval sets
    train_stream = data_pipeline(stream(train_data))
    eval_stream = data_pipeline(stream(eval_data))

    # show an example of the data pipeline
    # the stream generators will yield (input, target, weights).
    # let's just grab the input for inspection
    inp, target, mask = next(train_stream)

    # print the shape. format is (batch size, token length)
    print("input shape: ", inp.shape, end='\n\n')

    # detokenize the first element
    print("input: ", inp[0], end='\n\n')
    print("target: ", target[0], end='\n\n')
    print("mask: ", mask[0], end='\n\n')

    # show the Reformer model
    temp_model = ReformerLM()
    # display the model
    print(str(temp_model))
    del temp_model

    # delete the old model
    if os.path.exists('./model/'):
        shutil.rmtree('./model/')
    os.mkdir('./model/')

    # create the reformer model with the training tasks
    # building a training loop object then run the training
    loop = training_loop(ReformerLM, train_stream, eval_stream)
    # run the training loop
    loop.run(5)




