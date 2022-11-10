import os
import shutil

import nltk
import trax
from trax import layers as tl
from trax.supervised import training
from trax.fastmath import numpy as fastnp
import numpy as np
import pandas as pd
import random as rnd
from trax import shapes
import w4_unittest
from collections import defaultdict
from functools import partial

nltk.download('punkt')

''''  This code is to build and train Siamese network to detect question duplicates'''


# we will use a dataset from Quora, the shape of the dataset is CSV format
# each line will contain 2 questions and a label 0 or 1. 0 means not duplicate and 1 means duplicate
# the data contains 404351 question pairs


def data_generator(Q1, Q2, batch_size, pad=1, shuffle=True):
    """Generator function that yields batches of data

    Args:
        Q1 (list): List of transformed (to tensor) questions.
        Q2 (list): List of transformed (to tensor) questions.
        batch_size (int): Number of elements per batch.
        pad (int, optional): Pad character from the vocab. Defaults to 1.
        shuffle (bool, optional): If the batches should be randomnized or not. Defaults to True.
    Yields:
        tuple: Of the form (input1, input2) with types (numpy.ndarray, numpy.ndarray)
        NOTE: input1: inputs to your model [q1a, q2a, q3a, ...] i.e. (q1a,q1b) are duplicates
              input2: targets to your model [q1b, q2b,q3b, ...] i.e. (q1a,q2i) i!=a are not duplicates
    """

    input1 = []
    input2 = []
    idx = 0
    len_q = len(Q1)
    # this will be used to get items from the questions
    question_indexes = [*range(len_q)]
    # if we want to shuffle the items, then we only shuffle the indexes
    if shuffle:
        rnd.shuffle(question_indexes)
    # we loop over all the dataset
    while True:
        # every time we reach the end f the dataset we reset the index
        if idx >= len_q:
            # if idx is greater than or equal to len_q, set idx accordingly
            idx = 0
            # shuffle to get random batches if shuffle is set to True
            if shuffle:
                rnd.shuffle(question_indexes)

        # get questions at the `question_indexes[idx]` position in Q1 and Q2
        # we use the loop index to get the index of the question and its duplicate
        q1 = Q1[question_indexes[idx]]
        q2 = Q2[question_indexes[idx]]

        # increment idx by 1
        idx += 1
        # append q1
        input1.append(q1)
        # append q2
        input2.append(q2)
        # if we reached the desired batch size, then yield
        if len(input1) == batch_size:
            # determine max_len as the longest question in input1 & input 2
            # take max of input1 & input2 and then max out of the two of them.
            max_len = max(max([len(i) for i in input1]), max([len(j) for j in input2]))
            # pad to power-of-2
            # for computational reasons we prefer to make the length of the questions
            # power of 2
            max_len = 2 ** int(np.ceil(np.log2(max_len)))
            b1 = []
            b2 = []
            # here what we do is to add padding for all  the questions in each batch
            # to be equal to the longest question length
            for q1, q2 in zip(input1, input2):
                # add [pad] to q1 until it reaches max_len
                pad1 = [pad for i in range(max_len - len(q1))]
                q1 = q1 + pad1
                # add [pad] to q2 until it reaches max_len
                pad2 = [pad for i in range(max_len - len(q2))]
                q2 = q2 + pad2
                # append q1
                b1.append(q1)
                # append q2
                b2.append(q2)
            # use b1 and b2
            # convert to numpy array and yield them
            yield np.array(b1), np.array(b2)
            # reset the batches
            input1, input2 = [], []


def Siamese(vocab_size=41699, d_model=512, mode='train'):
    """Returns a Siamese model.

    Args:
        vocab_size (int, optional): Length of the vocabulary. Defaults to len(vocab).
        d_model (int, optional): Depth of the model. Defaults to 128.
        mode (str, optional): 'train', 'eval' or 'predict', predict mode is for fast inference. Defaults to 'train'.

    Returns:
        trax.layers.combinators.Parallel: A Siamese model.
    """

    # we define our own Trax layer by defining a function
    # we need to normalise the output vectors as a first step before
    # calculating the cosine similarity of the output
    def normalize(x):  # normalizes the vectors to have L2 norm 1
        return x / fastnp.sqrt(fastnp.sum(x * x, axis=-1, keepdims=True))

    # because siamese network contain 2 input that run in parallel
    # we define one network, and then use the parallel trax layer
    # we pass to it the same network twice

    # each of the sister network has embedding layer with dimension of vocab size * d_feature
    # meaning each word in the vocab will be represented by a vector has the depth of d_model (128) in default
    # then we build an LSTM layer with d_feature unit (the depth of the word vector)
    # we need also to average the output of the LSTM layer over the columns, so we use trax mean layer
    # meaning all words in the question will be back to one value which is the mean of the values of the d_feature
    # vector of each word
    # then we use the normalise for the whole question word values (each one is the mean value from before )
    # tl.Fn Layer with no weights that applies the function f, which should be specified using a lambda syntax
    q_processor = tl.Serial(  # Processor will run on Q1 and Q2.
        tl.Embedding(vocab_size, d_model),  # Embedding layer vocab_size * d_model
        tl.LSTM(d_model),  # LSTM layer, output will be (number of the words in the question) * d_model,
        tl.Mean(axis=1),  # Mean over columns (1 * d_model) for each question in the batch
        tl.Fn('Normalize', lambda x: normalize(x)),  # Apply normalize function, will not change the dimensions
    )  # Returns one vector of shape [batch_size, d_model].

    """ The model output 1 vector for each question with the size of [d_model]"""

    # Run on Q1 and Q2 in parallel.
    # build a parallel layer with 2 sister networks
    # takes 2 inputs and get 2 outputs
    # then we will use the output vectors to calculate the triplet loss
    # then backpropagation to adjust the weights of the network to
    # learn the similarity of the questions
    model = tl.Parallel(q_processor, q_processor)
    return model


def TripletLossFn(v1, v2, margin=0.25):
    """Custom Loss function.

    Args:
        v1 (numpy.ndarray): Array with dimension (batch_size, model_dimension) associated to Q1.
        v2 (numpy.ndarray): Array with dimension (batch_size, model_dimension) associated to Q2.
        margin (float, optional): Desired margin. Defaults to 0.25.

    Returns:
        jax.interpreters.xla.DeviceArray: Triplet Loss.
    """

    # loss is composed of two terms. One term utilizes the mean of all the non duplicates,
    # the second utilizes the closest negative. Our loss expression is then:
    # Loss1 = max(  −𝑐𝑜𝑠(𝐴,𝑃) + 𝑚𝑒𝑎𝑛𝑛𝑒𝑔 + 𝛼, 0 )
    # Loss2 = max(  −𝑐𝑜𝑠(𝐴,𝑃) + 𝑐𝑙𝑜𝑠𝑒𝑠𝑡𝑛𝑒𝑔 + 𝛼, 0 )
    # Loss = Mean(Loss1, Loss2)

    # because the model output vectors already normalised then
    # we calculate cosine similarity between the vectors using dot product
    # use fastnp to take the dot product of the two batches
    scores = fastnp.dot(v1, v2.T)  # pairwise cosine sim
    # calculate new batch size
    batch_size = len(scores)
    # use fastnp to grab all positive `diagonal` entries in `scores`
    positive = fastnp.diagonal(scores)  # the positive ones (duplicates)
    # subtract `fastnp.eye(batch_size)` out of 1.0 and do element-wise multiplication with `scores`
    negative_zero_on_duplicate = fastnp.multiply((1.0 - fastnp.eye(batch_size)), scores)
    # use `fastnp.sum` on `negative_zero_on_duplicate` for `axis= None
    mean_negative = fastnp.sum(negative_zero_on_duplicate, axis=1)
    # create a composition of two masks:
    #  the first mask to extract the diagonal elements,
    # the second mask to extract elements in the negative_zero_on_duplicate matrix that are
    # larger than the elements in the diagonal
    mask_exclude_positives = (fastnp.eye(batch_size) == 1) | \
                             (negative_zero_on_duplicate > positive.reshape(batch_size, 1))
    # multiply `mask_exclude_positives` with 2.0 and subtract it out of `negative_zero_on_duplicate`
    negative_without_positive = negative_zero_on_duplicate - (mask_exclude_positives * 2)
    # take the row by row `max` of `negative_without_positive`.
    closest_negative = negative_without_positive.max(axis=1)
    # compute `fastnp.maximum` among 0.0 and `A`
    # where A = subtract `positive` from `margin` and add `closest_negative`
    # IMPORTANT: DO NOT create an extra variable 'A'
    triplet_loss1 = fastnp.maximum(0.0, closest_negative - positive + margin)
    # compute `fastnp.maximum` among 0.0 and `B`
    # where B = ubtract `positive` from `margin` and add `mean_negative`
    # IMPORTANT: DO NOT create an extra variable 'B'
    triplet_loss2 = fastnp.maximum(0.0, mean_negative - positive + margin)
    # add the two losses together and take the `fastnp.sum` of it
    triplet_loss = fastnp.sum(triplet_loss1 + triplet_loss2)

    return triplet_loss


def train_model(model, TripletLoss
                , train_generator, val_generator, output_dir='model/'):
    """Training the Siamese Model

    Args:
        model :  Siamese model.
        TripletLoss (function): Function that defines the TripletLoss loss function.
        lr_schedule (function): Trax multifactor schedule function.
        train_generator (generator, optional): Training generator. Defaults to train_generator.
        val_generator (generator, optional): Validation generator. Defaults to val_generator.
        output_dir (str, optional): Path to save model to. Defaults to 'model/'.

    Returns:
        trax.supervised.training.Loop: Training loop for the model.
    """

    # if the provided path has ~
    output_dir = os.path.expanduser(output_dir)
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    # The TrainTask class is used for defining the training architecture
    # Specifically, it's used for defining the strategy behind:
    #     Loss function
    #     Any gradient optimizers, such as Adam
    #     Logging checkpoints for parameter and accuracy evaluations after an nnn number of steps have been taken
    # the data with the labels (we give a generator object)
    # n_steps_per_checkpoint will define how frequent will the metrics will be reported
    # if we give 1 then each step will get a report about the training values

    # lr_schedule: Learning rate (LR) schedules
    # lr_schedules.warmup(n_warmup_steps, max_value)
    # Returns an LR schedule with linear warm-up followed by constant value
    # So, the learning rate will change for 400 steps until it gets max_value
    # This reduces volatility in the early stages of training.

    train_task = training.TrainTask(
        labeled_data=train_generator,  # Use generator (train)
        loss_layer=TripletLoss(),  # Use triplet loss
        optimizer=trax.optimizers.Adam(0.01),
        lr_schedule=trax.lr.warmup_and_rsqrt_decay(400, 0.01),  # Use Trax multifactor schedule function
        n_steps_per_checkpoint=1  # this will let the training give a report each step
    )

    # The EvalTask class is used for defining the testing architecture
    # Similar to TrainTask, it defines:
    #     How to measure model performance as a function of steps
    #     When to measure model performance as a function of steps
    #     Determining which data to use
    #     Determining which metrics to report
    eval_task = training.EvalTask(
        labeled_data=val_generator,  # Use generator (val)
        metrics=[TripletLoss()],  # Use triplet loss.
    )

    # The Loop class is used for running and performing the core training loop
    # The number of steps taken by the training is given in the training task
    # The training parameters run by Loop are initialized randomly
    # First, we define the directory output_dir to which the output file will be written
    # Next, we implement a Loop object that does the following:
    #     Trains a given model on training data
    #     The training data is given in the training task
    #     Outlines the training architecture with a training task
    #     Outlines the testing architecture with an evaluation task
    # we give the model that we want to train with the object of the train task (shipped with the training data
    # and all other training parameters)
    # then we also give the evaluation task object shipped with the evaluation parameters
    # the output dir is for saving the training check points
    training_loop = training.Loop(model,
                                  train_task,
                                  eval_tasks=[eval_task],
                                  output_dir=output_dir)

    # return a loop object that ready to run the training
    return training_loop


def classify(test_Q1, test_Q2, y, threshold, model, vocab, data_generator=data_generator, batch_size=64):
    """Function to test the accuracy of the model.

    Args:
        test_Q1 (numpy.ndarray): Array of Q1 questions.
        test_Q2 (numpy.ndarray): Array of Q2 questions.
        y (numpy.ndarray): Array of actual target.
        threshold (float): Desired threshold.
        model (trax.layers.combinators.Parallel): The Siamese model.
        vocab (collections.defaultdict): The vocabulary used.
        data_generator (function): Data generator function. Defaults to data_generator.
        batch_size (int, optional): Size of the batches. Defaults to 64.

    Returns:
        float: Accuracy of the model.
    """

    accuracy = 0
    for i in range(0, len(test_Q1), batch_size):
        # Call the data generator  with shuffle= None
        # use batch size chuncks of questions as Q1 & Q2 arguments of the data generator. e.g x[i:i + batch_size]
        # Hint: use `vocab['<PAD>']` for the `pad` argument of the data generator
        q1, q2 = next(
            data_generator(test_Q1[i:i + batch_size], test_Q2[i:i + batch_size], batch_size, pad=vocab['<PAD>'],
                           shuffle=False))
        # use batch size chuncks of actual output targets (same syntax as example above)
        y_test = y[i:i + batch_size]
        # Call the model
        v1, v2 = model([q1, q2])

        for j in range(batch_size):
            # take dot product to compute cos similarity of each pair of entries, v1[j], v2[j]
            # don't forget to transpose the second argument
            d = fastnp.dot(v1[j], v2[j].T)
            # is d greater than the threshold?
            res = d > threshold
            # increment accuracy if y_test is equal `res`
            accuracy += (y_test[j] == res)
    # compute accuracy using accuracy and total length of test questions
    accuracy = accuracy / len(test_Q1)

    return accuracy


def predict(question1, question2, threshold, model, vocab, data_generator=data_generator, verbose=False):
    """Function for predicting if two questions are duplicates.

    Args:
        question1 (str): First question.
        question2 (str): Second question.
        threshold (float): Desired threshold.
        model (trax.layers.combinators.Parallel): The Siamese model.
        vocab (collections.defaultdict): The vocabulary used.
        data_generator (function): Data generator function. Defaults to data_generator.
        verbose (bool, optional): If the results should be printed out. Defaults to False.

    Returns:
        bool: True if the questions are duplicates, False otherwise.
    """

    # use `nltk` word tokenize function to tokenize
    q1 = nltk.word_tokenize(question1)  # tokenize
    q2 = nltk.word_tokenize(question2)  # tokenize

    Q1, Q2 = [], []
    for word in q1:  # encode q1
        # increment by checking the 'word' index in `vocab`
        Q1.append(vocab[word])
    for word in q2:  # encode q2
        # increment by checking the 'word' index in `vocab`
        Q2.append(vocab[word])

    # Call the data generator (built in Ex 01) using next()
    # pass [Q1] & [Q2] as Q1 & Q2 arguments of the data generator. Set batch size as 1
    # Hint: use `vocab['<PAD>']` for the `pad` argument of the data generator
    Q1, Q2 = next(data_generator([Q1], [Q2], 1, pad=vocab['<PAD>'], shuffle=False))
    # Call the model
    v1, v2 = model([Q1, Q2])
    # take dot product to compute cos similarity of each pair of entries, v1, v2
    # don't forget to transpose the second argument
    d = fastnp.dot(v1, v2.T)
    # is d greater than the threshold?
    res = d > threshold

    if (verbose):
        print("Q1  = ", Q1, "\nQ2  = ", Q2)
        print("d   = ", d)
        print("res = ", res)

    return res


if __name__ == "__main__":
    # set random seeds
    rnd.seed(34)

    """ Import and preprocess the dataset """

    # first step is to import the dataset and preprocess it
    data = pd.read_csv("data/questions.csv")
    N = len(data)
    print('Number of question pairs: ', N)
    print(data.head(5))

    # split the dataset to train and test 300k for train and 10k for test
    N_train = 300000
    N_test = 10 * 1024
    data_train = data[:N_train]
    data_test = data[N_train:N_train + N_test]
    print("Train set:", len(data_train), "Test set:", len(data_test))
    del (data)  # remove to free memory

    # what we want to do is to select only the question pairs that are duplicate to train the model
    # so from the train data we have to extract just the pairs that have the (1) label
    # to build the batches, the batch should be a tuple that has 2 arrays
    # the first question from the first array in the batch should be duplicate to the first question from the
    # second array and not duplicate with all others in the same batch

    # this line will give a array of True and false (true in the place that has a label of 1) (false other wise)
    td_index = (data_train['is_duplicate'] == 1).to_numpy()
    print("td_index", td_index)
    # here we loop and get just the index of the lines that has a label of 1 (duplicates questions)
    td_index = [i for i, x in enumerate(td_index) if x]
    print('number of duplicate questions: ', len(td_index))
    print('indexes of first ten duplicate questions:', td_index[:10])

    # basically the training dataset has a length of 111486 items
    # show example of duplicate
    print(data_train['question1'][5])
    print(data_train['question2'][5])
    print('is_duplicate: ', data_train['is_duplicate'][5])

    # Split the questions that are duplicates, notcie that they are at the same order
    # meaning Q1 and Q2 have the same questions based on their indexes
    Q1_train_words = np.array(data_train['question1'][td_index])
    Q2_train_words = np.array(data_train['question2'][td_index])
    # print(Q1_train_words[1])
    # print(Q2_train_words[1])

    # split also for the Test data
    Q1_test_words = np.array(data_test['question1'])
    Q2_test_words = np.array(data_test['question2'])
    y_test = np.array(data_test['is_duplicate'])
    # print(y_test)

    print('TRAINING QUESTIONS:\n')
    print('Question 1: ', Q1_train_words[0])
    print('Question 2: ', Q2_train_words[0], '\n')
    print('Question 1: ', Q1_train_words[5])
    print('Question 2: ', Q2_train_words[5], '\n')

    print('TESTING QUESTIONS:\n')
    print('Question 1: ', Q1_test_words[1])
    print('Question 2: ', Q2_test_words[1], '\n')
    print('is_duplicate =', y_test[1], '\n')

    # now we want to tokenize the questions

    # create arrays to hold the tokens
    Q1_train = np.empty_like(Q1_train_words)
    Q2_train = np.empty_like(Q2_train_words)

    Q1_test = np.empty_like(Q1_test_words)
    Q2_test = np.empty_like(Q2_test_words)

    # Building the vocabulary with the train set
    # we will use default dictionary to be the vocab, the reason is to easily handle the OOV (out of vocab) words
    # any word that not in the vocab will have a value of 0
    vocab = defaultdict(lambda: 0)
    # the numeric value of the padding tokens will be 1
    vocab['<PAD>'] = 1

    # we loop over the questions both Q1 and Q2
    # split them to words, add them to a vocab as the word is the key and the value is a unique index
    for idx in range(len(Q1_train_words)):
        # covert each question to words (tokenize)
        Q1_train[idx] = nltk.word_tokenize(Q1_train_words[idx])
        Q2_train[idx] = nltk.word_tokenize(Q2_train_words[idx])
        # combine the words from the 2 questions
        q = Q1_train[idx] + Q2_train[idx]
        # for each new word we add to the vocab with unique index
        for word in q:
            if word not in vocab:
                vocab[word] = len(vocab) + 1

    # there will be about 36268 word in the vocab
    # show some info
    print('The length of the vocabulary is: ', len(vocab))
    print(" Show some words in the vocab:")
    [print(k, ":", v) for i, (k, v) in enumerate(vocab.items()) if i <5]

    # now also tokenize the test questions but without adding them to the vocab
    for idx in range(len(Q1_test_words)):
        Q1_test[idx] = nltk.word_tokenize(Q1_test_words[idx])
        Q2_test[idx] = nltk.word_tokenize(Q2_test_words[idx])

    print('Train set has reduced to: ', len(Q1_train))
    print('Test set length: ', len(Q1_test))

    # we have to convert the questions to numbers (tensors) based on the vocab
    # Converting questions to array of integers
    # loop over questions
    for i in range(len(Q1_train)):
        # replace each question with a list of integers
        # each word in a question will be converted to integer based on the vocab
        Q1_train[i] = [vocab[word] for word in Q1_train[i]]
        Q2_train[i] = [vocab[word] for word in Q2_train[i]]

    # do the same as previous for the test questions
    # one thing to note, if the word was not in the vocab (oov) then it will get a value of 0
    for i in range(len(Q1_test)):
        Q1_test[i] = [vocab[word] for word in Q1_test[i]]
        Q2_test[i] = [vocab[word] for word in Q2_test[i]]

    print('first question in the train set:\n')
    print(Q1_train_words[0], '\n')
    print('encoded version:')
    print(Q1_train[0], '\n')

    print('first question in the test set:\n')
    print(Q1_test_words[0], '\n')
    print('encoded version:')
    print(Q1_test[0])

    # we then split the train dataset to train/validation dataset
    # Splitting the data
    # about 20% will go for validation and the rest will be for training
    cut_off = int(len(Q1_train) * .8)
    train_Q1, train_Q2 = Q1_train[:cut_off], Q2_train[:cut_off]
    val_Q1, val_Q2 = Q1_train[cut_off:], Q2_train[cut_off:]
    print('Number of duplicate questions: ', len(Q1_train))
    print("The length of the training set is:  ", len(train_Q1))
    print("The length of the validation set is: ", len(val_Q1))

    """                 Build the data Generator Function                  """
    # Now that you have your generator, you can just call it
    # and it will return tensors which correspond to your questions in the Quora data set.
    # test the batch generator, we could remove this later
    batch_size = 2
    res1, res2 = next(data_generator(train_Q1, train_Q2, batch_size))
    print("First questions  : ", '\n', res1, '\n')
    print("Second questions : ", '\n', res2)

    """         Defining the Siamese model                     """
    # A Siamese network is a neural network which uses the same
    # weights while working in tandem on two different input vectors
    # to compute comparable output vectors v1,v2
    # You get the question embedding, run it through an LSTM layer,
    # normalize v1 and v2, and finally use a triplet loss to get the
    # corresponding cosine similarity for each pair of questions
    model = Siamese()
    print(model)

    """            Define the Loss function                  """
    # to train the model for classifying the duplicate questions
    # we need a Loss function that calculate the loss in the way
    # that minimising the distance between  baseline question
    # and its duplicate and maximising the distance between the same base
    # line and other not duplicate questions

    # we do that by defining something called tripletLoss
    # The triplet loss makes use of a baseline (anchor) input
    # that is compared to a positive (truthy) input and a negative
    # (falsy) input. The distance from the baseline (anchor) input
    # to the positive (truthy) input is minimized, and the distance from
    # the baseline (anchor) input to the negative (falsy) input is maximized
    # L(𝐴,𝑃,𝑁)= max( ‖f(𝐴)−f(𝑃)‖2 − ‖f(𝐴)−f(𝑁)‖ 2 + 𝛼, 0)
    # 𝛼 is a margin (how much you want to push the duplicates from the non duplicates)
    # note: if alpha is smaller then the training will be easier
    # and if alpha is larger then the training will be harder
    v1 = np.array([[0.26726124, 0.53452248, 0.80178373], [-0.5178918, -0.57543534, -0.63297887]])
    v2 = np.array([[0.26726124, 0.53452248, 0.80178373], [0.5178918, 0.57543534, 0.63297887]])
    print("Triplet Loss:", TripletLossFn(v1, v2))

    """            Make a Trax layer from our yser defined Loss function           """


    # we want to make a trax layer that represent our user defined Loss function
    def TripletLoss(margin=0.25):
        triplet_loss_fn = partial(TripletLossFn, margin=margin)
        # using the trax layer we use the Fn (function) class by passing the function name
        # and what we want to name the layer
        return tl.Fn('TripletLoss', triplet_loss_fn)


    """                          Training                              """
    # we will use the our data generator to generate batches of 256 from the trading and validation dataset
    batch_size = 256
    train_generator = data_generator(train_Q1, train_Q2, batch_size, vocab['<PAD>'])
    val_generator = data_generator(val_Q1, val_Q2, batch_size, vocab['<PAD>'])
    print('train_Q1.shape ', train_Q1.shape)
    print('val_Q1.shape   ', val_Q1.shape)

    # define the number of steps for the trading
    train_steps = 1000
    # train_model is our user defined function to build the training object
    model = Siamese()
    training_loop = train_model(model, TripletLoss, train_generator, val_generator)
    # here what will start the training based on the loop object paramters
    training_loop.run(train_steps)

    """                   Testing and Evaluation                            """

    print("for our previous model the accuracy is:")
    # this takes around 1 minute, we pass the model, similarity threshold, test questions pairs and labels
    accuracy = classify(Q1_test, Q2_test, y_test, 0.7, model, vocab, batch_size=512)
    print("Accuracy", accuracy)

    # we will load a pretrained model for testing, because training a new model
    # will take very long time
    # input signature we take the shape of the input from the generator
    # we can manually pass  2 ndarray like this (  fastnp.array([[1]], dtype=np.int32)  )
    trained_model = Siamese()
    trained_model.init_from_file(file_name='model.pkl.gz', weights_only=True,
                         input_signature=shapes.signature(next(train_generator)))
    print("pretrained model is loaded ")

    # the test data, Q1_test, Q2_test and y_test, is setup as pairs of questions,
    # some of which are duplicates some are not. This routine will run all the test
    # question pairs through the model, compute the cosine simlarity of each pair,
    # threshold it and compare the result to y_test - the correct response from the data set.
    # The results are accumulated to produce an accuracy.

    print("for pretrained model the accuracy is: ")
    # this takes around 1 minute, we pass the model, similarity threshold, test questions pairs and labels
    accuracy = classify(Q1_test, Q2_test, y_test, 0.7, trained_model, vocab, batch_size=512)
    print("Accuracy", accuracy)

    """                           Test using our own questions                   """
    # create 2 questions
    question1 = "When will I see you?"
    question2 = "When can I see you again?"
    print("question1: ", question1)
    print("question2: ", question2)
    # takes in two questions, the model, and the vocabulary and returns whether the questions
    # are duplicates (1) or not duplicates (0) given a similarity threshold.
    # 1/True means it is duplicated, 0/False otherwise
    predict(question1, question2, 0.7, trained_model, vocab, verbose=True)

    question1 = "how to create a database?"
    question2 = "how to implement a database?"
    print("question1: ", question1)
    print("question2: ", question2)
    predict(question1, question2, 0.7, trained_model, vocab, verbose=True)

    question1 = "Do they enjoy eating the dessert?"
    question2 = "Do they like hiking in the desert?"
    print("question1: ", question1)
    print("question2: ", question2)
    predict(question1, question2, 0.7, trained_model, vocab, verbose=True)
