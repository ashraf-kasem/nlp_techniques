import os
import shutil
import trax
import trax.fastmath.numpy as np
import pickle
import numpy
import random as rnd
from trax import fastmath
from trax import layers as tl
import w2_unittest
from trax.supervised import training
import itertools

''' What I want to build is predict the next set of characters
    using the previous characters (and that could be used later
    the predict the next set of frames for a video):
    - You will start by converting a line of text into a tensor
    - Then you will create a generator to feed data into the model
    - You will train a neural network in order to predict the new
      set of characters of defined length.
    - You will use embeddings for each character and feed them as
      inputs to your model.

    Many natural language tasks rely on using embeddings for predictions.
     Your model will convert each character to its embedding, run the
     embeddings through a Gated Recurrent Unit GRU, and run it through
     a linear layer to predict the next set of characters.

    -You will get the embeddings;
    -Stack the embeddings on top of each other;
    -Run them through two layers with a relu activation in the middle;
    -Finally, you will compute the softmax.

To predict the next character:
    -Use the softmax output and identify the word with the highest probability.
    -The word with the highest probability is the prediction for the next word.'''


def line_to_tensor(line, EOS_int=1):
    """Turns a line of text into a tensor

    Args:
        line (str): A single line of text.
        EOS_int (int, optional): End-of-sentence integer. Defaults to 1.

    Returns:
        list: a list of integers (unicode values) for the characters in the `line`.
    """
    # Initialize the tensor as an empty list
    tensor = []
    # for each character:
    for c in line:
        # convert to unicode int
        c_int = ord(c)
        # append the unicode integer to the tensor list
        tensor.append(c_int)
    # include the end-of-sentence integer
    tensor.append(EOS_int)
    return tensor


def data_generator(batch_size, max_length, data_lines, line_to_tensor=line_to_tensor, shuffle=True):
    """Generator function that yields batches of data
    Args:
        batch_size (int): number of examples (in this case, sentences) per batch.
        max_length (int): maximum length of the output tensor.
        NOTE: max_length includes the end-of-sentence character that will be added
                to the tensor.
                Keep in mind that the length of the tensor is always 1 + the length
                of the original line of characters.
        data_lines (list): list of the sentences to group into batches.
        line_to_tensor (function, optional): function that converts line to tensor. Defaults to line_to_tensor.
        shuffle (bool, optional): True if the generator should generate random batches of data. Defaults to True.

    Yields:
        tuple: two copies of the batch (jax.interpreters.xla.DeviceArray) and mask (jax.interpreters.xla.DeviceArray).
        NOTE: jax.interpreters.xla.DeviceArray is trax's version of numpy.ndarray
    """
    # initialize the index that points to the current position in the lines index array
    index = 0
    # initialize the list that will contain the current batch
    cur_batch = []
    # count the number of lines in data_lines
    num_lines = len(data_lines)
    # create an array with the indexes of data_lines that can be shuffled
    lines_index = [*range(num_lines)]
    # shuffle line indexes if shuffle is set to True
    if shuffle:
        rnd.shuffle(lines_index)

    while True:
        # if the index is greater than or equal to the number of lines in data_lines
        if index >= num_lines:
            # then reset the index to 0
            index = 0
            # shuffle line indexes if shuffle is set to True
            if shuffle:
                rnd.shuffle(lines_index)

        # get a line at the `lines_index[index]` position in data_lines
        line = data_lines[lines_index[index]]
        # if the length of the line is less than max_length
        if len(line) < max_length:
            # append the line to the current batch
            cur_batch.append(line)
        # increment the index by one
        index += 1
        # if the current batch is now equal to the desired batch size
        if len(cur_batch) == batch_size:
            batch = []
            mask = []
            # go through each line (li) in cur_batch
            for li in cur_batch:
                # convert the line (li) to a tensor of integers
                tensor = line_to_tensor(li)
                # Create a list of zeros to represent the padding
                # so that the tensor plus padding will have length `max_length`
                pad = [0] * (max_length - len(tensor))
                # combine the tensor plus pad
                tensor_pad = tensor + pad
                # append the padded tensor to the batch
                batch.append(tensor_pad)
                # A mask for this tensor_pad is 1 whereever tensor_pad is not
                # 0 and 0 whereever tensor_pad is 0, i.e. if tensor_pad is
                # [1, 2, 3, 0, 0, 0] then example_mask should be
                # [1, 1, 1, 0, 0, 0]
                example_mask = [1] * len(tensor) + pad
                mask.append(example_mask)

            # convert the batch (data type list) to a numpy array
            batch_np_arr = np.array(batch)
            mask_np_arr = np.array(mask)
            # Yield two copies of the batch and mask.
            yield batch_np_arr, batch_np_arr, mask_np_arr
            # reset the current batch to an empty list
            cur_batch = []


def GRULM(vocab_size=256, d_model=512, n_layers=2, mode='train'):
    """Returns a GRU language model.
    Args:
        vocab_size (int, optional): Size of the vocabulary. Defaults to 256.
        d_model (int, optional): Depth of embedding (n_units in the GRU cell). Defaults to 512.
        n_layers (int, optional): Number of GRU layers. Defaults to 2.
        mode (str, optional): 'train', 'eval' or 'predict', predict mode is for fast inference. Defaults to "train".
    Returns:
        trax.layers.combinators.Serial: A GRU language model as a layer that maps from a tensor of tokens to activations
         over a vocab set.
    """

    # the Serial layer works as a combinator which stack the layers together
    # the first layer to add is the shift-right which (This is because when we
    # train the language model, it takes the input at time t and predict the
    # output at time t+1 . We shift right the input, so that the output can be
    # delayed by 1 timestamp)

    # about the embedding layer:
    # it will generate trainable weights of size ( vocab length * vector size )
    # vocab is the number of words or characters or in my future task the identical NLP frames
    # for each index between [0 ... len(vocab)-1] there will be a unique vector of size "vector size"
    # so, the general idea is, you feed the layer the size of your vocab and the size of the desired
    # output vector. something also to note that this embedding could be trained,

    # the GRU layer is already built and we just have to use it, we should give the GRU
    # layer the number of units which in our case is the depth of the output vector
    # in other words, the length of the embedding of each vocab item (vector size)

    # we add 2 layers of the GRU to increase the accuracy of the model

    # we add a dense layer of vocab size units (256 in our case) because we are predicting
    # a character by a time (the character is on of the vocab (256))

    # as I understood that using a layer of softmax or logSoftMax will be for the same
    # reason but logSoftMax will be better in numerical computations
    # normal softmax it may create issues sometimes as we get large
    # output values for small values of input
    # At the heart of using log-softmax over softmax is the use
    # of log probabilities over probabilities, a log probability
    # is simply a logarithm of a probability. The use of log probabilities
    # means representing probabilities on a logarithmic scale, instead of the standard

    model = tl.Serial(
        tl.ShiftRight(mode=mode),  # Stack the ShiftRight layer
        tl.Embedding(vocab_size, d_model),  # Stack the embedding layer
        [tl.GRU(d_model) for _ in range(n_layers)],
        # Stack GRU layers of d_model units keeping n_layer
        tl.Dense(vocab_size),  # Dense layer
        tl.LogSoftmax(),  # Log Softmax
    )
    return model


def n_used_lines(lines, max_length):
    '''
    Args:
    lines: all lines of text an array of lines
    max_length - max_length of a line in order to be considered an int
    output_dir - folder to save your file an int
    Return:
    number of efective examples
    '''
    n_lines = 0
    for l in lines:
        if len(l) <= max_length:
            n_lines += 1
    return n_lines


def train_model(model,
                data_generator,
                lines,
                eval_lines,
                batch_size=32,
                max_length=64,
                n_steps=1,
                output_dir='model/'):
    """Function that trains the model
    Args:
        model (trax.layers.combinators.Serial): GRU model.
        data_generator (function): Data generator function.
        batch_size (int, optional): Number of lines per batch. Defaults to 32.
        max_length (int, optional): Maximum length allowed for a line to be processed. Defaults to 64.
        lines (list): List of lines to use for training. Defaults to lines.
        eval_lines (list): List of lines to use for evaluation. Defaults to eval_lines.
        n_steps (int, optional): Number of steps to train. Defaults to 1.
        output_dir (str, optional): Relative path of directory to save model. Defaults to "model/".

    Returns:
        trax.supervised.training.Loop: Training loop for the model.
    """

    # here we are making the generator object for the training
    bare_train_generator = data_generator(batch_size=batch_size, max_length=max_length, data_lines=lines)
    infinite_train_generator = itertools.cycle(bare_train_generator)

    # here we are making the generator object for the evaluation
    bare_eval_generator = data_generator(batch_size=batch_size, max_length=max_length, data_lines=eval_lines)
    infinite_eval_generator = itertools.cycle(bare_eval_generator)

    # The TrainTask class is used for defining the training architecture
    # Specifically, it's used for defining the strategy behind:
    #     Loss function
    #     Any gradient optimizers, such as Adam
    #     Logging checkpoints for parameter and accuracy evaluations after an nnn number of steps have been taken
    # the data with the labels (we give a generator object)
    # n_steps_per_checkpoint will define how frequent will the metrics will be reported
    # if we give 1 then each step will get a report about the training values
    train_task = training.TrainTask(
        labeled_data=infinite_train_generator,  # Use infinite train data generator
        loss_layer=tl.CrossEntropyLoss(),  # Don't forget to instantiate this object
        optimizer=trax.optimizers.Adam(0.0005),  # Don't forget to add the learning rate parameter TO 0.0005
        n_steps_per_checkpoint=1  # this will let the training give a report each step
    )

    # The EvalTask class is used for defining the testing architecture
    # Similar to TrainTask, it defines:
    #     How to measure model performance as a function of steps
    #     When to measure model performance as a function of steps
    #     Determining which data to use
    #     Determining which metrics to report
    eval_task = training.EvalTask(
        labeled_data=infinite_eval_generator,  # Use infinite eval data generator
        metrics=[tl.CrossEntropyLoss(), tl.Accuracy()],  # Don't forget to instantiate these objects
        n_eval_batches=2  # For better evaluation accuracy in reasonable time
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

    # the actual training starts with the run function after giving the number of steps wanted
    training_loop.run(n_steps=n_steps)

    # We return this because it contains a handle to the model, which has the weights etc.
    return training_loop


def test_model(preds, target):
    """Function to test the model.
    Args:
        preds (jax.interpreters.xla.DeviceArray): Predictions of a list of batches of tensors corresponding to lines of text.
        target (jax.interpreters.xla.DeviceArray): Actual list of batches of tensors corresponding to lines of text.
    Returns:
        float: log_perplexity of the model.
    """
    log_p = np.sum(preds * tl.one_hot(target, preds.shape[-1]),
                   axis=-1)  # HINT: tl.one_hot() should replace one of the Nones
    non_pad = 1.0 - np.equal(target, 0)  # You should check if the target equals 0
    log_p = log_p * non_pad  # Get rid of the padding
    log_ppx = np.sum(log_p, axis=1) / np.sum(non_pad, axis=1)  # Remember to set the axis properly when summing up
    log_ppx = np.mean(log_ppx)  #  Compute the mean of the previous expression
    return -log_ppx


def gumbel_sample(log_probabilities: numpy.array,
                  temperature: float = 1.0) -> float:
    """Gumbel sampling from a categorical distribution
    Args:
     log_probabilities: model predictions for a given input
     temperature: fudge

    Returns:
     the maximum sample
    """
    # log_probabilities is something like this, shape is 256 because of the model output
    # [-13.177756    -7.322052   -15.147184   -12.973529   -14.1982765
    #  -13.71838    -14.414057   -14.498814   -14.087547    -9.10589
    #  -13.794556   -14.267869   -13.479736   -14.370044   -14.721134 .......... ]
    #
    # u will be something like this, also 256 long:
    # [0.06271247 0.22430466 0.2653089  0.14371353 0.02139075 0.18085115, .. ]
    u = numpy.random.uniform(low=1e-6, high=1.0 - 1e-6,
                             size=log_probabilities.shape)
    g = -numpy.log(-numpy.log(u))
    result = numpy.argmax(log_probabilities + g * temperature, axis=-1)
    return result


# this will be the predict function
def predict(num_chars, prefix):
    """Predicts characters
    Args:
     num_chars: how many characters to predict
     prefix: character to prompt the predictions (starting characters)
    Returns:
     prefix followed by predicted characters
    """
    # converting the input chars to numbers using ord functions
    # something like this [104, 101, 108, 108, 111, 44, 32, 109, 121]
    inp = [ord(c) for c in prefix]
    # the result will be the previous set of chars and then we append the predicted chars
    # somthing like this ['h', 'e', 'l', 'l', 'o', ',', ' ', 'm', 'y']
    result = [c for c in prefix]
    # the max length of the result will be the set of starting charters plus the length of the predicted chars
    max_len = len(prefix) + num_chars
    # loop over for each prediction
    # each iteration will
    for _ in range(num_chars):
        # generating a sequence having [ input_chars, 0,0,0,0 ...  ]
        # something like this [104 101 108 108 111  44  32 109 121   0   0   0   0   0   0   0   0   0  0]
        cur_inp = np.array(inp + [0] * (max_len - len(inp)))
        # get prediction based on the current sequence
        # note that there is no batch here (so the first dimension is none)
        outp = model(cur_inp[None, :])  # Add batch dim.
        # not that the output of the model has dimisnions of (1,input_sequence_length,256)
        # when we want to get the predection of the next char, we need the values of the last charcter of the
        # input which is in the len(inp) index, thats why we are doing the flowing step
        # gumbel feed will have the diminsion (256)
        gumbel_feed = outp[0, len(inp)]
        # get the next charchter based on the gumbel distribution
        next_char = gumbel_sample(gumbel_feed)
        # append the predicted char (numeric value) to the input sequence
        # and loop again to predict the next char
        inp += [int(next_char)]

        if inp[-1] == 1:
            break  # EOS
        # append the string value of the predicted char to the result with the previous
        # sentence input
        result.append(chr(int(next_char)))

    return "".join(result)


if __name__ == "__main__":

    # set random seed
    rnd.seed(32)

    """                  Importing the Data                      """
    # first part is to process the dataset
    # in our case there is a file for shakespeare
    # each line is a sentence, and the goal of the model
    # is to predict set of characters, so we have to process
    # each sentence and convert its characters to vectors (tensors)
    # in case we are predicting words, then we need to convert
    # each word in a sentence to a tensor

    # we are reading all lines and storing them in a list
    dirname = 'data/'
    filename = 'shakespeare_data.txt'
    lines = []  # storing all the lines in a variable.
    counter = 0
    with open(os.path.join(dirname, filename)) as files:
        for line in files:
            # remove leading and trailing whitespace
            # we want to convert them to a lower case to not make the
            # model choose between capital letter and small letter
            pure_line = line.strip().lower()
            # if pure_line is not the empty string,
            if pure_line:
                # append it to the list
                lines.append(pure_line)

    # number of lines are 125k
    print(lines[:5])

    # now we want to create a validation set and training set
    eval_lines = lines[-1000:]  # the last 1000 line for validation
    lines = lines[:-1000]  # the rest of the lines are for training

    # we want to convert the lines to tensors, and we do that by converting
    # each charter to a number, we can do that using the unicode of each
    # character (function called ord)
    # Testing your output
    # print(line_to_tensor('abc xyz'))
    #
    # # # Try out your data generator
    # tmp_lines = ['hi my name is ashraf',
    #              'how are you?',
    #              'hi',
    #              'sdvfsdfdsfsdf']
    # #
    # # # Get a batch size of 2, max length 10
    # tmp_data_gen = data_generator(batch_size=2,
    #                               max_length=15,
    #                               data_lines=tmp_lines,
    #                               shuffle=False)
    # # # get 10 batch
    # # for i in range(10):
    # #     print(next(tmp_data_gen))
    #
    # import itertools
    #
    # infinite_data_generator = itertools.cycle(
    #     data_generator(batch_size=2, max_length=10, data_lines=tmp_lines))
    # ten_lines = [next(tmp_data_gen) for _ in range(10)]
    # print(len(ten_lines))




    """                       Building the model                         """
    # build the model by stacking the layers
    # for building the model we are using a user defined function
    model = GRULM(vocab_size=256, d_model=512, n_layers=2, mode='train')
    # display the model
    print(model)

    # define the training paramters
    batch_size = 32
    max_length = 64

    # get rid from all the lines that has legth more than 32
    num_used_lines = n_used_lines(lines, 32)
    print('Number of used lines from the dataset:', num_used_lines)
    print('Batch size (a power of 2):', int(batch_size))
    steps_per_epoch = int(num_used_lines / batch_size)
    print('Number of steps to cover one epoch:', steps_per_epoch)

    # Train the model 1 step and keep the `trax.supervised.training.Loop` object.
    output_dir = './model/'
    try:
        shutil.rmtree(output_dir)
    except OSError as e:
        pass


    """                          Training                     """
    # this is the main step for training
    # we are using a user defined function for training
    # the function is build to take the model object and the name
    # of the function that will work as a data generator
    # the data its self and the evaluation data and the number of
    # epochs (iteration). batch size for the generator function
    # max length (max number of characters in each line) and the
    # lines that has lower number of the max will be padded with zeros
    # the model then will use these parameters
    # to run the training and after finishing it will return
    # an abject (loop object) that can be used for prediction
    training_loop = train_model(model, data_generator,
                                lines=lines,
                                eval_lines=eval_lines,
                                n_steps=5,
                                batch_size=32,
                                max_length=64,
                                output_dir=output_dir)

    # after training is finished
    # we can print some parameters about the training
    print("Training History: ", training_loop.history)

    """ we are using pretrained model because training a model will take very long time to complete"""



    """                                      Test the model                       """
    ## Testing
    # model = GRULM()
    # model.init_from_file('model.pkl.gz')
    batch = next(data_generator(batch_size, max_length, lines, shuffle=False))
    # preds = model(batch[0])
    model = training_loop.model
    preds = model(batch[0])
    log_ppx = test_model(preds, batch[1])
    print('The log perplexity and perplexity of your model are respectively', log_ppx, np.exp(log_ppx))

    # to show tensor board data we can easily run the flowing
    print("Tensorboard data has been generated automatically, please run this command to veiw:\n",
          "tensorboard --logdir model/")



    """                      Generate characters                     """
    # the generating part of the model
    # the Gumbel distribution is used to sample from a categorical distribution
    # The maximum value, which is what we choose as the prediction in the last step of a Recursive Neural Network RNN
    # we are using for text generation, in a sample of a random variable following an exponential distribution
    # approaches the Gumbel distribution when the sample increases asymptotically. For that reason, the Gumbel
    # distribution is used to sample from a categorical distribution

    # we are using already trained model beacuse of our model is not trained for enough steps
    model = GRULM()
    model.init_from_file('model.pkl.gz')

    # the prediction function takes the number of chars that we want to predict and the starting sequence
    # and it will predict char by char using the model output logs probabilities with the help of gumbel distribution
    # true set of chars should be: to make the weeper laughs
    prefix = "to make "
    print("prefix is :", prefix)
    pred = predict(num_chars=10, prefix=prefix)
    print("prefix and 10 characters prediction are:", pred)

    prefix = "she went to "
    print("prefix is :", prefix)
    pred = predict(num_chars=20, prefix=prefix)
    print("prefix and prediction are:", pred)

    prefix = "he likes "
    print("prefix is :", prefix)
    pred = predict(num_chars=20, prefix=prefix)
    print("prefix and prediction are:", pred)

    prefix = "he likes to play with "
    print("prefix is :", prefix)
    pred = predict(num_chars=20, prefix=prefix)
    print("prefix and prediction are:", pred)

    prefix = ""
    print("prefix is :", prefix)
    pred = predict(num_chars=20, prefix=prefix)
    print("prefix and prediction are:", pred)

    # In the generated text above, you can see that the model generates text
    # that makes sense capturing dependencies between words and without any input.
    # A simple n-gram model would have not been able to capture all of that in one sentence.
