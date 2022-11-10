import os.path
import shutil
from termcolor import colored
import random
import numpy as np
import trax
from trax import layers as tl
from trax.fastmath import numpy as fastnp
from trax.supervised import training

""" English-to-German neural machine translation (NMT) model 
    using Long Short-Term Memory (LSTM) networks with attention.
    
    Implementing this using just a Recurrent Neural Network (RNN)
     with LSTMs can work for short to medium length sentences but
    can result in vanishing gradients for very long sequences.
    To solve this, we will be adding an attention mechanism 
    to allow the decoder to access all relevant parts of the
    input sentence regardless of its length. """


# generator helper function to append EOS to each sentence
def append_eos(stream):
    for (inputs, targets) in stream:
        inputs_with_eos = list(inputs) + [EOS]
        targets_with_eos = list(targets) + [EOS]
        yield np.array(inputs_with_eos), np.array(targets_with_eos)


# Setup helper functions for tokenizing and detokenizing sentences
def tokenize(input_str, vocab_file=None, vocab_dir=None):
    """Encodes a string to an array of integers
    Args:
        input_str (str): human-readable string to encode
        vocab_file (str): filename of the vocabulary text file
        vocab_dir (str): path to the vocabulary file
    Returns:
        numpy.ndarray: tokenized version of the input string
    """
    # Set the encoding of the "end of sentence" as 1
    EOS = 1
    # Use the trax.data.tokenize method. It takes streams and returns streams,
    # we get around it by making a 1-element stream with `iter`.
    inputs = next(trax.data.tokenize(iter([input_str]),
                                     vocab_file=vocab_file, vocab_dir=vocab_dir))
    # Mark the end of the sentence with EOS
    inputs = list(inputs) + [EOS]
    # Adding the batch dimension to the front of the shape
    batch_inputs = np.reshape(np.array(inputs), [1, -1])
    return batch_inputs


def detokenize(integers, vocab_file=None, vocab_dir=None):
    """Decodes an array of integers to a human readable string
    Args:
        integers (numpy.ndarray): array of integers to decode
        vocab_file (str): filename of the vocabulary text file
        vocab_dir (str): path to the vocabulary file
    Returns:
        str: the decoded sentence.
    """
    # Remove the dimensions of size 1
    integers = list(np.squeeze(integers))
    # Set the encoding of the "end of sentence" as 1
    EOS = 1
    # Remove the EOS to decode only the original tokens
    if EOS in integers:
        integers = integers[:integers.index(EOS)]
    return trax.data.detokenize(integers, vocab_file=vocab_file, vocab_dir=vocab_dir)


def input_encoder_fn(input_vocab_size, d_model, n_encoder_layers):
    """ Input encoder runs on the input sentence and creates
    activations that will be the keys and values for attention.
    Args:
        input_vocab_size: int: vocab size of the input
        d_model: int:  depth of embedding (n_units in the LSTM cell)
        n_encoder_layers: int: number of LSTM layers in the encoder
    Returns:
        tl.Serial: The input encoder
    """
    # create a serial network
    input_encoder = tl.Serial(
        # create an embedding layer to convert tokens to vectors
        tl.Embedding(input_vocab_size, d_model),
        # feed the embeddings to the LSTM layers. It is a stack of n_encoder_layers LSTM layers
        [tl.LSTM(d_model) for _ in range(n_encoder_layers)]
    )
    return input_encoder


def pre_attention_decoder_fn(mode, target_vocab_size, d_model):
    """ Pre-attention decoder runs on the targets and creates
    activations that are used as queries in attention.
    Args:
        mode: str: 'train' or 'eval'
        target_vocab_size: int: vocab size of the target
        d_model: int:  depth of embedding (n_units in the LSTM cell)
    Returns:
        tl.Serial: The pre-attention decoder
    """
    # create a serial network
    pre_attention_decoder = tl.Serial(
        # shift right to insert start-of-sentence token and implement
        # teacher forcing during training
        tl.ShiftRight(),
        # run an embedding layer to convert tokens to vectors
        tl.Embedding(target_vocab_size, d_model),
        # feed to an LSTM layer
        tl.LSTM(d_model)
    )

    return pre_attention_decoder


def prepare_attention_input(encoder_activations, decoder_activations, inputs):
    """Prepare queries, keys, values and mask for attention.
    Args:
        encoder_activations fastnp.array(batch_size, padded_input_length, d_model): output from the input encoder
        decoder_activations fastnp.array(batch_size, padded_input_length, d_model): output from the pre-attention decoder
        inputs fastnp.array(batch_size, padded_input_length): input tokens
    Returns:
        queries, keys, values and mask for attention.
    """
    # set the keys and values to the encoder activations
    keys = encoder_activations
    values = encoder_activations
    # set the queries to the decoder activations
    queries = decoder_activations
    # generate the mask to distinguish real tokens from padding
    # inputs is positive for real tokens and 0 where they are padding
    mask = inputs[:] > 0
    # add axes to the mask for attention heads and decoder length.
    mask = fastnp.reshape(mask, (mask.shape[0], 1, 1, mask.shape[1]))
    # broadcast so mask shape is [batch size, attention heads, decoder-len, encoder-len].
    # note: for this assignment, attention heads is set to 1.
    mask = mask + fastnp.zeros((1, 1, decoder_activations.shape[1], 1))
    return queries, keys, values, mask


def NMTAttn(input_vocab_size=33300,
            target_vocab_size=33300,
            d_model=1024,
            n_encoder_layers=2,
            n_decoder_layers=2,
            n_attention_heads=4,
            attention_dropout=0.0,
            mode='train'):
    """Returns an LSTM sequence-to-sequence model with attention.
    The input to the model is a pair (input tokens, target tokens), e.g.,
    an English sentence (tokenized) and its translation into German (tokenized).
    Args:
    input_vocab_size: int: vocab size of the input
    target_vocab_size: int: vocab size of the target
    d_model: int:  depth of embedding (n_units in the LSTM cell)
    n_encoder_layers: int: number of LSTM layers in the encoder
    n_decoder_layers: int: number of LSTM layers in the decoder after attention
    n_attention_heads: int: number of attention heads
    attention_dropout: float, dropout for the attention layer
    mode: str: 'train', 'eval' or 'predict', predict mode is for fast inference
    Returns:
    An LSTM sequence-to-sequence model with attention.
    """
    # Step 0: call the helper function to create layers for the input encoder
    input_encoder = input_encoder_fn(input_vocab_size, d_model, n_encoder_layers)
    # Step 0: call the helper function to create layers for the pre-attention decoder
    pre_attention_decoder = pre_attention_decoder_fn(mode, target_vocab_size, d_model)
    # Step 1: create a serial network
    model = tl.Serial(
        # Step 2: copy input tokens and target tokens as they will be needed later.
        tl.Select([0, 1, 0, 1]),
        # Step 3: run input encoder on the input and pre-attention decoder the target.
        tl.Parallel(input_encoder, pre_attention_decoder),
        # Step 4: prepare queries, keys, values and mask for attention.
        tl.Fn('PrepareAttentionInput', prepare_attention_input, n_out=4),
        # Step 5: run the AttentionQKV layer
        # nest it inside a Residual layer to add to the pre-attention decoder activations(i.e. queries)
        tl.Residual(tl.AttentionQKV(d_model, n_heads=n_attention_heads, dropout=attention_dropout, mode=None)),
        # Step 6: drop attention mask (i.e. index = None
        tl.Select([0, 2]),
        # Step 7: run the rest of the RNN decoder
        [tl.LSTM(d_model) for _ in range(n_decoder_layers)],
        # Step 8: prepare output by making it the right size
        tl.Dense(target_vocab_size),
        # Step 9: Log-softmax for output
        tl.LogSoftmax()
    )
    return model


if __name__ == "__main__":

    """ Part 1: Data Preparation """

    # we are using a small dataset (which we can replace with larger one)
    # English to German translation subset specified as opus/medical which
    # has medical related texts.

    # trax.data.TFDS will easily download and build a generator function for
    # the dataset that we choose. it will yielding tuples.
    # Use the keys argument to select what appears at which position in the tuple.

    # Get generator function for the training set
    # This will download the train dataset if no data_dir is specified.
    train_stream_fn = trax.data.TFDS('opus/medical',
                                     data_dir='./data/',
                                     keys=('en', 'de'),
                                     eval_holdout_size=0.01,  # 1% for eval
                                     train=True
                                     )

    # Get generator function for the eval set
    eval_stream_fn = trax.data.TFDS('opus/medical',
                                    data_dir='./data/',
                                    keys=('en', 'de'),
                                    eval_holdout_size=0.01,  # 1% for eval
                                    train=False
                                    )

    print(type(eval_stream_fn))
    # convert the generator function to a generator object
    train_stream = train_stream_fn()
    print(colored('train data (en, de) tuple:', 'green'), next(train_stream), '\n')
    print()
    eval_stream = eval_stream_fn()
    print(colored('eval data (en, de) tuple:', 'green'), next(eval_stream), '\n')

    # we have already a vocab file, with subwords in german and in english
    # now what we want to do is to tokenize the english-german sentences
    # then convert them to integer vocab using the vocab file

    # global variables that state the filename and directory of the vocabulary file
    VOCAB_FILE = 'ende_32k.subword'
    VOCAB_DIR = 'data/'
    # This function assumes that stream generates either strings or tuples/dicts
    # containing strings at some keys. This function maps these strings to numpy
    # arrays of integers – the tokenized version of each string.
    """ stream – A python generator yielding strings, tuples or dicts.
    keys – which keys of the tuple/dict to tokenize (by default: all)
    vocab_type – Type of vocabulary, one of: ‘subword’, ‘sentencepiece’, ‘char’.
    vocab_file – Name of the vocabulary file.
    vocab_dir – Directory which contains the vocabulary file.
    n_reserved_ids – An int, offset added so 0, …, n_reserved_ids-1 are unused; This is common for example when 
    reserving the 0 for padding and 1 for EOS, but it’s only needed if these symbols are not already included 
    (and thus reserved) in the vocab_file."""
    # # Tokenize the dataset.
    tokenized_train_stream = trax.data.Tokenize(vocab_file=VOCAB_FILE, vocab_dir=VOCAB_DIR)(train_stream)
    tokenized_eval_stream = trax.data.Tokenize(vocab_file=VOCAB_FILE, vocab_dir=VOCAB_DIR)(eval_stream)
    # show an example
    print(next(tokenized_train_stream), '\n')

    # now we want to append the EOS token for the end of each sentence
    # to mark the end of a sentence. This will be useful in inference/prediction
    # so we'll know that the model has completed the translation.
    # Integer assigned as end-of-sentence (EOS)
    EOS = 1
    # append EOS to the train data
    tokenized_train_stream = append_eos(tokenized_train_stream)
    # append EOS to the eval data
    tokenized_eval_stream = append_eos(tokenized_eval_stream)

    # Filter long sentences: We will place a limit on the number
    # of tokens per sentence to ensure we won't run out of memory.
    # This is done with the trax.data.FilterByLength()

    # Filter too long sentences to not run out of memory.
    # length_keys=[0, 1] means we filter both English and German sentences, so
    # both must be not longer that 512 tokens for training / 512 for eval.
    filtered_train_stream = trax.data.FilterByLength(
        max_length=512, length_keys=[0, 1])(tokenized_train_stream)
    filtered_eval_stream = trax.data.FilterByLength(
        max_length=512, length_keys=[0, 1])(tokenized_eval_stream)

    # print a sample input-target pair of tokenized sentences
    train_input, train_target = next(filtered_train_stream)
    print(colored(f'Single tokenized example input:', 'green'), train_input, '\n')
    print(colored(f'Single tokenized example target:', 'green'), train_target, '\n')

    # Bucketing: before training we usually make all the sentences have the same length
    # this is done by padding the sentences to match the longest  sentence in the dataset
    # but that will waste computational power for the samall sentences. so, bucketing
    # is to group all the sentences that have close length together and padd them
    # the nearest power of 2 and build the batches in a way that all the sentences
    # of a batch have the same length (belongs to the same bucket)

    # Bucketing to create streams of batches.

    # Buckets are defined in terms of boundaries and batch sizes.
    # Batch_sizes[i] determines the batch size for items with length < boundaries[i]
    # So below, we'll take a batch of 256 sentences of length < 8, 128 if length is
    # between 8 and 16, and so on -- and only 2 if length is over 512.
    boundaries = [8, 16, 32, 64, 128, 256, 512]
    batch_sizes = [256, 128, 64, 32, 16, 8, 4, 2]

    # Create the generators.
    train_batch_stream = trax.data.BucketByLength(
        boundaries, batch_sizes,
        length_keys=[0, 1]  # As before: count inputs and targets to length.
    )(filtered_train_stream)

    eval_batch_stream = trax.data.BucketByLength(
        boundaries, batch_sizes,
        length_keys=[0, 1]  # As before: count inputs and targets to length.
    )(filtered_eval_stream)

    # Add masking for the padding (0s).
    train_batch_stream = trax.data.AddLossWeights(id_to_mask=0)(train_batch_stream)
    eval_batch_stream = trax.data.AddLossWeights(id_to_mask=0)(eval_batch_stream)

    # now the generator will yeild batches of tuples
    input_batch, target_batch, mask_batch = next(train_batch_stream)

    # let's see the data type of a batch
    print("input_batch data type: ", type(input_batch))
    print("target_batch data type: ", type(target_batch))

    # let's see the shape of this particular batch (batch length, sentence length)
    print("input_batch shape: ", input_batch.shape)
    print("target_batch shape: ", target_batch.shape)

    # The input_batch and target_batch are Numpy arrays consisting of tokenized
    # English sentences and German sentences respectively. These tokens will later
    # be used to produce embedding vectors for each word in the sentence (so the
    # embedding for a sentence will be a matrix). The number of sentences in each
    # batch is usually a power of 2 for optimal computer memory usage

    # pick a random index less than the batch size.
    index = random.randrange(len(input_batch))

    # use the index to grab an entry from the input and target batch
    print(colored('THIS IS THE ENGLISH SENTENCE: \n', 'red'),
          detokenize(input_batch[index], vocab_file=VOCAB_FILE, vocab_dir=VOCAB_DIR), '\n')
    print(colored('THIS IS THE TOKENIZED VERSION OF THE ENGLISH SENTENCE: \n ', 'red'), input_batch[index], '\n')
    print(colored('THIS IS THE GERMAN TRANSLATION: \n', 'red'),
          detokenize(target_batch[index], vocab_file=VOCAB_FILE, vocab_dir=VOCAB_DIR), '\n')
    print(colored('THIS IS THE TOKENIZED VERSION OF THE GERMAN TRANSLATION: \n', 'red'), target_batch[index], '\n')

    # print("the mask is :", mask_batch[index])

    """Part 2: Neural Machine Translation with Attention"""

    # The model we will be building uses an encoder-decoder architecture. This Recurrent Neural Network (RNN)
    # will take in a tokenized version of a sentence in its encoder, then passes it on to the decoder for translation.
    # There are different ways to implement attention and the one we'll use for this assignment is the
    # Scaled Dot Product Attention which has the form:
    # Attention(Q,K,V) = Softmax( Q * K ) * V
    # computing scores using queries (Q) and keys (K), followed by a multiplication of values (V)
    # to get a context vector at a particular time step of the decoder. This context vector is fed
    # to the decoder RNN to get a set of probabilities for the next predicted word.

    # Input encoder
    # The input encoder runs on the input tokens, creates its embeddings, and feeds it to an LSTM network.
    # This outputs the activations that will be the keys and values for attention

    # Pre-attention decoder
    # The pre-attention decoder runs on the targets and creates activations that are used as queries in attention

    # Preparing the attention input
    # This function will prepare the inputs to the attention layer. We want to take in the encoder
    # and pre-attention decoder activations and assign it to the queries, keys, and values.
    # In addition, another output here will be the mask to distinguish real tokens from padding tokens

    # NMTAttn function
    model = NMTAttn()
    # print your model
    print(model)

    """ Part 3: Training """

    train_task = training.TrainTask(
        # use the train batch stream as labeled data
        labeled_data=train_batch_stream,
        # use the cross entropy loss
        loss_layer=tl.CrossEntropyLoss(),
        # use the Adam optimizer with learning rate of 0.01
        optimizer=trax.optimizers.Adam(0.01),
        # use the `trax.lr.warmup_and_rsqrt_decay` as the learning rate schedule
        # have 1000 warmup steps with a max value of 0.01
        lr_schedule=trax.lr.warmup_and_rsqrt_decay(1000, 0.01),
        # have a checkpoint every 10 steps
        n_steps_per_checkpoint=1)

    eval_task = training.EvalTask(
        # use the eval batch stream as labeled data
        labeled_data=eval_batch_stream,
        # use the cross entropy loss and accuracy as metrics
        metrics=[tl.CrossEntropyLoss(), tl.Accuracy()], )

    # define the output directory
    output_dir = 'output_dir/'

    # remove old model if it exists. restarts training.
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # define the training loop
    training_loop = training.Loop(NMTAttn(mode='train'),
                                  train_task,
                                  eval_tasks=[eval_task],
                                  output_dir=output_dir)
    # run training for 5 steps
    training_loop.run(10)