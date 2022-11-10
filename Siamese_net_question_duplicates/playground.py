import os
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


def Siamese(vocab_size=41699, d_model=128, mode='train'):
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
    # meaning each word in the question will be back to one value which is the mean of the d_feature vector
    # then we use the normalise for the whole question word values (each one is the mean value from before )
    # tl.Fn Layer with no weights that applies the function f, which should be specified using a lambda syntax
    q_processor = tl.Serial(  # Processor will run on Q1 and Q2.
        tl.Embedding(vocab_size, d_model),  # Embedding layer
        tl.LSTM(d_model),  # LSTM layer
        # tl.Mean(axis=1),  # Mean over columns
        tl.Fn('Normalize', lambda x: normalize(x)),  # Apply normalize function
    )  # Returns one vector of shape [batch_size, d_model].

    # Run on Q1 and Q2 in parallel.
    # build a parallel layer with 2 sister networks
    # takes 2 inputs and get 2 outputs
    # then we will use the output vectors to calculate the triplet loss
    # then backpropagation to adjust the weights of the network to
    # learn the similarity of the questions
    model = tl.Parallel(q_processor, q_processor)
    return q_processor

if __name__ == "__main__":
    model = Siamese()
    print(model)

    in2 = fastnp.array([[1, 30, 40]], dtype=np.int32)

    model.init_weights_and_state(input_signature=trax.shapes.signature(in2))

    v1 = model(in2)

    print(v1)
    print(v1.shape)


