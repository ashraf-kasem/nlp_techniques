import numpy
import trax
import trax.fastmath.numpy as np
from trax import layers as tl

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

def GRULM(vocab_size=256, d_model=512, n_layers=2, mode='train'):
    """Returns a GRU language model.
    Args:
        vocab_size (int, optional): Size of the vocabulary. Defaults to 256.
        d_model (int, optional): Depth of embedding (n_units in the GRU cell). Defaults to 512.
        n_layers (int, optional): Number of GRU layers. Defaults to 2.
        mode (str, optional): 'train', 'eval' or 'predict', predict mode is for fast inference. Defaults to "train".
    Returns:
        trax.layers.combinators.Serial: A GRU language model as a layer that maps from a tensor of tokens to activations over a vocab set.
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

    # we add a dense layer of vocab size units (256 in our case) because we are predictining
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
        # Stack GRU layers of d_model units keeping n_layer parameter in mind
        tl.Dense(vocab_size),  # Dense layer
        tl.LogSoftmax(),  # Log Softmax
    )
    return model


if __name__ == "__main__":


    # I want to load a trained model from a pickl file
    model = GRULM()
    model.init_from_file('model.pkl.gz')
    print(type(model))
    print(model)


    max_length = 64
    # convert the line (li) to a tensor of integers
    input_sen = "all kinds of arguments and question deep"
    tensor = line_to_tensor(input_sen)
    print("input sentence is: ", input_sen)
    # Create a list of zeros to represent the padding
    # so that the tensor plus padding will have length `max_length`
    pad = [0] * (max_length - len(tensor))
    # combine the tensor plus pad
    tensor_pad = tensor + pad
    # convert to device array
    input_sentence = np.array([tensor_pad])
    print("input_sentence shape", len(input_sentence[0]))


    # use the model to get the output
    prediction = model(input_sentence)

    # we have to convert the predictions to set of characters
    # shape will be (1,64,256)
    # meaning 64 char with 256 embedding for each char
    # we have to convert the embedding for each char to t char
    # what I understood is for each char predection we get a 256 vector
    # we find the index of the max value in this vector and take the index
    # the index will point to a char based on our converting method (the ord function)
    # and because we don't have more than 256 char, the depth of the vector is fine
    print("prediction length:", len(prediction[0]))

    # for one char
    print("the prediction vector for the first char: ", prediction[0][0])
    print("the max value of the vector: ", max(prediction[0][0]))
    print("the index of the max value: ", numpy.argmax(prediction[0][0], axis=-1))
    print("it maps to the char: ", chr(int(numpy.argmax(prediction[0][0], axis=-1))))


    # get the over all predection
    result = numpy.argmax(prediction, axis=-1)
    print("the overall prediction int vector: ", result)
    # char result
    ch_result = "".join([chr(int(i)) for i in result[0]])
    print("the predated sequence of chars: ", ch_result)

    # exit(0)

    # this next folowing lines are to explain how the emmbedding layer of trax works
    # it will generate trainable weights of size ( vocab length * vector size )
    # vocab is the number of words or characters or in my future task the identical NLP frames
    # for each index between [0 ... len(vocab)-1] there will be a uniqe vector of size "vector size"
    # so, the general idea is, you feed the layer the size of your vocab and the size of the desired
    # output vector. somthing also to note that this emmbedding could be trained,

    # we chose our paramters
    vocab_size = 10
    vector_size = 32

    # we build the layer object
    embedding_layer = tl.Embedding(vocab_size, vector_size)
    print(embedding_layer)

    # to initlise the weights of the embbeding we need to specify the signature
    # of the input, it should be a tensor, and in our case it should be a tensor has
    # the index of the item from the vocabe that we want to get its emmbedding
    id1 = np.array([1], dtype=np.int32)
    embedding_layer.init(trax.shapes.signature(id1))


    # now we can generate input and get the related emmbedding
    # the input is any index in range of [0 ... size of the vocab]
    # the important thing is that we should make it an array (tensor) of type devicearray
    # we can do that using the trax version of numpy
    id2 = np.array([2],  dtype=np.int32)
    id3 = np.array([0, -1, 8, 9,10 ,11], dtype=np.int32)
    print(type(id1))


    # to get the output vector of an index, easy
    # just use the object name of the layer or use the forward method!
    print("embedding of id2: ", embedding_layer(id2))
    print("embedding of id2: ", embedding_layer.forward(id2))
    print("embedding of id3: ", embedding_layer.forward(id3))
    print("embedding of id1: ", embedding_layer.forward(id1))

