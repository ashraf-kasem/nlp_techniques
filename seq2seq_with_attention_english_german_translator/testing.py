import os.path
import shutil
from termcolor import colored
import random
import numpy as np
import trax
from trax import layers as tl
from trax.fastmath import numpy as fastnp
from trax.supervised import training
from collections import Counter

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


def next_symbol(NMTAttn, input_tokens, cur_output_tokens, temperature):
    """Returns the index of the next token.
    Args:
        NMTAttn (tl.Serial): An LSTM sequence-to-sequence model with attention.
        input_tokens (np.ndarray 1 x n_tokens): tokenized representation of the input sentence
        cur_output_tokens (list): tokenized representation of previously translated words
        temperature (float): parameter for sampling ranging from 0.0 to 1.0.
            0.0: same as argmax, always pick the most probable token
            1.0: sampling from the distribution (can sometimes say random things)
    Returns:
        int: index of the next token in the translated sentence
        float: log probability of the next symbol
    """
    # set the length of the current output tokens
    token_length = len(cur_output_tokens)
    # calculate next power of 2 for padding length
    padded_length = 2 ** int(np.ceil(np.log2(token_length + 1)))
    # pad cur_output_tokens up to the padded_length
    padded = cur_output_tokens + [0 for _ in range(padded_length - len(cur_output_tokens))]
    # model expects the output to have an axis for the batch size in front so
    # convert `padded` list to a numpy array with shape (1, <padded_length>)
    padded_with_batch = np.reshape(np.array(padded), (1, len(padded)))
    # get the model prediction
    output, _ = NMTAttn((input_tokens, padded_with_batch))
    # get log probabilities from the last token output
    log_probs = output[0, token_length, :]
    # get the next symbol by getting a logsoftmax sample (*hint: cast to an int)
    symbol = int(tl.logsoftmax_sample(log_probs, temperature))
    return symbol, float(log_probs[symbol])


def sampling_decode(input_sentence, NMTAttn=None, temperature=0.0, vocab_file=None, vocab_dir=None,
                    next_symbol=next_symbol, tokenize=tokenize, detokenize=detokenize):
    """Returns the translated sentence.
    Args:
        input_sentence (str): sentence to translate.
        NMTAttn (tl.Serial): An LSTM sequence-to-sequence model with attention.
        temperature (float): parameter for sampling ranging from 0.0 to 1.0.
            0.0: same as argmax, always pick the most probable token
            1.0: sampling from the distribution (can sometimes say random things)
        vocab_file (str): filename of the vocabulary
        vocab_dir (str): path to the vocabulary file
    Returns:
        tuple: (list, str, float)
            list of int: tokenized version of the translated sentence
            float: log probability of the translated sentence
            str: the translated sentence
    """
    # encode the input sentence
    input_tokens = tokenize(input_sentence, vocab_file=vocab_file, vocab_dir=vocab_dir)
    # initialize an empty the list of output tokens
    cur_output_tokens = []
    # initialize an integer that represents the current output index
    cur_output = 0
    # Set the encoding of the "end of sentence" as 1
    EOS = 1
    # check that the current output is not the end of sentence token
    while cur_output != EOS:
        # update the current output token by getting the index of the next word
        cur_output, log_prob = next_symbol(NMTAttn, input_tokens, cur_output_tokens, temperature)
        # append the current output token to the list of output tokens
        cur_output_tokens.append(cur_output)
        # detokenize the output tokens
    sentence = detokenize(cur_output_tokens, vocab_file=vocab_file, vocab_dir=vocab_dir)
    return cur_output_tokens, log_prob, sentence


def greedy_decode_test(sentence, NMTAttn=None, vocab_file=None, vocab_dir=None, sampling_decode=sampling_decode,
                       next_symbol=next_symbol, tokenize=tokenize, detokenize=detokenize):
    """Prints the input and output of our NMTAttn model using greedy decode
    Args:
        sentence (str): a custom string.
        NMTAttn (tl.Serial): An LSTM sequence-to-sequence model with attention.
        vocab_file (str): filename of the vocabulary
        vocab_dir (str): path to the vocabulary file
    Returns:
        str: the translated sentence
    """
    _, _, translated_sentence = sampling_decode(sentence, NMTAttn=NMTAttn, vocab_file=vocab_file, vocab_dir=vocab_dir,
                                                next_symbol=next_symbol, tokenize=tokenize, detokenize=detokenize)
    print("English: ", sentence)
    print("German: ", translated_sentence)
    return translated_sentence


def generate_samples(sentence, n_samples, NMTAttn=None, temperature=0.6, vocab_file=None, vocab_dir=None,
                     sampling_decode=sampling_decode, next_symbol=next_symbol, tokenize=tokenize,
                     detokenize=detokenize):
    """Generates samples using sampling_decode()
    Args:
        sentence (str): sentence to translate.
        n_samples (int): number of samples to generate
        NMTAttn (tl.Serial): An LSTM sequence-to-sequence model with attention.
        temperature (float): parameter for sampling ranging from 0.0 to 1.0.
            0.0: same as argmax, always pick the most probable token
            1.0: sampling from the distribution (can sometimes say random things)
        vocab_file (str): filename of the vocabulary
        vocab_dir (str): path to the vocabulary file
    Returns:
        tuple: (list, list)
            list of lists: token list per sample
            list of floats: log probability per sample
    """
    # define lists to contain samples and probabilities
    samples, log_probs = [], []
    # run a for loop to generate n samples
    for _ in range(n_samples):
        # get a sample using the sampling_decode() function
        sample, logp, _ = sampling_decode(sentence, NMTAttn, temperature, vocab_file=vocab_file, vocab_dir=vocab_dir,
                                          next_symbol=next_symbol)
        # append the token list to the samples list
        samples.append(sample)
        # append the log probability to the log_probs list
        log_probs.append(logp)
    return samples, log_probs


def jaccard_similarity(candidate, reference):
    """Returns the Jaccard similarity between two token lists
    Args:
        candidate (list of int): tokenized version of the candidate translation
        reference (list of int): tokenized version of the reference translation
    Returns:
        float: overlap between the two token lists
    """
    # convert the lists to a set to get the unique tokens
    can_unigram_set, ref_unigram_set = set(candidate), set(reference)
    # get the set of tokens common to both candidate and reference
    joint_elems = can_unigram_set.intersection(ref_unigram_set)
    # get the set of all tokens found in either candidate or reference
    all_elems = can_unigram_set.union(ref_unigram_set)
    # divide the number of joint elements by the number of all elements
    overlap = len(joint_elems) / len(all_elems)
    return overlap


def rouge1_similarity(system, reference):
    """Returns the ROUGE-1 score between two token lists
    Args:
        system (list of int): tokenized version of the system translation
        reference (list of int): tokenized version of the reference translation
    Returns:
        float: overlap between the two token lists
    """
    # make a frequency table of the system tokens (hint: use the Counter class)
    sys_counter = Counter(system)
    # make a frequency table of the reference tokens (hint: use the Counter class)
    ref_counter = Counter(reference)
    # initialize overlap to 0
    overlap = 0
    # run a for loop over the sys_counter object (can be treated as a dictionary)
    for token in sys_counter:
        # lookup the value of the token in the sys_counter dictionary (hint: use the get() method)
        token_count_sys = sys_counter.get(token, 0)
        # lookup the value of the token in the ref_counter dictionary (hint: use the get() method)
        token_count_ref = ref_counter.get(token, 0)
        # update the overlap by getting the smaller number between the two token counts above
        overlap += min(token_count_sys, token_count_ref)
    # get the precision
    precision = overlap / len(reference)
    # get the recall
    overlap2 = 0
    # run a for loop over the sys_counter object (can be treated as a dictionary)
    for token in ref_counter:
        # lookup the value of the token in the sys_counter dictionary (hint: use the get() method)
        token_count_sys = sys_counter.get(token, 0)
        # lookup the value of the token in the ref_counter dictionary (hint: use the get() method)
        token_count_ref = ref_counter.get(token, 0)
        # update the overlap by getting the smaller number between the two token counts above
        overlap2 += min(token_count_sys, token_count_ref)
    recall = overlap2 / len(system)
    if precision + recall != 0:
        # compute the f1-score
        rouge1_score = 2 * (precision * recall) / (precision + recall)
    else:
        rouge1_score = 0
    return rouge1_score


def average_overlap(similarity_fn, samples, *ignore_params):
    """Returns the arithmetic mean of each candidate sentence in the samples
    Args:
        similarity_fn (function): similarity function used to compute the overlap
        samples (list of lists): tokenized version of the translated sentences
        *ignore_params: additional parameters will be ignored
    Returns:
        dict: scores of each sample
            key: index of the sample
            value: score of the sample
    """
    # initialize dictionary
    scores = {}
    # run a for loop for each sample
    for index_candidate, candidate in enumerate(samples):
        # initialize overlap
        overlap = 0
        # run a for loop for each sample
        for index_sample, sample in enumerate(samples):
            # skip if the candidate index is the same as the sample index
            if index_candidate == index_sample:
                continue
            # get the overlap between candidate and sample using the similarity function
            sample_overlap = similarity_fn(candidate, sample)
            # add the sample overlap to the total overlap
            overlap += sample_overlap
        # get the score for the candidate by computing the average
        score = overlap / (len(samples) - 1)
        # save the score in the dictionary. use index as the key.
        scores[index_candidate] = score
    return scores


def weighted_avg_overlap(similarity_fn, samples, log_probs):
    """Returns the weighted mean of each candidate sentence in the samples
    Args:
        samples (list of lists): tokenized version of the translated sentences
        log_probs (list of float): log probability of the translated sentences
    Returns:
        dict: scores of each sample
            key: index of the sample
            value: score of the sample
    """
    # initialize dictionary
    scores = {}
    # runa for loop for each sample
    for index_candidate, candidate in enumerate(samples):
        # initialize overlap and weighted sum
        overlap, weight_sum = 0.0, 0.0
        # run a for loop for each sample
        for index_sample, (sample, logp) in enumerate(zip(samples, log_probs)):
            # skip if the candidate index is the same as the sample index
            if index_candidate == index_sample:
                continue
            # convert log probability to linear scale
            sample_p = float(np.exp(logp))
            # update the weighted sum
            weight_sum += sample_p
            # get the unigram overlap between candidate and sample
            sample_overlap = similarity_fn(candidate, sample)
            # update the overlap
            overlap += sample_p * sample_overlap
        # get the score for the candidate
        score = overlap / weight_sum
        # save the score in the dictionary. use index as the key.
        scores[index_candidate] = score
    return scores


def mbr_decode(sentence, n_samples, score_fn, similarity_fn, NMTAttn=None, temperature=0.6,
               vocab_file=None, vocab_dir=None, generate_samples=generate_samples,
               sampling_decode=sampling_decode, next_symbol=next_symbol, tokenize=tokenize,
               detokenize=detokenize):
    """Returns the translated sentence using Minimum Bayes Risk decoding
    Args:
        sentence (str): sentence to translate.
        n_samples (int): number of samples to generate
        score_fn (function): function that generates the score for each sample
        similarity_fn (function): function used to compute the overlap between a pair of samples
        NMTAttn (tl.Serial): An LSTM sequence-to-sequence model with attention.
        temperature (float): parameter for sampling ranging from 0.0 to 1.0.
            0.0: same as argmax, always pick the most probable token
            1.0: sampling from the distribution (can sometimes say random things)
        vocab_file (str): filename of the vocabulary
        vocab_dir (str): path to the vocabulary file
    Returns:
        str: the translated sentence
    """
    # generate samples
    samples, log_probs = generate_samples(sentence, n_samples, NMTAttn=NMTAttn, temperature=temperature,
                                          vocab_file=vocab_file,
                                          vocab_dir=vocab_dir, sampling_decode=sampling_decode,
                                          next_symbol=next_symbol,
                                          tokenize=tokenize, detokenize=detokenize)
    # use the scoring function to get a dictionary of scores
    # pass in the relevant parameters as shown in the function definition of
    # the mean methods you developed earlier
    scores = score_fn(similarity_fn, samples, log_probs)
    # find the key with the highest score
    max_score_key = max(scores, key=scores.get)
    # detokenize the token list associated with the max_score_key
    translated_sentence = detokenize(samples[max_score_key], vocab_file=vocab_file, vocab_dir=vocab_dir)
    return (translated_sentence, max_score_key, scores)


if __name__ == "__main__":

    # global variables that state the filename and directory of the vocabulary file
    VOCAB_FILE = 'ende_32k.subword'
    VOCAB_DIR = 'data/'

    """  Testing """

    # load the model
    # We will start by first loading in a pre-trained copy of the model

    print("model is loading..")
    # instantiate the model we built in eval mode
    model = NMTAttn(mode='eval')

    # initialize weights from a pre-trained model
    model.init_from_file("model.pkl.gz", weights_only=True)
    model = tl.Accelerate(model)
    print("model is loaded!")

    """     Decoding     """

    #  there are several ways to get the next token when translating a sentence.
    #  For instance, we can just get the most probable token at each step (i.e. greedy decoding)
    #  or get a sample from a distribution.
    # Run it several times with each setting and see how often the output changes.
    your_sentence = 'I go to the office everyday.'
    cur_output_tokens, log_prob, sentence = sampling_decode(your_sentence,
                                                            NMTAttn=model,
                                                            temperature=0.5,
                                                            vocab_file=VOCAB_FILE,
                                                            vocab_dir=VOCAB_DIR)
    # show the output
    print( "English sentence is: ", your_sentence)
    print("with random sampling decoding and temperature of 0.5:")
    print("translation will be: ", cur_output_tokens, log_prob, sentence, sep="\n")

    # when the temperature is 0 then it's greedy decoding
    cur_output_tokens, log_prob, sentence = sampling_decode(your_sentence,
                                                            NMTAttn=model,
                                                            temperature=0.0,
                                                            vocab_file=VOCAB_FILE,
                                                            vocab_dir=VOCAB_DIR)
    print("English sentence is: ", your_sentence)
    print("with greedy decoding when temperature is 0.0:")
    print("translation will be: ", cur_output_tokens, log_prob, sentence, sep="\n")

    """ Minimum Bayes-Risk Decoding """

    # getting the most probable token at each step may not necessarily produce
    # the best results. Another approach is to do Minimum Bayes Risk Decoding or MBR.
    # The general steps to implement this are:
    # take several random samples
    # score each sample against all other samples
    # select the one with the highest score

    TEMPERATURE = 1.0
    # put a custom string here
    your_sentence = 'She speaks English and French'
    translated_sentence, max_score_key, scores = mbr_decode(your_sentence,
                                                            n_samples=4,
                                                            score_fn=average_overlap,
                                                            similarity_fn=rouge1_similarity,
                                                            NMTAttn=model,
                                                            temperature=TEMPERATURE,
                                                            vocab_file=VOCAB_FILE,
                                                            vocab_dir=VOCAB_DIR)

    print("English sentence is: ", your_sentence)
    print("with MBR decoding with  (temperature=1, similarity_fn=rouge, score_fn=average_overlap): ")
    print("translation will be: ", translated_sentence)

