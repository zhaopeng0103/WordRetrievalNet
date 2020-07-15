#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-

import tqdm
import numpy as np
import scipy.fftpack as sf


def get_uni_grams_from_words(word_strings, split_character=None):
    """ count unique char set from all words
        Input:
            word_strings   : all annotation words
            split_character: split char between single word, word split to char
        Output:
            uni_grams      : a unique char set
    """
    if split_character is not None:
        uni_grams = [elem for word_string in word_strings for elem in word_string.replace(" ", "").split(split_character)]
    else:
        uni_grams = [elem for word_string in word_strings for elem in word_string.replace(" ", "")]
    uni_grams = sorted(set(uni_grams))
    print("Get_Uni_Grams_From_Words::uni_grams is {0}".format(uni_grams))
    print("Get_Uni_Grams_From_Words::uni_grams size is {0}".format(len(uni_grams)))
    return uni_grams


def get_n_grams(word, len_ngram):
    """ Calculates list of n_grams for a given word.
        Input:
            word     : Word to calculate n_grams for. (str)
            len_ngram: Maximal ngram size: n=3 extracts 1-, 2- and 3-grams. (int)
        Output:
            n_grams  : List of ngrams as strings.
    """
    return [word[i:i + len_ngram] for i in range(len(word) - len_ngram + 1)]


def get_most_common_n_grams(word_strings, num_results=50, len_ngram=2):
    """ Calculates the 50 (default) most common bi_grams (default) from a list of pages
        Input:
            word_strings : (list of str) List containing the words from which to extract the bi_grams
            num_results  : Number of n-grams returned.
            len_ngram    : length of n-grams.
        Output:
            n_grams      : most common <n>-grams
    """
    n_grams = {}
    for word in word_strings:
        word_n_grams = get_n_grams(word, len_ngram)
        for ngram in word_n_grams:
            n_grams[ngram] = n_grams.get(ngram, 0) + 1
    sorted_list = sorted(n_grams.items(), key=lambda x: x[1], reverse=True)
    top_n_grams = sorted_list[:num_results]
    return {k: i for i, (k, _) in enumerate(top_n_grams)}


def build_embedding_descriptor(words, embedding_uni_gram_levels=(1, 2, 4, 8),
                               embedding_bi_grams=None, embedding_bi_gram_levels=None,
                               resolution=3, embedding_type='dctow', on_unknown_uni_gram='nothing'):
    """ Calculate Discrete Cosine Transform of Words (DCToW) embedding descriptor.
        Input:
            words              : word to calculate descriptor for (list of str)
            embedding_uni_gram_levels    : the levels to use in the embedding (list of int)
            on_unknown_uni_gram : What to do if a unigram appearing in a word is not among the supplied phoc_unigrams.
                Possible: 'warn', 'error', 'nothing' (str)
            embedding_type     : the type of the embedding to be build. The default is the phoc. Possible: phoc, spoc
        Output:
            embedding_size     : embedding size of given words
            embeddings_map     : the embedding map for the given words
    """
    if on_unknown_uni_gram not in ['error', 'warn', 'nothing']:
        raise ValueError('I don\'t know the on_unknown_uni_gram parameter \'%s\'' % on_unknown_uni_gram)
    if embedding_type not in ['dctow', 'phoc', 'spoc']:
        raise ValueError('I don\'t know the embedding_type parameter \'%s\'' % embedding_type)

    embedding_uni_grams = [chr(i) for i in list(range(ord('a'), ord('z') + 1)) + list(range(ord('0'), ord('9') + 1))]

    embeddings_map = {}
    if embedding_type == "dctow":
        embedding_uni_grams = "".join(embedding_uni_grams)
        embedding_size = len(embedding_uni_grams) * resolution

        # iterate through all the words
        for word in tqdm.tqdm(words):
            im = np.zeros([len(embedding_uni_grams), len(word)], 'single')
            F = np.zeros([len(embedding_uni_grams), len(word)], 'single')
            for jj in range(0, len(word)):
                c = word[jj]
                im[str.find(embedding_uni_grams, c), jj] = 1.0

            for ii in range(0, len(embedding_uni_grams)):
                F[ii, :] = sf.dct(im[ii, :])

            A = F[:, 0:resolution]
            B = np.zeros([len(embedding_uni_grams), max(0, resolution - len(word))])
            embeddings_map[word] = np.hstack((A, B)).flatten()
    else:
        embedding_size = len(embedding_uni_grams) * np.sum(embedding_uni_gram_levels)
        if embedding_bi_grams is not None:
            embedding_size += len(embedding_bi_grams) * np.sum(embedding_bi_gram_levels)

        # prepare some lambda functions
        occupancy = lambda k, m: [float(k) / m, float(k + 1) / m]
        overlap = lambda a, b: [max(a[0], b[0]), min(a[1], b[1])]
        size = lambda the_region: the_region[1] - the_region[0]

        # map from character to alphabet position
        char_indices = {d: i for i, d in enumerate(embedding_uni_grams)}

        # iterate through all the words
        for word in tqdm.tqdm(words):
            embeddings_map[word] = np.zeros(embedding_size)
            n = len(word)
            for index, char in enumerate(word):
                char_occ = occupancy(index, n)
                if char not in char_indices:
                    if on_unknown_uni_gram == 'warn':
                        print('The uni_gram \'{0}\' is unknown, skipping this character'.format(char))
                        continue
                    elif on_unknown_uni_gram == 'error':
                        print('The uni_gram \'{0}\' is unknown'.format(char))
                        raise ValueError()
                    else:
                        continue
                char_index = char_indices[char]
                for level in embedding_uni_gram_levels:
                    for region in range(level):
                        region_occ = occupancy(region, level)
                        if size(overlap(char_occ, region_occ)) / size(char_occ) >= 0.5:
                            feat_vec_index = sum([l for l in embedding_uni_gram_levels if l < level]) * len(embedding_uni_grams) + \
                                             region * len(embedding_uni_grams) + char_index
                            if embedding_type == 'phoc':
                                embeddings_map[word][feat_vec_index] = 1
                            elif embedding_type == 'spoc':
                                embeddings_map[word][feat_vec_index] += 1
                            else:
                                raise ValueError('The embedding_type \'%s\' is unknown' % embedding_type)
            # add bi_grams
            if embedding_bi_grams is not None:
                n_gram_features = np.zeros(len(embedding_bi_grams) * np.sum(embedding_bi_gram_levels))
                n_gram_occupancy = lambda k, m: [float(k) / m, float(k + 2) / m]
                for i in range(n - 1):
                    ngram = word[i:i + 2]
                    if embedding_bi_grams.get(ngram, 0) == 0:
                        continue
                    occ = n_gram_occupancy(i, n)
                    for level in embedding_bi_gram_levels:
                        for region in range(level):
                            region_occ = occupancy(region, level)
                            overlap_size = size(overlap(occ, region_occ)) / size(occ)
                            if overlap_size >= 0.5:
                                if embedding_type == 'phoc':
                                    n_gram_features[region * len(embedding_bi_grams) + embedding_bi_grams[ngram]] = 1
                                elif embedding_type == 'spoc':
                                    n_gram_features[region * len(embedding_bi_grams) + embedding_bi_grams[ngram]] += 1
                                else:
                                    raise ValueError('The phoc_type \'%s\' is unknown' % embedding_type)
                embeddings_map[word][-n_gram_features.shape[0]:] = n_gram_features
    print("Build_Embedding_Descriptor::embedding type is {0}; embedding size is {1}".format(embedding_type, embedding_size))
    return embedding_size, embeddings_map
