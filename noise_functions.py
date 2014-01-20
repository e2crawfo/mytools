#noise functions
import numpy as np
import matplotlib.pyplot as plt
import hrr

def default_noise(vec):
    vector = np.random.rand(len(vec)) - 0.5
    vector = vector / np.linalg.norm(vector)
    return vector

def flip_noise(vec):
    return -vec

def ortho_vector(vec):
    dim = len(vec)
    ortho = hrr.HRR(dim).v
    proj = np.dot(vec, ortho) * vec
    ortho -= proj
    assert np.allclose([np.dot(ortho, vec)], [0])
    return ortho

import random
def make_hrr_noise(D, num):
    def hrr_noise(input_vec):
        noise_vocab = hrr.Vocabulary(D)
        keys = [noise_vocab.parse(str(x)) for x in range(2*num+1)]

        input_vec = hrr.HRR(data=input_vec)
        partner_key = random.choice(keys)

        pair_keys = filter(lambda x: x != partner_key, keys)

        pairs = random.sample(pair_keys, 2 * num)
        p0 = (pairs[x] for x in range(0,len(pairs),2))
        p1 = (pairs[x] for x in range(1,len(pairs),2))
        S = map(lambda x, y: noise_vocab[x].convolve(noise_vocab[y]), p0, p1)

        S = reduce(lambda x, y: x + y, S, noise_vocab[partner_key].convolve(input_vec))
        S.normalize()

        vec_hrr = S.convolve(~noise_vocab[partner_key])
        return vec_hrr.v
    return hrr_noise

def output(trial_length, main, main_vector, alternate, noise_func=default_noise):
    tick = 0
    vector = main_vector
    main_hrr = hrr.HRR(data=main_vector)

    while True:
        if tick == trial_length :
            tick = 0
            if main:
                vector = main_vector
            else:
                vector = noise_func(main_vector)
                u = hrr.HRR(data=vector)
                similarity = u.compare(main_hrr)
                print "Sim:", similarity

            if alternate:
                main = not main

        tick += 1

        yield vector

def interpolator(end_time, start_vec, end_vec, time_func=lambda x: x):
    tick = 0
    while True:
        t = time_func(tick)
        t = min(t, end_time)

        vector = (end_time - t) * start_vec + t * end_vec
        vector = vector / np.linalg.norm(vector)
        tick += 1

        yield vector


def make_f(generators, times):
    last_time = [0.0]
    def f(t):
        if len(generators) > 1 and t > times[0] + last_time[0]:
            generators.pop(0)
            last_time[0] += times.pop(0)
        return generators[0].next()
    return f

