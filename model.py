import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.nn.utils import clip_grad_norm
import torchtext.data as data
import random
import matplotlib.pyplot as plt

class Encoder(nn.Module):

    def __init__(self, vocab_size, d_embed, d_hidden):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_embed)
        self.lstm = nn.LSTMCell(d_embed, d_hidden)
        self.d_hidden = d_hidden

    def forward(self, x_seq, cuda=False):
        o = []
        e_seq = self.embedding(x_seq)  # seq x batch x dim
        tt = torch.cuda if cuda else torch  # use cuda tensor or not
        # create initial hidden state and initial cell state
        h = Variable(tt.FloatTensor(e_seq.size(1), self.d_hidden).zero_())
        c = Variable(tt.FloatTensor(e_seq.size(1), self.d_hidden).zero_())
        
        for e in e_seq.chunk(e_seq.size(0), 0):
            e = e.squeeze(0)
            h, c = self.lstm(e, (h, c))
            o.append(h)
        return torch.stack(o, 0), h, c
    
    

class Attention(nn.Module):
    """Dot global attention from https://arxiv.org/abs/1508.04025"""
    def __init__(self, dim):
        super(Attention, self).__init__()
        self.dim = dim
        self.linear = nn.Linear(dim*2, dim, bias=False)
        
    def forward(self, x, context=None):
        if context is None:
            return x
        assert x.size(0) == context.size(0)  # x: batch x dim
        assert x.size(1) == context.size(2)  # context: batch x seq x dim
        # print("context: ", context.size()) # context:  torch.Size([10, 12, 500])
        # print("x: ", x.size()) # x:  torch.Size([10, 500])
        # print("x.unsqueeze(2): ", x.unsqueeze(2).size()) # x.unsqueeze(2):  torch.Size([10, 500, 1])
        # print("for softmax: ", context.bmm(x.unsqueeze(2)).squeeze(2).size()) # for softmax:  torch.Size([10, 12])
        attn = F.softmax(context.bmm(x.unsqueeze(2)).squeeze(2), dim=1) # version update
        # print("attn.unsqueeze(1): ", attn.unsqueeze(1).size()) # batch x 1 x seq 
        # print("context: ", context.size())
        weighted_context = attn.unsqueeze(1).bmm(context).squeeze(1)
        o = self.linear(torch.cat((x, weighted_context), 1))
        return torch.tanh(o) # version update
    
    
    
class Decoder(nn.Module):
    
    def __init__(self, vocab_size, d_embed, d_hidden):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_embed)
        self.lstm = nn.LSTMCell(d_embed, d_hidden)
        self.attn = Attention(d_hidden)
        self.linear = nn.Linear(d_hidden, vocab_size)

    def forward(self, x_seq, h, c, context=None):
        o = []
        e_seq = self.embedding(x_seq)
        for e in e_seq.chunk(e_seq.size(0), 0):
            e = e.squeeze(0)
            h, c = self.lstm(e, (h, c))
            o.append(self.attn(h, context))
        o = torch.stack(o, 0)
        # print("o: ", o.view(-1, h.size(1)).size()) # o:  torch.Size([110, 500])
        o = self.linear(o.view(-1, h.size(1)))
        # print("o2: ", o.size()) # o2:  torch.Size([110, 73])
        return F.log_softmax(o, dim=1).view(x_seq.size(0), -1, o.size(1)), h, c # version update
    
    

class G2P(nn.Module):
    
    def __init__(self, config):
        super(G2P, self).__init__()
        self.encoder = Encoder(config.g_size, config.d_embed,
                               config.d_hidden)
        self.decoder = Decoder(config.p_size, config.d_embed,
                               config.d_hidden)
        self.config = config
        
    def forward(self, g_seq, p_seq=None):
        o, h, c = self.encoder(g_seq, self.config.cuda)
        # print(o.size()) # o.size() = seq x batch x dim
        # context: batch x seq x dim
        # transpose for dim batch and dim seq by permute()
        context = o.permute(1, 0, 2) if self.config.attention else None # version update
        if p_seq is not None:  # not generate
            return self.decoder(p_seq, h, c, context)
        else:
            assert g_seq.size(1) == 1  # make sure batch_size = 1
            return self._generate(h, c, context)
        
    def _generate(self, h, c, context):
        beam = Beam(self.config.beam_size, cuda=self.config.cuda)
        # Make a beam_size batch.
        h = h.expand(beam.size, h.size(1))  
        c = c.expand(beam.size, c.size(1))
        context = context.expand(beam.size, context.size(1), context.size(2))
        
        for i in range(self.config.max_len):  # max_len = 20
            x = beam.get_current_state()
            o, h, c = self.decoder(Variable(x.unsqueeze(0)), h, c, context)
            if beam.advance(o.data.squeeze(0)):
                break
            h.data.copy_(h.data.index_select(0, beam.get_current_origin()))
            c.data.copy_(c.data.index_select(0, beam.get_current_origin()))
        tt = torch.cuda if self.config.cuda else torch
        return Variable(tt.LongTensor(beam.get_hyp(0)))
    
    
    
# Based on https://github.com/MaximumEntropy/Seq2Seq-PyTorch/
class Beam(object):
    """Ordered beam of candidate outputs."""

    def __init__(self, size, pad=1, bos=2, eos=3, cuda=False):
        """Initialize params."""
        self.size = size
        self.done = False
        self.pad = pad
        self.bos = bos
        self.eos = eos
        self.tt = torch.cuda if cuda else torch

        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(size).zero_()

        # The backpointers at each time-step.
        self.prevKs = []

        # The outputs at each time-step.
        self.nextYs = [self.tt.LongTensor(size).fill_(self.pad)]
        self.nextYs[0][0] = self.bos

    # Get the outputs for the current timestep.
    def get_current_state(self):
        """Get state of beam."""
        return self.nextYs[-1]

    # Get the backpointers for the current timestep.
    def get_current_origin(self):
        """Get the backpointer to the beam at this step."""
        return self.prevKs[-1]

    def advance(self, workd_lk):
        """Advance the beam."""
        num_words = workd_lk.size(1)

        # Sum the previous scores.
        if len(self.prevKs) > 0:
            beam_lk = workd_lk + self.scores.unsqueeze(1).expand_as(workd_lk)
        else:
            beam_lk = workd_lk[0]

        flat_beam_lk = beam_lk.view(-1)

        bestScores, bestScoresId = flat_beam_lk.topk(self.size, 0,
                                                     True, True)
        self.scores = bestScores

        # bestScoresId is flattened beam x word array, so calculate which
        # word and beam each score came from
        prev_k = bestScoresId // num_words
        self.prevKs.append(prev_k)
        self.nextYs.append(bestScoresId - prev_k * num_words)
        # End condition is when top-of-beam is EOS.
        if self.nextYs[-1][0] == self.eos:
            self.done = True
        return self.done

    def get_hyp(self, k):
        """Get hypotheses."""
        hyp = []
        # print(len(self.prevKs), len(self.nextYs), len(self.attn))
        for j in range(len(self.prevKs) - 1, -1, -1):
            hyp.append(self.nextYs[j + 1][k])
            k = self.prevKs[j][k]
        return hyp[::-1]
    
    
# Based on https://github.com/SeanNaren/deepspeech.pytorch/blob/master/decoder.py.
import Levenshtein  # https://github.com/ztane/python-Levenshtein/

def phoneme_error_rate(p_seq1, p_seq2):
    p_vocab = set(p_seq1 + p_seq2)
    p2c = dict(zip(p_vocab, range(len(p_vocab))))
    c_seq1 = [chr(p2c[p]) for p in p_seq1]
    c_seq2 = [chr(p2c[p]) for p in p_seq2]
    return Levenshtein.distance(''.join(c_seq1),
                                ''.join(c_seq2)) / len(c_seq2)


def adjust_learning_rate(optimizer, lr_decay):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= lr_decay
        
        
def grapheme_add_noise(prob,batch, dic):
    #TODO
    # rework on the adding noise function
    
    # find index of vowel consonant
    vowel = []
    consonant = []
    for i, letter in enumerate(dic):
        if not letter.isalpha():
            continue
        else:
            if letter in ['a', 'o', 'e', 'u', 'i']:
                vowel.append(i)
            else:
                consonant.append(i)
        pass
    
    grapheme = batch.T
    
    for i, g in enumerate(grapheme):
        for j, letter in enumerate(g):
            #if the letter is in alphabet
            if dic[letter].isalpha():
                # randomly flip prob percent of letter
                rand = random.randint(1,100)

                if rand <= prob*100:
                    #flip the char
                    if letter in vowel:
                        rand_index = random.randint(0, len(vowel) - 1)
                        grapheme[i][j] = vowel[rand_index]
                    else:
                        rand_index = random.randint(0, len(consonant) - 1)
                        grapheme[i][j] = consonant[rand_index]
    
    return grapheme.T


class CMUDict(data.Dataset):

    def __init__(self, data_lines, g_field, p_field):
        fields = [('grapheme', g_field), ('phoneme', p_field)]
        examples = []  # maybe ignore '...-1' grapheme
        for line in data_lines:
            grapheme, phoneme = line.split(maxsplit=1)
            examples.append(data.Example.fromlist([grapheme, phoneme],
                                                  fields))
        self.sort_key = lambda x: len(x.grapheme)
        super(CMUDict, self).__init__(examples, fields)

        
        
def prepare_data(train_lines, val_lines, test_lines, args):
    #initialize g_field and p_field
    g_field = data.Field(init_token='<s>',
                         tokenize=(lambda x: list(x.split('(')[0])[::-1]))
    p_field = data.Field(init_token='<os>', eos_token='</os>',
                         tokenize=(lambda x: x.split('#')[0].split()))
    
    
    # Build dataset
    train_data = CMUDict(train_lines, g_field, p_field)
    val_data = CMUDict(val_lines, g_field, p_field)
    test_data = CMUDict(test_lines, g_field, p_field)
    
    g_field.build_vocab(train_data, val_data, test_data)
    p_field.build_vocab(train_data, val_data, test_data)
    
    # Build dataset iter
    device = torch.device("cuda")# None if args.cuda else -1  # None is current gpu
    train_iter = data.BucketIterator(train_data, batch_size=args.batch_size,
                                     repeat=False, device=device)
    val_iter = data.Iterator(val_data, batch_size=1,
                             train=False, sort=False, device=device)
    test_iter = data.Iterator(test_data, batch_size=1,
                              train=False, shuffle=True, device=device)
    
    return train_iter, val_iter, test_iter, g_field, p_field


def grapheme_add_noise2(prob, test_l):
    vowel = ['a','o','e','i','u']
    consonant = ['q','w','r','t','y','p','s','d','f','g','h','j','k','l','z','x','c','v','b','n','m']
    
    for i,pair in enumerate(test_l):
        for j,char in enumerate(pair):
            #process only if we haven't encounter space (process grapheme)
            if not char.isspace():
                #replace only if char is alphabet (ignore special characters)
                if char.isalpha():
                    rand = random.randint(1,100)
                    #replace only if random number is within the prob
                    if rand <= prob*100:
                        #replace with random non-capital alphabetical letters
                        if char in vowel:
                            rand_index = random.randint(0,4)
                            test_l[i] = test_l[i][:j] + vowel[rand_index] +test_l[i][j+1:]
                        else:
                            rand_index = random.randint(0,20)
                            test_l[i] = test_l[i][:j] + consonant[rand_index] +test_l[i][j+1:]       
                        
            #if we encounter space break to next pair of grapheme and phoneme
            else:
                break
    
    return test_l


def prepare_data2(test_lines, noise,g_field, p_field):

    #add noise to test set
    test_l = test_lines[:]
    test_l = grapheme_add_noise2(noise, test_l)
    
    # Build dataset
    test_data = CMUDict(test_l, g_field, p_field)
    
    # Build dataset iter
    device = torch.device("cuda")# None if args.cuda else -1  # None is current gpu
    test_iter = data.Iterator(test_data, batch_size=1,
                              train=False, shuffle=True, device=device)
    
    return test_iter