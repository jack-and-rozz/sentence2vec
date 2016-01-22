# -*- coding: utf-8 -*-
#
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html


"""

"""

import logging
import sys
import os
import argparse
import codecs
from word2vec import Word2Vec, Sent2Vec, LineSentence

logging.basicConfig(format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s', level=logging.INFO)
logging.info("running %s" % " ".join(sys.argv))

def main(args):
    sv_file = args.sv_file
    wv_file = args.wv_file
    size = int(args.size)
    window=int(args.window)
    min_count=int(args.min_count)
    cbow = int(args.cbow)
    sg = 0 if cbow == 1 else 1
    n_workers = 24
    hs = 0 # とりあえずhsは使わないでおく
    print wv_file
    if wv_file:
        model = Word2Vec.load_word2vec_format(wv_file + '.wv', fvocab=wv_file +'.vocab')
    else:
        wv_file = sv_file
        model = Word2Vec(LineSentence(wv_file), size=size, window=window, sg=sg, min_count=min_count, workers=n_workers)

    model.save(wv_file + '.wv_model')

    model = Sent2Vec(LineSentence(sv_file), model_file=wv_file + '.wv_model',sg=sg, hs=hs)
    model.save_sent2vec_format(sv_file + '.sv')

    program = os.path.basename(sys.argv[0])
    logging.info("finished running %s" % program)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('sv_file', default=None )
    parser.add_argument('--wv_file', default=None )
    parser.add_argument('--size', default=200 ,help ='')
    parser.add_argument('--cbow', default=0 ,help ='')
    parser.add_argument('--window', default=5 ,help ='')
    parser.add_argument('--min_count', default=1 ,help ='')
    
    args = parser.parse_args()
    main(args)
