# -*- coding: utf-8 -*-
#
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html


"""

"""

import logging
import sys
import os
from word2vec import Word2Vec, Sent2Vec, LineSentence

logging.basicConfig(format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s', level=logging.INFO)
logging.info("running %s" % " ".join(sys.argv))

DIR = 'w2vdata/'

wordvec_file = DIR + 'jtest.lemma'
#model = Word2Vec(LineSentence(wordvec_file), size=100, window=5, sg=1, min_count=1, workers=8)
#model.save(wordvec_file + '.model')
#model.save_word2vec_format(wordvec_file + '.wv')

# load pretrained wv 
#model = Word2Vec(None , size=100, window=5, sg=0, min_count=5, workers=8)
model = Word2Vec.load_word2vec_format(wordvec_file + '.wv', fvocab=wordvec_file + '.vocab')

model.save(wordvec_file + '.model')

sent_file = wordvec_file
#model = Sent2Vec(LineSentence(sent_file), model_file=wordvec_file + '.model',sg=1, hs=0)

#学習済みの単語ベクトルを用いる場合は sg, hsなどのオプションを合わせる
model = Sent2Vec(LineSentence(sent_file), model_file=wordvec_file + '.model',sg=1, hs=0)
model.save_sent2vec_format(sent_file + '.sv')

program = os.path.basename(sys.argv[0])
logging.info("finished running %s" % program)
