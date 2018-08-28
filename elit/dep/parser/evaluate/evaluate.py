import os
import tempfile
import time
from functools import reduce

import sys

from elit.dep.common.utils import stdchannel_redirected
from elit.dep.parser.biaffine_parser import BiaffineParser

with stdchannel_redirected(sys.stderr, os.devnull):
    import dynet as dy

from elit.dep.parser.common import DataLoader


def evaluate_official_script(parser: BiaffineParser, vocab, num_buckets_test, test_batch_size, test_file, output_file,
                             debug=False, documents=None):
    if output_file is None:
        output_file = tempfile.NamedTemporaryFile().name
    data_loader = DataLoader(test_file, num_buckets_test, vocab, documents)
    record = data_loader.idx_sequence
    results = [None] * len(record)
    idx = 0
    seconds = time.time()
    for chars, cased_words, words, tags, arcs, rels in data_loader.get_batches(batch_size=test_batch_size,
                                                                               shuffle=False):
        dy.renew_cg()
        outputs = parser.run(chars, cased_words, words, tags, is_train=False)
        for output in outputs:
            sent_idx = record[idx]
            results[sent_idx] = output
            idx += 1
    seconds = time.time() - seconds
    speed = len(record) / seconds

    arcs = reduce(lambda x, y: x + y, [list(result[0]) for result in results])
    rels = reduce(lambda x, y: x + y, [list(result[1]) for result in results])
    idx = 0
    with open(test_file) as f:
        if debug:
            f = f.readlines()[:1000]
        with open(output_file, 'w') as fo:
            for line in f:
                info = line.strip().split()
                if info:
                    assert len(info) == 10, 'Illegal line: %s' % line
                    info[6] = str(arcs[idx])
                    info[7] = vocab.id2rel(rels[idx])
                    fo.write('\t'.join(info) + '\n')
                    idx += 1
                else:
                    fo.write('\n')

    os.system('perl %s -q -b -g %s -s %s -o tmp' % (
        os.path.join(os.path.dirname(os.path.realpath(__file__)), 'eval.pl'), test_file, output_file))
    os.system('tail -n 3 tmp > score_tmp')
    LAS, UAS = [float(line.strip().split()[-2]) for line in open('score_tmp').readlines()[:2]]
    # print('UAS %.2f, LAS %.2f' % (UAS, LAS))
    os.system('rm tmp score_tmp')
    os.remove(output_file)
    return UAS, LAS, speed


if __name__ == '__main__':
    cmd = 'perl %s -q -b -g %s -s %s -o tmp' % (
        os.path.join(os.path.dirname(os.path.realpath(__file__)), 'eval.pl'), 'test_file',
        'output_file')
    print(cmd)
