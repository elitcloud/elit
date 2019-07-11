# coding: utf-8

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import os
import tempfile
import time
from collections import namedtuple
from functools import reduce
import numpy as np
from elit.component.dep.common.data import DataLoader
from elit.component.dep.common.utils import _save_conll
from elit.component.sdp.data import SDPDataLoader


def prf(correct, pred_sum, gold_sum):
    if pred_sum:
        p = correct / pred_sum
    else:
        p = 0
    if gold_sum:
        r = correct / gold_sum
    else:
        r = 0
    if p + r:
        f = 2 * p * r / (p + r)
    else:
        f = 0
    return p, r, f


def evaluate_official_script(parser, vocab, num_buckets_test, test_batch_size, test_file, output_file,
                             debug=False):
    """Evaluate parser on a data set

    Parameters
    ----------
    parser : BiaffineParser
        biaffine parser
    vocab : ParserVocabulary
        vocabulary built from data set
    num_buckets_test : int
        size of buckets (cluster sentences into this number of clusters)
    test_batch_size : int
        batch size
    test_file : str
        gold test file
    output_file : str
        output result to this file
    debug : bool
        only evaluate first 1000 sentences for debugging

    Returns
    -------
    tuple
        UAS, LAS, speed
    """
    if output_file is None:
        output_file = tempfile.NamedTemporaryFile().name
    if isinstance(test_file, list):
        tmp_file = tempfile.NamedTemporaryFile().name
        _save_conll(test_file, tmp_file)
        test_file = tmp_file

    data_loader = DataLoader(test_file, num_buckets_test, vocab)
    record = data_loader.idx_sequence
    results = [None] * len(record)
    idx = 0
    seconds = time.time()
    for words, tags, arcs, rels in data_loader.get_batches(batch_size=test_batch_size,
                                                           shuffle=False):
        outputs = parser.forward(words, tags)
        for output in outputs:
            sent_idx = record[idx]
            results[sent_idx] = output
            idx += 1
    assert idx == len(results), 'parser swallowed some sentences'
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
                    arc_offset = 5
                    rel_offset = 6
                    if len(info) == 10:  # conll or conllx
                        arc_offset = 6
                        rel_offset = 7
                    # assert len(info) == 10, 'Illegal line: %s' % line
                    info[arc_offset] = str(arcs[idx])
                    info[rel_offset] = vocab.id2rel(rels[idx])
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


def compute_F1(gold_file, sys_file, labeled=False):
    """
    Adopted from https://github.com/tdozat/Parser-v3/blob/master/scripts/semdep_eval.py#L29
    :param gold_file:
    :param sys_file:
    :param labeled:
    :return:
    """

    correct = 0
    predicted = 0
    actual = 0
    n_tokens = 0
    n_sequences = 0
    current_seq_correct = False
    n_correct_sequences = 0
    current_fp = 0
    current_sent = 0

    with open(gold_file, encoding='utf-8') as gf, open(sys_file, encoding='utf-8') as sf:
        gold_line = gf.readline()
        gold_i = 1
        sys_i = 0
        while gold_line:
            while gold_line.startswith('#'):
                current_sent += 1
                gold_i += 1
                n_sequences += 1
                n_correct_sequences += current_seq_correct
                current_seq_correct = True
                gold_line = gf.readline()
            if gold_line.rstrip() != '':
                sys_line = sf.readline()
                sys_i += 1
                while sys_line.startswith('#') or sys_line.rstrip() == '' or sys_line.split('\t')[0] == '0':
                    sys_line = sf.readline()
                    sys_i += 1

                gold_line = gold_line.rstrip().split('\t')
                sys_line = sys_line.rstrip().split('\t')
                assert len(sys_line) == 10, 'line {} format corrupted {}'.format(sys_i, '\t'.join(sys_line))
                assert sys_line[1] == gold_line[1], 'Files are misaligned at lines {}, {}'.format(gold_i, sys_i)

                # Compute the gold edges
                gold_node = gold_line[8]
                if gold_node != '_':
                    gold_node = gold_node.split('|')
                    if labeled:
                        gold_edges = set(tuple(gold_edge.split(':', 1)) for gold_edge in gold_node)
                    else:
                        gold_edges = set(gold_edge.split(':', 1)[0] for gold_edge in gold_node)
                else:
                    gold_edges = set()

                # Compute the sys edges
                sys_node = sys_line[8]
                if sys_node != '_':
                    sys_node = sys_node.split('|')
                    if labeled:
                        sys_edges = set(tuple(sys_edge.split(':', 1)) for sys_edge in sys_node)
                    else:
                        sys_edges = set(sys_edge.split(':', 1)[0] for sys_edge in sys_node)
                else:
                    sys_edges = set()

                correct_edges = gold_edges & sys_edges
                if len(correct_edges) != len(gold_edges):
                    current_seq_correct = False
                correct += len(correct_edges)
                predicted += len(sys_edges)
                actual += len(gold_edges)
                n_tokens += 1

                # current_fp += len(sys_edges) - len(gold_edges & sys_edges)
            gold_line = gf.readline()
            gold_i += 1
        # print(correct, predicted - correct, actual - correct)
    Accuracy = namedtuple('Accuracy', ['precision', 'recall', 'F1', 'seq_acc'])
    precision = correct / (predicted + 1e-12)
    recall = correct / (actual + 1e-12)
    F1 = 2 * precision * recall / (precision + recall + 1e-12)
    seq_acc = n_correct_sequences / n_sequences
    return Accuracy(precision, recall, F1, seq_acc)


def evaluate_sdp(parser, vocab, num_buckets_test, test_batch_size, test_file, output_file,
                 bert=None, debug=False):
    """Evaluate parser on a data set

    Parameters
    ----------
    parser : BiaffineParser
        biaffine parser
    vocab : ParserVocabulary
        vocabulary built from data set
    num_buckets_test : int
        size of buckets (cluster sentences into this number of clusters)
    test_batch_size : int
        batch size
    test_file : str
        gold test file
    output_file : str
        output result to this file
    debug : bool
        verify the scorer

    Returns
    -------
    tuple
        LF, speed
    """
    # if output_file is None:
    #     output_file = tempfile.NamedTemporaryFile().name
    data_loader = SDPDataLoader(test_file, num_buckets_test, vocab, bert=bert)
    record = data_loader.idx_sequence
    results = [None] * len(record)
    idx = 0
    seconds = time.time()
    uc, up, ug, lc, lp, lg = 0, 0, 0, 0, 0, 0
    for words, bert, tags, arcs, rels in data_loader.get_batches(batch_size=test_batch_size,
                                                                 shuffle=False):
        outputs = parser.forward(words, bert, tags)
        for output, gold_arc, gold_rel in zip(outputs, arcs.transpose([2, 0, 1]), rels.transpose([2, 0, 1])):
            pred_arc = output[0][:, 1:].asnumpy()
            pred_rel = output[1][:, 1:].asnumpy()
            length = pred_arc.shape[0]
            gold_arc = gold_arc[:length, 1:length]
            gold_rel = gold_rel[:length, 1:length]

            gold_mask = np.greater(gold_rel, 0)
            correct = np.sum(np.equal(pred_rel, gold_rel) * gold_mask)
            pred_sum = np.sum(np.greater(pred_rel, 0))
            gold_sum = gold_mask.sum()
            lc += correct
            lp += pred_sum
            lg += gold_sum

            correct = np.sum(pred_arc * gold_arc)
            pred_sum = pred_arc.sum()
            gold_sum = gold_arc.sum()
            uc += correct
            up += pred_sum
            ug += gold_sum

        for output in outputs:
            sent_idx = record[idx]
            results[sent_idx] = output
            idx += 1
    UP, UR, UF = prf(uc, up, ug)
    LP, LR, LF = prf(lc, lp, lg)
    # print('%.1f' % (LF * 100))
    assert idx == len(results), 'parser swallowed some sentences'
    seconds = time.time() - seconds
    speed = len(record) / seconds
    if not debug:
        return UF, LF, speed

    idx = 0
    with open(test_file) as f:
        # if debug:
        #     f = f.readlines()[:1000]
        with open(output_file, 'w') as out:
            sent = []
            for line in f:
                if line.startswith('#'):
                    continue
                info = line.strip().split()
                if info:
                    assert len(info) == 10, 'Illegal line: %s' % line
                    sent.append(info[:6])
                else:
                    # now we have read one sentence, output it with our prediction
                    output_sent(data_loader, idx, out, results, sent)
                    idx += 1
                    sent = []
            if info:
                output_sent(data_loader, idx, out, results, sent)

    # os.remove(output_file)
    UAS = compute_F1(test_file, output_file, labeled=False)
    LAS = compute_F1(test_file, output_file, labeled=True)
    if abs(UAS.F1 - UF) > 1e-3 or abs(LAS.F1 - LF) > 1e-3:
        print('Warning: UAS.F1=%.1f UF=%.1f LAS.F1=%.1f LF=%.1f' % (UAS.F1 * 100, UF * 100, LAS.F1 * 100, LF * 100))
    return UAS.F1, LAS.F1, speed


def output_sent(data_loader, idx, out, results, sent):
    arcs, rels = results[idx]
    length = arcs.shape[0]
    for i in range(1, length):
        head_rel = []
        for j in range(0, length):
            if arcs[j, i]:
                head_rel.append((j, data_loader.vocab.id2rel(int(rels[j, i].asscalar()))))
        out.write('\t'.join(sent[i - 1]))
        if len(head_rel) == 0:  # headless
            out.write('\t_\t_\t_\t_\n')
            continue
        head, rel = head_rel[0]
        out.write('\t{}'.format(str(head)))
        out.write('\t{}'.format(rel))
        out.write('\t')
        out.write('|'.join([str(head) + ':' + rel for (head, rel) in head_rel]))
        out.write('\t_\n')

    out.write('\n')


def evaluate_chinese_sdp(parser, vocab, num_buckets_test, test_batch_size, test_file, output_file,
                         bert=None, debug=False):
    """Evaluate parser on a data set
        Re-implementation of evaluation scripts of A Neural Transition-Based Approach for Semantic Dependency Graph Parsing
        See https://github.com/HITalexwang/lstm-sdparser/blob/master/lstmsdparser/lstm-parse.cc#L1127
    Parameters
    ----------
    parser : BiaffineParser
        biaffine parser
    vocab : ParserVocabulary
        vocabulary built from data set
    num_buckets_test : int
        size of buckets (cluster sentences into this number of clusters)
    test_batch_size : int
        batch size
    test_file : str
        gold test file
    output_file : str
        output result to this file
    debug : bool
        verify the scorer

    Returns
    -------
    tuple
        LF, speed
    """
    # if output_file is None:
    #     output_file = tempfile.NamedTemporaryFile().name
    data_loader = DataLoader(test_file, num_buckets_test, vocab, bert=bert)
    punc = data_loader.vocab.tag2id('PU')
    record = data_loader.idx_sequence
    seconds = time.time()
    correct_arcs = 0  # unlabeled
    correct_arcs_wo_punc = 0
    correct_rels = 0  # labeled
    correct_rels_wo_punc = 0

    correct_labeled_graphs_wo_punc = 0
    correct_unlabeled_graphs_wo_punc = 0

    sum_gold_arcs = 0
    sum_gold_arcs_wo_punc = 0
    sum_pred_arcs = 0
    sum_pred_arcs_wo_punc = 0

    correct_labeled_flag_wo_punc = True
    correct_unlabeled_flag_wo_punc = True

    correct_non_local_arcs = 0
    correct_non_local_rels = 0

    sum_non_local_gold_arcs = 0
    sum_non_local_pred_arcs = 0
    for words, bert, tags, arcs, rels in data_loader.get_batches(batch_size=test_batch_size,
                                                                 shuffle=False):
        outputs = parser.forward(words, bert, tags)
        for output, gold_arc, gold_rel, pos_tags in zip(outputs, arcs.transpose([2, 0, 1]),
                                                        rels.transpose([2, 0, 1]), tags.transpose([1, 0])):
            pred_arc = output[0].asnumpy()
            pred_rel = output[1].asnumpy()
            length = pred_arc.shape[0]
            gold_rel = gold_rel[:length, :length]

            gold_head = [0] * length
            pred_head = [0] * length

            for j in range(0, length):  # head
                for k in range(1, length):  # dep
                    if gold_rel[j][k]:
                        sum_gold_arcs += 1
                        if pos_tags[k] != punc:
                            sum_gold_arcs_wo_punc += 1
                            gold_head[k] += 1
                        if pred_rel[j][k]:
                            correct_arcs += 1
                            if pos_tags[k] != punc:
                                correct_arcs_wo_punc += 1
                            if gold_rel[j][k] == pred_rel[j][k]:
                                correct_rels += 1
                                if pos_tags[k] != punc:
                                    correct_rels_wo_punc += 1
                            elif pos_tags[k] != punc:
                                correct_labeled_flag_wo_punc = False
                        elif pos_tags[k] != punc:
                            correct_labeled_flag_wo_punc = False
                            correct_unlabeled_flag_wo_punc = False
                    if pred_rel[j][k]:
                        sum_pred_arcs += 1
                        if pos_tags[k] != punc:
                            sum_pred_arcs_wo_punc += 1
                            pred_head[k] += 1
            if correct_unlabeled_flag_wo_punc:
                correct_unlabeled_graphs_wo_punc += 1
                if correct_labeled_flag_wo_punc:
                    correct_labeled_graphs_wo_punc += 1
            for c in range(1, length):
                if gold_head[c] == 1 and pred_head[c] == 1:
                    continue
                sum_non_local_gold_arcs += gold_head[c]
                sum_non_local_pred_arcs += pred_head[c]
                for h in range(0, length):
                    if gold_rel[h][c] and pos_tags[c] != punc and pred_rel[h][c]:
                        correct_non_local_arcs += 1
                        if gold_rel[h][c] == pred_rel[h][c]:
                            correct_non_local_rels += 1

    result = {}
    result["UR"] = correct_arcs_wo_punc * 100.0 / sum_gold_arcs_wo_punc
    result["UP"] = correct_arcs_wo_punc * 100.0 / sum_pred_arcs_wo_punc
    result["LR"] = correct_rels_wo_punc * 100.0 / sum_gold_arcs_wo_punc
    result["LP"] = correct_rels_wo_punc * 100.0 / sum_pred_arcs_wo_punc

    result["NUR"] = correct_non_local_arcs * 100.0 / sum_non_local_gold_arcs
    result["NUP"] = correct_non_local_arcs * 100.0 / sum_non_local_pred_arcs
    result["NLR"] = correct_non_local_rels * 100.0 / sum_non_local_gold_arcs
    result["NLP"] = correct_non_local_rels * 100.0 / sum_non_local_pred_arcs

    if sum_pred_arcs_wo_punc == 0:
        result["LP"] = 0
        result["UP"] = 0

    result["UF"] = 2 * result["UR"] * result["UP"] / (result["UR"] + result["UP"])
    result["LF"] = 2 * result["LR"] * result["LP"] / (result["LR"] + result["LP"])

    result["NUF"] = 2 * result["NUR"] * result["NUP"] / (result["NUR"] + result["NUP"])
    result["NLF"] = 2 * result["NLR"] * result["NLP"] / (result["NLR"] + result["NLP"])

    if result["LR"] == 0 and result["LP"] == 0:
        result["LF"] = 0
    if result["UR"] == 0 and result["UP"] == 0:
        result["UF"] = 0
    if result["NLR"] == 0 and result["NLP"] == 0:
        result["NLF"] = 0
    if result["NUR"] == 0 and result["NUP"] == 0:
        result["NUF"] = 0
    seconds = time.time() - seconds
    speed = len(record) / seconds
    return result, speed
