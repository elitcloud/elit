# ========================================================================
# Copyright 2017 Emory University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ========================================================================
import itertools
import glob
import sys
import os

__author__ = 'Jinho D. Choi'


def read_labels(fin):
    l = [line.split() for line in fin]
    return [list(y) for x, y in itertools.groupby(l, lambda z: len(z) == 0) if not x]

def read_transitions(fin):
    l = [line.strip() for line in fin]
    return [list(y) for x, y in itertools.groupby(l, lambda z: z == '') if not x]

def get_deprel(transition):
    idx = transition.rfind('-')
    return transition[idx + 1:] if idx >= 0 else ''

def shift(stack, input):
    stack.append(input)

def reduce(stack):
    if len(stack) > 1: stack.pop()

def left_arc(stack, input, transition, labels):
    if len(stack) > 1:
        deprel = get_deprel(transition)
        labels[stack.pop()] = (str(input), deprel)

def right_arc(stack, input, transition, labels):
    deprel = get_deprel(transition)
    labels[input] = (str(stack[-1]), deprel)
    stack.append(input)

def evaluate(gold_labels, transitions):
    sys_labels = [(None, None) for i in range(len(gold_labels)+1)]
    stack = [0]
    input = 1

    for transition in transitions:
        if input >= len(sys_labels): break

        if transition == 'SHIFT':
            shift(stack, input)
            input += 1
        elif transition == 'REDUCE':
            reduce(stack)
        elif transition.startswith('LEFT'):
            left_arc(stack, input, transition, sys_labels)
        elif transition.startswith('RIGHT'):
            right_arc(stack, input, transition, sys_labels)
            input += 1

    las = uas = 0
    for i, gold_label in enumerate(gold_labels, 1):
        sys_label = sys_labels[i]
        if gold_label[5] == sys_label[0]:
            uas += 1
            if gold_label[6] == sys_label[1]:
                las += 1

    return uas, las

def run(gold_labels, sys_file):
    fin = open(sys_file)
    sys_transitions = read_transitions(fin)
    user = os.path.basename(sys_file)
    fin.close()

    guas = glas = total = 0
    for i, gold in enumerate(gold_labels):
        total += len(gold)
        if i < len(sys_transitions):
            uas, las = evaluate(gold, sys_transitions[i])
            guas += uas
            glas += las

    uas = 100.0 * guas / total
    las = 100.0 * glas / total
    match = len(gold_labels) == len(sys_transitions)
    print('%20s: UAS: %6.2f (%6d/%6d), LAS: %6.2f (%6d/%6d), Sequence Match: %r' % (user, uas, guas, total, las, glas, total, match))


# gold_file = sys.argv[1]
# label_file = sys.argv[2]
gold_file = '/Users/jdchoi/Downloads/HW5_CS571_Spring2017/wsj-dep.trn.gold.tsv'
sys_dir = '/Users/jdchoi/Downloads/HW5_CS571_Spring2017/out/'

# gold labels
fin = open(gold_file)
gold_labels = read_labels(fin)
fin.close()
print('Gold: %d' % len(gold_labels))

for sys_file in sorted(glob.glob(os.path.join(sys_dir, '*.out'))):
    run(gold_labels, sys_file)