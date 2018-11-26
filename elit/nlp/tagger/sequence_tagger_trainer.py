# ========================================================================
# Copyright 2018 ELIT
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
# -*- coding:utf-8 -*-
# Author：ported from PyTorch implementation of flair: https://github.com/zalandoresearch/flair to MXNet
# Date: 2018-09-22 18:24
import datetime
import os
import random
import re
import sys
from subprocess import run, PIPE
from typing import List

import mxnet as mx
from mxnet import autograd, gluon

from elit.nlp.tagger.corpus import TaggedCorpus, Sentence
from elit.nlp.tagger.mxnet_util import mxnet_prefer_gpu
from elit.nlp.tagger.reduce_lr_on_plateau import ReduceLROnPlateau
from elit.nlp.tagger.sequence_tagger_model import SequenceTagger


class Metric(object):

    def __init__(self, name):
        self.name = name

        self._tp = 0.0
        self._fp = 0.0
        self._tn = 0.0
        self._fn = 0.0

    def tp(self):
        self._tp += 1

    def tn(self):
        self._tn += 1

    def fp(self):
        self._fp += 1

    def fn(self):
        self._fn += 1

    def precision(self):
        if self._tp + self._fp > 0:
            return self._tp / (self._tp + self._fp)
        return 0.0

    def recall(self):
        if self._tp + self._fn > 0:
            return self._tp / (self._tp + self._fn)
        return 0.0

    def f_score(self):
        if self.precision() + self.recall() > 0:
            return 2 * (self.precision() * self.recall()) / (self.precision() + self.recall())
        return 0.0

    def accuracy(self):
        if self._tp + self._tn + self._fp + self._fn > 0:
            return (self._tp + self._tn) / (self._tp + self._tn + self._fp + self._fn)
        return 0.0

    def __str__(self):
        return '{0:<20}\tprecision: {1:.4f} - recall: {2:.4f} - accuracy: {3:.4f} - f1-score: {4:.4f}'.format(
            self.name, self.precision(), self.recall(), self.accuracy(), self.f_score())

    def print(self):
        print('{0:<20}\tprecision: {1:.4f} - recall: {2:.4f} - accuracy: {3:.4f} - f1-score: {4:.4f}'.format(
            self.name, self.precision(), self.recall(), self.accuracy(), self.f_score()))


class SequenceTaggerTrainer:
    def __init__(self, model: SequenceTagger, corpus: TaggedCorpus, test_mode: bool = False) -> None:
        self.model = model
        self.corpus = corpus
        self.test_mode = test_mode

    def train(self,
              base_path: str,
              learning_rate: float = 0.1,
              mini_batch_size: int = 32,
              max_epochs: int = 100,
              anneal_factor: float = 0.5,
              patience: int = 2,
              save_model: bool = True,
              embeddings_in_memory: bool = True,
              train_with_dev: bool = False,
              context: mx.Context = None) -> float:
        """

        :param base_path: a folder to store model, log etc.
        :param learning_rate:
        :param mini_batch_size:
        :param max_epochs:
        :param anneal_factor:
        :param patience:
        :param save_model:
        :param embeddings_in_memory:
        :param train_with_dev:
        :return: best dev f1
        """
        evaluation_method = 'F1'
        if self.model.tag_type in ['ner', 'np', 'srl']:
            evaluation_method = 'span-F1'
        if self.model.tag_type in ['pos', 'upos']:
            evaluation_method = 'accuracy'
        print(evaluation_method)

        os.makedirs(base_path, exist_ok=True)

        loss_txt = os.path.join(base_path, "loss.txt")
        open(loss_txt, "w", encoding='utf-8').close()

        anneal_mode = 'min' if train_with_dev else 'max'
        train_data = self.corpus.train

        # if training also uses dev data, include in training set
        if train_with_dev:
            train_data.extend(self.corpus.dev)

        # At any point you can hit Ctrl + C to break out of training early.
        try:
            with mx.Context(context if context else mxnet_prefer_gpu()):
                self.model.initialize()
                scheduler = ReduceLROnPlateau(lr=learning_rate, verbose=True, factor=anneal_factor,
                                              patience=patience, mode=anneal_mode)
                optimizer = mx.optimizer.SGD(learning_rate=learning_rate, lr_scheduler=scheduler, clip_gradient=5.0)
                trainer = gluon.Trainer(self.model.collect_params(), optimizer=optimizer)
                for epoch in range(0, max_epochs):
                    current_loss = 0
                    if not self.test_mode:
                        random.shuffle(train_data)

                    batches = [train_data[x:x + mini_batch_size] for x in
                               range(0, len(train_data), mini_batch_size)]

                    batch_no = 0

                    for batch in batches:
                        batch = batch
                        batch_no += 1

                        if batch_no % 100 == 0:
                            print("%d of %d (%f)" % (batch_no, len(batches), float(batch_no / len(batches))))

                        # Step 4. Compute the loss, gradients, and update the parameters by calling optimizer.step()
                        with autograd.record():
                            loss = self.model.neg_log_likelihood(batch, self.model.tag_type)

                        current_loss += loss.sum().asscalar()

                        loss.backward()

                        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)

                        # optimizer.step()
                        trainer.step(len(batch))

                        sys.stdout.write('.')
                        sys.stdout.flush()

                        if not embeddings_in_memory:
                            self.clear_embeddings_in_batch(batch)

                    current_loss /= len(train_data)

                    if not train_with_dev:
                        print('.. evaluating... dev... ')
                        dev_score, dev_fp, dev_result = self.evaluate(self.corpus.dev, base_path,
                                                                      evaluation_method=evaluation_method,
                                                                      embeddings_in_memory=embeddings_in_memory)
                    else:
                        dev_fp = 0
                        dev_result = '_'

                    # anneal against train loss if training with dev, otherwise anneal against dev score
                    scheduler.step(current_loss) if train_with_dev else scheduler.step(dev_score)

                    # save if model is current best and we use dev data for model selection
                    if save_model and not train_with_dev and dev_score == scheduler.best:
                        self.model.save(base_path)
                    summary = '%d' % epoch + '\t({:%H:%M:%S})'.format(datetime.datetime.now()) \
                              + '\t%f\t%d\t%f\tDEV   %d\t' % (
                                  current_loss, scheduler.num_bad_epochs, learning_rate, dev_fp) + dev_result
                    summary = summary.replace('\n', '')
                    if self.corpus.test and len(self.corpus.test):
                        print('test... ')
                        test_score, test_fp, test_result = self.evaluate(self.corpus.test, base_path,
                                                                         evaluation_method=evaluation_method,
                                                                         embeddings_in_memory=embeddings_in_memory)
                        summary += '\tTEST   \t%d\t' % test_fp + test_result
                    with open(loss_txt, "a") as loss_file:
                        loss_file.write('%s\n' % summary)
                        loss_file.close()
                    print(summary)

                    if self.corpus.test and len(self.corpus.test):
                        print('test... ')
                        test_score, test_fp, test_result = self.evaluate(self.corpus.test, base_path,
                                                                         evaluation_method=evaluation_method,
                                                                         embeddings_in_memory=embeddings_in_memory)
                        summary += '\tTEST   \t%d\t' % test_fp + test_result
                    print(summary)

            # if we do not use dev data for model selection, save final model
            if save_model and train_with_dev:
                self.model.save(base_path)

            return scheduler.best  # return maximum dev f1

        except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from training early')
            print('saving model')
            self.model.save(base_path + "/final-model")
            print('done')

    def evaluate(self, evaluation: List[Sentence], out_path=None, evaluation_method: str = 'F1',
                 embeddings_in_memory: bool = True):

        tp = 0
        fp = 0

        batch_no = 0
        mini_batch_size = 32
        batches = [evaluation[x:x + mini_batch_size] for x in
                   range(0, len(evaluation), mini_batch_size)]

        metric = Metric('')

        lines = []

        for batch in batches:
            batch_no += 1

            self.model.embeddings.embed(batch)

            for sentence in batch:

                sentence = sentence

                # Step 3. Run our forward pass.
                score, tag_seq = self.model.predict_scores(sentence)

                # Step 5. Compute predictions
                predicted_id = tag_seq
                for (token, pred_id) in zip(sentence.tokens, predicted_id):
                    token = token
                    # get the predicted tag
                    predicted_tag = self.model.tag_dictionary.get_item_for_index(pred_id)
                    token.add_tag('predicted', predicted_tag)

                    # get the gold tag
                    gold_tag = token.get_tag(self.model.tag_type)

                    # append both to file for evaluation
                    eval_line = token.text + ' ' + gold_tag + ' ' + predicted_tag + "\n"

                    # positives
                    if predicted_tag != '':
                        # true positives
                        if predicted_tag == gold_tag:
                            metric.tp()
                        # false positive
                        if predicted_tag != gold_tag:
                            metric.fp()

                    # negatives
                    if predicted_tag == '':
                        # true negative
                        if predicted_tag == gold_tag:
                            metric.tn()
                        # false negative
                        if predicted_tag != gold_tag:
                            metric.fn()

                    lines.append(eval_line)

                lines.append('\n')

            if not embeddings_in_memory:
                self.clear_embeddings_in_batch(batch)

        if out_path is not None:
            test_tsv = os.path.join(out_path, "test.tsv")
            with open(test_tsv, "w", encoding='utf-8') as outfile:
                outfile.write(''.join(lines))

        if evaluation_method == 'span-F1':
            # get the eval script
            eval_script = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'conll03_eval_script.pl')
            os.chmod(eval_script, 0o777)

            eval_data = ''.join(lines)

            p = run(eval_script, stdout=PIPE, input=eval_data, encoding='utf-8')
            main_result = p.stdout
            print(main_result)

            main_result = main_result.split('\n')[1]

            # parse the result file
            main_result = re.sub(';', ' ', main_result)
            main_result = re.sub('precision', 'p', main_result)
            main_result = re.sub('recall', 'r', main_result)
            main_result = re.sub('accuracy', 'acc', main_result)

            f_score = float(re.findall(r'\d+\.\d+$', main_result)[0])
            return f_score, metric._fp, main_result

        if evaluation_method == 'accuracy':
            score = metric.accuracy()
            return score, metric._fp, str(score)

        if evaluation_method == 'F1':
            score = metric.f_score()
            return score, metric._fp, str(metric)

    def clear_embeddings_in_batch(self, batch: List[Sentence]):
        for sentence in batch:
            for token in sentence.tokens:
                token.clear_embeddings()
