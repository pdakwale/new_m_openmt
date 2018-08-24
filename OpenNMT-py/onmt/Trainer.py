from __future__ import division
"""
This is the loadable seq2seq trainer library that is
in charge of training details, loss compute, and statistics.
See train.py for a use case of this library.

Note!!! To make this a general library, we implement *only*
mechanism things here(i.e. what to do), and leave the strategy
things to users(i.e. how to do it). Also see train.py(one of the
users of this library) for the strategy things we do.
"""
import time
import sys
import math
import torch
import torch.nn as nn

import onmt
import onmt.modules


class Statistics(object):
    """
    Train/validate loss statistics.
    """
    def __init__(self, loss=0, n_words=0, n_correct=0):
        self.loss = loss
        self.n_words = n_words
        self.n_correct = n_correct
        self.n_src_words = 0
        self.start_time = time.time()

    def update(self, stat):
        self.loss += stat.loss
        self.n_words += stat.n_words
        self.n_correct += stat.n_correct

    def accuracy(self):
        return 100 * (self.n_correct / self.n_words)

    def ppl(self):
        return math.exp(min(self.loss / self.n_words, 100))

    def elapsed_time(self):
        return time.time() - self.start_time

    def output(self, epoch, batch, n_batches, start):
        t = self.elapsed_time()
        print(("Epoch %2d, %5d/%5d; acc: %6.2f; ppl: %6.2f; " +
               "%3.0f src tok/s; %3.0f tgt tok/s; %6.0f s elapsed") %
              (epoch, batch,  n_batches,
               self.accuracy(),
               self.ppl(),
               self.n_src_words / (t + 1e-5),
               self.n_words / (t + 1e-5),
               time.time() - start))
        sys.stdout.flush()

    def log(self, prefix, experiment, optim):
        t = self.elapsed_time()
        experiment.add_scalar_value(prefix + "_ppl", self.ppl())
        experiment.add_scalar_value(prefix + "_accuracy", self.accuracy())
        experiment.add_scalar_value(prefix + "_tgtper",  self.n_words / t)
        experiment.add_scalar_value(prefix + "_lr", optim.lr)


class Trainer(object):
    def __init__(self, model1, model2, train_iter, valid_iter,
                 train_loss1, valid_loss1, train_loss2, valid_loss2, optim1, optim2,
                 trunc_size, shard_size):
        """
        Args:
            model: the seq2seq model.
            train_iter: the train data iterator.
            valid_iter: the validate data iterator.
            train_loss: the train side LossCompute object for computing loss.
            valid_loss: the valid side LossCompute object for computing loss.
            optim: the optimizer responsible for lr update.
            trunc_size: a batch is divided by several truncs of this size.
            shard_size: compute loss in shards of this size for efficiency.
        """
        # Basic attributes.
        self.model1 = model1
        self.model2 = model2
        self.train_iter = train_iter
        self.valid_iter = valid_iter
        self.train_loss1 = train_loss1
        self.valid_loss1 = valid_loss1
        self.train_loss2 = train_loss2
        self.valid_loss2 = valid_loss2
        self.optim1 = optim1
        self.optim2 = optim2
        self.trunc_size = trunc_size
        self.shard_size = shard_size

        # Set model in training mode.
        self.model1.train()
        self.model2.train()

    def bottle(self, v):
        return v.view(-1, v.size(2))

    def unbottle(self, v, batch_size):
        return v.view(-1, batch_size, v.size(1))

    def train(self, epoch, report_func=None):
        """ Called for each epoch to train. """
        total_stats1 = Statistics()
        report_stats1 = Statistics()
        total_stats2 = Statistics()
        report_stats2 = Statistics()

        for i, batch in enumerate(self.train_iter):
            target_size = batch.tgt.size(0)
            # Truncated BPTT
            trunc_size = self.trunc_size if self.trunc_size else target_size

            dec_state1 = None
            dec_state2 = None
            _, src_lengths = batch.src

            src = onmt.IO.make_features(batch, 'src')
            tgt_outer = onmt.IO.make_features(batch, 'tgt')
            report_stats1.n_src_words += src_lengths.sum()
            report_stats2.n_src_words += src_lengths.sum()

            for j in range(0, target_size-1, trunc_size):
                # 1. Create truncated target.
                tgt = tgt_outer[j: j + trunc_size]

                # 2. F-prop all but generator.
                self.model1.zero_grad()
                self.model2.zero_grad()
                outputs1, attns1, dec_state1 = \
                    self.model1(src, tgt, src_lengths, dec_state1)

                outputs2, attns2, dec_state2 = \
                    self.model2(src, tgt, src_lengths, dec_state2)
                range_ = (j, j + trunc_size) 
               # print(src)
               # print(tgt)
               # print(attns1)
               # print(attns2)
               # print('----------------------------')
                # 3. Compute loss in shards for memory efficiency.
                gen_state1 = onmt.Loss.make_gen_state(
                outputs1, batch, attns1, range_)

                gen_state2 = onmt.Loss.make_gen_state(
                outputs2, batch, attns2, range_)
                
                scores1 = self.model1.generator(self.bottle(outputs1))
                scores2 = self.model2.generator(self.bottle(outputs2))
                pred1 = scores1.max(1)[1]
                pred2 = scores2.max(1)[1]
                comp = pred1.ne(pred2).view(-1)
                target1 = gen_state1["target"].view(-1)
                target2 = gen_state2["target"].view(-1)
                #if comp.any() > 0:
                #mask = comp.nonzero().view(-1)
                #target1 = target1.index_select(0, mask)
                #target2 = target2.index_select(0, mask)
                #scores1 = scores1.index_select(0, mask)
                #scores2 = scores2.index_select(0, mask)
                #print(batch.batch_size)
                loss1, batch_stats1 = self.train_loss1.compute_loss(batch, scores1, target1)
                loss2, batch_stats2 = self.train_loss2.compute_loss(batch, scores2, target2)
                loss1.div(batch.batch_size).backward()
                loss2.div(batch.batch_size).backward()
                #print(scores1.grad)
                #batch_stats1 = self.train_loss1.sharded_compute_loss(
                #        batch, outputs1, attns1, j,
                #        trunc_size, self.shard_size)
                #batch_stats2 = self.train_loss2.sharded_compute_loss(
                #        batch, outputs2, attns2, j,
                #        trunc_size, self.shard_size)
               
                # 4. Update the parameters and statistics.
                self.optim1.step()
                self.optim2.step()
                total_stats1.update(batch_stats1)
                report_stats1.update(batch_stats1)
  
                total_stats2.update(batch_stats2)
                report_stats2.update(batch_stats2)

                # If truncated, don't backprop fully.
                if dec_state1 is not None:
                    dec_state1.detach()
                if dec_state2 is not None:
                    dec_state2.detach()

            if report_func is not None:
                report_func(epoch, i, len(self.train_iter),
                            total_stats1.start_time, self.optim1.lr,
                            report_stats1)
                report_stats1 = Statistics()

            if report_func is not None:
                report_func(epoch, i, len(self.train_iter),
                            total_stats2.start_time, self.optim2.lr,
                            report_stats2)
                report_stats2 = Statistics()

        return total_stats1, total_stats2

    def validate(self):
        """ Called for each epoch to validate. """
        # Set model in validating mode.
        self.model1.eval()
        self.model2.eval()

        stats1 = Statistics()
        stats2 = Statistics()
        for batch in self.valid_iter:
            _, src_lengths = batch.src
            src = onmt.IO.make_features(batch, 'src')
            tgt = onmt.IO.make_features(batch, 'tgt')

            # F-prop through the model.
            outputs1, attns1, _ = self.model1(src, tgt, src_lengths)
            outputs2, attns2, _ = self.model2(src, tgt, src_lengths)
        #    print(src) 
            # Compute loss.
            gen_state1 = onmt.Loss.make_gen_state(
                outputs1, batch, attns1, (0, batch.tgt.size(0)))

            gen_state2 = onmt.Loss.make_gen_state(
                outputs2, batch, attns2, (0, batch.tgt.size(0)))

            scores1 = self.model1.generator(self.bottle(outputs1))
            scores2 = self.model2.generator(self.bottle(outputs2))
             
            target1 = gen_state1["target"].view(-1)
            target2 = gen_state2["target"].view(-1)

            loss1, batch_stats1 = self.valid_loss1.compute_loss(batch, scores1, target1)
            loss2, batch_stats2 = self.valid_loss2.compute_loss(batch, scores2, target2)
            # Update statistics.
            stats1.update(batch_stats1)
            stats2.update(batch_stats2)

        # Set model back to training mode.
        self.model1.train()
        self.model2.train()

        return stats1, stats2

    def epoch_step(self, ppl, epoch):
        """ Called for each epoch to update learning rate. """
        return self.optim1.updateLearningRate(ppl, epoch), self.optim2.updateLearningRate(ppl, epoch)

    def drop_checkpoint(self, opt, epoch, fields1, fields2,  valid_stats1, valid_stats2):
        """ Called conditionally each epoch to save a snapshot. """
        real_model1 = (self.model1.module
                      if isinstance(self.model1, nn.DataParallel)
                      else self.model1)
        real_generator1 = (real_model1.generator.module
                          if isinstance(real_model1.generator, nn.DataParallel)
                          else real_model1.generator)

        real_model2 = (self.model2.module
                      if isinstance(self.model2, nn.DataParallel)
                      else self.model2)
        real_generator2 = (real_model2.generator.module
                          if isinstance(real_model2.generator, nn.DataParallel)
                          else real_model2.generator)


        model_state_dict1 = real_model1.state_dict()
        model_state_dict1 = {k: v for k, v in model_state_dict1.items()
                            if 'generator' not in k}
        generator_state_dict1 = real_generator1.state_dict()

        model_state_dict2 = real_model2.state_dict()
        model_state_dict2 = {k: v for k, v in model_state_dict2.items()
                            if 'generator' not in k}
        generator_state_dict2 = real_generator2.state_dict()

        checkpoint1 = {
            'model': model_state_dict1,
            'generator': generator_state_dict1,
            'vocab': onmt.IO.ONMTDataset.save_vocab(fields1),
            'opt': opt,
            'epoch': epoch,
            'optim': self.optim1
        }
        checkpoint2 = {
            'model': model_state_dict2,
            'generator': generator_state_dict2,
            'vocab': onmt.IO.ONMTDataset.save_vocab(fields2),
            'opt': opt,
            'epoch': epoch,
            'optim': self.optim2
        }
        torch.save(checkpoint1,
                   '%s_acc_%.2f_ppl_%.2f_e%d.model1.pt'
                   % (opt.save_model, valid_stats1.accuracy(),
                      valid_stats1.ppl(), epoch))
        torch.save(checkpoint2,
                   '%s_acc_%.2f_ppl_%.2f_e%d.model2.pt'
                   % (opt.save_model, valid_stats2.accuracy(),
                      valid_stats2.ppl(), epoch))