from __future__ import division

import os
import argparse
import torch
import torch.nn as nn
from torch import cuda

import onmt
import onmt.Models
import onmt.ModelConstructor
import onmt.modules
from onmt.Utils import aeq, use_gpu
import opts

parser = argparse.ArgumentParser(description='train.py')

# opts.py
opts.add_md_help_argument(parser)
opts.model_opts(parser)
opts.train_opts(parser)

opt = parser.parse_args()
if opt.word_vec_size != -1:
    opt.src_word_vec_size = opt.word_vec_size
    opt.tgt_word_vec_size = opt.word_vec_size

if opt.layers != -1:
    opt.enc_layers = opt.layers
    opt.dec_layers = opt.layers

opt.brnn = (opt.encoder_type == "brnn")
if opt.seed > 0:
    torch.manual_seed(opt.seed)

if opt.rnn_type == "SRU" and not opt.gpuid:
    raise AssertionError("Using SRU requires -gpuid set.")

if torch.cuda.is_available() and not opt.gpuid:
    print("WARNING: You have a CUDA device, should run with -gpuid 0")

if opt.gpuid:
    cuda.set_device(opt.gpuid[0])
    if opt.seed > 0:
        torch.cuda.manual_seed(opt.seed)


# Set up the Crayon logging server.
if opt.exp_host != "":
    from pycrayon import CrayonClient
    cc = CrayonClient(hostname=opt.exp_host)

    experiments = cc.get_experiment_names()
    print(experiments)
    if opt.exp in experiments:
        cc.remove_experiment(opt.exp)
    experiment = cc.create_experiment(opt.exp)


def report_func(epoch, batch, num_batches,
                start_time, lr, report_stats):
    """
    This is the user-defined batch-level traing progress
    report function.
    Args:
        epoch(int): current epoch count.
        batch(int): current batch count.
        num_batches(int): total number of batches.
        start_time(float): last report time.
        lr(float): current learning rate.
        report_stats(Statistics): a Statistics instance.
    """
    if batch % opt.report_every == -1 % opt.report_every:
        report_stats.output(epoch, batch+1, num_batches, start_time)
        if opt.exp_host:
            report_stats.log("progress", experiment, lr)


def make_train_data_iter(train_data, opt):
    """
    This returns user-defined train data iterator for the trainer
    to iterate over during each train epoch. We implement simple
    ordered iterator strategy here, but more sophisticated strategy
    like curriculum learning is ok too.
    """
    return onmt.IO.OrderedIterator(
                dataset=train_data, batch_size=opt.batch_size,
                device=opt.gpuid[0] if opt.gpuid else -1,
                repeat=False)


def make_valid_data_iter(valid_data, opt):
    """
    This returns user-defined validate data iterator for the trainer
    to iterate over during each validate epoch. We implement simple
    ordered iterator strategy here, but more sophisticated strategy
    is ok too.
    """
    return onmt.IO.OrderedIterator(
                dataset=valid_data, batch_size=opt.batch_size,
                device=opt.gpuid[0] if opt.gpuid else -1,
                train=False, sort=True)


def make_loss_compute(model1, model2, tgt_vocab, dataset, opt):
    """
    This returns user-defined LossCompute object, which is used to
    compute loss in train/validate process. You can implement your
    own *LossCompute class, by subclassing LossComputeBase.
    """
    if opt.copy_attn:
        compute1 = onmt.modules.CopyGeneratorLossCompute(
            model1.generator, tgt_vocab, dataset, opt.copy_attn_force)
        compute2 = onmt.modules.CopyGeneratorLossCompute(
            model2.generator, tgt_vocab, dataset, opt.copy_attn_force)
    else:
        compute1 = onmt.Loss.NMTLossCompute(model1.generator, tgt_vocab)
        compute2 = onmt.Loss.NMTLossCompute(model2.generator, tgt_vocab)
    if use_gpu(opt):
        compute1.cuda()
        compute2.cuda()
    return compute1, compute2


def train_model(model1, model2, train_data, valid_data, fields1, fields2,  optim1, optim2):

    train_iter = make_train_data_iter(train_data, opt)
    valid_iter = make_valid_data_iter(valid_data, opt)

    train_loss1, train_loss2 = make_loss_compute(model1, model2, fields1["tgt"].vocab,
                                   train_data, opt)
    valid_loss1, valid_loss2 = make_loss_compute(model1, model2, fields1["tgt"].vocab,
                                   valid_data, opt)

    trunc_size = opt.truncated_decoder  # Badly named...
    shard_size = opt.max_generator_batches

    trainer = onmt.Trainer(model1, model2,  train_iter, valid_iter,
                           train_loss1, valid_loss1, train_loss2, valid_loss2, optim1, optim2,
                           trunc_size, shard_size)

    for epoch in range(opt.start_epoch, opt.epochs + 1):
        print('')

        # 1. Train for one epoch on the training set.
        train_stats1, train_stats2 = trainer.train(epoch, report_func)
        print('Train perplexity 1: %g' % train_stats1.ppl())
        print('Train accuracy 1: %g' % train_stats1.accuracy())

        print('Train perplexity 2: %g' % train_stats2.ppl())
        print('Train accuracy 2: %g' % train_stats2.accuracy())

        # 2. Validate on the validation set.
        valid_stats1, valid_stats2 = trainer.validate()
        print('Validation perplexity 1: %g' % valid_stats1.ppl())
        print('Validation accuracy 1: %g' % valid_stats1.accuracy())

        print('Validation perplexity 2: %g' % valid_stats2.ppl())
        print('Validation accuracy 2: %g' % valid_stats2.accuracy())

        # 3. Log to remote server.
        if opt.exp_host:
            train_stats1.log("train", experiment, optim1.lr)
            valid_stats1.log("valid", experiment, optim1.lr)
            train_stats2.log("train", experiment, optim2.lr)
            valid_stats2.log("valid", experiment, optim2.lr)

        # 4. Update the learning rate
        trainer.epoch_step(valid_stats1.ppl(), epoch)
        trainer.epoch_step(valid_stats2.ppl(), epoch)

        # 5. Drop a checkpoint if needed.
        if epoch >= opt.start_checkpoint_at:
            trainer.drop_checkpoint(opt, epoch, fields1, fields2, valid_stats1, valid_stats2)
            


def check_save_model_path():
    save_model_path = os.path.abspath(opt.save_model)
    model_dirname = os.path.dirname(save_model_path)
    if not os.path.exists(model_dirname):
        os.makedirs(model_dirname)


def tally_parameters(model):
    n_params = sum([p.nelement() for p in model.parameters()])
    print('* number of parameters: %d' % n_params)
    enc = 0
    dec = 0
    for name, param in model.named_parameters():
        if 'encoder' in name:
            enc += param.nelement()
        elif 'decoder' or 'generator' in name:
            dec += param.nelement()
    print('encoder: ', enc)
    print('decoder: ', dec)


def load_fields(train, valid, checkpoint, train_from):
    fields = onmt.IO.ONMTDataset.load_fields(
                torch.load(opt.data + '.vocab.pt'))
    fields = dict([(k, f) for (k, f) in fields.items()
                  if k in train.examples[0].__dict__])
    train.fields = fields
    valid.fields = fields

    if train_from:
        print('Loading vocab from checkpoint at %s.' % train_from)
        fields = onmt.IO.ONMTDataset.load_fields(checkpoint['vocab'])

    print(' * vocabulary size. source = %d; target = %d' %
          (len(fields['src'].vocab), len(fields['tgt'].vocab)))

    return fields


def collect_features(train, fields):
    # TODO: account for target features.
    # Also, why does fields need to have the structure it does?
    src_features = onmt.IO.ONMTDataset.collect_features(fields)
    aeq(len(src_features), train.nfeatures)

    return src_features


def build_model(model_opt, opt, fields, checkpoint):
    print('Building model...')
    model = onmt.ModelConstructor.make_base_model(model_opt, fields,
                                                  use_gpu(opt), checkpoint)
    if len(opt.gpuid) > 1:
        print('Multi gpu training ', opt.gpuid)
        model = nn.DataParallel(model, device_ids=opt.gpuid, dim=1)
    #print(model)

    return model


def build_optim(model, checkpoint, train_from):
    if train_from:
        print('Loading optimizer from checkpoint.')
        optim = checkpoint['optim']
        optim.optimizer.load_state_dict(
            checkpoint['optim'].optimizer.state_dict())
    else:
        # what members of opt does Optim need?
        optim = onmt.Optim(
            opt.optim, opt.learning_rate, opt.max_grad_norm,
            lr_decay=opt.learning_rate_decay,
            start_decay_at=opt.start_decay_at,
            opt=opt
        )

    optim.set_parameters(model.parameters())

    return optim


def main():

    # Load train and validate data.
    print("Loading train and validate data from '%s'" % opt.data)
    train = torch.load(opt.data + '.train.pt')
    valid = torch.load(opt.data + '.valid.pt')
    print(' * number of training sentences: %d' % len(train))
    print(' * maximum batch size: %d' % opt.batch_size)

    # Load checkpoint if we resume from a previous training.
    if opt.train_from1:
        print('Loading checkpoint from %s' % opt.train_from1)
        checkpoint1 = torch.load(opt.train_from1,
                                map_location=lambda storage, loc: storage)
        model_opt1 = checkpoint1['opt']
        # I don't like reassigning attributes of opt: it's not clear
        opt.start_epoch = checkpoint1['epoch'] + 1
    else:
        checkpoint1 = None
        model_opt1 = opt

    # Load fields generated from preprocess phase.
    fields1 = load_fields(train, valid, checkpoint1, opt.train_from1)

    # Collect features.
    src_features1 = collect_features(train, fields1)
    for j, feat in enumerate(src_features1):
        print(' * src feature %d size = %d' % (j, len(fields1[feat].vocab)))
    


    if opt.train_from2:
        print('Loading checkpoint from %s' % opt.train_from2)
        checkpoint2 = torch.load(opt.train_from2,
                                map_location=lambda storage, loc: storage)
        model_opt2 = checkpoint2['opt']
        # I don't like reassigning attributes of opt: it's not clear
        opt.start_epoch = checkpoint2['epoch'] + 1
    else:
        checkpoint2 = None
        model_opt2 = opt

    # Load fields generated from preprocess phase.
    fields2 = load_fields(train, valid, checkpoint2, opt.train_from2)

    # Collect features.
    src_features2 = collect_features(train, fields2)
    for j, feat in enumerate(src_features2):
        print(' * src feature %d size = %d' % (j, len(fields2[feat].vocab)))
    # Build model.
    #model_opt1.param_init = 0.1
    model1 = build_model(model_opt1, opt, fields1, checkpoint2)
    #model_opt2.param_init = 0.05
    model2 = build_model(model_opt2, opt, fields2, checkpoint2)
    tally_parameters(model1)
    tally_parameters(model2)
    check_save_model_path()

    # Build optimizer.
    optim1 = build_optim(model1, checkpoint1, opt.train_from1)
    optim2 = build_optim(model2, checkpoint2, opt.train_from2)
    print(opt)
    # Do training.
    train_model(model1, model2, train, valid, fields1, fields2, optim1, optim2)


if __name__ == "__main__":
    main()
