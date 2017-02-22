#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from __future__ import division
import tensorflow as tf
import data
import model
from timer import Timer
import datetime
from six.moves import xrange
import os
import config as cfg

batch_size = cfg.BATCH_SIZE
initial_learning_rate = cfg.INITIAL_LEARNING_RATE
max_steps = cfg.MAX_STEPS
log_dir = cfg.LOG_DIR


def run_val(sess, img1_placeholder, img2_placeholder, flo_placeholder, loss, validation):
    loss_count = 0
    steps_per_epoch = validation.num_examples // batch_size
    num_examples = steps_per_epoch * batch_size
    for step in xrange(steps_per_epoch):
        feed_dict = model.fill_feed_dict(validation, img1_placeholder, img2_placeholder, flo_placeholder)
        loss_count += sess.run(loss, feed_dict=feed_dict)
    average = float(loss_count) / num_examples
    print('  Num examples: %d Average loss: %0.04f' % (num_examples, average))


def train():
    train, validation = data.get_nameset()
    with tf.Graph().as_default():
        img1_placeholder, img2_placeholder, flo_placeholder = model.placeholder_inputs()
        predict6, predict5, predict4, predict3, predict2 = model.inference(img1_placeholder, img2_placeholder)
        loss = model.loss(predict6, predict5, predict4, predict3, predict2, flo_placeholder)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step, decay_steps=200000, decay_rate=0.1)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.minimize(loss, global_step=global_step)

        summary = tf.summary.merge_all()
        init = tf.initialize_all_variables()
        saver = tf.train.Saver()
        sess = tf.Session()

        train_timer = Timer()

        sess.run(init)
        for step in xrange(max_steps):
            train_timer.tic()
            feed_dict = model.fill_feed_dict(train, img1_placeholder, img2_placeholder, flo_placeholder)
            _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
            train_timer.toc()

            if step % 100 == 0:
                if step % 20 == 0:
                    log_str = ('{} Epoch: {}, Step: {}, Learning rate: {},'
                            ' Loss: {:5.3f}\nSpeed: {:.3f}s/iter, Remain: {}').format(
                            datetime.datetime.now().strftime('%m/%d %H:%M:%S'),
                            train.epochs_completed,
                            int(step),
                            learning_rate.eval(session=sess),
                            loss_value,
                            train_timer.average_time,
                            train_timer.remain(step, max_steps))
                    print log_str
                summary_str = sess.run(summary, feed_dict=feed_dict)
                summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()

            if (step + 1) % 1000 == 0 or (step + 1) == max_steps:
                checkpoint_file = os.path.join(log_dir, 'model.ckpt')
                saver.save(sess, checkpoint_file, global_step=step)
                print('Validation Data Eval:')
                run_val(sess, img1_placeholder, img2_placeholder, flo_placeholder, loss, validation)


train()
