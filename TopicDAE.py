#!/usr/bin/env python3

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os
import shutil
import sys
import time
from datetime import datetime

import bottleneck as bn
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import sparse
from tensorflow.contrib.layers import apply_regularization, l2_regularizer

gpu_id = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id

ARG = argparse.ArgumentParser()
ARG.add_argument('--data', type=str, required=True)
ARG.add_argument('--mode', type=str, default='tst', help='trn/tst, for training/testing.')
ARG.add_argument('--logdir', type=str, default='checkpoints')
ARG.add_argument('--seed', type=int, default=98765, help='Random seed')
ARG.add_argument('--epoch', type=int, default=500, help='Number of training epochs.')
ARG.add_argument('--batch', type=int, default=500, help='Training batch size.')
ARG.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate.')
ARG.add_argument('--rg', type=float, default=0.0, help='L2 regularization.')
ARG.add_argument('--keep', type=float, default=0.5, help='Keep probability for dropout, in (0,1].')
ARG.add_argument('--tau', type=float, default=0.1, help='Temperature of sigmoid/softmax, in (0,oo).')
ARG.add_argument('--lam', type=float, default=0.0001, help='the coefficient of contrastive loss.')
ARG.add_argument('--tfac', type=int, default=5, help='Number of topics.')
ARG.add_argument('--dfac', type=int, default=20, help='Dimension of emb.')
ARG = ARG.parse_args()

if ARG.seed < 0:
    ARG.seed = int(time.time())
dt = datetime.now()
LOG_DIR = '%s-%s-%d-%d-%d-%d-%d' % (ARG.data.split('/')[2], 'topicvae', dt.month, dt.day, dt.hour, dt.minute, dt.second)
LOG_DIR = os.path.join(ARG.logdir, LOG_DIR)
if ARG.mode == 'tst':
    # Industrial-topicdae-4-13-22-48-29
    LOG_DIR = 'checkpoints/Industrial-topicdae-4-13-22-48-29'

def set_rng_seed(seed):
    np.random.seed(seed)
    tf.set_random_seed(seed)

class TopicDAE(object):
    def __init__(self, num_items, item_words, word_embs):
        tfac, dfac = ARG.tfac, ARG.dfac
        self.rg = ARG.rg
        self.lr = ARG.lr
        self.lam = ARG.lam
        self.random_seed = ARG.seed
        self.temp = 0.2

        self.n_items = num_items
        self.item_words = item_words

        # The first fc layer of the encoder is the context embedding table.
        self.q_dims = [num_items, dfac, dfac]
        self.weights_q, self.biases_q = [], []
        for i, (d_in, d_out) in enumerate(zip(self.q_dims[:-1], self.q_dims[1:])):
            weight_key = "weight_q_{}to{}".format(i, i + 1)
            self.weights_q.append(tf.get_variable(
                name=weight_key, shape=[d_in, d_out],
                initializer=tf.contrib.layers.xavier_initializer(
                    seed=self.random_seed)))
            bias_key = "bias_q_{}".format(i + 1)
            self.biases_q.append(tf.get_variable(
                name=bias_key, shape=[d_out],
                initializer=tf.truncated_normal_initializer(
                    stddev=0.001, seed=self.random_seed)))

        self.word_embs = tf.Variable(word_embs, name="word_embs", dtype=tf.float32, trainable=True)
        # self.word_embs = tf.convert_to_tensor(wordEmb, dtype=tf.float32)

        self.topic_embs = tf.get_variable(
            name="topic_embs", shape=[tfac, dfac],
            initializer=tf.contrib.layers.xavier_initializer(seed=self.random_seed))

        self.g = tf.get_variable(
            name='g', shape=[tfac, self.word_embs.shape[1], dfac],
            initializer=tf.contrib.layers.xavier_initializer(seed=self.random_seed))

        self.transform = tf.get_variable(
            name="transform", shape=[tfac, dfac],
            initializer=tf.contrib.layers.xavier_initializer(seed=self.random_seed))

        self.info_w = tf.get_variable(
            name="info_w", shape=[dfac, dfac],
            initializer=tf.contrib.layers.xavier_initializer(seed=self.random_seed))

        self.input_ph = tf.placeholder(dtype=tf.float32, shape=[None, num_items])
        self.keep_prob_ph = tf.placeholder_with_default(1., shape=None)
        self.is_training_ph = tf.placeholder_with_default(0., shape=None)

    def build_graph(self):
        saver, logits, recon_loss, info_nec_loss = self.forward_pass()

        reg_var = apply_regularization(
            l2_regularizer(self.rg),
            self.weights_q + [self.topic_embs, self.g, self.transform, self.info_w])
        # the l2 norm multiply 2 so that it is back in the same scale
        neg_elbo = recon_loss + self.lam * info_nec_loss + 2. * reg_var

        train_op = tf.train.AdamOptimizer(self.lr).minimize(neg_elbo)

        # add summary statistics
        tf.summary.scalar('trn/recon_loss', recon_loss)
        tf.summary.scalar('trn/info_nce', info_nec_loss)
        tf.summary.scalar('trn/reg_var', reg_var)
        tf.summary.scalar('trn/neg_elbo', neg_elbo)
        merged = tf.summary.merge_all()

        return saver, logits, train_op, merged

    def q_graph_k(self, x):
        h = tf.nn.l2_normalize(x, 1)
        h = tf.nn.dropout(h, self.keep_prob_ph)
        for i, (w, b) in enumerate(zip(self.weights_q, self.biases_q)):
            h = tf.matmul(h, w, a_is_sparse=(i == 0)) + b
            if i != len(self.weights_q) - 1:
                h = tf.nn.tanh(h)
        return h

    def forward_pass(self):
        topic_embs = tf.nn.l2_normalize(self.topic_embs, axis=1)
        g = tf.nn.l2_normalize(self.g, axis=1)
        W = tf.nn.l2_normalize(self.transform, axis=1)
        m_list, z_list = None, None
        probs = None
        for t in range(ARG.tfac):
            word_lat_emb = tf.matmul(self.word_embs, g[t])
            item_word_lat_emb = tf.nn.embedding_lookup(word_lat_emb, self.item_words)

            aspIds = np.empty((self.n_items, 1), dtype=int)
            aspIds.fill(t)
            aspIds = aspIds.tolist()  # (n*1)

            topic_emb = tf.nn.embedding_lookup(topic_embs, aspIds)
            topic_emb = tf.transpose(topic_emb, (0, 2, 1))

            item_word_topic_attention = tf.matmul(item_word_lat_emb, topic_emb)
            item_word_topic_attention = tf.nn.softmax(item_word_topic_attention)
            # item_word_topic_attention = tf.sigmoid(item_word_topic_attention)

            item_word_topic_emb = item_word_lat_emb * item_word_topic_attention
            items_t = tf.reduce_sum(item_word_topic_emb, 1)
            items_t = tf.nn.l2_normalize(items_t, axis=1)

            topic_probability = tf.matmul(items_t, W, transpose_b=True) / ARG.tau
            topic_probability = tf.nn.softmax(topic_probability)

            # encoder
            h_t = tf.reshape(topic_probability[:, t], (1, -1))
            x_t = self.input_ph * h_t
            z_t = self.q_graph_k(x_t)

            # decoder
            m_t = tf.matmul(x_t, items_t)
            m_t = tf.nn.l2_normalize(m_t, axis=1)
            z_t = tf.nn.l2_normalize(z_t, axis=1)
            if t == 0:
                m_list = tf.expand_dims(m_t, axis=1)
                z_list = tf.expand_dims(z_t, axis=1)
            else:
                m_list = tf.concat([m_list, tf.expand_dims(m_t, axis=1)], axis=1)
                z_list = tf.concat([z_list, tf.expand_dims(z_t, axis=1)], axis=1)
            logits_t = tf.matmul(z_t, items_t, transpose_b=True) / ARG.tau
            probs_t = tf.exp(logits_t)
            probs_t = probs_t * h_t
            probs = (probs_t if (probs is None) else (probs + probs_t))

        logits = tf.log(probs)
        logits = tf.nn.log_softmax(logits)
        recon_loss = tf.reduce_mean(tf.reduce_sum(-logits * self.input_ph, axis=-1))

        z_list = tf.matmul(z_list, self.info_w)
        m_list = tf.matmul(m_list, self.info_w)
        m_list = tf.nn.l2_normalize(m_list, axis=-1)
        z_list = tf.nn.l2_normalize(z_list, axis=-1)
        pos_score = tf.reduce_sum(tf.multiply(z_list, m_list), axis=2)
        ttl_score = tf.matmul(z_list, m_list, transpose_a=False, transpose_b=True)
        pos_score = tf.exp(pos_score / self.temp)
        ttl_score = tf.reduce_sum(tf.exp(ttl_score / self.temp), axis=1)
        info_nec_loss = -tf.reduce_sum(tf.log(pos_score / ttl_score))
        # info_nec_loss = 0.0
        return tf.train.Saver(), logits, recon_loss, info_nec_loss

def load_data(data_dir):
    train_data, test_data, n_users, n_items = \
        load_tr_te_data(os.path.join(data_dir, 'train.csv'),
                        os.path.join(data_dir, 'test.csv'))
    tst_data_tr = train_data

    item_cid = pd.read_csv(os.path.join(data_dir, 'item_cid.csv'))
    item_cids = item_cid['cid'].values.tolist()
    item_words = []
    for cids in item_cids:
        wids = [int(i) for i in cids[1: len(cids) - 1].split(',')]
        item_words.append(wids)
    word_embs = np.load(os.path.join(data_dir, 'embedding_words.npy'))

    assert n_items == train_data.shape[1]
    assert n_items == tst_data_tr.shape[1]
    assert n_items == test_data.shape[1]

    return (n_items, train_data, tst_data_tr, test_data, item_words, word_embs)

def load_tr_te_data(csv_file_tr, csv_file_te):
    tp_tr = pd.read_csv(csv_file_tr)
    tp_te = pd.read_csv(csv_file_te)

    u_start_idx = min(tp_tr['uid'].min(), tp_te['uid'].min())
    u_end_idx = max(tp_tr['uid'].max(), tp_te['uid'].max())
    i_start_idx = min(tp_tr['iid'].min(), tp_te['iid'].min())
    i_end_idx = max(tp_tr['iid'].max(), tp_te['iid'].max())
    num_users = u_end_idx - u_start_idx + 1
    num_items = i_end_idx - i_start_idx + 1

    rows_tr, cols_tr = tp_tr['uid'] - u_start_idx, tp_tr['iid'] - i_start_idx
    rows_te, cols_te = tp_te['uid'] - u_start_idx, tp_te['iid'] - i_start_idx

    data_tr = sparse.csr_matrix((np.ones_like(rows_tr),
                                 (rows_tr, cols_tr)), dtype='float64',
                                shape=(num_users, num_items))
    data_te = sparse.csr_matrix((np.ones_like(rows_te),
                                 (rows_te, cols_te)), dtype='float64',
                                shape=(num_users, num_items))
    return data_tr, data_te, num_users, num_items

def ndcg_binary_at_k_batch(x_pred, heldout_batch, k=20):
    """
    normalized discounted cumulative gain@k for binary relevance
    ASSUMPTIONS: all the 0's in heldout_data indicate 0 relevance
    """
    batch_users = x_pred.shape[0]
    idx_topk_part = bn.argpartition(-x_pred, k, axis=1)
    topk_part = x_pred[np.arange(batch_users)[:, np.newaxis],
                       idx_topk_part[:, :k]]
    idx_part = np.argsort(-topk_part, axis=1)
    idx_topk = idx_topk_part[np.arange(batch_users)[:, np.newaxis], idx_part]
    # build the discount template
    tp = 1. / np.log2(np.arange(2, k + 2))

    dcg = (heldout_batch[np.arange(batch_users)[:, np.newaxis], idx_topk].toarray() * tp).sum(axis=1)
    idcg = np.array([(tp[:min(n, k)]).sum() for n in heldout_batch.getnnz(axis=1)])
    ndcg = dcg / idcg
    ndcg[np.isnan(ndcg)] = 0
    return ndcg


def recall_at_k_batch(x_pred, heldout_batch, k=100):
    batch_users = x_pred.shape[0]
    idx = bn.argpartition(-x_pred, k, axis=1)
    x_pred_binary = np.zeros_like(x_pred, dtype=bool)
    x_pred_binary[np.arange(batch_users)[:, np.newaxis], idx[:, :k]] = True

    x_true_binary = (heldout_batch > 0).toarray()
    tmp = (np.logical_and(x_true_binary, x_pred_binary).sum(axis=1)).astype(
        np.float32)
    # recall = tmp / np.minimum(k, x_true_binary.sum(axis=1))
    recall = tmp / x_true_binary.sum(axis=1)
    recall[np.isnan(recall)] = 0
    return recall


def main_trn(model, train_data, vad_data_tr, vad_data_te):
    set_rng_seed(ARG.seed)
    n = train_data.shape[0]
    idxlist = list(range(n))
    n_vad = vad_data_tr.shape[0]
    idxlist_vad = list(range(n_vad))
    num_batches = int(np.ceil(float(n) / ARG.batch))

    saver, logits_var, train_op_var, merged_var = model.build_graph()

    ndcg_var = tf.Variable(0.0)
    ndcg_best_var = tf.placeholder(dtype=tf.float64, shape=None)
    ndcg_summary = tf.summary.scalar('vad/ndcg', ndcg_var)
    ndcg_best_summary = tf.summary.scalar('vad/ndcg_best', ndcg_best_var)
    merged_valid = tf.summary.merge([ndcg_summary, ndcg_best_summary])

    if os.path.exists(LOG_DIR):
        shutil.rmtree(LOG_DIR)
    summary_writer = tf.summary.FileWriter(LOG_DIR, graph=tf.get_default_graph())
    if not os.path.isdir(LOG_DIR):
        os.makedirs(LOG_DIR)

    with tf.device('/gpu:' + gpu_id):
        config = tf.ConfigProto(allow_soft_placement=True)
        with tf.Session(config=config) as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            best_ndcg = -np.inf
            update_count = 0.0
            count = 0
            for epoch in range(ARG.epoch):
                np.random.shuffle(idxlist)
                for bnum, st_idx in enumerate(range(0, n, ARG.batch)):
                    end_idx = min(st_idx + ARG.batch, n)
                    x = train_data[idxlist[st_idx:end_idx]]
                    if sparse.isspmatrix(x):
                        x = x.toarray()
                    x = x.astype('float32')

                    feed_dict = {model.input_ph: x,
                                 model.keep_prob_ph: ARG.keep,
                                 model.is_training_ph: 1}
                    sess.run(train_op_var, feed_dict=feed_dict)
                    if bnum % 100 == 0:
                        summary_train = sess.run(merged_var, feed_dict=feed_dict)
                        summary_writer.add_summary(
                            summary_train,
                            global_step=epoch * num_batches + bnum)
                    update_count += 1

                ndcg_dist = []
                for bnum, st_idx in enumerate(range(0, n_vad, ARG.batch)):
                    end_idx = min(st_idx + ARG.batch, n_vad)
                    x = vad_data_tr[idxlist_vad[st_idx:end_idx]]
                    if sparse.isspmatrix(x):
                        x = x.toarray()
                    x = x.astype('float32')
                    pred_val = sess.run(logits_var, feed_dict={model.input_ph: x})
                    pred_val[x.nonzero()] = -np.inf
                    ndcg_dist.append(
                        ndcg_binary_at_k_batch(
                            pred_val, vad_data_te[idxlist_vad[st_idx:end_idx]]))
                ndcg_dist = np.concatenate(ndcg_dist)
                ndcg = ndcg_dist.mean()
                print('epoch:', epoch, ndcg)
                if ndcg > best_ndcg:
                    saver.save(sess, '{}/chkpt'.format(LOG_DIR))
                    count=0
                    best_ndcg = ndcg
                else:
                    count = count+1
                if count > 50 and epoch > 200:
                    break
                merged_valid_val = sess.run(
                    merged_valid,
                    feed_dict={ndcg_var: ndcg, ndcg_best_var: best_ndcg})
                summary_writer.add_summary(merged_valid_val, epoch)

    return best_ndcg


def main_tst(model, tst_data_tr, tst_data_te):
    set_rng_seed(ARG.seed)

    n_test = tst_data_tr.shape[0]
    idxlist_test = list(range(n_test))
    saver, logits_var, _, _ = model.build_graph()

    n20_list, n50_list, r20_list, r50_list = [], [], [], []
    with tf.Session() as sess:
        saver.restore(sess, '{}/chkpt'.format(LOG_DIR))
        for bnum, st_idx in enumerate(range(0, n_test, ARG.batch)):
            end_idx = min(st_idx + ARG.batch, n_test)
            x = tst_data_tr[idxlist_test[st_idx:end_idx]]
            if sparse.isspmatrix(x):
                x = x.toarray()
            x = x.astype('float32')
            pred_val = sess.run(logits_var, feed_dict={model.input_ph: x})
            pred_val[x.nonzero()] = -np.inf

            n20_list.append(
                ndcg_binary_at_k_batch(
                    pred_val, tst_data_te[idxlist_test[st_idx:end_idx]], k=20))
            n50_list.append(
                ndcg_binary_at_k_batch(
                    pred_val, tst_data_te[idxlist_test[st_idx:end_idx]], k=50))
            r20_list.append(
                recall_at_k_batch(
                    pred_val, tst_data_te[idxlist_test[st_idx:end_idx]], k=20))
            r50_list.append(
                recall_at_k_batch(
                    pred_val, tst_data_te[idxlist_test[st_idx:end_idx]], k=50))

    n20_list = np.concatenate(n20_list)
    n50_list = np.concatenate(n50_list)
    r20_list = np.concatenate(r20_list)
    r50_list = np.concatenate(r50_list)

    print("Test NDCG@20=%.5f (%.5f)" % (
        n20_list.mean(), np.std(n20_list) / np.sqrt(len(n20_list))),
          file=sys.stderr)
    print("Test NDCG@50=%.5f (%.5f)" % (
        n50_list.mean(), np.std(n50_list) / np.sqrt(len(n50_list))),
          file=sys.stderr)
    print("Test Recall@20=%.5f (%.5f)" % (
        r20_list.mean(), np.std(r20_list) / np.sqrt(len(r20_list))),
          file=sys.stderr)
    print("Test Recall@50=%.5f (%.5f)" % (
        r50_list.mean(), np.std(r50_list) / np.sqrt(len(r50_list))),
          file=sys.stderr)
    return n20_list.mean()

def main():
    (n_items, train_data, tst_data_tr, test_data, item_words, word_embs) = load_data(ARG.data)
    val, tst = 0, 0
    if ARG.mode in ('trn',):
        tf.reset_default_graph()
        model = TopicDAE(n_items, item_words, word_embs)
        val = main_trn(model, train_data, tst_data_tr, test_data)
    if ARG.mode in ('trn', 'tst'):
        tf.reset_default_graph()
        model = TopicDAE(n_items, item_words, word_embs)
        tst = main_tst(model, tst_data_tr, test_data)
        print('(%.5f, %.5f)' % (val, tst))

if __name__ == '__main__':
    main()
