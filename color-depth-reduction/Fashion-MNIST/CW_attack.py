import tensorflow as tf
import numpy as np
from sklearn.neighbors.kde import KernelDensity
from util import preprocess

class CWAttack:
    def __init__(self, model, num_steps, step_size, epsilon, codes, batch_size, alpha):
        self.model = model
        self.num_steps = num_steps
        self.step_size = step_size
        self.codes = codes

        self.xs = tf.Variable(np.zeros((batch_size, 784), dtype=np.float32),
                                    name='modifier')
        self.orig_xs = tf.placeholder(tf.float32, [batch_size, 784])

        self.ys = tf.placeholder(tf.int32, [batch_size])

        self.epsilon = epsilon

        delta = tf.clip_by_value(self.xs, 0, 1) - self.orig_xs
        delta = tf.clip_by_value(delta, -self.epsilon, self.epsilon)

        self.do_clip_xs = tf.assign(self.xs, self.orig_xs+delta)

        w = []
        cw = []
        for i in range(codes.shape[0]):
            wt = tf.exp(-alpha*tf.abs(self.xs-codes[i]))
            w.append(wt)
            cw.append(codes[i]*wt)
        self.z = sum(cw)/sum(w)

        logits = self.model.forward(self.z)
        label_mask = tf.one_hot(self.ys, 10)
        correct_logit = tf.reduce_sum(label_mask * logits, axis=1)
        wrong_logit = tf.reduce_max((1-label_mask) * logits - 1e4*label_mask, axis=1)
        self.loss = (correct_logit - wrong_logit)
        start_vars = set(x.name for x in tf.global_variables())
        optimizer = tf.train.AdamOptimizer(step_size*1)

        grad,var = optimizer.compute_gradients(self.loss, [self.xs])[0]
        self.train = optimizer.apply_gradients([(tf.sign(grad),var)])

        end_vars = tf.global_variables()
        self.new_vars = [x for x in end_vars if x.name not in start_vars]
        self.new_vars_initializer = tf.variables_initializer(self.new_vars)

    def perturb(self, x, y, sess):
        sess.run(self.new_vars_initializer)
        sess.run(self.xs.initializer)
        sess.run(self.do_clip_xs,
                 {self.orig_xs: x})

        for i in range(self.num_steps):
            imgs = sess.run(self.xs)
            points = imgs.reshape((-1,1))
            t = preprocess(imgs, self.codes)
            sess.run(self.train, feed_dict={self.ys: y,
                                            self.z: t})
            sess.run(self.do_clip_xs,
                     {self.orig_xs: x})

        return sess.run(self.xs)
