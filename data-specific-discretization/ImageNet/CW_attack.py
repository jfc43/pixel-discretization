import tensorflow as tf
import numpy as np
from sklearn.neighbors.kde import KernelDensity
from sklearn.neighbors import KDTree
from tensorflow.contrib.slim.nets import inception

slim = tf.contrib.slim

class CWAttack:
    def __init__(self, model, num_steps, step_size, epsilon, codes, batch_size, alpha):
        self.model = model
        self.num_steps = num_steps
        self.step_size = step_size
        self.codes = codes

        self.xs = tf.Variable(np.zeros((batch_size, 299, 299, 3), dtype=np.float32),
                                    name='modifier')
        self.orig_xs = tf.placeholder(tf.float32, [batch_size, 299, 299, 3])

        self.ys = tf.placeholder(tf.int32, [batch_size])

        self.epsilon = epsilon

        delta = tf.clip_by_value(self.xs, 0, 255) - self.orig_xs
        delta = tf.clip_by_value(delta, -self.epsilon, self.epsilon)

        self.do_clip_xs = tf.assign(self.xs, self.orig_xs+delta)

        w = []
        cw = []
        for i in range(codes.shape[0]):
            wt = tf.exp(-alpha*tf.abs(self.xs-codes[i]))
            w.append(wt)
            cw.append(codes[i]*wt)
        self.z = sum(cw)/sum(w)

        logits = model.forward(self.z)

        label_mask = tf.one_hot(self.ys, 1001)
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

    def preprocess(self, images0):
        images = np.copy(images0).astype(float)
        kd = KDTree(self.codes, metric='infinity')
        new_images = []
        for img in images:
            points = img.reshape(-1,3)
            inds = np.squeeze(kd.query(points,return_distance=False))
            new_images.append(self.codes[inds].reshape(img.shape))
        return np.array(new_images)

    def perturb(self, x, y, sess):
        sess.run(self.new_vars_initializer)
        sess.run(self.xs.initializer)
        sess.run(self.do_clip_xs,
                 {self.orig_xs: x})

        for i in range(self.num_steps):
            imgs = sess.run(self.xs)
            points = imgs.reshape((-1,3))
            t = self.preprocess(imgs)
            sess.run(self.train, feed_dict={self.ys: y,
                                            self.z: t})
            sess.run(self.do_clip_xs,
                     {self.orig_xs: x})

        return sess.run(self.xs)

class KMCWAttack:
    def __init__(self, model, num_steps, step_size, epsilon, batch_size, num_clusters, alpha=0.1):
        self.model = model
        self.num_steps = num_steps
        self.step_size = step_size
        self.num_clusters = num_clusters
        self.eng = matlab.engine.start_matlab('-nodisplay')
        self.codes = tf.placeholder(tf.float32,shape=[num_clusters,3])
        self.xs = tf.Variable(np.zeros((batch_size, 299, 299, 3), dtype=np.float32),
                                    name='modifier')
        self.orig_xs = tf.placeholder(tf.float32, [batch_size, 299, 299, 3])

        self.ys = tf.placeholder(tf.int32, [batch_size])

        self.epsilon = epsilon

        delta = tf.clip_by_value(self.xs, 0, 255) - self.orig_xs
        delta = tf.clip_by_value(delta, -self.epsilon, self.epsilon)

        self.do_clip_xs = tf.assign(self.xs, self.orig_xs+delta)

        w = []
        cw = []
        for i in range(self.num_clusters):
          wt = tf.exp(-alpha*tf.norm(self.xs-self.codes[i],ord=np.inf,axis=-1))
          w.append(tf.expand_dims(wt,axis=-1))
          cw.append(self.codes[i]*tf.expand_dims(wt,axis=-1))
        self.z = sum(cw)/(sum(w))
        logits = self.model.forward(self.z)
        label_mask = tf.one_hot(self.ys, 1001)
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

    def preprocess(self, images0):
        images = np.copy(images0).astype(float)
        new_images = []
        for img in images:
            points = img.reshape((-1,3))
            idx, C = self.eng.kmedoids(matlab.double(points.tolist()),self.num_clusters,'Distance','chebychev',nargout=2)
            idx = np.squeeze(np.array(idx))
            C = np.array(C)
            for i in range(self.num_clusters):
                points[np.where(idx==(i+1))] = C[i]
            new_images.append(points.reshape(img.shape))
        return np.array(new_images), C

    def perturb(self, x, y, sess):
        sess.run(self.new_vars_initializer)
        sess.run(self.xs.initializer)
        sess.run(self.do_clip_xs,
                 {self.orig_xs: x})

        for i in range(self.num_steps):
            imgs = sess.run(self.xs)
            points = imgs.reshape((-1,3))
            t, codes = self.preprocess(imgs)
            sess.run(self.train, feed_dict={self.ys: y,
                                            self.z: t,
                                            self.codes: codes})
            sess.run(self.do_clip_xs,
                     {self.orig_xs: x})

        return sess.run(self.xs)
