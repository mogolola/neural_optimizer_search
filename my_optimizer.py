from keras.optimizers import Optimizer
from keras import backend as K
from keras.legacy import interfaces
import numpy as np
import tensorflow as tf

#Search space for action
operands = {1: 'g', 2: 'g2', 3: 'g3', 4: 'm', 5: 'v', 6: 'y', 7: 'sign(g)', 8: 'sign(m)', 9: '1', 10: 'noise',
            11: '10-4w', 12: '10-3w', 13: '10-2w', 14: '10-1w', 15: 'ADAM', 16: 'RMSProp'}

unarys = {1: '1', 2: '-1', 3: 'exp', 4: 'log', 5: 'clip10-5', 6: 'clip10-4', 7: 'clip10-3', 8: 'drop0.1', 9:'drop0.3',
          10: 'drop0.5', 11: 'sign'}

binarys = {1: 'add', 2: 'sub', 3: 'mul', 4: 'div', 5: 'keep_left'}

class my_optimizer(Optimizer):
    def __init__(self, lr=0.0001, beta_1=0.9, beta_2=0.999, beta_3=0.999,
                 epsilon=None, decay=0., amsgrad=False, strings=None, **kwargs):
        super(my_optimizer, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.beta_1 = K.variable(beta_1, name='beta_1') #learning rate for m_t
            self.beta_2 = K.variable(beta_2, name='beta_2') #learning rate for v_t
            self.beta_3 = K.variable(beta_3, name='beta_3') #learning rate for y_t
            self.decay = K.variable(decay, name='decay')
        if epsilon is None:
            epsilon = K.epsilon()
        self.epsilon = epsilon
        self.initial_decay = decay
        self.amsgrad = amsgrad

        print (type(strings))


        self.op1, self.op2, self.unop1, self.unop2, self.biops = strings


    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):

        grads = self.get_gradients(loss, params)
        accumulators = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr *= (1. / (1. + self.decay * K.cast(self.iterations,
                                                  K.dtype(self.decay))))

        t = K.cast(self.iterations, K.floatx()) + 1
        lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) /
                     (1. - K.pow(self.beta_1, t)))

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        ys = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]



        if self.amsgrad:
            vhats = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        else:
            vhats = [K.zeros(1) for _ in params]
        self.weights = [self.iterations] + ms + vs + vhats

        for p, g, m, v, y, vhat, a in zip(params, grads, ms, vs, ys, vhats, accumulators):
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)
            y_t = (self.beta_3 * y) + (1. - self.beta_3) * K.pow(g, 3)

            new_a = self.beta_1 * a + (1. - self.beta_1) * K.square(g)
            self.updates.append(K.update(a, new_a))

            noise = tf.random_normal(
                K.int_shape(p),
                mean=0.0,
                stddev=0.1,
                dtype=K.dtype(p),
            )

            delta_adam = m_t / (K.sqrt(v_t) + self.epsilon)

            delta_rms = g / (K.sqrt(new_a) + self.epsilon)
############################################
            if self.op1 == 1:
                o1 = g
            elif self.op1 == 2:
                o1 = tf.square(g)
            elif self.op1 == 3:
                o1 = tf.pow(g, 3)
            elif self.op1 == 4:
                o1 = m_t
            elif self.op1 == 5:
                o1 = v_t
            elif self.op1 == 6:
                o1 = y_t
            elif self.op1 == 7:
                o1 = K.sign(g)
            elif self.op1 == 8:
                o1 = K.sign(m_t)
            elif self.op1 == 9:
                o1 = tf.constant(1, dtype=tf.float32)
            elif self.op1 == 10:
                o1 = noise
            elif self.op1 == 11:
                o1 = (10 ** (-4)) * p
            elif self.op1 == 12:
                o1 = (10 ** (-3)) * p
            elif self.op1 == 13:
                o1 = (10 ** (-2)) * p
            elif self.op1 ==14:
                o1 = (10 ** (-1)) * p
            elif self.op1 == 15:
                o1 = delta_adam
            elif self.op1 == 16:
                o1 = delta_rms
   ####################################
            if self.op2 == 1:
                o2 = g
            elif self.op2 == 2:
                o2 = tf.square(g)
            elif self.op2 == 3:
                o2 = tf.pow(g, 3)
            elif self.op2 == 4:
                o2 = m_t
            elif self.op2 == 5:
                o2 = v_t
            elif self.op2 == 6:
                o2 = y_t
            elif self.op2 == 7:
                o2 = K.sign(g)
            elif self.op2 == 8:
                o2 = K.sign(m_t)
            elif self.op2 == 9:
                o2 = tf.constant(1, dtype=tf.float32)
            elif self.op2 == 10:
                o2 = noise
            elif self.op2 == 11:
                o2 = (10 ** (-4)) * p
            elif self.op2 == 12:
                o2 = (10 ** (-3)) * p
            elif self.op2 == 13:
                o2 = (10 ** (-2)) * p
            elif self.op2 ==14:
                o2 = (10 ** (-1)) * p
            elif self.op2 == 15:
                o2 = delta_adam
            elif self.op2 == 16:
                o2 = delta_rms
##############################################
            if self.unop1 == 1:
                u1 = o1
            elif self.unop1 == 2:
                u1 = -o1
            elif self.unop1 == 3:
                u1 = K.exp(o1)
            elif self.unop1 == 4:
                u1 = K.log(K.abs(o1))
            elif self.unop1 == 5:
                u1 = K.clip(o1, -(10 ** (-5)), 10 ** (-5))
            elif self.unop1 == 6:
                u1 = K.clip(o1, -(10 ** (-4)), 10 ** (-4))
            elif self.unop1 == 7:
                u1 = K.clip(o1, -(10 ** (-3)), 10 ** (-3))
            elif self.unop1 -- 8:
                u1 = K.dropout(o1,0.9)
            elif self.unop1 == 9:
                u1 = K.dropout(o1, 0.7)
            elif self.unop1 == 10:
                u1 = K.dropout(o1, 0.5)
            elif self.unop1 == 11:
                u1 = K.sign(o1)
 ##############################################

            if self.unop2 == 1:
                u2 = o2
            elif self.unop2 == 2:
                u2 = -o2
            elif self.unop2 == 3:
                u2 = K.exp(o2)
            elif self.unop2 == 4:
                u2 = K.log(K.abs(o2))
            elif self.unop2 == 5:
                u2 = K.clip(o2, -(10 ** (-5)), 10 ** (-5))
            elif self.unop2 == 6:
                u2 = K.clip(o2, -(10 ** (-4)), 10 ** (-4))
            elif self.unop2 == 7:
                u2 = K.clip(o2, -(10 ** (-3)), 10 ** (-3))
            elif self.unop2 == 8:
                u2 = K.dropout(o2,0.9)
            elif self.unop2 == 9:
                u2 = K.dropout(o2, 0.7)
            elif self.unop2 == 10:
                u2 = K.dropout(o2, 0.5)
            elif self.unop2 == 11:
                u2 = K.sign(o2)

#################################################


            if self.biops == 1:
                delta = u1 + u2
            elif self.biops == 2:
                delta = u1 - u2
            elif self.biops == 3:
                delta = u1 * u2
            elif self.biops == 4:
                delta = u1 / (u2 + self.epsilon)
            elif self.biops == 5:
                delta = u1
###################################################


            p_t = p - self.lr * delta





            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))
            self.updates.append(K.update(y, y_t))
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates




def get_config(self):
    config = {'lr': float(K.get_value(self.lr)),
              'beta_1': float(K.get_value(self.beta_1)),
              'beta_2': float(K.get_value(self.beta_2)),
              'decay': float(K.get_value(self.decay)),
              'epsilon': self.epsilon,
              'amsgrad': self.amsgrad}
    base_config = super(my_optimizer, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))