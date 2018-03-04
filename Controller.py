import tensorflow as tf
import random
import numpy as np

log_dir = 'log'
state_space = {'size': [16, 16, 11, 11, 5],
               'space': [[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16], [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16],
                         [1,2,3,4,5,6,7,8,9,10,11], [1,2,3,4,5,6,7,8,9,10,11], [1,2,3,4,5] ]}


class Policy_network():
    def __init__(self, sess, optimizer, global_step,
                 reg_param=0.001,
                 discount_factor=0.99,
                 exploration=0.8,
                 controller_cells=32
                 ):
        self.sess = sess
        self.optimizer = optimizer
        self.reg_param = reg_param
        self.discount_factor = discount_factor
        self.controller_cells = controller_cells
        self.global_step = global_step
        self.exploration = exploration
        self.cell_outputs = []
        self.policy_classifiers = []
        self.policy_actions = []
        self.policy_labels = []


        self.reward_buffer = []
        self.state_buffer = []

        self.build_policy_network()
        var_lists = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        self.sess.run(tf.variables_initializer(var_lists))
        self.writer = tf.summary.FileWriter(log_dir, graph=tf.get_default_graph())

    def build_policy_network(self):
        with tf.name_scope("policy_network"):
            # use NAS cell, can be replaced by any other cell, we can also use other RNN cell
            nas_cell = tf.contrib.rnn.NASCell(self.controller_cells)
            cell_state = nas_cell.zero_state(batch_size=1, dtype=tf.float32)

            #initially, cell input will be the state input
            with tf.name_scope('state_input'):
                state_input = tf.placeholder(dtype=tf.float32, shape=(1, None, 1), name='state_input')
            self.state_input = state_input
            cell_input = state_input



            for i in range(5):
                size = state_space['size'][i]

                with tf.name_scope('controller_output_%d' % i):

                    outputs, final_state = tf.nn.dynamic_rnn(
                        cell=nas_cell,
                        inputs=cell_input,
                        initial_state=cell_state,
                        dtype=tf.float32
                    )

                    #add a new classifier for each layers output
                    classifier = tf.layers.dense(inputs=outputs[:, -1, :], units=size, name='classifier_%d' % i, reuse=False)
                    #print('outputs_%d' % i, outputs)
                    #print ('classifier_%d' % i, classifier)

                    preds = tf.nn.softmax(classifier)

                    #feed next layer with current output, as well as state
                    cell_input = tf.expand_dims(classifier, -1, name='cell_output_%d' % i)
                    #print('input_%d' % (i+1), cell_input)
                    cell_state = final_state

                # store tensors for later loss computations
                self.cell_outputs.append(cell_input)
                self.policy_classifiers.append(classifier)
                self.policy_actions.append(preds)




        # collect all variables for regularization loss
        policy_network_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='policy_network')

        # compute loss and gradients
        with tf.name_scope("compute_gradients"):
            # gradients for selecting action from policy network
            self.discounted_rewards = tf.placeholder(tf.float32, (None,), name="discounted_rewards")
            tf.summary.scalar('discounted_reward', tf.reduce_sum(self.discounted_rewards))

            #calculate sum of cross entrophy loss of all the individual classifiers
            cross_entrophy_loss = 0

            for i in range(5):
                classifier = self.policy_classifiers[i]
                size = state_space['size'][i]

                with tf.name_scope('state_%d' % (i+1)):
                    labels = tf.placeholder(dtype=tf.float32, shape=(None, size), name='cell_label_%d' % i)
                    self.policy_labels .append(labels)

                    one_cross_entropy_loss = tf.nn.softmax_cross_entropy_with_logits(logits=classifier, labels=labels)
                    print (classifier, labels)
                cross_entrophy_loss += one_cross_entropy_loss
            pg_loss = tf.reduce_mean(cross_entrophy_loss)
            reg_loss = tf.reduce_sum([tf.reduce_sum(tf.square(x)) for x in policy_network_variables]) # regularization

            self.total_loss = pg_loss + self.reg_param * reg_loss
            tf.summary.scalar('total_loss', self.total_loss)

            # compute gradients
            self.gradients = self.optimizer.compute_gradients(self.total_loss)


            # compute policy gradients
            for i, (grad, var) in enumerate(self.gradients):
                if grad is not None:
                    self.gradients[i] = (grad * self.discounted_rewards, var)

            # training update
            with tf.name_scope("train_policy_network"):
                # apply gradients to update policy network
                self.train_op = self.optimizer.apply_gradients(self.gradients, global_step=self.global_step)






    def get_action(self, state):
        #return self.sess.run(self.predicted_action, {self.states: state})
        if random.random() < self.exploration:
            return np.array([random.choice(range(1, 17)), random.choice(range(1, 17)), random.choice(range(1, 12)),
                               random.choice(range(1, 12)), random.choice(range(1, 6))])
        else:
            preds = self.sess.run(self.policy_actions, {self.state_input: state})
            action = []
            for i, pred in enumerate(preds):
                print (pred)
                one_action = np.random.choice(
                    state_space['space'][i],
                    1,
                    p=pred[0]
                )
                action.append(one_action[0])
            return action




    def storeRollout(self, state, reward):
        self.reward_buffer.append(reward)
        self.state_buffer.append(state)

    def train_step(self, steps_count):
        states = np.array(self.state_buffer[-steps_count:])
        feed_dict = {}
        labels = []
        for i, state in enumerate(states[0]):
            one_hot = np.zeros(state_space['size'][i])
            one_hot[state-1] = 1.
            one_hot = np.expand_dims(one_hot, 0)
            labels.append(one_hot)
            feed_dict[self.policy_labels[i]] = one_hot
        print ('states:', states)
        states = np.expand_dims(states, -1)
        rewars = self.reward_buffer[-steps_count:]
        feed_dict[self.state_input] = states
        feed_dict[self.discounted_rewards] = rewars


        tf.summary.scalar('reward', rewars[0])
        merged = tf.summary.merge_all()
        _, ls, summary_str, global_step = self.sess.run([self.train_op, self.total_loss, merged, self.global_step],
                                                        {self.state_input: states,
                                                         self.discounted_rewards: rewars,
                                                         self.policy_labels[0]: labels[0],
                                                         self.policy_labels[1]: labels[1],
                                                         self.policy_labels[2]: labels[2],
                                                         self.policy_labels[3]: labels[3],
                                                         self.policy_labels[4]: labels[4]})
        self.writer.add_summary(summary_str)


        # epsilon greedy with decay
        if global_step != 0 and global_step % 20 == 0 and self.exploration > 0.2:
            self.exploration *= 0.97

        return ls