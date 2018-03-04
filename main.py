import numpy as np
import tensorflow as tf
import datetime
from train_target import Conv
from Controller import Policy_network

log_dir = 'log'


MAX_EPISODES = 2500  # maximum episode of trials
MAX_EPOCHS = 5  # maximum of epochs to train target network
EXPLORATION = 0.8   # initial rate of exploration, epsilon greedy
REGULARIZATION = 1e-3   # regularization rate
CONTROLLER_CELLS = 32   # number of cells in rnn controller
LEARNING_RATES = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1] #candidate learning rate for target network



def main():

    global args
    sess = tf.Session()
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(0.99, global_step,
                                               500, 0.96, staircase=True)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    policy_network = Policy_network(sess, optimizer, global_step,
                                    exploration=EXPLORATION,
                                    reg_param=REGULARIZATION,
                                    controller_cells=CONTROLLER_CELLS)

    step = 0
    # define initial input state
    state = np.array([[1, 9, 1, 1, 3]], dtype=np.float32)
    state = np.expand_dims(state, -1)

    total_rewards = 0
    moving_acc = 0.0    # approximate average accurency
    beta = 0.9  # parameter to update moving accurency

    for i_episode in range(MAX_EPISODES):
        action = policy_network.get_action(state)
        print("ca:", action)

        #train target network and get reward (accuracy on hold-out set)
        convsess = tf.Session()
        target = Conv(convsess)
        target_acc_rec = []
        for i, lr in enumerate(LEARNING_RATES):
            print('train target network with learning rate %d ========>' % lr)
            target.train_one_epoche(lr=lr, action=action)
            target_acc = target.test()
            target_acc_rec.append(target_acc)
        best_lr_index = int(np.argmax(target_acc_rec))
        best_lr = LEARNING_RATES[best_lr_index]

        print ('train target network with the best learning rate %d ===========>' % best_lr)
        target.train(lr=best_lr, action=action, epochs=MAX_EPOCHS)
        acc = target.test()
        tf.summary.scalar('accurency', acc)
        convsess.close()
        moving_acc = moving_acc * beta + acc * (1 - beta)
        reward = acc - moving_acc



        print("reward=====>", reward)

        total_rewards += reward

        # action is equal to state
        state = action
        policy_network.storeRollout(state, reward)
        state = np.expand_dims(state, 0)
        state = np.expand_dims(state, -1)

        step += 1
        ls = policy_network.train_step(1)
        log_str = "current time:  " + str(datetime.datetime.now().time()) + " episode:  " + \
                  str(i_episode) + " loss:  " + str(ls) + " last_state:  " + str(np.squeeze(state)) + " last_reward:  " + \
                  str(reward) + " last_acc " + str(acc) + " moving_acc " + str(moving_acc) + 'learning_rate' + str(best_lr) + "\n"
        log = open("lg3.txt", "a+")
        log.write(log_str)
        log.close()
        print(log_str)



if __name__ == '__main__':
    main()














