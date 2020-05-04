from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import time
import tensorflow.keras.backend as K
import os
from tensorflow.keras.layers import \
        Conv2D, MaxPool2D, Dropout, Flatten, Dense

# print("Number of GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

class Model(tf.keras.Model):
    def __init__(self):

        super(Model, self).__init__()

        self.batch_size = 64

        self.architecture = [
            # # Block 1
            # Conv2D(64, 3, 1, padding="same", activation="relu", name="block1_conv1"),
            # Conv2D(64, 3, 1, padding="same", activation="relu", name="block1_conv2"),
            # MaxPool2D(2, name="block1_pool"),
            # # Block 2
            # Conv2D(128, 3, 1, padding="same", activation="relu", name="block2_conv1"),
            # Conv2D(128, 3, 1, padding="same", activation="relu", name="block2_conv2"),
            # MaxPool2D(2, name="block2_pool"),
            # # Block 3
            # Conv2D(256, 3, 1, padding="same", activation="relu", name="block3_conv1"),
            # Conv2D(256, 3, 1, padding="same", activation="relu", name="block3_conv2"),
            # Conv2D(256, 3, 1, padding="same", activation="relu", name="block3_conv3"),
            # MaxPool2D(2, name="block3_pool"),
            # # Block 4
            # Conv2D(512, 3, 1, padding="same", activation="relu", name="block4_conv1"),
            # Conv2D(512, 3, 1, padding="same", activation="relu", name="block4_conv2"),
            # Conv2D(512, 3, 1, padding="same", activation="relu", name="block4_conv3"),
            # MaxPool2D(2, name="block4_pool"),
            # # Block 5
            # Conv2D(512, 3, 1, padding="same", activation="relu", name="block5_conv1"),
            # Conv2D(512, 3, 1, padding="same", activation="relu", name="block5_conv2"),
            # Conv2D(512, 3, 1, padding="same", activation="relu", name="block5_conv3"),
            # MaxPool2D(2, name="block5_pool"),
            
            Conv2D(64, 3, 3, padding="same", activation="relu", name="conv1"),
            # # Conv2D(64, 3, 1, padding="same", activation="relu", name="conv11"),
            MaxPool2D(2, name="pool1"),

            Conv2D(128, 3, 3, padding="same", activation="relu", name="conv2"),
            # Conv2D(128, 3, 1, padding="same", activation="relu", name="conv22"),
            MaxPool2D(2, name="pool2"),

            Conv2D(256, 3, 3, padding="same", activation="relu", name="conv3"),
            # Conv2D(128, 3, 1, padding="same", activation="relu", name="conv4"),
            MaxPool2D(2, name="pool3"),

            # Conv2D(32, 3, 1, padding="same", activation="relu", name="conv4"),
            # Conv2D(128, 3, 1, padding="same", activation="relu", name="conv4"),
            # MaxPool2D(2, name="pool4"),

            # # Dropout(rate=0.5),

            Flatten(),

            # # Dense(256, activation="relu"),
            
            # # Dropout(rate=0.5),

            Dense(128, activation="relu"),

            Dropout(rate=0.5),

            Dense(64, activation="relu"),

            # Dropout(rate=0.5),

            # Dense(16, activation="relu")
            # Dense(250, activation="softmax")

        ]

        self.dense1 = tf.keras.layers.Dense(64, activation="relu")
        self.dense2 = tf.keras.layers.Dense(16, activation="relu")
        self.dense3 = tf.keras.layers.Dense(2, activation=None)
        # self.optimizer = tf.keras.optimizers.Adam(0.001)
        self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=1e-3, momentum=0.9)
        
    def call(self, inputs, img_db):
        # inputs (N, 2)
        # self.E (num_of_imgs, h, w, 3)
        img_db = tf.convert_to_tensor(img_db)

        img1 = tf.nn.embedding_lookup(img_db, inputs[:,0])
        img2 = tf.nn.embedding_lookup(img_db, inputs[:,1]) # (batch, h, w, 3)

        for layer in self.architecture:
            img1 = layer(img1)
            img2 = layer(img2)

        # (batch, features)

        h = tf.concat([img1,img2], axis=1) #(batch, features*2)

        logits = self.dense1(h)
        logits = self.dense2(logits)
        logits = self.dense3(logits)

        # mold
#        v1 = tf.reduce_sum(tf.multiply(encoder_q1, encoder_q1), axis=1)
#        v2 = tf.reduce_sum(tf.multiply(encoder_q2, encoder_q2), axis=1)
#        r = tf.math.maximum(v1/v2, v2/v1) #(1,+)
#        logits = -tf.math.log(r-1+1e-8)
#        logits = tf.reshape(logits, [-1, 1])
#        logits = tf.concat([-logits, logits], axis=1)

        return tf.math.sigmoid(logits)

    def loss(self, logits, labels):
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=labels, y_pred=logits)
        loss = tf.reduce_mean(loss)
        return loss

def train(model, train_inputs, train_labels, img_db, test_inputs, test_labels):
    # (N, window, 2)
    num_epochs = 100
    train_size = train_inputs.shape[0]
    index = [i for i in range(train_size)]
    index = tf.random.shuffle(index)
    train_inputs = tf.gather(train_inputs, index)
    train_labels = tf.gather(train_labels, index)
    print('Total steps: ', int(train_size/model.batch_size))
    for epoch in range(num_epochs):
        train_loss = 0
        step = 0
        for start, end in zip(range(0, train_size - model.batch_size, model.batch_size), range(model.batch_size, train_size, model.batch_size)):
            batch_data = train_inputs[start:end]
            batch_labels = train_labels[start:end]
            with tf.GradientTape() as tape:
                logits = model.call(batch_data, img_db)
                loss = model.loss(logits, batch_labels)

            train_loss += loss
            step += 1
            if step % 50 == 0:
                print('Step %d \t Loss: %.3f' % (step, train_loss / step))
            gradients = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        if epoch % 1 == 0:
            test(model, test_inputs, test_labels, img_db)
            # manager.save()
            print("--------------------------------------------------------------")
            print('Epoch %d \t Loss: %.3f' % (epoch, train_loss / step))
            print("--------------------------------------------------------------")

def test(model, test_inputs, test_labels, img_db):
    # print(test_labels)
    test_size = test_inputs.shape[0]
    print('Total steps: ', int(test_size/model.batch_size))
    count = 0
    accuracy = 0.0
    f1 = 0.0
    for start, end in zip(range(0, test_size - model.batch_size, model.batch_size), range(model.batch_size, test_size, model.batch_size)):
        cur_inputs = test_inputs[start:end]
        cur_labels = test_labels[start:end]
        logits = model.call(cur_inputs, img_db)
        pred = tf.argmax(logits, axis=1)
        f1 += f1_score(cur_labels, pred)
        result = tf.dtypes.cast(tf.math.equal(pred, cur_labels), tf.float32)
        accuracy += tf.reduce_mean(result)
        count += 1
    print('Acc:', accuracy.numpy()/count)
    print('F1 Score:', f1.numpy()/count)

def f1_score(y_true, y_pred):
    y_true = tf.dtypes.cast(K.flatten(y_true), tf.float64)
    y_pred = tf.dtypes.cast(K.flatten(y_pred), tf.float64)
    return 2*(K.sum(y_true * y_pred)+K.epsilon()) / (K.sum(y_true) + K.sum(y_pred) + K.epsilon())


def main():
    img_db_fp = 'img_db.npy'
    train_data_fp = 'train_data.npy'
    train_labels_fp = 'train_labels.npy'

    test_data_fp = 'test_data.npy'
    test_labels_fp = 'test_labels.npy'
    
    train_data = np.load(train_data_fp).astype(np.int32) #(N, 2)
    train_labels = np.load(train_labels_fp).astype(np.int32) #(N,)
    test_data = np.load(test_data_fp).astype(np.int32)
    test_labels = np.load(test_labels_fp).astype(np.int32)
    print('train data: ', train_data.shape)
    print('train labels: ', train_labels.shape)
    print('test data: ', test_data.shape)
    print('test labels: ', test_labels.shape)

    img_db = np.load(img_db_fp)
    
    model = Model()

    start = time.time()

    train(model, train_data, train_labels, img_db, test_data, test_labels)
    print('Training process takes %.4f minutes' % ((time.time()-start)/60))

    # print(accuracy)
    
if __name__ == '__main__':
    main()
