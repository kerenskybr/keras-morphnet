from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import tensorflow as tf

from model import MorphNetModel
from morph_net.network_regularizers import flop_regularizer
from morph_net.tools import structure_exporter

import os

batch_size = 128
num_classes = 10
epochs = 1


# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

(x_train, labels_not_encoded), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

print("Y ONE HOT", y_test)

def create_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model

model = create_model()

#print("LAYERS", model.layers[7].input)

#inputs = model.inputs[:2]
#dense = model.get_layer('NSP-Dense').output
#outputs = keras.layers.Dense(units=2, activation='softmax')(dense)

# Adding Morphnet babysitting process
logits = model.get_layer(index=7).output

network_regularizer = flop_regularizer.GammaFlopsRegularizer(
    output_boundary=[logits.op], #Logits
    input_boundary=[x_train.shape, y_train.shape], #Inputs.op, labels.op
    gamma_threshold=1e-3
)

regularization_strength = 1e-10
regularizer_loss = (network_regularizer.get_regularization_term() * regularization_strength)

model_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.cast(labels_not_encoded, tf.int32), logits=logits)

optimizer = tf.train.MomentumOptimizer(learning_rate=0.01, momentum=0.9)

train_op = optimizer.minimize(model_loss + regularizer_loss)

tf.summary.scalar('RegularizationLoss', regularizer_loss)
tf.summary.scalar(network_regularizer.cost_name, network_regularizer.get_cost())



# Saving the graph
session = tf.InteractiveSession()
writer = tf.summary.FileWriter(os.getcwd() + '/graphs', session.graph)

m_model = MorphNetModel(
    base_model=model,
    num_classes=10,
    learning_rate=1e-3,
    batch_size=256,
    main_train_device="/cpu:0",
    main_eval_device="/cpu:0",
    morphnet_regularizer_algorithm=regularizer_loss,
    morphnet_target_cost="FLOPs",
    morphnet_regularizer_threshold=1e-2,
    morphnet_regularization_strength=1e-2,
    log_dir='/morphnet_log')




# Compiling the model
#model.compile(loss=keras.losses.categorical_crossentropy,
#              optimizer=keras.optimizers.Adadelta(),
#              metrics=['accuracy'])

'''
model.compile(loss=model_loss,
              optimizer=train_op,
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
'''




'''
# Saving the pruned architecture
train_dir = os.getcwd() + '/json'

exporter = structure_exporter.StructureExporter(
    network_regularizer.op_regularizer_manager)

with tf.Session() as sess:
    tf.compat.v1.global_variables_initializer().run()
    for step in range(epochs):
        _, structure_exporter_tensors = sess.run([train_op, exporter.tensors])
    if (step % 1 == 0):
        exporter.populate_tensor_values(structure_exporter_tensors)
        exporter.create_file_and_save_alive_counts(train_dir, step)
#tensorboard --logdir=graphs
'''