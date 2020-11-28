from __future__ import absolute_import, division, print_function, unicode_literals
import os

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import time
import datetime
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.tensorflow import balanced_batch_generator
from imblearn.over_sampling import RandomOverSampler
from termcolor import colored
import PIL.Image
import sys


# parameters for loading
data_path = sys.argv[1]
log_dir = sys.argv[2]
checkpoint_directory = log_dir + '/training_checkpoints'

# parameters for architecture
image_shape = list(map(int, sys.argv[3].strip('[]').split(',')))
input_shape = (image_shape[0], image_shape[1], 3)
classes = [name for name in os.listdir(data_path) if
           (os.path.isdir(data_path + '/' + name) and (not 'training_checkpoints' in name))]
print(classes)
num_classes = len(classes)
print(num_classes)

# parameters for training
epochs = int(sys.argv[4])
learning_rate = float(sys.argv[5])
batch_size = int(sys.argv[6])
val_test_split = float(sys.argv[7])
test_size = float(sys.argv[8])
seed = 1


def transform(X_i):
    return ((X_i / 127.5) - 1)


def detransform(X_i):
    return ((X_i + 1) * 127.5)


# tf2 version
def load_imgs_with_lbl(path, classes, shape):
    X = []
    y = []
    for lbl, cls in enumerate(classes):
        files = tf.io.gfile.glob(path + '/' + cls + '/*')
        for myFile in files:
            image = np.array(PIL.Image.open(tf.io.gfile.GFile(myFile, 'rb')).convert(
                'RGB').resize(shape, PIL.Image.BILINEAR), dtype=np.float32)
            X.append(image)
            y.append(lbl)
    return np.array(X), np.array(y)


def timer(start, end):
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    return ("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))


class CF(tf.keras.Model):
    def __init__(self):
        super(CF, self).__init__()
        self.bn = tf.keras.layers.BatchNormalization(input_shape=input_shape)
        basemodel = tf.keras.applications.VGG16(include_top=False, weights='imagenet')
        basemodel.trainable = True
        self.base = basemodel
        self.identity_gradcam = tf.keras.layers.Lambda(lambda x: x, name='identity_gradcam')
        self.gap = tf.keras.layers.GlobalAveragePooling2D()
        self.dense1 = tf.keras.layers.Dense(512, activation='relu')
        self.identity_tcav = tf.keras.layers.Lambda(lambda x: x, name='identity_tcav')
        self.dense2 = tf.keras.layers.Dense(num_classes)
        self.prediction = tf.keras.layers.Activation('softmax')

    @tf.function
    def call(self, x):
        x = self.bn(x)
        x = self.base(x)
        x = self.identity_gradcam(x)
        x = self.gap(x)
        x = self.dense1(x)
        x = self.identity_tcav(x)
        x = self.dense2(x)
        return self.prediction(x)


# Create an instance of the model
model = CF()

loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, amsgrad=False)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
val_loss = tf.keras.metrics.Mean(name='val_loss')
val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')


@tf.function
def train_step(images, labels, model):
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)

    return gradients, tf.math.not_equal(tf.math.argmax(predictions, axis=1), labels)


@tf.function
def eval_step(images, labels, model):
    predictions = model(images, training=False)
    t_loss = loss_object(labels, predictions)

    val_loss(t_loss)
    val_accuracy(labels, predictions)
    return predictions, labels


@tf.function
def t_step(images, labels, model):
    predictions = model(images, training=False)
    test_accuracy(labels, predictions)


if not os.path.exists(checkpoint_directory):
    os.makedirs(checkpoint_directory)
checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
manager = tf.train.CheckpointManager(checkpoint, directory=checkpoint_directory, max_to_keep=1)

status = checkpoint.restore(manager.latest_checkpoint)
print('Checkpoint status: ', status)
if manager.latest_checkpoint:
    print('Restored from checkpoints!')


X, y = load_imgs_with_lbl(data_path, classes, shape=(input_shape[0], input_shape[1]))
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=val_test_split, random_state=seed)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=test_size, random_state=seed)

train_dataset_org = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(len(X_train)).batch(batch_size)
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size)

training_start_time = time.time()
best_acc_epoch_batch = [0, 0, 0]
sampler = RandomOverSampler(sampling_strategy='not majority')

log_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
batch_counter = 0

datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=360,
    brightness_range=[0.95, 1.05],
    width_shift_range=0.01,
    height_shift_range=0.01,
    zoom_range=[0.95, 1.05],
    horizontal_flip=True,
    vertical_flip=True,
    preprocessing_function=transform
)


for epoch in range(epochs):
    train_dataset_org.shuffle(len(X_train)).batch(batch_size)
    runs_per_epoch = len(list(train_dataset_org))

    for c, (train_images, train_labels) in enumerate(train_dataset_org):
        if np.unique(train_labels).size == 1:
            print(colored('Skipped batch!', 'red'))
            continue
        num_examples = len(train_images.numpy())
        train_images = train_images.numpy().reshape(num_examples, -1)
        training_generator, steps_per_epoch = balanced_batch_generator(X=train_images, y=train_labels.numpy(),
                                                                       sample_weight=None,
                                                                       sampler=sampler,
                                                                       batch_size=batch_size)
        train_images_rs, train_labels_rs = next(training_generator)
        train_images_rs = train_images_rs.reshape(train_images_rs.shape[0], input_shape[0], input_shape[1], 3)

        for train_images, train_labels in datagen.flow(x=train_images_rs, y=train_labels_rs,
                                                       batch_size=batch_size):
            print(train_labels)

            grad, fails = train_step(train_images, train_labels, model)
            batch_counter += 1

            print('Epoch: {}/{}, Batch: {}/{}'.format(epoch + 1, epochs, c + 1, runs_per_epoch))
            template = 'Training Loss: {}, Training Accuracy: {}'
            print(template.format(train_loss.result(), train_accuracy.result() * 100))
            norm_grad = tf.linalg.global_norm(grad)
            print('Norm of Gradient: {}'.format(norm_grad))

            # Reset the metrics
            train_loss.reset_states()
            train_accuracy.reset_states()
            break

        for val_images, val_labels in val_dataset:

            val_images = tf.map_fn(transform, val_images, dtype=tf.float32)
            preds, lbls = eval_step(val_images, val_labels, model)

            fail_list = []
            for z, pred in enumerate(preds):
                if tf.math.argmax(tf.reshape(pred, shape=(num_classes, 1)),
                                  axis=0).numpy()[0] != lbls[z].numpy():
                    fail_list.append((tf.math.argmax(tf.reshape(pred, shape=(num_classes, 1)),
                                                     axis=0).numpy()[0],
                                      lbls[z].numpy(), val_images[z].numpy()))

        val_acc_step = val_accuracy.result().numpy()
        if (val_acc_step >= best_acc_epoch_batch[0]):
            print('New checkpoints stored!')
            manager.save()
            best_acc_epoch_batch[0] = val_acc_step
            best_acc_epoch_batch[1] = epoch + 1
            best_acc_epoch_batch[2] = c + 1

        template = 'Val Loss: {}, Val Accuracy: {}'
        print(template.format(val_loss.result(),
                              val_accuracy.result() * 100))

        print('Best Val Accuracy: {}, Best Epoch: {}, Best Batch: {}'.format(best_acc_epoch_batch[0] * 100,
                                                                             best_acc_epoch_batch[1],
                                                                             best_acc_epoch_batch[2]))
        # Reset the metrics
        val_loss.reset_states()
        val_accuracy.reset_states()

        print('Training time: {}'.format(timer(training_start_time, time.time()) + '\n'))


status = checkpoint.restore(manager.latest_checkpoint)
status.assert_existing_objects_matched()
print('Best model restored from checkpoints for testing!')

for test_images, test_labels in test_dataset:
    test_images = tf.map_fn(transform, test_images, dtype=tf.float32)
    t_step(test_images, test_labels, model)
print('Test acc: ', test_accuracy.result())

# convert Keras model to ConcreteFunction
full_model = tf.function(lambda x: model(x, training=False))

# prepare model
full_model = full_model.get_concrete_function(
    tf.TensorSpec((None, input_shape[0], input_shape[1], 3), model.inputs[0].dtype))

# get frozen ConcreteFunction
frozen_func = convert_variables_to_constants_v2(full_model)
frozen_func.graph.as_graph_def()

layers = [op.name for op in frozen_func.graph.get_operations()]
print("-" * 50)
print("Frozen model layers: ")
for layer in layers:
    print(layer)

print("-" * 50)
print("Frozen model inputs: ")
print(frozen_func.inputs)
print("Frozen model outputs: ")
print(frozen_func.outputs)

log_directory = log_dir + '/' + log_time
if not os.path.exists(log_directory):
    os.makedirs(log_directory)

# save frozen graph from frozen ConcreteFunction
tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                  logdir=log_directory,
                  name="frozen_graph.pb",
                  as_text=False)