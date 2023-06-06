
import importlib, pkg_resources
importlib.reload(pkg_resources)


# %%
import tensorflow as tf
import tensorflow_quantum as tfq

import cirq
import sympy
import numpy as np
import seaborn as sns
import collections

# visualization tools
import matplotlib.pyplot as plt
from cirq.contrib.svg import SVGCircuit

# %% [markdown]
# ## 1. Load the data
# 
# In this tutorial you will build a binary classifier to distinguish between the digits 3 and 6, following <a href="https://arxiv.org/pdf/1802.06002.pdf" class="external">Farhi et al.</a> This section covers the data handling that:
# 
# - Loads the raw data from Keras.
# - Filters the dataset to only 3s and 6s.
# - Downscales the images so they fit can fit in a quantum computer.
# - Removes any contradictory examples.
# - Converts the binary images to Cirq circuits.
# - Converts the Cirq circuits to TensorFlow Quantum circuits. 

# %% [markdown]
# ### 1.1 Load the raw data

# %% [markdown]
# Load the MNIST dataset distributed with Keras. 

# %%
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Rescale the images from [0,255] to the [0.0,1.0] range.
x_train, x_test = x_train[..., np.newaxis]/255.0, x_test[..., np.newaxis]/255.0

print("Number of original training examples:", len(x_train))
print("Number of original test examples:", len(x_test))

# %% [markdown]
# Filter the dataset to keep just the 3s and 6s,  remove the other classes. At the same time convert the label, `y`, to boolean: `True` for `3` and `False` for 6. 

# %%
def filter_36(x, y):
    keep = (y == 3) | (y == 6)
    x, y = x[keep], y[keep]
    y = y == 3
    return x,y

# %%
x_train, y_train = filter_36(x_train, y_train)
x_test, y_test = filter_36(x_test, y_test)

print("Number of filtered training examples:", len(x_train))
print("Number of filtered test examples:", len(x_test))

# %% [markdown]
# Show the first example:

# %%
# print(y_train[0])

# plt.imshow(x_train[0, :, :, 0])
# plt.colorbar()

# %% [markdown]
# ### 1.2 Downscale the images

# %% [markdown]
# An image size of 28x28 is much too large for current quantum computers. Resize the image down to 4x4:

# %%
x_train_small = tf.image.resize(x_train, (4,4)).numpy()
x_test_small = tf.image.resize(x_test, (4,4)).numpy()

# %% [markdown]
# Again, display the first training example—after resize: 

# %%
# print(y_train[0])

# plt.imshow(x_train_small[0,:,:,0], vmin=0, vmax=1)
# plt.colorbar()

# %% [markdown]
# ### 1.3 Remove contradictory examples

# %% [markdown]
# From section *3.3 Learning to Distinguish Digits* of <a href="https://arxiv.org/pdf/1802.06002.pdf" class="external">Farhi et al.</a>, filter the dataset to remove images that are labeled as belonging to both classes.
# 
# This is not a standard machine-learning procedure, but is included in the interest of following the paper.

# %%
def remove_contradicting(xs, ys):
    mapping = collections.defaultdict(set)
    orig_x = {}
    # Determine the set of labels for each unique image:
    for x,y in zip(xs,ys):
       orig_x[tuple(x.flatten())] = x
       mapping[tuple(x.flatten())].add(y)
    
    new_x = []
    new_y = []
    for flatten_x in mapping:
      x = orig_x[flatten_x]
      labels = mapping[flatten_x]
      if len(labels) == 1:
          new_x.append(x)
          new_y.append(next(iter(labels)))
      else:
          # Throw out images that match more than one label.
          pass
    
    num_uniq_3 = sum(1 for value in mapping.values() if len(value) == 1 and True in value)
    num_uniq_6 = sum(1 for value in mapping.values() if len(value) == 1 and False in value)
    num_uniq_both = sum(1 for value in mapping.values() if len(value) == 2)

    return np.array(new_x), np.array(new_y)

# %% [markdown]
# The resulting counts do not closely match the reported values, but the exact procedure is not specified.
# 
# It is also worth noting here that applying filtering contradictory examples at this point does not totally prevent the model from receiving contradictory training examples: the next step binarizes the data which will cause more collisions. 

# %%
x_train_nocon, y_train_nocon = remove_contradicting(x_train_small, y_train)

# %% [markdown]
# ### 1.4 Encode the data as quantum circuits
# 
# To process images using a quantum computer, <a href="https://arxiv.org/pdf/1802.06002.pdf" class="external">Farhi et al.</a> proposed representing each pixel with a qubit, with the state depending on the value of the pixel. The first step is to convert to a binary encoding.

# %%
THRESHOLD = 0.1

x_train_bin = np.array(x_train_nocon > THRESHOLD, dtype=np.float32)
x_test_bin = np.array(x_test_small > THRESHOLD, dtype=np.float32)

# %% [markdown]
# %% [markdown]

# Second, use a custiom `hinge_accuracy` metric that correctly handles `[-1, 1]` as the `y_true` labels argument. 
# `tf.losses.BinaryAccuracy(threshold=0.0)` expects `y_true` to be a boolean, and so can't be used with hinge loss).

# %%
def hinge_accuracy(y_true, y_pred):
    y_true = tf.squeeze(y_true) > 0.0
    y_pred = tf.squeeze(y_pred) > 0.0
    result = tf.cast(y_true == y_pred, tf.float32)

    return tf.reduce_mean(result)

# %%
def convert_to_circuit(image):
    """Encode truncated classical image into quantum datapoint."""
    values = np.ndarray.flatten(image)
    qubits = cirq.GridQubit.rect(4, 4)
    circuit = cirq.Circuit()
    for i, value in enumerate(values):
        if value:
            circuit.append(cirq.X(qubits[i]))
    return circuit


x_train_circ = [convert_to_circuit(x) for x in x_train_bin]
x_test_circ = [convert_to_circuit(x) for x in x_test_bin]


# %% [markdown]
# Convert these `Cirq` circuits to tensors for `tfq`:

# %%
x_train_tfcirc = tfq.convert_to_tensor(x_train_circ)
x_test_tfcirc = tfq.convert_to_tensor(x_test_circ)

# %% [markdown]
# ## 2. Quantum neural network
# 
# There is little guidance for a quantum circuit structure that classifies images. Since the classification is based on the expectation of the readout qubit, <a href="https://arxiv.org/pdf/1802.06002.pdf" class="external">Farhi et al.</a> propose using two qubit gates, with the readout qubit always acted upon. This is similar in some ways to running small a <a href="https://arxiv.org/abs/1511.06464" class="external">Unitary RNN</a> across the pixels.

# %% [markdown]
# ### 2.1 Build the model circuit
# 
# This following example shows this layered approach. Each layer uses *n* instances of the same gate, with each of the data qubits acting on the readout qubit.
# 
# Start with a simple class that will add a layer of these gates to a circuit:

# %%
class CircuitLayerBuilder():
    def __init__(self, data_qubits, readout):
        self.data_qubits = data_qubits
        self.readout = readout
    
    def add_layer(self, circuit, gate, prefix):
        for i, qubit in enumerate(self.data_qubits):
            symbol = sympy.Symbol(prefix + '-' + str(i))
            circuit.append(gate(qubit, self.readout)**symbol)


# %% [markdown]
# Now build a two-layered model, matching the data-circuit size, and include the preparation and readout operations.
def create_quantum_model():
    """Create a QNN model circuit and readout operation to go along with it."""
    data_qubits = cirq.GridQubit.rect(4, 4)  # a 4x4 grid.
    readout = cirq.GridQubit(-1, -1)         # a single qubit at [-1,-1]
    circuit = cirq.Circuit()
    
    # Prepare the readout qubit.
    circuit.append(cirq.X(readout))
    circuit.append(cirq.H(readout))
    
    builder = CircuitLayerBuilder(
        data_qubits = data_qubits,
        readout=readout)

    # Then add layers (experiment by adding more).
    builder.add_layer(circuit, cirq.XX, "xx1")
    builder.add_layer(circuit, cirq.ZZ, "zz1")

    # Finally, prepare the readout qubit.
    circuit.append(cirq.H(readout))

    return circuit, cirq.Z(readout)

model_circuit, model_readout = create_quantum_model()
# %%
### 2.2 Wrap the model-circuit in a tfq-keras model

# Build the Keras model with the quantum components. This model is fed the "quantum data", from `x_train_circ`, that encodes the classical data. It uses a *Parametrized Quantum Circuit* layer, `tfq.layers.PQC`, to train the model circuit, on the quantum data.

# To classify these images, <a href="https://arxiv.org/pdf/1802.06002.pdf" class="external">Farhi et al.</a> proposed taking the expectation of a readout qubit in a parameterized circuit. The expectation returns a value between 1 and -1.
# Build the Keras model.
import qsimcirq
model = tf.keras.Sequential([
    # The input is the data-circuit, encoded as a tf.string
    tf.keras.layers.Input(shape=(), dtype=tf.string),
    # The PQC layer returns the expected value of the readout gate, range [-1,1].
    tfq.layers.PQC(model_circuit, model_readout,
                   backend=qsimcirq.QSimSimulator(),
                   ),
])
model.compile(
    loss=tf.keras.losses.Hinge(),
    optimizer=tf.keras.optimizers.Adam(),
    metrics=[hinge_accuracy])
# %% [markdown]
# Next, describe the training procedure to the model, using the `compile` method.
# 
# Since the the expected readout is in the range `[-1,1]`, optimizing the hinge loss is a somewhat natural fit. 
# 
# Note: Another valid approach would be to shift the output range to `[0,1]`, and treat it as the probability the model assigns to class `3`. This could be used with a standard a `tf.losses.BinaryCrossentropy` loss.
# 
# To use the hinge loss here you need to make two small adjustments. First convert the labels, `y_train_nocon`, from boolean to `[-1,1]`, as expected by the hinge loss.

# %%
y_train_hinge = 2.0*y_train_nocon-1.0
y_test_hinge = 2.0*y_test-1.0


# %% [markdown]
# ### Train the quantum model
# 
# Now train the model—this takes about 45 min. If you don't want to wait that long, use a small subset of the data (set `NUM_EXAMPLES=500`, below). This doesn't really affect the model's progress during training (it only has 32 parameters, and doesn't need much data to constrain these). Using fewer examples just ends training earlier (5min), but runs long enough to show that it is making progress in the validation logs.

# %%
EPOCHS = 16
BATCH_SIZE = 32

# NUM_EXAMPLES = len(x_train_tfcirc)
NUM_EXAMPLES = 1000

# %%
x_train_tfcirc_sub = x_train_tfcirc[:NUM_EXAMPLES]
y_train_hinge_sub = y_train_hinge[:NUM_EXAMPLES]

# %% [markdown]
# Training this model to convergence should achieve >85% accuracy on the test set.

import datetime
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "_1000_samples_32_param_X,Y_qsim_Simulator"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

qnn_history = model.fit(
      x_train_tfcirc_sub, y_train_hinge_sub,
      batch_size=32,
      epochs=EPOCHS,
      verbose=1,
      validation_data=(x_test_tfcirc, y_test_hinge),
      callbacks=[tensorboard_callback])
# %% [markdown]
# Note: The training accuracy reports the average over the epoch. The validation accuracy is evaluated at the end of each epoch.

# %% [markdown]
# ## 3. Classical neural network
# 
# While the quantum neural network works for this simplified MNIST problem, a basic classical neural network can easily outperform a QNN on this task. After a single epoch, a classical neural network can achieve >98% accuracy on the holdout set.
# 
# In the following example, a classical neural network is used for for the 3-6 classification problem using the entire 28x28 image instead of subsampling the image. This easily converges to nearly 100% accuracy of the test set.

# # %%
# def create_classical_model():
#     # A simple model based off LeNet from https://keras.io/examples/mnist_cnn/
#     model = tf.keras.Sequential()
#     model.add(tf.keras.layers.Conv2D(32, [3, 3], activation='relu', input_shape=(28,28,1)))
#     model.add(tf.keras.layers.Conv2D(64, [3, 3], activation='relu'))
#     model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
#     model.add(tf.keras.layers.Dropout(0.25))
#     model.add(tf.keras.layers.Flatten())
#     model.add(tf.keras.layers.Dense(128, activation='relu'))
#     model.add(tf.keras.layers.Dropout(0.5))
#     model.add(tf.keras.layers.Dense(1))
#     return model


# model = create_classical_model()
# model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
#               optimizer=tf.keras.optimizers.Adam(),
#               metrics=['accuracy'])

# model.summary()

# # %%
# model.fit(x_train,
#           y_train,
#           batch_size=128,
#           epochs=1,
#           verbose=1,
#           validation_data=(x_test, y_test))

# cnn_results = model.evaluate(x_test, y_test)

# # %% [markdown]
# # The above model has nearly 1.2M parameters. For a more fair comparison, try a 37-parameter model, on the subsampled images:

# # %%
# def create_fair_classical_model():
#     # A simple model based off LeNet from https://keras.io/examples/mnist_cnn/
#     model = tf.keras.Sequential()
#     model.add(tf.keras.layers.Flatten(input_shape=(4,4,1)))
#     model.add(tf.keras.layers.Dense(2, activation='relu'))
#     model.add(tf.keras.layers.Dense(1))
#     return model


# model = create_fair_classical_model()
# model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
#               optimizer=tf.keras.optimizers.Adam(),
#               metrics=['accuracy'])

# model.summary()

# # %%



# import datetime
# log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "_1000_samples_37_param_classical_model"
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# model.fit(x_train_bin,
#           y_train_nocon,
#           batch_size=128,
#           epochs=20,
#           verbose=2,
#           validation_data=(x_test_bin, y_test),
#                 callbacks=[tensorboard_callback])


# fair_nn_results = model.evaluate(x_test_bin, y_test)

# %% [markdown]
# ## 4. Comparison
# 
# Higher resolution input and a more powerful model make this problem easy for the CNN. While a classical model of similar power (~32 parameters) trains to a similar accuracy in a fraction of the time. One way or the other, the classical neural network easily outperforms the quantum neural network. For classical data, it is difficult to beat a classical neural network.

# # %%
# qnn_accuracy = qnn_results[1]
# cnn_accuracy = cnn_results[1]
# fair_nn_accuracy = fair_nn_results[1]

# sns.barplot(x=["Quantum", "Classical, full", "Classical, fair"],
#             y=[qnn_accuracy, cnn_accuracy, fair_nn_accuracy])

# %% [markdown]
# ## 5. Квантовая нейросеть с фазовым кодированием. 

# %%
# def convert_to_circuit_2(image):
#     """Encode truncated classical image into quantum datapoint."""
#     values = np.ndarray.flatten(image)
#     qubits = cirq.GridQubit.rect(4, 4)
#     circuit = cirq.Circuit()
#     for i, value in enumerate(values):
#         if value:
#             circuit.append(cirq.rx(value*np.pi)(qubits[i]))
#     return circuit


# x_train_circ_2 = [convert_to_circuit_2(x) for x in x_train_nocon]
# x_test_circ_2 = [convert_to_circuit_2(x) for x in x_test_small]

# # %%
# SVGCircuit(x_train_circ_2[0])

# # %% [markdown]
# # Compare this circuit to the indices where the image value exceeds the threshold:

# # %%
# bin_img = x_train_nocon[0,:,:,0]
# indices = np.array(bin_img)
# indices

# # %% [markdown]
# # Convert these `Cirq` circuits to tensors for `tfq`:

# # %%
# x_train_tfcirc_2 = tfq.convert_to_tensor(x_train_circ_2)
# x_test_tfcirc_2 = tfq.convert_to_tensor(x_test_circ_2)

# # %% [markdown]
# # ### Train the quantum model
# # 
# # Now train the model—this takes about 45 min. If you don't want to wait that long, use a small subset of the data (set `NUM_EXAMPLES=500`, below). This doesn't really affect the model's progress during training (it only has 32 parameters, and doesn't need much data to constrain these). Using fewer examples just ends training earlier (5min), but runs long enough to show that it is making progress in the validation logs.

# # %%
# EPOCHS = 10
# BATCH_SIZE = 32

# # NUM_EXAMPLES = len(x_train_tfcirc)
# NUM_EXAMPLES = 1000

# # %%
# x_train_tfcirc_sub_2 = x_train_tfcirc_2[:NUM_EXAMPLES]
# y_train_hinge_sub = y_train_hinge[:NUM_EXAMPLES]

# # %%
# qnn_history = model.fit(
#       x_train_tfcirc_sub_2, y_train_hinge_sub,
#       batch_size=32,
#       epochs=EPOCHS,
#       verbose=1,
#       validation_data=(x_test_tfcirc_2, y_test_hinge))

# qnn_results = model.evaluate(x_test_tfcirc_2, y_test)

# %% [markdown]
# ### try to repeat paper success

# %%
# class CircuitLayerBuilder():
#     def __init__(self, data_qubits, readout):
#         self.data_qubits = data_qubits
#         self.readout = readout
    
#     def add_layer(self, circuit, gate, prefix):
#         for i, qubit in enumerate(self.data_qubits):
#             symbol = sympy.Symbol(prefix + '-' + str(i))
#             circuit.append(gate(qubit, self.readout)**symbol)

# # %%
# def create_quantum_model():
#     """Create a QNN model circuit and readout operation to go along with it."""
#     data_qubits = cirq.GridQubit.rect(4, 4)  # a 4x4 grid.
#     readout = cirq.GridQubit(-1, -1)         # a single qubit at [-1,-1]
#     circuit = cirq.Circuit()
    
#     # Prepare the readout qubit.
#     circuit.append(cirq.X(readout))
#     circuit.append(cirq.H(readout))
    
#     builder = CircuitLayerBuilder(
#         data_qubits=data_qubits,
#         readout=readout)

#     # Then add layers (experiment by adding more).
#     builder.add_layer(circuit, cirq.ZZ, "zz1")
#     builder.add_layer(circuit, cirq.ZZ, "zz2")
#     builder.add_layer(circuit, cirq.ZZ, "zz3")
#     builder.add_layer(circuit, cirq.XX, "xx1")
#     builder.add_layer(circuit, cirq.XX, "xx2")
#     builder.add_layer(circuit, cirq.XX, "xx3")

#     # Finally, prepare the readout qubit.
#     circuit.append(cirq.H(readout))

#     return circuit, cirq.Z(readout)

# # %%
# model_circuit, model_readout = create_quantum_model()

# # %%
# # Build the Keras model.
# model = tf.keras.Sequential([
#     # The input is the data-circuit, encoded as a tf.string
#     tf.keras.layers.Input(shape=(), dtype=tf.string),
#     # The PQC layer returns the expected value of the readout gate, range [-1,1].
#     tfq.layers.PQC(model_circuit, model_readout),
# ])

# # %%
# model.compile(
#     loss=tf.keras.losses.Hinge(),
#     optimizer=tf.keras.optimizers.Adam(),
#     metrics=[hinge_accuracy])

# # %%
# print(model.summary())

# # %%
# len(x_train_tfcirc)

# # %%
# EPOCHS = 3
# BATCH_SIZE = 32

# NUM_EXAMPLES = len(x_train_tfcirc)
# # NUM_EXAMPLES = 1000

# # %%
# x_train_tfcirc_sub = x_train_tfcirc[:NUM_EXAMPLES]
# y_train_hinge_sub = y_train_hinge[:NUM_EXAMPLES]

# # %%
# qnn_history = model.fit(
#       x_train_tfcirc_sub, y_train_hinge_sub,
#       batch_size=32,
#       epochs=EPOCHS,
#       verbose=1,
#       validation_data=(x_test_tfcirc, y_test_hinge))

# qnn_results = model.evaluate(x_test_tfcirc, y_test)

# # %% [markdown]
# # The same on small subset of dataset

# # %%
# # Build the Keras model.
# model = tf.keras.Sequential([
#     # The input is the data-circuit, encoded as a tf.string
#     tf.keras.layers.Input(shape=(), dtype=tf.string),
#     # The PQC layer returns the expected value of the readout gate, range [-1,1].
#     tfq.layers.PQC(model_circuit, model_readout),
# ])

# # %%
# model.compile(
#     loss=tf.keras.losses.Hinge(),
#     optimizer=tf.keras.optimizers.Adam(),
#     metrics=[hinge_accuracy])

# # %%
# NUM_EXAMPLES = 1000

# x_train_tfcirc_sub = x_train_tfcirc[:NUM_EXAMPLES]
# y_train_hinge_sub = y_train_hinge[:NUM_EXAMPLES]


# # %%

# qnn_history = model.fit(
#       x_train_tfcirc_sub, y_train_hinge_sub,
#       batch_size=32,
#       epochs=EPOCHS,
#       verbose=1,
#       validation_data=(x_test_tfcirc, y_test_hinge))

# qnn_results = model.evaluate(x_test_tfcirc, y_test)

# # %% [markdown]
# # ## Декомпозиция ZX

# # %%
# class CircuitLayerBuilder():
#     def __init__(self, data_qubits, readout):
#         self.data_qubits = data_qubits
#         self.readout = readout
    
#     def add_layer(self, circuit, gate, prefix):
#         for i, qubit in enumerate(self.data_qubits):
#             symbol = sympy.Symbol(prefix + '-' + str(i))
#             circuit.append(gate(qubit, self.readout)**symbol)
#     def add_ZX_gate(self, circuit, prefix):
#         for i, qubit in enumerate(self.data_qubits):
#             symbol = sympy.Symbol(prefix + '-' + str(i))
#             circuit.append(cirq.H(qubit))
#             circuit.append(cirq.X(qubit).controlled_by(self.readout))
#             circuit.append(cirq.rz(symbol).on(qubit))
#             circuit.append(cirq.X(qubit).controlled_by(self.readout))
#             circuit.append(cirq.H(qubit))
        

# # %%
# def create_quantum_model():
#     """Create a QNN model circuit and readout operation to go along with it."""
#     data_qubits = cirq.GridQubit.rect(4, 4)  # a 4x4 grid.
#     readout = cirq.GridQubit(-1, -1)         # a single qubit at [-1,-1]
#     circuit = cirq.Circuit()
    
#     # Prepare the readout qubit.
#     circuit.append(cirq.X(readout))
#     circuit.append(cirq.H(readout))
    
#     builder = CircuitLayerBuilder(
#         data_qubits=data_qubits,
#         readout=readout)

#     # Then add layers (experiment by adding more).
#     builder.add_ZX_gate(circuit, "zx1")
#     builder.add_layer(circuit, cirq.XX, "xx1")
#     builder.add_ZX_gate(circuit, "zx2")
#     builder.add_layer(circuit, cirq.XX, "xx2")
#     builder.add_ZX_gate(circuit, "zx3")
#     builder.add_layer(circuit, cirq.XX, "xx3")

#     # Finally, prepare the readout qubit.
#     circuit.append(cirq.H(readout))

#     return circuit, cirq.Z(readout)

# # %%
# model_circuit, model_readout = create_quantum_model()

# # %%
# # Build the Keras model.
# model = tf.keras.Sequential([
#     # The input is the data-circuit, encoded as a tf.string
#     tf.keras.layers.Input(shape=(), dtype=tf.string),
#     # The PQC layer returns the expected value of the readout gate, range [-1,1].
#     tfq.layers.PQC(model_circuit, model_readout),
# ])

# # %%
# SVGCircuit(model_circuit)

# # %%
# model.compile(
#     loss=tf.keras.losses.Hinge(),
#     optimizer=tf.keras.optimizers.Adam(),
#     metrics=[hinge_accuracy])

# # %%
# qnn_history = model.fit(
#       x_train_tfcirc_sub, y_train_hinge_sub,
#       batch_size=32,
#       epochs=EPOCHS,
#       verbose=1,
#       validation_data=(x_test_tfcirc, y_test_hinge))

# qnn_results = model.evaluate(x_test_tfcirc, y_test)

# # %% [markdown]
# # (сверху результаты для сети предложенной в туториале)

# # %% [markdown]
# # addintg tensrboard

# # %%
# import datetime
# log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# qnn_history = model.fit(
#       x_train_tfcirc_sub, y_train_hinge_sub,
#       batch_size=32,
#       epochs=EPOCHS,
#       verbose=1,
#       validation_data=(x_test_tfcirc, y_test_hinge),
#       callbacks=[tensorboard_callback])


# # %% [markdown]
# # #### Прямо в точности такая же схема, как в статье:

# # %%
# def create_quantum_model():
#     """Create a QNN model circuit and readout operation to go along with it."""
#     data_qubits = cirq.GridQubit.rect(4, 4)  # a 4x4 grid.
#     readout = cirq.GridQubit(-1, -1)         # a single qubit at [-1,-1]
#     circuit = cirq.Circuit()
    

#     builder = CircuitLayerBuilder(
#         data_qubits=data_qubits,
#         readout=readout)

#     # Then add layers (experiment by adding more).
#     builder.add_ZX_gate(circuit, "zx1")
#     builder.add_layer(circuit, cirq.XX, "xx1")
#     builder.add_ZX_gate(circuit, "zx2")
#     builder.add_layer(circuit, cirq.XX, "xx2")
#     builder.add_ZX_gate(circuit, "zx3")
#     builder.add_layer(circuit, cirq.XX, "xx3")

#     return circuit, cirq.Z(readout)

# # %%
# model_circuit, model_readout = create_quantum_model()

# # %%
# # Build the Keras model.
# model = tf.keras.Sequential([
#     # The input is the data-circuit, encoded as a tf.string
#     tf.keras.layers.Input(shape=(), dtype=tf.string),
#     # The PQC layer returns the expected value of the readout gate, range [-1,1].
#     tfq.layers.PQC(model_circuit, model_readout),
# ])

# # %%
# model.compile(
#     loss=tf.keras.losses.Hinge(),
#     optimizer=tf.keras.optimizers.Adam(),
#     metrics=[hinge_accuracy])

# # %%
# import datetime
# log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# qnn_history = model.fit(
#       x_train_tfcirc_sub, y_train_hinge_sub,
#       batch_size=32,
#       epochs=EPOCHS,
#       verbose=1,
#       validation_data=(x_test_tfcirc, y_test_hinge),
#       callbacks=[tensorboard_callback])



