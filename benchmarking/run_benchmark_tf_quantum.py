import time
import numpy as np
import sympy
import cirq
import tensorflow_quantum as tfq
import tensorflow as tf
from typing import Callable, List
from tqdm.auto import tqdm
import pandas as pd


def run_benchmark(qubits_array: List, test_time: Callable, file_name):
    '''runs given time measuring function test_time
    with different qubits number from qubits_array list
    and saves results to csv file.'''
    data = []
    for n_qubits in tqdm(qubits_array):
        duration = test_time(n_qubits)

        data.append([n_qubits, duration])

        df = pd.DataFrame(data, columns=['n_qubits', "time[ms]"])
        df.to_csv(file_name, index=False)


# %%


# ### Define function to create scalable hard to sim circuit.

def create_circuit(n_qubits):
    circuit = cirq.Circuit()
    qubits = cirq.LineQubit.range(n_qubits)

    # wall of single qubit operations
    for i in range(n_qubits):
        symbol = sympy.Symbol("Y" + '-' + str(i))
        circuit.append(cirq.Y(qubits[i])**symbol)
    # ladder of entangling operations
    for i in range(n_qubits):
        symbol = sympy.Symbol("XX"+"-"+str(i))
        circuit.append(cirq.XX(qubits[i], qubits[(i+1) % n_qubits])**symbol)
    measurements = []
    for i in range(1):
        measurements.append(cirq.Z(qubits[i]))

    return circuit, measurements


quantum_data = tfq.convert_to_tensor([
    cirq.Circuit()
])

steps = 200
file_name = "tensorflow.csv"


def test_time(n_qubits):
    circuit, measurements = create_circuit(n_qubits)

    model = tf.keras.Sequential([
        # The input is the data-circuit, encoded as a tf.string
        tf.keras.layers.Input(shape=(), dtype=tf.string),
        # The PQC layer returns the expected value of the readout gate, range [-1,1].
        tfq.layers.PQC(circuit, measurements),
    ])

    model.compile(
        loss='mse',
        optimizer=tf.keras.optimizers.Adam())
    beg_steps = 5

    model.fit(quantum_data, y=np.ones((1, 1)),
              epochs=beg_steps,
              batch_size=1,
              verbose=0,)

    beg = time.time()
    model.fit(quantum_data, y=np.ones((1, 1)),
              epochs=steps,
              batch_size=1,
              verbose=0,)
    end = time.time()
    return (end-beg)*1000


run_benchmark(range(2, 22), test_time, file_name)

# %% [markdown]
# ## Testing with qsim from cirq


# %%
steps = 200
file_name = "tensorflow_qsim_default.csv"


def test_time(n_qubits):
    circuit, measurements = create_circuit(n_qubits)
    import qsimcirq

    options = {}

    qsim_simulator = qsimcirq.QSimSimulator(options)

    model = tf.keras.Sequential([
        # The input is the data-circuit, encoded as a tf.string
        tf.keras.layers.Input(shape=(), dtype=tf.string),
        # The PQC layer returns the expected value of the readout gate, range [-1,1].
        tfq.layers.PQC(circuit, measurements, backend=qsim_simulator),
    ])

    model.compile(
        loss='mse',
        optimizer=tf.keras.optimizers.Adam())
    beg_steps = 5

    model.fit(quantum_data, y=np.ones((1, 1)),
              epochs=beg_steps,
              batch_size=1,
              verbose=0,)

    beg = time.time()
    model.fit(quantum_data, y=np.ones((1, 1)),
              epochs=steps,
              batch_size=1,
              verbose=0,)
    end = time.time()
    return (end-beg)*1000


run_benchmark(range(2, 18), test_time, file_name)
