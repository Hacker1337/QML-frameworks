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
        
        
import pennylane as qml
from pennylane import numpy as np
import torch
import time


def create_circuit(n_qubits):
    '''Example function for creating test circuit'''

    dev = qml.device("lightning.gpu", wires=n_qubits)
    
    @qml.qnode(dev, interface="torch", diff_method="adjoint")
    def circuit(params):
        # wall of single qubit operations
        for i in range(n_qubits):
            qml.RY(params[0, i]*np.pi, wires=i)
            
        # ladder of entangling operations
        for i in range(n_qubits):
            qml.IsingXX(params[1, i]*np.pi, wires=[i, (i+1)%n_qubits])

        return qml.expval(qml.PauliZ(0))
    
    return circuit


steps = 200
file_name="pennylane_lightning_gpu.csv"

def test_time(n_qubits):
    circuit = create_circuit(n_qubits)
    params = torch.rand((2, n_qubits), requires_grad=True)
    def cost(params):
        return (1 - circuit(params))**2
    
    opt = torch.optim.Adam([params], lr = 0.1)
    beg_steps = 5
    for i in range(steps+beg_steps):
        if (i == beg_steps):
            beg = time.time()
        opt.zero_grad()
        loss = cost(params)
        loss.backward()
        opt.step()
    end = time.time()
    return (end-beg)*1000


run_benchmark(range(2, 22), test_time, file_name)


# # so far it was tested with such configurations

# "pennylane_default.csv"
# dev = qml.device("default.qubit", wires=n_qubits)
# @qml.qnode(dev, interface="torch")

# "pennylane_qsim.csv"
# dev = qml.device("cirq.qsim", wires=n_qubits)    
# @qml.qnode(dev, interface="torch")

# "pennylane.csv"
# dev = qml.device("lightning.qubit", wires=n_qubits)
# @qml.qnode(dev, interface="torch")
    
# "pennylane_default_adjoint.csv"
# dev = qml.device("default.qubit", wires=n_qubits)
# @qml.qnode(dev, interface="torch", diff_method="adjoint")
    
# "pennylane_qsim_adjoint.csv"
# dev = qml.device("cirq.qsim", wires=n_qubits)
# @qml.qnode(dev, interface="torch", diff_method="adjoint")
    
# "pennylane_qsim_default.csv"
# dev = qml.device("cirq.qsim", wires=n_qubits)
# @qml.qnode(dev, interface="torch")
    
# file_name="pennylane_qiskit.csv"
# dev = qml.device("qiskit.basicaer", wires=n_qubits)
# @qml.qnode(dev, interface="torch", diff_method="adjoint")

# file_name="pennylane_qiskit_adjoint.csv"
# dev = qml.device("qiskit.basicaer", wires=n_qubits)
# @qml.qnode(dev, interface="torch", diff_method="adjoint")

# file_name="pennylane_qulacs.csv"
# dev = qml.device("qulacs.simulator", wires=n_qubits)
# @qml.qnode(dev, interface="torch")

# file_name="pennylane_qulacs_adjoint.csv"
# dev = qml.device("qulacs.simulator", wires=n_qubits)
# @qml.qnode(dev, interface="torch", diff_method="adjoint")

# file_name="pennylane_lightning_gpu.csv"
# dev = qml.device("lightning.gpu", wires=n_qubits)
# @qml.qnode(dev, interface="torch")



