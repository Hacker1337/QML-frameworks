from tqdm.notebook import tqdm
import pandas as pd
def run_benchmark(qubits_array, test_time, file_name):
    data = []
    for n_qubits in tqdm(qubits_array):
        duration = test_time(n_qubits)
        
        data.append([n_qubits, duration])
        print([n_qubits, duration])

        
        df = pd.DataFrame(data, columns=['n_qubits', "time[ms]"])
        df.to_csv(file_name, index=False)