import sys
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

N_qubits = int(sys.argv[1])
data_set_train = pd.read_csv(f"Usable_data_sets/trainning_data_set.csv")
data_set_test = pd.read_csv(f"Usable_data_sets/test_data_set.csv")
data_set_train.rename(columns={list(data_set_train)[0]:"OldIndex"}, inplace=True)
data_set_test.rename(columns={list(data_set_test)[0]:"OldIndex"}, inplace=True)
FEATURES_NAMES = list(data_set_train)[2:]

COUNTER = 0
NUM_FEATURES = 12
NUM_SAMPLES_train = len(data_set_train)
NUM_SAMPLES_test = len(data_set_test)

def get_values(index):

	values = np.empty(NUM_FEATURES)

	for i, feature in zip(range(NUM_FEATURES), FEATURES_NAMES):
		values[i] = data_set.at[index, feature]

	if data_set.at[index, "CLASSE"] == "R":
		classe = 1 
	elif data_set.at[index, "CLASSE"] == "W":
		classe = -1
		
	return values, classe

def normalize_amp(array):
    r = array.copy()
    for i, datapoint in enumerate(array):
        soma_sq = sum([x**2 for x in datapoint])
        r[i] /= np.sqrt(soma_sq)
    return r

def padding(values, num_qubits):
    if NUM_FEATURES > 2**num_qubits:
        print("PAD não necessário")
        return values
    else:
        dataset_pad = []
        for datapoint in values:
            datapoint_pad = np.zeros(2**num_qubits)
            for i in range(NUM_FEATURES):
                datapoint_pad[i] = datapoint[i]
            dataset_pad += [datapoint_pad]
        return np.array(dataset_pad)

def main():
    train_data_set_features = data_set_train.loc[:, FEATURES_NAMES]
    train_classes = data_set_train.loc[:, ["CLASSE"]].values
    test_data_set_features = data_set_test.loc[:, FEATURES_NAMES]
    test_classes = data_set_test.loc[:, ["CLASSE"]].values

    scaler = StandardScaler()
    x_train_data = scaler.fit_transform(train_data_set_features)# Apply transform to both the training set and the test set.
    x_test_data = scaler.transform(test_data_set_features)

    if N_qubits**2 < 12: 
        pca = PCA(n_components=2**N_qubits)
        x_train_data = pca.fit_transform(x_train_data)
        x_test_data = pca.fit_transform(x_test_data)

    train_classes_encode = [-1 if classe == "W" else 1 for classe in train_classes]
    test_classes_encode = [-1 if classe == "W" else 1 for classe in test_classes]

    train_np_data_set_features_std = normalize_amp(x_train_data)
    train_np_data_set_features_std = padding(train_np_data_set_features_std, N_qubits)
   
    test_np_data_set_features_std = normalize_amp(x_test_data)
    test_np_data_set_features_std = padding(test_np_data_set_features_std, N_qubits)

    np.savetxt(f"Encode_data/amp_enc_data_set_trainning_values.csv", train_np_data_set_features_std, delimiter=";")
    np.savetxt(f"Encode_data/amp_enc_data_set_trainning_classes.csv", train_classes_encode, delimiter=";")
    np.savetxt(f"Encode_data/amp_enc_data_set_test_values.csv", test_np_data_set_features_std, delimiter=";")
    np.savetxt(f"Encode_data/amp_enc_data_set_test_classes.csv", test_classes_encode, delimiter=";")

    print(f"Amplitude Enconding guardado em \"Encode_data/amp_enc_data_set_trainning_values.csv\"")
    print(f"Amplitude Enconding guardado em \"Encode_data/amp_enc_data_set_trainning_classes.csv\"")
    print(f"Amplitude Enconding guardado em \"Encode_data/amp_enc_data_set_test_values.csv\"")
    print(f"Amplitude Enconding guardado em \"Encode_data/amp_enc_data_set_test_classes.csv\"")

if __name__ == '__main__':
	main()
