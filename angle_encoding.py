import sys
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

PCA_num_features = int(sys.argv[1])
data_set_train = pd.read_csv(f"Usable_data_sets/trainning_data_set.csv")
data_set_test = pd.read_csv(f"Usable_data_sets/test_data_set.csv")
data_set_train.rename(columns={list(data_set_train)[0]:"OldIndex"}, inplace=True)
data_set_test.rename(columns={list(data_set_test)[0]:"OldIndex"}, inplace=True)
FEATURES_NAMES = list(data_set_train)[2:]

COUNTER = 0
NUM_FEATURES = 12
NUM_SAMPLES_train = len(data_set_train)
NUM_SAMPLES_test = len(data_set_test)

def normalize_ang(array):
    r = array.copy()
    for i, elem in enumerate(array):
        maxi = max([abs(x) for x in elem])
        for j, num in enumerate(elem):
            r[i][j] = num * np.pi / maxi
    return r            

def normalize(array):
    r = array.copy()
    for i, datapoint in enumerate(array):
        soma_sq = sum([x for x in datapoint])
        r[i] /= soma_sq
    return r

def main():

    train_data_set_features = data_set_train.loc[:, FEATURES_NAMES]
    train_classes = data_set_train.loc[:, ["CLASSE"]].values
    test_data_set_features = data_set_test.loc[:, FEATURES_NAMES]
    test_classes = data_set_test.loc[:, ["CLASSE"]].values

    if PCA_num_features != 12:
        scaler = StandardScaler()
        x_train_data = scaler.fit_transform(train_data_set_features)# Apply transform to both the training set and the test set.
        x_test_data = scaler.transform(test_data_set_features)

        pca = PCA(n_components=PCA_num_features)
        train_data_set_features = pca.fit_transform(x_train_data)
        test_data_set_features = pca.fit_transform(x_test_data)
    
    train_classes_encode = [-1 if classe == "W" else 1 for classe in train_classes]
    test_classes_encode = [-1 if classe == "W" else 1 for classe in test_classes]

    train_np_data_set_features_std = normalize_ang(train_data_set_features)
    test_np_data_set_features_std = normalize_ang(test_data_set_features)

    np.savetxt(f"Encode_data/ang_enc_data_set_trainning_values.csv", train_np_data_set_features_std, delimiter=";")
    np.savetxt(f"Encode_data/ang_enc_data_set_trainning_classes.csv", train_classes_encode, delimiter=";")

    np.savetxt(f"Encode_data/ang_enc_data_set_test_values.csv", test_np_data_set_features_std, delimiter=";")
    np.savetxt(f"Encode_data/ang_enc_data_set_test_classes.csv", test_classes_encode, delimiter=";")

    print(f"Amplitude Enconding guardado em \"Encode_data/ang_enc_data_set_trainning_values.csv\"")
    print(f"Amplitude Enconding guardado em \"Encode_data/ang_enc_data_set_trainning_classes.csv\"")
    print(f"Amplitude Enconding guardado em \"Encode_data/ang_enc_data_set_test_values.csv\"")
    print(f"Amplitude Enconding guardado em \"Encode_data/ang_enc_data_set_test_classes.csv\"")         

if __name__ == '__main__':
    main()

