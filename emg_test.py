from scipy.io import loadmat
import numpy as np
from scipy.stats import mode
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

window_size = 20
step_size = 10

def process_subject(filename, downsample_rest=False):
    data = loadmat(filename)
    variable_name = data['emg']
    graph = data['restimulus'][:, 0]

    rest_counter = 0
    windows = []
    labels = []
    start = 0

    while (start + window_size) <= len(graph):
        label_slice = graph[start : start + window_size]
        slice_label = mode(label_slice, keepdims=False).mode

        if downsample_rest and slice_label == 0:
            rest_counter += 1
            if rest_counter % 10 == 0:
                slices = variable_name[start : start + window_size]
                windows.append(slices)
                labels.append(slice_label)
        else:
            slices = variable_name[start : start + window_size]
            windows.append(slices)
            labels.append(slice_label)

        start += step_size

    features = []
    for index in range(len(windows)):
        window_features = []
        for channel in range(10):
            ch_data = windows[index][:, channel]
            mean_value = np.mean(ch_data)
            standard_deviation = np.std(ch_data)
            max_value = np.max(ch_data)
            window_features.extend([mean_value, standard_deviation, max_value])
        features.append(window_features)

    return np.array(features), np.array(labels)

features_train, labels_train = process_subject('S1_A1_E1.mat', downsample_rest=True)
features_test, labels_test = process_subject('S1_A1_E2.mat', downsample_rest=False)

scaler = StandardScaler()
features_train_scaled = scaler.fit_transform(features_train)
features_test_scaled = scaler.transform(features_test)

model = SVC(kernel='rbf', C=10.0, gamma='scale')
model.fit(features_train_scaled, labels_train)

predictions = model.predict(features_test_scaled)
accuracy = accuracy_score(labels_test, predictions)

print(f"Same-Subject Accuracy (S1 E1 -> S1 E2): {accuracy * 100:.2f}%")