import ramanspy as rp

# Load training and testing datasets
X_train, y_train = rp.datasets.bacteria("train", path_to_data="path/to/data")
X_test, y_test = rp.datasets.bacteria("test", path_to_data="path/to/data"))

# Load the names of the species and antibiotics corresponding to the 30 classes
y_labels, antibiotics_labels = rp.datasets.bacteria("labels")