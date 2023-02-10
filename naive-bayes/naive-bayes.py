import csv
import numpy as np

def naive_bayes_classify(header, train_x, train_y, test_x, test_y=None):
    print('='*9, 'Conditions', '='*9)
    unique_labels = np.unique(train_y)
    p_labels = []
    for label in unique_labels:
        count = np.count_nonzero(train_y == label)
        p_labels.append(count / len(train_y))

    p_features_given_label = []
    for label in unique_labels:
        relevant_rows = train_x[train_y == label]
        for feature in range(len(test_x)):
            count = np.count_nonzero(relevant_rows[:, feature] == test_x[feature])
            p = count / len(relevant_rows)
            p_features_given_label.append(p)
            print("P({}|{}) = {}".format(header[feature], 'purchase' if label == '1' else 'not purchase', p))

    print()
    print('='*9, 'Priors', '='*9)
    p_label_given_features = []
    for i in range(len(unique_labels)):
        p = p_labels[i]
        for j in range(len(test_x)):
            p *= p_features_given_label[i * len(test_x) + j]
            print("P({}|{}) = {}".format('purchase' if unique_labels[i] == '1' else 'not purchase', header[j], p))
        p_label_given_features.append(p)

    print()
    print('='*9, 'Results', '='*9)
    print("P(purchase|{}) = {}".format(test_x, p_label_given_features[0]))
    print("P(not purchase|{}) = {}".format(test_x, p_label_given_features[1]))
    print("Predicted Label:", 'purchase' if unique_labels[np.argmax(p_label_given_features)] == '1' else 'not purchase')
    if test_y is not None:
        print("Actual Label:", 'purchase' if test_y == '1' else 'not purchase')


def split_data(data, ratio):
    np.random.shuffle(data)
    train_size = int(len(data) * ratio)
    return data[:train_size], data[train_size:]


def main():
    f = open('data.csv')
    data = np.array(list(csv.reader(f, delimiter=',')))

    # Cut the first row (headers) and first column (id)
    header = data[0, 1:]
    data = data[1:, 1:]

    # Split data into training and testing sets
    train, test = split_data(data, 0.8)

    # Split data into features and labels
    train_x = train[:, :-1]
    train_y = train[:, -1]
    test_x = test[:, :-1]
    test_y = test[:, -1]

    naive_bayes_classify(header, train_x, train_y, test_x[0], test_y[0])


if __name__ == "__main__":
    main()

