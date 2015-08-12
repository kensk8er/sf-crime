from collections import defaultdict
import csv
import re
import click
from datetime import datetime
import scipy
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

__author__ = 'kensk8er'


def delete_row_csr(matrix, row_index):
    if not isinstance(matrix, scipy.sparse.csr_matrix):
        raise ValueError("works only for CSR format -- use .tocsr() first")
    n = matrix.indptr[row_index + 1] - matrix.indptr[row_index]
    if n > 0:
        matrix.data[matrix.indptr[row_index]:-n] = matrix.data[matrix.indptr[row_index+1]:]
        matrix.data = matrix.data[:-n]
        matrix.indices[matrix.indptr[row_index]:-n] = matrix.indices[matrix.indptr[row_index+1]:]
        matrix.indices = matrix.indices[:-n]
    matrix.indptr[row_index:-1] = matrix.indptr[row_index + 1:]
    matrix.indptr[row_index:] -= n
    matrix.indptr = matrix.indptr[:-1]
    matrix._shape = (matrix._shape[0] - 1, matrix._shape[1])
    return matrix


def align_classes(X_train, X_valid, y_train, y_valid):
    train_classes = {class_ for class_ in y_train}
    valid_classes = {class_ for class_ in y_valid}

    for index in xrange(len(y_train) - 1, -1, -1):
        if y_train[index] not in valid_classes:
            X_train = delete_row_csr(X_train, index)
            del y_train[index]

    for index in xrange(len(y_valid) - 1, -1, -1):
        if y_valid[index] not in train_classes:
            X_valid = delete_row_csr(X_valid, index)
            del y_valid[index]

    return X_train, X_valid, y_train, y_valid


def validate(X, y):
    print('Split the data...')
    (X_train, X_valid, y_train, y_valid) = train_test_split(X, y, test_size=0.1, random_state=0)
    (X_train, X_valid, y_train, y_valid) = align_classes(X_train, X_valid, y_train, y_valid)

    classifier = fit(X_train, y_train)

    print('Predict the labels on validation set...')
    y_pred = classifier.predict_proba(X_valid)
    class_indices = classifier.classes_
    y_pred = add_prior_probability(y_pred, class_indices)
    print("Log Loss: {}".format(log_loss(y_valid, y_pred)))


def fit(X, y):
    print('Fit the classifier...')
    classifier = LogisticRegression(penalty='l2', C=1.0)
    classifier.fit(X, y)
    return classifier


def parse_time(date_str):
    date = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
    hour = date.hour
    day = date.day
    month = date.month
    year = date.year
    return hour, day, month, year


_address_pattern = re.compile(r"[A-Z][A-Z ]+")


def parse_coordinate(X, Y):
    X = round(float(X), 1)
    Y = round(float(Y), 1)
    coordinate_feature = "{0}_{1}".format(X, Y)
    return coordinate_feature


def load(file_path, labeled, debug=False):
    if labeled is True:
        y = []
    features = []

    print('Loading {} ...'.format(file_path))
    with open(file_path) as train_file:
        csv_reader = csv.reader(train_file)
        header = csv_reader.next()

        for index, row in enumerate(csv_reader):
            datum = {key: val for key, val in zip(header, row)}

            if debug is True and index > 10000:
                break

            if labeled is True:
                y.append(datum['Category'])

            (hour, day, month, year) = parse_time(datum['Dates'])
            # coordinate_feature = parse_coordinate(datum['X'], datum['Y'])

            feature = {
                "DayOfWeek_{}".format(datum['DayOfWeek']): 1,
                "PdDistrict_{}".format(datum['PdDistrict']): 1,
                "hour_{}".format(hour): 1,
                "day_{}".format(day): 1,
                "month_{}".format(month): 1,
                "year_{}".format(year): 1,
                # "coordinate_{}".format(coordinate_feature): 1,
                'X': datum['X'],
                'Y': datum['Y'],
            }
            address = _address_pattern.findall(datum['Address'])
            if len(address) > 0:
                feature["Address_{}".format(address[0])] = 1.
            features.append(feature)

    return (y, features) if labeled is True else features


def train(validate_model, predict_result, debug):
    (y, features) = load('data/train.csv', labeled=True, debug=debug)

    print('Vectorizing features...')
    vectorizer = DictVectorizer()
    X = vectorizer.fit_transform(features)

    if validate_model is True:
        print('Validate the model...')
        validate(X, y)

    if predict_result is True:
        classifier = fit(X, y)
    else:
        classifier = None

    return classifier, vectorizer


def predict(classifier, vectorizer):
    features = load('data/test.csv', labeled=False)

    print('Vectorizing features...')
    X = vectorizer.transform(features)

    print('Predicting the y for test set...')
    P = classifier.predict_proba(X)
    class_indices = classifier.classes_
    P = add_prior_probability(P, class_indices)

    return P, class_indices


class_statistics = {'ARSON': 1513., 'ASSAULT': 76876., 'BAD CHECKS': 406., 'BRIBERY': 289., 'BURGLARY': 36755.,
                    'DISORDERLY CONDUCT': 4320., 'DRIVING UNDER THE INFLUENCE': 2268., 'DRUG/NARCOTIC': 53971.,
                    'DRUNKENNESS': 4280, 'EMBEZZLEMENT': 1166., 'EXTORTION': 256., 'FAMILY OFFENSES': 491.,
                    'FORGERY/COUNTERFEITING': 10609., 'FRAUD': 16679., 'GAMBLING': 146., 'KIDNAPPING': 2341.,
                    'LARCENY/THEFT': 174900., 'LIQUOR LAWS': 1903., 'LOITERING': 1225., 'MISSING PERSON': 25989.,
                    'NON-CRIMINAL': 92304., 'OTHER OFFENSES': 126182., 'PORNOGRAPHY/OBSCENE MAT': 22.,
                    'PROSTITUTION': 7484., 'RECOVERED VEHICLE': 3138., 'ROBBERY': 23000., 'RUNAWAY': 1946.,
                    'SECONDARY CODES': 9985., 'SEX OFFENSES FORCIBLE': 4388., 'SEX OFFENSES NON FORCIBLE': 148.,
                    'STOLEN PROPERTY': 4540., 'SUICIDE': 508., 'SUSPICIOUS OCC': 31414., 'TREA': 6., 'TRESPASS': 7326.,
                    'VANDALISM': 44725., 'VEHICLE THEFT': 53781., 'WARRANTS': 42214., 'WEAPON LAWS': 8555., }
class_prior = {class_: statistic / sum(class_statistics.values()) for class_, statistic in class_statistics.items()}


def output(P, class_indices, file_path):
    with open(file_path, 'w') as output_file:
        csv_writer = csv.writer(output_file)

        classes = class_statistics.keys()
        class_name2index = {class_name: index for index, class_name in enumerate(class_indices)}
        header = ['Id'] + classes
        csv_writer.writerow(header)

        for id_, p in enumerate(P):
            row = [id_]

            for class_name in classes:
                try:
                    row.append(p[class_name2index[class_name]])
                except KeyError:
                    row.append(0)

            csv_writer.writerow(row)


def add_prior_probability(P, class_indices, class_prior_weight=0.2):
    classes = class_statistics.keys()
    class_name2index = {class_name: index for index, class_name in enumerate(class_indices)}

    for sample_id, p in enumerate(P):
        for class_name in classes:
            try:
                p[class_name2index[class_name]] = p[class_name2index[class_name]] * (1 - class_prior_weight) + \
                                                  class_prior[class_name] * class_prior_weight
            except KeyError:
                pass
        P[sample_id] = p

    return P


@click.command()
@click.option('--validate_model', prompt='Validate the model? (y/n)', default='n')
@click.option('--predict_result', prompt='Predict the submission results? (y/n)', default='n')
@click.option('--debug', prompt='Run on debug mode? (y/n)', default='n')
def main(validate_model, predict_result, debug):
    validate_model = True if validate_model == 'y' else False
    predict_result = True if predict_result == 'y' else False
    debug = True if debug == 'y' else False

    (classifier, vectorizer) = train(validate_model, predict_result, debug)

    if predict_result is True:
        (P, class_indices) = predict(classifier, vectorizer)
        output(P, class_indices, 'results/submission.csv')


if __name__ == '__main__':
    main()
