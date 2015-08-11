import csv
import re
import click
from datetime import datetime
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import classification_report

__author__ = 'kensk8er'


def validate(X, y):
    print('Split the data...')
    (X_train, X_valid, y_train, y_valid) = train_test_split(X, y, test_size=0.1, random_state=0)

    classifier = fit(X_train, y_train)

    print('Predict the labels on validation set...')
    y_pred = classifier.predict(X_valid)
    print(classification_report(y_valid, y_pred))


def fit(X, y):
    print('Fit the classifier...')
    classifier = RandomForestClassifier(n_jobs=-1)  # TODO: Tune the parameters
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


def load(file_path, labeled, debug=False):
    if labeled is True:
        y = []
    features = []

    print('Loading {} ...'.format(file_path))
    with open(file_path) as train_file:
        csv_reader = csv.reader(train_file)
        header = csv_reader.next()

        for index, row in enumerate(csv_reader):
            if debug is True and index == 10000:
                break

            datum = {key: val for key, val in zip(header, row)}

            if labeled is True:
                y.append(datum['Category'])

            (hour, day, month, year) = parse_time(datum['Dates'])

            feature = {
                "DayOfWeek_{}".format(datum['DayOfWeek']): 1,
                "PdDistrict_{}".format(datum['PdDistrict']): 1,
                "hour_{}".format(hour): 1,
                "day_{}".format(day): 1,
                "month_{}".format(month): 1,
                "year_{}".format(year): 1,
                "longitude": int(float(datum['X'])),
                "latitude": int(float(datum['Y'])),
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


def output(P, class_indices, file_path, class_prior_weight):
    classes = ['ARSON', 'ASSAULT', 'BAD CHECKS', 'BRIBERY', 'BURGLARY', 'DISORDERLY CONDUCT',
               'DRIVING UNDER THE INFLUENCE', 'DRUG/NARCOTIC', 'DRUNKENNESS', 'EMBEZZLEMENT', 'EXTORTION',
               'FAMILY OFFENSES', 'FORGERY/COUNTERFEITING', 'FRAUD', 'GAMBLING', 'KIDNAPPING', 'LARCENY/THEFT',
               'LIQUOR LAWS', 'LOITERING', 'MISSING PERSON', 'NON-CRIMINAL', 'OTHER OFFENSES',
               'PORNOGRAPHY/OBSCENE MAT', 'PROSTITUTION', 'RECOVERED VEHICLE', 'ROBBERY', 'RUNAWAY', 'SECONDARY CODES',
               'SEX OFFENSES FORCIBLE', 'SEX OFFENSES NON FORCIBLE', 'STOLEN PROPERTY', 'SUICIDE', 'SUSPICIOUS OCC',
               'TREA', 'TRESPASS', 'VANDALISM', 'VEHICLE THEFT', 'WARRANTS', 'WEAPON LAWS']

    class_name2index = {class_name: index for index, class_name in enumerate(class_indices)}

    with open(file_path, 'w') as output_file:
        csv_writer = csv.writer(output_file)

        header = ['Id'] + classes
        csv_writer.writerow(header)

        for id_, p in enumerate(P):
            row = [id_]

            for class_name in classes:
                try:
                    row.append(p[class_name2index[class_name]] * (1 - class_prior_weight) + class_prior[
                        class_name] * class_prior_weight)
                except KeyError:
                    row.append(class_prior[class_name] * class_prior_weight)

            csv_writer.writerow(row)


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
        output(P, class_indices, 'results/submission.csv', class_prior_weight=0.5)


if __name__ == '__main__':
    main()
