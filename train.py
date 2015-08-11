import csv
from scipy.sparse import hstack
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC, SVC

__author__ = 'kensk8er'


def validate(X, y):
    print('Split the data...')
    (X_train, X_valid, y_train, y_valid) = train_test_split(X, y, test_size=0.1, random_state=0)

    print('Fit the classifier...')
    classifier = LinearSVC()
    classifier.fit(X_train, y_train)

    print('Predict the labels on validation set...')
    y_pred = classifier.predict(X_valid)
    print(classification_report(y_valid, y_pred))


def fit(X, y):
    print('Fit the classifier...')
    # classifier = LinearSVC()
    # classifier = SVC(kernel='linear', probability=True)
    classifier = RandomForestClassifier(n_jobs=-1)  # TODO: Tune the parameters
    classifier.fit(X, y)
    return classifier


def load(file_path, labeled):
    if labeled is True:
        y = []
    day_of_weeks = []
    pd_districts = []
    addresses = []

    print('Loading {} ...'.format(file_path))
    with open(file_path) as train_file:
        csv_reader = csv.reader(train_file)
        header = csv_reader.next()

        for index, row in enumerate(csv_reader):
            datum = {key: val for key, val in zip(header, row)}
            if labeled is True:
                y.append(datum['Category'])
            day_of_weeks.append({datum['DayOfWeek']: 1})
            pd_districts.append({datum['PdDistrict']: 1})
            addresses.append({datum['Address']: 1})
            # TODO: Integrate seasonal (or month) information
            # TODO: Integrate time (or morning/noon/night) information
            # TODO: Address needs to be treated together with PdDistrict? Probably ignore the street number. Maybe pick up capital letters only.
            # TODO: Integrate latitude and longitude (RandomForest can utilize these without any feature engineering??)
            # TODO: Integrate day information (maybe more theft on a day when pay day is likely to be)
            # TODO: Integrate year information

    return (y, day_of_weeks, pd_districts, addresses) if labeled is True else (day_of_weeks, pd_districts, addresses)


def train():
    (y, day_of_weeks, pd_districts, addresses) = load('data/train.csv', labeled=True)

    print('Vectorizing DayOfWeek...')
    day_of_weeks_encoder = DictVectorizer()
    X_day_of_weeks = day_of_weeks_encoder.fit_transform(day_of_weeks)

    print('Vectorizing PdDistrict...')
    pd_districts_encoder = DictVectorizer()
    X_pd_districts = pd_districts_encoder.fit_transform(pd_districts)

    print('Vectorizing Address...')
    addresses_encoder = DictVectorizer()
    X_addresses = addresses_encoder.fit_transform(addresses)

    print('Concatenating features...')
    X = hstack([X_day_of_weeks, X_pd_districts, X_addresses])

    # validate(X, y)
    classifier = fit(X, y)

    return classifier, day_of_weeks_encoder, pd_districts_encoder, addresses_encoder


def predict(classifier, day_of_weeks_encoder, pd_districts_encoder, addresses_encoder):
    (day_of_weeks, pd_districts, addresses) = load('data/test.csv', labeled=False)

    print('Vectorizing DayOfWeek...')
    X_day_of_weeks = day_of_weeks_encoder.transform(day_of_weeks)

    print('Vectorizing PdDistrict...')
    X_pd_districts = pd_districts_encoder.transform(pd_districts)

    print('Vectorizing Address...')
    X_addresses = addresses_encoder.transform(addresses)

    print('Concatenating features...')
    X = hstack([X_day_of_weeks, X_pd_districts, X_addresses])

    print('Predicting the y for test set...')
    P = classifier.predict_proba(X)
    class_indices = classifier.classes_

    return P, class_indices


class_statistics = {'ARSON': 1513., 'ASSAULT': 76876., 'BAD CHECKS': 406., 'BRIBERY': 289., 'BURGLARY': 36755.,
                    'DISORDERLY CONDUCT': 4320., 'DRIVING UNDER THE INFLUENCE': 2268., 'DRUG/NARCOTIC': 53971.,
                    'EMBEZZLEMENT': 1166., 'EXTORTION': 256., 'FAMILY OFFENSES': 491., 'FORGERY/COUNTERFEITING': 10609.,
                    'FRAUD': 16679., 'GAMBLING': 146., 'KIDNAPPING': 2341., 'LARCENY/THEFT': 174900.,
                    'LIQUOR LAWS': 1903., 'LOITERING': 1225., 'MISSING PERSON': 25989., 'NON-CRIMINAL': 92304.,
                    'OTHER OFFENSES': 126182., 'PORNOGRAPHY/OBSCENE MAT': 22., 'PROSTITUTION': 7484.,
                    'RECOVERED VEHICLE': 3138., 'ROBBERY': 23000., 'RUNAWAY': 1946., 'SECONDARY CODES': 9985.,
                    'SEX OFFENSES FORCIBLE': 4388., 'SEX OFFENSES NON FORCIBLE': 148., 'STOLEN PROPERTY': 4540.,
                    'SUICIDE': 508., 'SUSPICIOUS OCC': 31414., 'TREA': 6., 'TRESPASS': 7326., 'VANDALISM': 44725.,
                    'VEHICLE THEFT': 53781., 'WARRANTS': 42214., 'WEAPON LAWS': 8555., }
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
                    row.append(p[class_name2index[class_name]] + class_prior[class_name] * class_prior_weight)
                except KeyError:
                    row.append(class_prior[class_name] * class_prior_weight)

            csv_writer.writerow(row)


if __name__ == '__main__':
    # TODO: Find out why submission.csv doens't have enough number of rows
    (classifier, day_of_weeks_encoder, pd_districts_encoder, addresses_encoder) = train()
    (P, class_indices) = predict(classifier, day_of_weeks_encoder, pd_districts_encoder, addresses_encoder)
    output(P, class_indices, 'results/submission.csv', class_prior_weight=0.5)
