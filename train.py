import csv
from scipy.sparse import hstack
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC

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
    classifier = LinearSVC()
    classifier.fit(X, y)
    return classifier


def load(file_path, labeled):
    if labeled is True:
        y = []
    day_of_weeks = []
    pd_districts = []
    addresses = []

    print('Loading {} ...'.format(file_path))
    with open('data/train.csv') as train_file:
        csv_reader = csv.reader(train_file)
        header = csv_reader.next()

        for index, row in enumerate(csv_reader):
            datum = {key: val for key, val in zip(header, row)}
            if labeled is True:
                y.append(datum['Category'])
            day_of_weeks.append({datum['DayOfWeek']: 1})
            pd_districts.append({datum['PdDistrict']: 1})
            addresses.append({datum['Address']: 1})

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
    X_day_of_weeks = day_of_weeks_encoder.fit_transform(day_of_weeks)

    print('Vectorizing PdDistrict...')
    X_pd_districts = pd_districts_encoder.fit_transform(pd_districts)

    print('Vectorizing Address...')
    X_addresses = addresses_encoder.fit_transform(addresses)

    print('Concatenating features...')
    X = hstack([X_day_of_weeks, X_pd_districts, X_addresses])

    print('Predicting the y for test set...')
    y = classifier.predict(X)

    return y


def output(y, file_path):
    classes = ['ARSON', 'ASSAULT', 'BAD CHECKS', 'BRIBERY', 'BURGLARY', 'DISORDERLY CONDUCT',
               'DRIVING UNDER THE INFLUENCE', 'DRUG/NARCOTIC', 'DRUNKENNESS', 'EMBEZZLEMENT', 'EXTORTION',
               'FAMILY OFFENSES', 'FORGERY/COUNTERFEITING', 'FRAUD', 'GAMBLING', 'KIDNAPPING', 'LARCENY/THEFT',
               'LIQUOR LAWS', 'LOITERING', 'MISSING PERSON', 'NON-CRIMINAL', 'OTHER OFFENSES',
               'PORNOGRAPHY/OBSCENE MAT', 'PROSTITUTION', 'RECOVERED VEHICLE', 'ROBBERY', 'RUNAWAY', 'SECONDARY CODES',
               'SEX OFFENSES FORCIBLE', 'SEX OFFENSES NON FORCIBLE', 'STOLEN PROPERTY', 'SUICIDE', 'SUSPICIOUS OCC',
               'TREA', 'TRESPASS', 'VANDALISM', 'VEHICLE THEFT', 'WARRANTS', 'WEAPON LAWS']

    with open(file_path, 'w') as output_file:
        csv_writer = csv.writer(output_file)

        header = ['Id'] + classes
        csv_writer.writerow(header)

        for id_, pred_class in enumerate(y):
            row = [id_]

            for class_ in classes:
                if class_ == pred_class:
                    row.append(1)
                else:
                    row.append(0)

            csv_writer.writerow(row)


if __name__ == '__main__':
    (classifier, day_of_weeks_encoder, pd_districts_encoder, addresses_encoder) = train()
    y = predict(classifier, day_of_weeks_encoder, pd_districts_encoder, addresses_encoder)
    output(y, 'results/submission.csv')
