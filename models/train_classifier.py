# import libraries
import sys
import pandas as pd
import numpy as np
import pickle
from sqlalchemy import create_engine
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import nltk
nltk.download('punkt')
nltk.download('wordnet')


def load_data(database_filepath):
    """
    Load data from the database created with process_data.py and already splits into response and predicting variables.
    """
    path = ['sqlite:///', database_filepath]
    engine = create_engine("".join(path))
    df = pd.read_sql("SELECT * FROM DisasterResponse", engine)
    X = df.message
    Y = df.iloc[:, 4:]
    category_names = list(Y.columns.values)

    return X, Y, category_names

def tokenize(text):
    """
    Tokenize, lemmatize and clean text messages.
    """
    tokens = nltk.word_tokenize(text)
    lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
    Transform text and classifies it using a random forest algorithm, with two sets of parameters.
    """
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultiOutputClassifier(RandomForestClassifier()))])

    parameters = parameters = {
    'clf__estimator__min_samples_split': [2, 3]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate model outputs using f1 scores, precision and recall.
    """
    Y_pred = model.predict(X_test)
    Y_pred_df = pd.DataFrame(Y_pred)
    Y_pred_df.columns = category_names

    for category in category_names:
        print('\n---- {} ----\n{}\n'.format(category, classification_report(Y_test[category], Y_pred_df[category])))


def save_model(model, model_filepath):
    """
    Save model in a pickle file.
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    """
    Runs all functions above in the correct order.
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
