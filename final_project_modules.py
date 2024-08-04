"""
Contains functions to load and preprocess data for machine learning, train models, and evaluate performance.

Includes:
- load_data(filename)

- generate_wordclouds(X, y)
- generate_single_wordcloud(dataframe)
- generate_histograms(X, y, characteristic)
- generate_single_histogram(dataframe, characteristic_type)

- create_derived_features(X)
- find_http_or_bitly(string)
- preprocess_data(X_data, verbose)
- drop_instances_with_empty_text(X, y)
- split_data(X_data, y_data, desired_random_state)
- vectorize_text_data(X_train, X_test, X_val, method, word2vec_model)

- fit_model(X, y, modelname, desired_random_state)
- evaluate_model(model_name, y_true, y_pred)

- reintroduce_derived_features(X_train_original, X_test_original, X_val_original,
                                X_train_vectors, X_test_vectors, X_val_vector)

- create_dataframe_of_aggregate_data(label_list, list_of_evaluation_dictionaries)
- graph_aggregate_data(evaluation_df, fig_width, fig_height, legend_number_of_columns, title, outfile_name)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import gensim
import gensim.downloader as gensim_api

from wordcloud import WordCloud

import multiprocessing

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.feature_extraction.text import TfidfVectorizer



def load_data(filename):
    """
    Reads in input file and loads data.

    filename: csv file
    return: X and y dataframe
    """

    # 1. Load data from dataset
    print(f'********** Loading data **********')
    in_filename = filename
    df = pd.read_csv(in_filename, encoding='latin-1')
    df.rename(columns={'SMSes': 'text', 'Labels': 'label'}, inplace=True)

    print(f'\nNo of Rows: {df.shape[0]}')
    print(f'No of Columns: {df.shape[1]}')

    print(f'\nGeneral info:')
    print(f'{df.info()}')

    # Make sure that text data in selected columns are strings
    # Trim unnecessary spaces for strings in 'SMSes' column
    df.loc[:, 'text'] = df.loc[:, 'text'].apply(lambda x: str(x))
    df.loc[:, 'text'] = df.loc[:, 'text'].apply(lambda x: x.strip())

    # 3. Drop null values
    df.dropna(axis=0, inplace=True)
    print(f'\nNo of rows (After dropping null): {df.shape[0]}')
    print(f'No of columns: {df.shape[1]}')

    # 5. Remove duplicates using 'text' column and keeping first occurrence
    df.drop_duplicates(subset=['text'], keep='first', inplace=True)

    print(f'\nNo of rows (After removing duplicates): {df.shape[0]}')
    print(f'No of columns: {df.shape[1]}')

    print('\nData View: First Few Instances')
    pd.set_option('display.max_colwidth', 100)
    print(df.head())
    pd.set_option('display.max_colwidth', 50)

    print('\nClass Counts (label, row): Total')
    print('0.0 = legitimate, 1.0 = spam')
    print(df['label'].value_counts())

    # 6. Split data from df_selected into X_data and y_data
    X_data = df.iloc[:, :-1]
    y_data = df.iloc[:, -1]

    return X_data, y_data


def generate_wordclouds(X, y):
    """
    Generates 3 wordclouds based on y labels (all, 0.0 (legitimate), 1.0 (spam))

    :param X: dataframe containing text, column label 'text'
    :param y: dataframe containing labels, column label 'label'
    :return: 3 wordcloud images
    """
    df = pd.concat([X, y], axis=1)

    df['text'] = df['text'].astype(str)
    df['text'] = df['text'].map(lambda x: x.lower())

    print(f'\nWordcloud for All Texts:')
    generate_single_wordcloud(df)

    print(f'\nWordcloud for Legitimate Texts:')
    df_legitimate = df[df['label'] == 0.0]
    generate_single_wordcloud(df_legitimate)

    print(f'\nWordcloud for Spam Texts:')
    df_spam = df[df['label'] == 1.0]
    generate_single_wordcloud(df_spam)


def generate_single_wordcloud(dataframe):
    """
    Creates a single wordcloud from a given dataframe.

    :param dataframe: dataframe with column titled 'text'
    :return: 1 wordcloud image
    """
    stop_words = set(stopwords.words('english'))
    comment_words = ''

    for value in dataframe['text']:
        tokens = value.split()
        comment_words += " ".join(tokens) + " "

    single_wordcloud = WordCloud(width=400,
                                 height=400,
                                 background_color='white',
                                 stopwords=stop_words,
                                 min_font_size=10).generate(comment_words)

    plt.figure(figsize=(4, 4), facecolor=None)
    plt.imshow(single_wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()


def generate_histograms(X, y, characteristic):
    """
    Generates 3 histograms based on y labels (all, 0.0 (legitimate), 1.0 (spam))

    :param X: dataframe containing text, column label 'text'
    :param y: dataframe containing labels, column label 'label'
    :param characteristic: text characteristic to base histogram on, either:
        'Number of Characters', 'Word Count', or 'Average Word Length'
    :return: 3 histograms for number of characters
    """
    df = pd.concat([X, y], axis=1)

    df['text'] = df['text'].astype(str)
    df['text'] = df['text'].map(lambda x: x.lower())

    print(f'Note: X limits have been implemented for each histogram type')

    print(f'\n{characteristic} Histogram for All Texts:')
    generate_single_histogram(df, characteristic)

    print(f'\n{characteristic} Histogram for Legitimate Texts:')
    df_legitimate = df[df['label'] == 0.0]
    generate_single_histogram(df_legitimate, characteristic)

    print(f'\n{characteristic} Histogram for Spam Texts:')
    df_spam = df[df['label'] == 1.0]
    generate_single_histogram(df_spam, characteristic)


def generate_single_histogram(dataframe, characteristic_type):
    """
    Creates histogram based on characteristic type for text data

    :param dataframe: dataframe with column 'text'
    :param characteristic_type: characteristic on which to base histogram, either:
        'Number of Characters', 'Word Count', or 'Average Word Length'
    :return: image of histogram
    """
    fig, ax = plt.subplots()

    if characteristic_type == 'Number of Characters':
        plt.hist(x=dataframe['text'].str.len(), bins='auto')
        ax.set_xlim(0, 500)
        ax.set_xlabel('Number of characters in text')
        ax.set_ylabel('Frequency')
        plt.show()

    elif characteristic_type == 'Word Count':
        plt.hist(x=dataframe['text'].str.split().str.len(), bins='auto')
        ax.set_xlim(0, 125)
        ax.set_xlabel('Number of words in text')
        ax.set_ylabel('Frequency')
        plt.show()

    elif characteristic_type == 'Average Word Length':
        plt.hist(x=dataframe['text'].str.split().apply(lambda x: [len(i) for i in x]).map(lambda x: np.mean(x)),
                 bins='auto')
        ax.set_xlim(0, 15)
        ax.set_xlabel('Average characters per word in text')
        ax.set_ylabel('Frequency')
        plt.show()

    else:
        print(f'Error: invalid characteristic type, "{characteristic_type}".')
        print(f'Expecting "Number of Characters", "Word Count", or "Average Word Length"')


def create_derived_features(X):
    """
    Creates columns for word count and presence of a URL to be added to X dataframe

    :param X: X dataframe with raw text data, column label 'text'
    :return: X dataframe with raw text 'text', word count 'word_count', and URL presence 'url_presence'
    """
    print(f'********** Creating Derived Features **********')
    df = X

    print(f'\nAdding column for word count.')
    df['word_count'] = df['text'].str.split().str.len()

    print(f'\nAdding column for URL presence: 1 if URL substring found, 0 otherwise.')
    df['url_presence'] = df['text'].apply(lambda x: find_http_or_bitly(x))
    print(f'\n{df["url_presence"].value_counts()}')

    print(f'\nUpdated Data View:')
    pd.set_option('display.max_colwidth', 100)
    print(df)
    pd.set_option('display.max_colwidth', 50)

    return df


def find_http_or_bitly(string):
    """
    Locate 'http' or 'bit.ly' substring within existing string.
    :return: 1 if URL substring found, 0 if not.
    """
    if ('http' or 'bit.ly') in string:
        result = 1
    else:
        result = 0
    return result


def preprocess_data(X_data, verbose):
    """
    Preprocesses data with lowercase conversion, punctuation removal, stopword removal,
    tokenization, and stemming or lemmatization

    X_data: X data in dataframe, text in column labeled 'text'
    verbose: specify whether to print step-by-step. 1 = print, 0 = no print
    return: 2 X dataframes, one with stemming applied and one with lemmatization applied
    """

    print(f'\n********** Preprocessing **********')

    # 1. Make sure that text data in selected column are string
    X_data_text = X_data['text'].astype(str)
    if verbose == 1:
        print(f'\nConvert to string:')
        print(X_data_text.head(3))

    # 2. Convert all characters to lowercase
    X_data_text = X_data_text.map(lambda x: x.lower())
    if verbose == 1:
        print(f'\nConvert to lowercase:')
        print(X_data_text.head(3))

    # 3. Remove punctuation
    X_data_text = X_data_text.str.replace('[^\w\s]', '', regex=True)
    if verbose == 1:
        print(f'\nRemove punctuation:')
        print(X_data_text.head(3))

    # 4. Tokenize sentence
    X_data_text = X_data_text.apply(nltk.word_tokenize)
    if verbose == 1:
        print(f'\nTokenize:')
        print(X_data_text.head(3))

    # 5. Remove stopwords
    stopword_list = stopwords.words("english")
    X_data_text = X_data_text.apply(lambda x: [word for word in x if word not in stopword_list])
    if verbose == 1:
        print(f'\nRemove stopwords:')
        print(X_data_text.head(3))

    # 6. Stemming
    stemmer = PorterStemmer()
    X_data_text_stem = X_data_text.apply(lambda x: [stemmer.stem(y) for y in x])
    if verbose == 1:
        print(f'\nStemming:')
        print(X_data_text_stem.head(3))

    # 7. Lemmatizing
    lemmatizer = WordNetLemmatizer()
    X_data_text_lem = X_data_text.apply(lambda x: [lemmatizer.lemmatize(y) for y in x])
    if verbose == 1:
        print(f'\nLemmatization:')
        print(X_data_text_lem.head(3))

    # 7. Convert back from list
    X_data_text_stem = X_data_text_stem.apply(lambda x: " ".join(x))
    X_data_text_lem = X_data_text_lem.apply(lambda x: " ".join(x))
    if verbose == 1:
        print(f'\nConvert back from list:')
        print('Stemmed data:')
        print(X_data_text_stem.head(3))
        print('Lemmatized data:')
        print(X_data_text_lem.head(3))

    X_preprocessed_stem = pd.concat([X_data_text_stem, X_data['word_count'], X_data['url_presence']], axis=1)
    X_preprocessed_lem = pd.concat([X_data_text_lem, X_data['word_count'], X_data['url_presence']], axis=1)

    print(f'\n*** Preprocessed Data View: Stemmed data ***')
    print(X_preprocessed_stem.head(10))

    print(f'\n*** Preprocessed Data View: Lemmatized data ***')
    print(X_preprocessed_lem.head(10))

    return X_preprocessed_stem, X_preprocessed_lem


def drop_instances_with_empty_text(X, y):
    """
    Drops instances and label rows where text was completely removed by preprocessing.

    :param X: Preprocessed X dataframe with column 'text' that contains some empty strings
    :param y: y dataframe containing labels for each X instance
    :return: cleaned X and y dataframes with empty text instances removed
    """
    print('********** Dropping instances with empty text after preprocessing **********')
    df = pd.concat([X, y], axis=1)

    print(f'\nNo of rows (before dropping empty strings): {df.shape[0]}')
    print(f'No of columns: {df.shape[1]}')

    # Drop instances with empty text
    df_cleaned = df[df['text'] != '']

    print(f'\nNo of rows (after dropping empty strings): {df_cleaned.shape[0]}')
    print(f'No of columns: {df_cleaned.shape[1]}')

    # Show label distribution of dropped instances
    print(f'\nLabel distribution of removed instances:')
    df_empty = df[df['text'] == '']
    print(f'{df_empty["label"].value_counts()}')

    X_data_cleaned = df_cleaned.iloc[:, :-1]
    y_data_cleaned = df_cleaned.iloc[:, -1]

    return X_data_cleaned, y_data_cleaned


def split_data(X_data, y_data, desired_random_state):
    """
    Splits data into the subsets of train, validation, and test with a Ratio of 8:1:1 and resets indexes.

    X_data, y_data: X and y dataframes
    desired_random_state: random state to set train_test_split
    return: X_train, X_val, X_test, y_train, y_val, y_test dataframes
    """
    print(f'********** Splitting Data **********')

    # 1. Split data
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2,
                                                        random_state=desired_random_state, stratify=y_data)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5,
                                                    random_state=desired_random_state, stratify=y_test)

    print(f'\nTrain Data: {X_train.shape}')
    print(f'Val Data: {X_val.shape}')
    print(f'Test Data: {X_test.shape}')

    print('\nClass Counts(label, row): Train')
    print(y_train.value_counts())
    print('\nClass Counts(label, row): Val')
    print(y_val.value_counts())
    print('\nClass Counts(label, row): Test')
    print(y_test.value_counts())

    # 2. Reset indexes
    # Train Data
    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)

    # Validation Data
    X_val = X_val.reset_index(drop=True)
    y_val = y_val.reset_index(drop=True)

    # Test Data
    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    print("\nData View: X Train")
    print(X_train.head(3))
    print("\nData View: X Val")
    print(X_val.head(3))
    print("\nData View: X Test")
    print(X_test.head(3))

    return X_train, X_val, X_test, y_train, y_val, y_test


def vectorize_text_data(X_train, X_test, X_val, method, word2vec_model):
    """
    Vectorizes text data and returns dataframe or array with vectors. Drops any column except for 'text'.

    :param X_train: X train dataframe
    :param X_test: X test dataframe
    :param X_val: X validation dataframe
    :param method: either 'TF-IDF' or 'Word2Vec', selects vectorization method
    :param word2vec_model: preloaded Word2Vec model for vectorization. If method is 'TF-IDF', this value is 0
    :return: vectorized dataframe or array ready for model training.
    """
    print(f'\n********** Vectorizing Data Using {method} **********')
    print(f'Derived features are dropped during this step and will be reintroduced later.')
    X_train_text = X_train['text']
    X_test_text = X_test['text']
    X_val_text = X_val['text']

    if method == 'TF-IDF':
        vectorizer = TfidfVectorizer()

        X_train_vectors = vectorizer.fit_transform(X_train_text)
        X_test_vectors = vectorizer.transform(X_test_text)
        X_val_vectors = vectorizer.transform(X_val_text)

    elif method == 'Word2Vec':
        vectorizer = word2vec_model

        X_train_token = X_train_text.apply(nltk.word_tokenize)
        X_test_token = X_test_text.apply(nltk.word_tokenize)
        X_val_token = X_val_text.apply(nltk.word_tokenize)

        X_train_vectors_nested = X_train_token.apply(lambda x: vectorizer.get_mean_vector(x))
        X_test_vectors_nested = X_test_token.apply(lambda x: vectorizer.get_mean_vector(x))
        X_val_vectors_nested = X_val_token.apply(lambda x: vectorizer.get_mean_vector(x))

        X_train_vectors = np.stack(X_train_vectors_nested)
        X_test_vectors = np.stack(X_test_vectors_nested)
        X_val_vectors = np.stack(X_val_vectors_nested)

    else:
        print(f'ERROR: Unexpected method selected. Expecting either "TF-IDF" or "Word2Vec". You chose "{method}".')

    print(f'\n Data View: First Few Instances of X_train')
    print(pd.DataFrame(X_train_vectors).head(3))
    print('\n')
    print(pd.DataFrame(X_train_vectors).info())

    return X_train_vectors, X_test_vectors, X_val_vectors


def fit_model(X, y, modelname, desired_random_state):
    """
    Conducts model fitting on input data.

    X: preprocessed X training data, which has already undergone feature extraction and vectorization.
    y: labels corresponding to X data
    modelname: model type to fit. Can be:
        - 'SVM' Support Vector Machine
        - 'RF' Random Forest.
    desired_random_state: selected random state, for reproducibility
    return: fitted model
    """
    print(f'********** Training {modelname} Model **********')

    if modelname == 'SVM':
        model = SVC(random_state=desired_random_state)

    elif modelname == 'RF':
        model = RandomForestClassifier(random_state=desired_random_state)

    else:
        print(f'Error: model type unexpected. You put "{modelname}"; was expecting "SVM" or "RF".')

    model.fit(X, y)
    print('*** Model Training Complete ***')

    return model


def evaluate_model(model_name, y_true, y_pred):
    """
    Provides evaluation metrics for a given model

    :param model_name: Name of model as string, for comparison purposes
    :param y_true: actual y values
    :param y_pred: model-predicted y values
    :return: classification report dictionary and display of confusion matrix and classification report
    """
    matrix = confusion_matrix(y_true, y_pred)
    class_report = classification_report(y_true, y_pred, digits=4)
    class_report_dict = classification_report(y_true, y_pred, digits=4, output_dict=True)

    print(f'\n************** Model Evaluation for {model_name} **************\n')
    print('Confusion Matrix:')
    print(f'{matrix}')

    print('\nClassification Report:')
    print(f'{class_report}')

    confusion_display = ConfusionMatrixDisplay(confusion_matrix=matrix)
    confusion_display.plot()
    plt.title('Confusion Matrix')
    plt.show()

    return class_report_dict


def reintroduce_derived_features(X_train_original, X_test_original, X_val_original,
                                 X_train_vectors, X_test_vectors, X_val_vectors):
    """
    Adds columns back to dataframes relating to word count and url presence

    :param X_train_original: original X_train dataframe with 'text', 'word_count', and 'url_presence' columns
    :param X_test_original: original X_test dataframe with 'text', 'word_count', and 'url_presence' columns
    :param X_val_original: original X_val dataframe with 'text', 'word_count', and 'url_presence' columns
    :param X_train_vectors: X_train dataframe corresponding to vectors of original X_train 'text' column
    :param X_test_vectors: X_test dataframe corresponding to vectors of original X_test 'text' column
    :param X_val_vectors: X_val dataframe corresponding to vectors of original X_val 'text' column
    :return: X_train, X_test, and X_val dataframes with word2vec vectors and word_count and url_presence columns
    """
    print(f'********** Reintroducing Derived Features **********')
    X_train_vectors_df = pd.DataFrame(X_train_vectors)
    X_test_vectors_df = pd.DataFrame(X_test_vectors)
    X_val_vectors_df = pd.DataFrame(X_val_vectors)

    X_train_vectorized_plus = pd.concat([X_train_vectors_df, X_train_original[['word_count', 'url_presence']]], axis=1)
    X_test_vectorized_plus = pd.concat([X_test_vectors_df, X_test_original[['word_count', 'url_presence']]], axis=1)
    X_val_vectorized_plus = pd.concat([X_val_vectors_df, X_val_original[['word_count', 'url_presence']]], axis=1)

    X_train_vectorized_plus.rename(columns={'word_count': 300, 'url_presence': 301}, inplace=True)
    X_test_vectorized_plus.rename(columns={'word_count': 300, 'url_presence': 301}, inplace=True)
    X_val_vectorized_plus.rename(columns={'word_count': 300, 'url_presence': 301}, inplace=True)

    print(f'\n Data View: First Few Instances of X_train with Derived Features')
    print(pd.DataFrame(X_train_vectorized_plus).head(3))
    print('\n')
    print(pd.DataFrame(X_train_vectorized_plus).info())

    return X_train_vectorized_plus, X_test_vectorized_plus, X_val_vectorized_plus


def create_dataframe_of_aggregate_data(label_list, list_of_evaluation_dictionaries):
    """
    Creates dataframe of evaluation metrics to compare models.
    :param label_list: List of model names, in same order as evaluation dictionaries
    :param list_of_evaluation_dictionaries: List of evaluation dictionaries
    :return: Dataframe of evaluation metrics
    """
    class_zero_precision = []
    for model_evaluation in list_of_evaluation_dictionaries:
        precision = model_evaluation.get('0.0').get('precision')
        class_zero_precision.append(precision)

    class_one_precision = []
    for model_evaluation in list_of_evaluation_dictionaries:
        precision = model_evaluation.get('1.0').get('precision')
        class_one_precision.append(precision)

    class_zero_recall = []
    for model_evaluation in list_of_evaluation_dictionaries:
        recall = model_evaluation.get('0.0').get('recall')
        class_zero_recall.append(recall)

    class_one_recall = []
    for model_evaluation in list_of_evaluation_dictionaries:
        recall = model_evaluation.get('1.0').get('recall')
        class_one_recall.append(recall)

    class_zero_f1 = []
    for model_evaluation in list_of_evaluation_dictionaries:
        f1 = model_evaluation.get('0.0').get('f1-score')
        class_zero_f1.append(f1)

    class_one_f1 = []
    for model_evaluation in list_of_evaluation_dictionaries:
        f1 = model_evaluation.get('1.0').get('f1-score')
        class_one_f1.append(f1)

    accuracy_list = []
    for model_evaluation in list_of_evaluation_dictionaries:
        accuracy = model_evaluation.get('accuracy')
        accuracy_list.append(accuracy)

    combined_dictionary = {'Model': label_list,
                           'Class 0 Precision': class_zero_precision,
                           'Class 1 Precision': class_one_precision,
                           'Class 0 Recall': class_zero_recall,
                           'Class 1 Recall': class_one_recall,
                           'Class 0 F1': class_zero_f1,
                           'Class 1 F1': class_one_f1,
                           'Accuracy': accuracy_list}

    column_names_excluding_model = ['Class 0 Precision', 'Class 1 Precision', 'Class 0 Recall', 'Class 1 Recall',
                                    'Class 0 F1', 'Class 1 F1', 'Accuracy']
    evaluation_df = pd.DataFrame(combined_dictionary)
    evaluation_df.style.background_gradient(subset=column_names_excluding_model, cmap='RdYlGn', axis=0)

    return evaluation_df


def graph_aggregate_data(evaluation_df, fig_width, fig_height, legend_number_of_columns, title, outfile_name):
    """
    Creates graph comparison of evaluation metrics dataframe
    :param evaluation_df: Dataframe of evaluation metrics
    :param fig_width: Figure width as int
    :param fig_height: Figure height as int
    :param legend_number_of_columns: Legend number of columns as int
    :param title: Title as string
    :param outfile_name: Output file name as string, empty string if not saving
    """
    evaluation_df.plot(x='Model',
                       kind='bar',
                       stacked=False,
                       title=title,
                       colormap='Spectral',
                       figsize=(fig_width, fig_height))
    plt.legend(ncols=legend_number_of_columns)
    plt.ylabel('Score')
    plt.show()

    if outfile_name != '':
        plt.savefig(outfile_name, bbox_inches='tight')
