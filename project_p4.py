# CS421: Natural Language Processing
# University of Illinois at Chicago
# Fall 2022
# Project Part 4
#
# Do not rename/delete any functions or global variables provided in this template and write your solution
# in the specified sections. Use the main function to test your code when running it from a terminal.
# Avoid writing that code in the global scope; however, you should write additional functions/classes
# as needed in the global scope. These templates may also contain important information and/or examples
# in comments so please read them carefully. If you want to use external packages (not specified in
# the assignment) then you need prior approval from course staff.
# This part of the assignment will be graded automatically using Gradescope.
# =========================================================================================================



from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import numpy as np
import pickle as pkl
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import string
import re
import csv
import nltk


# Before running code that makes use of Word2Vec, you will need to download the provided w2v.pkl file
# which contains the pre-trained word2vec representations from Blackboard
#
# If you store the downloaded .pkl file in the same directory as this Python
# file, leave the global EMBEDDING_FILE variable below as is.  If you store the
# file elsewhere, you will need to update the file path accordingly.
EMBEDDING_FILE = "w2v.pkl"


# Function: load_w2v
# filepath: path of w2v.pkl
# Returns: A dictionary containing words as keys and pre-trained word2vec representations as numpy arrays of shape (300,)
def load_w2v(filepath):
    with open(filepath, 'rb') as fin:
        return pkl.load(fin)


# Function: load_as_list(fname)
# fname: A string indicating a filename
# Returns: Two lists: one a list of document strings, and the other a list of integers
#
# This helper function reads in the specified, specially-formatted CSV file
# and returns a list of documents (documents) and a list of binary values (label).
def load_as_list(fname):
    df = pd.read_csv(fname)
    documents = df['review'].values.tolist()
    labels = df['label'].values.tolist()
    return documents, labels


# Function: extract_user_info, see project statement for more details
# user_input: A string of arbitrary length
# Returns: name as string
def extract_user_info(user_input):
    name = ""

    # [YOUR CODE HERE]
    regex = r"^(.*\s)*[A-Z][A-Za-z.&'-]* [A-Z][A-Za-z.&'-]*( [A-Z][A-Za-z.&'-]*){0,2}($|\s.*$|[.&'-]+$)$"

    p = re.compile(regex)

    match = p.match(user_input)
    if match is not None:
        #name = re.match("[A-Z][a-z.&'-]*",user_input,0)
        name = re.search("(^|\s)[A-Z][A-Za-z.&'-]* [A-Z][A-Za-z.&'-]*( [A-Z][A-Za-z.&'-]*){0,2}",user_input,0).group().strip()

    return name


# Function to convert a given string into a list of tokens
# Args:
#   inp_str: input string 
# Returns: token list, dtype: list of strings
def get_tokens(inp_str):
    return inp_str.split()


# Function: preprocessing, see project statement for more details
# Args:
#   user_input: A string of arbitrary length
# Returns: A string of arbitrary length
def preprocessing(user_input):
    modified_input = ""
    # [YOUR CODE HERE]
    # get tokens
    modified_input = get_tokens(user_input)
    tokenized = []

    # remove punctuation tokens
    for tkn in modified_input:
        if (tkn not in string.punctuation):
            tokenized.append(tkn)

    # convert to lowercase
    tokenized = list(map(lambda i: i.lower(), tokenized))
    tokenized = ' '.join(tokenized)
    return tokenized


# Function: vectorize_train, see project statement for more details
# training_documents: A list of strings
# Returns: An initialized TfidfVectorizer model, and a document-term matrix, dtype: scipy.sparse.csr.csr_matrix
def vectorize_train(training_documents):
    # Initialize the TfidfVectorizer model and document-term matrix
    vectorizer = TfidfVectorizer()
    tfidf_train = None
    # [YOUR CODE HERE]
    for doc in training_documents:
        tfidf_train.append(preprocessing(doc))
    # apply transform
    tfidf_train = vectorizer.fit_transform(tfidf_train)
    return vectorizer, tfidf_train


# Function: vectorize_test, see project statement for more details
# vectorizer: A trained TFIDF vectorizer
# user_input: A string of arbitrary length
# Returns: A sparse TFIDF representation of the input string of shape (1, X), dtype: scipy.sparse.csr.csr_matrix
#
# This function computes the TFIDF representation of the input string, using
# the provided TfidfVectorizer.
def vectorize_test(vectorizer, user_input):
    # Initialize the TfidfVectorizer model and document-term matrix
    tfidf_test = None

    # [YOUR CODE HERE]
    tfidf_test = None

    # [YOUR CODE HERE]
    prepr_input = []
    prepr_input.append(preprocessing(user_input))
    tfidf_test = vectorizer.transform(prepr_input)

    return tfidf_test


# Function: train_nb_model(training_documents, training_labels)
# training_data: A sparse TfIDF document-term matrix, dtype: scipy.sparse.csr.csr_matrix
# training_labels: A list of integers (0 or 1)
# Returns: A trained model
def train_nb_model(training_data, training_labels):
    # Initialize the GaussianNB model and the output label
    nb_model = GaussianNB()

    # Write your code here.  You will need to make use of the GaussianNB fit()
    # function.  You probably need to transfrom your data into a dense numpy array.
    # [YOUR CODE HERE]
    dense_matrix = training_data.todense()
    nb_model = nb_model.fit(dense_matrix, training_labels)
    return nb_model

# Function: get_model_prediction(nb_model, tfidf_test)
# nb_model: A trained GaussianNB model
# tfidf_test: A sparse TFIDF representation of the input string of shape (1, X), dtype: scipy.sparse.csr.csr_matrix
# Returns: A predicted label for the provided test data (int, 0 or 1)
def get_model_prediction(nb_model, tfidf_test):
    # Initialize the output label
    label = 0

    # Write your code here.  You will need to make use of the GaussianNB
    # predict() function. You probably need to transfrom your data into a dense numpy array.
    # [YOUR CODE HERE]
    label = nb_model.predict(tfidf_test.todense())
    return label


# Function: w2v(word2vec, token)
# word2vec: The pretrained Word2Vec representations as dictionary
# token: A string containing a single token
# Returns: The Word2Vec embedding for that token, as a numpy array of size (300,)
#
# This function provides access to 300-dimensional Word2Vec representations
# pretrained on Google News.  If the specified token does not exist in the
# pretrained model, it should return a zero vector; otherwise, it returns the
# corresponding word vector from the word2vec dictionary.
def w2v(word2vec, token):
    word_vector = np.zeros(300,)

    # [YOUR CODE HERE]
    if token in word2vec.keys():
        word_vector = word2vec[token]
    return word_vector


# Function: string2vec(word2vec, user_input)
# word2vec: The pretrained Word2Vec model
# user_input: A string of arbitrary length
# Returns: A 300-dimensional averaged Word2Vec embedding for that string
#
# This function preprocesses the input string, tokenizes it using get_tokens, extracts a word embedding for
# each token in the string, and averages across those embeddings to produce a
# single, averaged embedding for the entire input.
def string2vec(word2vec, user_input):
    embedding = np.zeros(300,)

    # [YOUR CODE HERE]
    # preprocess input
    text = preprocessing(user_input)
    # get tokens from input
    tokens = get_tokens(text)
    # extract word embedding for each element of the list
    word_embeddings = []
    for token in tokens:
        word_embeddings.append(w2v(word2vec, token))
    # make the average
    embedding = np.mean(word_embeddings, axis=0)
    return embedding


# Function: instantiate_models()
# This function does not take any input
# Returns: Three instantiated machine learning models
#
# This function instantiates the three imported machine learning models, and
# returns them for later downstream use.  You do not need to train the models
# in this function.
def instantiate_models():
    logistic = None
    svm = None
    mlp = None

    # [YOUR CODE HERE]
    logistic = LogisticRegression(random_state=100)
    svm = LinearSVC(random_state=100)
    mlp = MLPClassifier(random_state=100)

    return logistic, svm, mlp


# Function: train_model(model, word2vec, training_documents, training_labels)
# model: An instantiated machine learning model
# word2vec: A pretrained Word2Vec model
# training_data: A list of training documents
# training_labels: A list of integers (all 0 or 1)
# Returns: A trained version of the input model
#
# This function trains an input machine learning model using averaged Word2Vec
# embeddings for the training documents.
def train_model(model, word2vec, training_documents, training_labels):
    # Write your code here:
    emb_vectors = []
    for document in training_documents:
        emb_vectors.append(string2vec(word2vec, document))
    model = model.fit(emb_vectors, training_labels)
    return model


# Function: test_model(model, word2vec, training_documents, training_labels)
# model: An instantiated machine learning model
# word2vec: A pretrained Word2Vec model
# test_data: A list of test documents
# test_labels: A list of integers (all 0 or 1)
# Returns: Precision, recall, F1, and accuracy values for the test data
#
# This function tests an input machine learning model by extracting features
# for each preprocessed test document and then predicting an output label for
# that document.  It compares the predicted and actual test labels and returns
# precision, recall, f1, and accuracy scores.
def test_model(model, word2vec, test_documents, test_labels):
    precision = None
    recall = None
    f1 = None
    accuracy = None

    # Write your code here:
    emb_vectors = []
    for document in test_documents:
        emb_vectors.append(string2vec(word2vec, document))
    # predict the labels
    predicted_test_labels = model.predict(emb_vectors)
    # get the metrics
    precision = precision_score(test_labels, predicted_test_labels)
    recall = recall_score(test_labels, predicted_test_labels)
    f1 = f1_score(test_labels, predicted_test_labels)
    accuracy = accuracy_score(test_labels, predicted_test_labels)
    return precision, recall, f1, accuracy


# Function: count_words(user_input)
# user_input: A string of arbitrary length
# Returns: An integer value
#
# This function counts the number of words in the input string.
def count_words(user_input):
    num_words = 0
    # [YOUR CODE HERE]
    tokens = nltk.tokenize.word_tokenize(user_input)
    tokens = list(filter(lambda x: x not in string.punctuation, tokens))
    num_words = len(tokens)
    return num_words

# Function: words_per_sentence(user_input)
# user_input: A string of arbitrary length
# Returns: A floating point value
#
# This function computes the average number of words per sentence
def words_per_sentence(user_input):
    wps = 0.0
    # [YOUR CODE HERE]
    sentences = nltk.tokenize.sent_tokenize(user_input)
    words = []
    for sentence in sentences:
        words.append(count_words(sentence))
    wps = np.mean(words)
    return wps


# Function: get_pos_tags(user_input)
# user_input: A string of arbitrary length
# Returns: A list of (token, POS) tuples
#
# This function tags each token in the user_input with a Part of Speech (POS) tag from Penn Treebank.
def get_pos_tags(user_input):
    tagged_input = []
    # [YOUR CODE HERE]
    tokens = nltk.tokenize.word_tokenize(user_input)
    tagged_input = nltk.pos_tag(tokens)
    return tagged_input


# Function: get_pos_categories(tagged_input)
# tagged_input: A list of (token, POS) tuples
# Returns: Seven integers, corresponding to the number of pronouns, personal
#          pronouns, articles, past tense verbs, future tense verbs,
#          prepositions, and negations in the tagged input
#
# This function counts the number of tokens corresponding to each of six POS tag
# groups, and returns those values.  The Penn Treebag tags corresponding that
# belong to each category can be found in Table 2 of the project statement.
def get_pos_categories(tagged_input):
    num_pronouns = 0
    num_prp = 0
    num_articles = 0
    num_past = 0
    num_future = 0
    num_prep = 0

    # [YOUR CODE HERE]
    for (token,POS) in tagged_input:
        if(POS=='PRP' or POS=='WP' or POS=='PRP$' or POS=='WP$'):
            num_pronouns += 1
        if(POS=='PRP'):
            num_prp += 1
        if(POS=='DT'):
            num_articles += 1
        if(POS=='VBD' or POS=='VBN'):
            num_past += 1
        if (POS == 'MD'):
            num_future += 1
        if (POS == 'IN'):
            num_prep += 1
    return num_pronouns, num_prp, num_articles, num_past, num_future, num_prep


# Function: count_negations(user_input)
# user_input: A string of arbitrary length
# Returns: An integer value
#
# This function counts the number of negation terms in a user input string
def count_negations(user_input):
    num_negations = 0
    # [YOUR CODE HERE]
    negations = ['no', 'not', 'never']
    tokens = nltk.tokenize.word_tokenize(user_input)
    for token in tokens:
        if (token in negations or 'n\'t' in token):
            num_negations += 1
    return num_negations


# Function: summarize_analysis(num_words, wps, num_pronouns, num_prp, num_articles, num_past, num_future, num_prep, num_negations)
# num_words: An integer value
# wps: A floating point value
# num_pronouns: An integer value
# num_prp: An integer value
# num_articles: An integer value
# num_past: An integer value
# num_future: An integer value
# num_prep: An integer value
# num_negations: An integer value
# Returns: A list of three strings
#
# This function identifies the three most informative linguistic features from
# among the input feature values, and returns the psychological correlates for
# those features.  num_words and/or wps should be included if, and only if,
# their values exceed predetermined thresholds.  The remainder of the three
# most informative features should be filled by the highest-frequency features
# from among num_pronouns, num_prp, num_articles, num_past, num_future,
# num_prep, and num_negations.
def summarize_analysis(num_words, wps, num_pronouns, num_prp, num_articles, num_past, num_future, num_prep, num_negations):
    informative_correlates = []

    # Creating a reference dictionary with keys = linguistic features, and values = psychological correlates.
    # informative_correlates should hold a subset of three values from this dictionary.
    # DO NOT change these values for autograder to work correctly
    psychological_correlates = {}
    psychological_correlates["num_words"] = "Talkativeness, verbal fluency"
    psychological_correlates["wps"] = "Verbal fluency, cognitive complexity"
    psychological_correlates["num_pronouns"] = "Informal, personal"
    psychological_correlates["num_prp"] = "Personal, social"
    psychological_correlates["num_articles"] = "Use of concrete nouns, interest in objects/things"
    psychological_correlates["num_past"] = "Focused on the past"
    psychological_correlates["num_future"] = "Future and goal-oriented"
    psychological_correlates["num_prep"] = "Education, concern with precision"
    psychological_correlates["num_negations"] = "Inhibition"

    # Set thresholds
    num_words_threshold = 100
    wps_threshold = 20

    # [YOUR CODE HERE]
    ties = {'num_pronouns':6, 'num_prp':5, 'num_articles':4, 'num_past':3, 'num_future':2, 'num_prep':1, 'num_negations':0}
    if num_words > num_words_threshold:
        informative_correlates.append(psychological_correlates["num_words"])
    if wps > wps_threshold:
        informative_correlates.append(psychological_correlates["wps"])
    info = [(num_pronouns, 'num_pronouns'), (num_prp, 'num_prp'), (num_articles, 'num_articles'), (num_past, 'num_past'), (num_future, 'num_future'), (num_prep, 'num_prep'), (num_negations, 'num_negations')]
    while (len(informative_correlates)<3):
        max_element = max(info)
        maxima = []
        for element in info:
            if element[0]==max_element[0]:
                maxima.append(element)
        for element in maxima:
            if ties[max_element[1]] < ties[element[1]]:
                max_element = element
        info.remove(max_element)
        informative_correlates.append(psychological_correlates[max_element[1]])
    return informative_correlates


# -------------------------- New in Project Part 4 --------------------------
# Function: welcome_state
# This function does not take any input
# Returns: A string indicating the next state
#
# This function implements the chatbot's welcome states.  Feel free to customize
# the welcome message!  In this state, the chatbot greets the user.
def welcome_state():
    # Display a welcome message to the user
    # *** Replace the line below with your updated welcome message from Project Part 1 ***
    print("Welcome! I am Julia, a simple Chatbot at your disposal.")
    return "get_name"


# Function: get_name_state
# This function does not take any input
# Returns: A string indicating the next state
#
# This function implements a state that requests the user's name and then
# processes the user's response to extract the name.
def get_name_state():
    # Request the user's name and accept a user response of arbitrary length
    user_input = input("What is your name?\n")

    # Extract the user's name
    name = extract_user_info(user_input)

    # Show name and thank the user
    user_input = print(f"Thanks {name}!")

    return "sentiment_analysis"


# Function: sentiment_analysis_state
# model: The trained classification model used for predicting sentiment (best one)
# word2vec: The word2vec dictionary
# first_time (bool): indicates whether the state is active for the first time. HINT: use this parameter to determine next state.
# Returns: A string indicating the next state
#
# This function implements a state that asks the user for input and predicts their sentiment
def sentiment_analysis_state(model, word2vec, first_time=False):
    # Check the user's current sentiment
    user_input = input("What do you want to talk about today? Tell me something in detail...\n")

    # Predict user's sentiment
    w2v_test = string2vec(word2vec, user_input)

    label = mlp.predict(w2v_test.reshape(1, -1)) # Use this if you select one of the other models (swap mlp for svm, etc.)

    if label == 0:
        print("Hmm, it seems like you're feeling a bit down.")
    elif label == 1:
        print("It sounds like you're in a positive mood!")
    else:
        print("Hmm, that's weird.  My classifier predicted a value of: {0}".format(label))
    if first_time:
        return "stylistic_analysis"
    else:
        return "check_next"


# Function: stylistic_analysis_state
# This function does not take any arguments
# Returns: A string indicating the next state
#
# This function implements a state that asks the user what's on their mind, and
# then analyzes their response to identify informative linguistic correlates to
# psychological status.
def stylistic_analysis_state():
    user_input = input("Well, let's now perform a quick stylistic analysis. What's on your mind today?\n")

    num_words = count_words(user_input)
    wps = words_per_sentence(user_input)
    pos_tags = get_pos_tags(user_input)
    num_pronouns, num_prp, num_articles, num_past, num_future, num_prep = get_pos_categories(pos_tags)
    num_negations = count_negations(user_input)

    # Uncomment the code below to view your output from each individual function
    # print("num_words:\t{0}\nwps:\t{1}\npos_tags:\t{2}\nnum_pronouns:\t{3}\nnum_prp:\t{4}"
    #      "\nnum_articles:\t{5}\nnum_past:\t{6}\nnum_future:\t{7}\nnum_prep:\t{8}\nnum_negations:\t{9}".format(
    #    num_words, wps, pos_tags, num_pronouns, num_prp, num_articles, num_past, num_future, num_prep, num_negations))

    # Generate a stylistic analysis of the user's input
    informative_correlates = summarize_analysis(num_words, wps, num_pronouns,
                                                num_prp, num_articles, num_past,
                                                num_future, num_prep, num_negations)
    print("Thanks! Based on my stylistic analysis, I've identified the following psychological correlates in your response:")
    for correlate in informative_correlates:
        print("- {0}".format(correlate))

    return "check_next"


# Function: check_next_state()
# This function does not take any input
# Returns: A string indicating the next state
#
# This function implements a state that checks to see what the user would like
# to do next.  The user should be able to indicate that they would like to quit
# (in which case the state should be "quit"), redo the sentiment analysis
# ("sentiment_analysis"), or redo the stylistic analysis ("stylistic_analysis").
def check_next_state():
    next_state = ""
    # [YOUR CODE HERE]
    user_input = input("What would you like to do next? Type in one of the following actions:\n\t- quit\n\t- sentiment analysis\n\t- stylistic analysis\n")
    processed_input = preprocessing(user_input)
    if "quit" in processed_input and ("stylistic" not in processed_input) and ("sentiment" not in processed_input):
        next_state = "quit"
    elif "stylistic" in processed_input and ("quit" not in processed_input) and ("sentiment" not in processed_input):
        next_state = "stylistic_analysis"
    elif "sentiment" in processed_input and ("quit" not in processed_input) and ("stylistic" not in processed_input):
        next_state = "sentiment_analysis"
    else:
        print("Sorry, I was not able to understand your willings! Please try again.")
        next_state = "check_next"
    return next_state


# Function: run_chatbot
# model: A trained classification model
# word2vec: The pretrained Word2Vec dictionary (leave empty if not using word2vec based model)
# Returns: This function does not return any values
#
# This function implements the main chatbot system --- it runs different
# dialogue states depending on rules governed by the internal dialogue
# management logic, with each state handling its own input/output and internal
# processing steps.  The dialogue management logic should be implemented as
# follows:
# welcome_state() (IN STATE) -> get_info_state() (OUT STATE)
# get_info_state() (IN STATE) -> sentiment_analysis_state() (OUT STATE)
# sentiment_analysis_state() (IN STATE) -> stylistic_analysis_state() (OUT STATE - First time sentiment_analysis_state() is run)
#                                    check_next_state() (OUT STATE - Subsequent times sentiment_analysis_state() is run)
# stylistic_analysis_state() (IN STATE) -> check_next_state() (OUT STATE)
# check_next_state() (IN STATE) -> sentiment_analysis_state() (OUT STATE option 1) or
#                                  stylistic_analysis_state() (OUT STATE option 2) or
#                                  terminate chatbot
def run_chatbot(model, word2vec):
    # [YOUR CODE HERE]
    next_state = welcome_state()
    first_time = True
    while next_state != "quit":
        if next_state=="stylistic_analysis":
            first_time = False
        try:
            next_state = eval(next_state + '_state()')
        except:
            try:
                next_state = eval(next_state + '_state(model,word2vec,first_time)')
            except:
                next_state = eval(next_state + '_state(model,word2vec)')
    print("See you soon!")
    pass
# ----------------------------------------------------------------------------




# Use this main function to test your code. Sample code is provided to assist with the assignment,
# feel free to change/remove it. In project components, this function might be graded, see rubric for details.
if __name__ == "__main__":

    # Set things up ahead of time by training the TfidfVectorizer and Naive Bayes model
    documents, labels = load_as_list("dataset.csv")

    # Load the Word2Vec representations so that you can make use of it later
    word2vec = load_w2v(EMBEDDING_FILE)

    # Instantiate and train the machine learning models
    logistic, svm, mlp = instantiate_models()
    logistic = train_model(logistic, word2vec, documents, labels)
    svm = train_model(svm, word2vec, documents, labels)
    mlp = train_model(mlp, word2vec, documents, labels)

    # Uncomment the line below to test out the w2v() function.  Make sure to try a few words that are unlikely to
    # exist in its dictionary (e.g., "covid") to see how it handles those.
    # print("Word2Vec embedding for {0}:\t{1}".format("vaccine", w2v(word2vec, "vaccine")))

    # Test the machine learning models to see how they perform on the small test set provided.
    # Write a classification report to a CSV file with this information.
    # Loading the dataset
    test_documents, test_labels = load_as_list("test.csv")
    models = [logistic, svm, mlp]
    model_names = ["Logistic Regression", "SVM", "Multilayer Perceptron"]
    outfile = open("classification_report.csv", "w", newline='\n')
    outfile_writer = csv.writer(outfile)
    outfile_writer.writerow(["Name", "Precision", "Recall", "F1", "Accuracy"]) # Header row
    i = 0
    while i < len(models): # Loop through other results
        p, r, f, a = test_model(models[i], word2vec, test_documents, test_labels)
        if models[i] == None: # Models will be None if functions have not yet been implemented
            outfile_writer.writerow([model_names[i],"N/A", "N/A", "N/A", "N/A"])
        else:
            outfile_writer.writerow([model_names[i], p, r, f, a])
        i += 1
    outfile.close()

    # For reference, let us also compute the accuracy for the Naive Bayes model from Project Part 1
    # Fill in the code templates from your previous submission and uncomment the code below
    # vectorizer, tfidf_train = vectorize_train(documents)
    # lexicon = [preprocessing(d) for d in test_documents]
    # tfidf_test = vectorizer.transform(lexicon)
    # naive = train_nb_model(tfidf_train, labels)
    # predictions = naive.predict(tfidf_test.toarray())
    # acc = np.sum(np.array(test_labels) == predictions) / len(test_labels)
    # print("Naive Bayes Accuracy:", acc)

    # Reference code to run the chatbot
    # Replace MLP with your best performing model
    run_chatbot(mlp, word2vec)