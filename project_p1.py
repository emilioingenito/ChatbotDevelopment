# CS421: Natural Language Processing
# University of Illinois at Chicago
# Fall 2022
# Project Part 1
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
import string
import re


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
    #get tokens
    modified_input = get_tokens(user_input)
    tokenized = []

    #remove punctuation tokens
    for tkn in modified_input:
        if(tkn not in string.punctuation):
            tokenized.append(tkn)

    #convert to lowercase
    tokenized = list(map(lambda i: i.lower(), tokenized))
    tokenized = ' '.join(tokenized)
    return tokenized


# Function: vectorize_train, see project statement for more details
# training_documents: A list of strings
# Returns: An initialized TfidfVectorizer model, and a document-term matrix, dtype: scipy.sparse.csr.csr_matrix
def vectorize_train(training_documents):
    # Initialize the TfidfVectorizer model and document-term matrix
    vectorizer = TfidfVectorizer()
    tfidf_train = []
    # [YOUR CODE HERE]
    for doc in training_documents:
        tfidf_train.append(preprocessing(doc))
    print(tfidf_train)
    #apply transform
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
    prepr_input = []
    prepr_input.append(preprocessing(user_input))
    tfidf_test = vectorizer.transform(prepr_input)
    #print(tfidf_test)
    return tfidf_test


# Function: train_model(training_documents, training_labels)
# training_data: A sparse TfIDF document-term matrix, dtype: scipy.sparse.csr.csr_matrix
# training_labels: A list of integers (0 or 1)
# Returns: A trained model
def train_model(training_data, training_labels):
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


# Use this main function to test your code. Sample code is provided to assist with the assignment,
# feel free to change/remove it. In project components, this function might be graded, see rubric for details.
if __name__ == "__main__":
    # Display a welcome message to the user, and accept a user response of arbitrary length
    user_input = input("Welcome! I am Julia, a simple Chatbot at your disposal.\nWhat is your name?\n")

    # Extract the user's name
    name = extract_user_info(user_input)

    # Query the user for a response
    user_input = input(f"Thanks {name}!  What do you want to talk about today?\n")

    # Set things up ahead of time by training the TfidfVectorizer and Naive Bayes model
    documents, labels = load_as_list("dataset.csv")
    vectorizer, tfidf_train = vectorize_train(documents)
    nb_model = train_model(tfidf_train, labels)

    # Predict whether the user's sentiment is positive or negative
    tfidf_test = vectorize_test(vectorizer, user_input)

    label = get_model_prediction(nb_model, tfidf_test)
    if label == 0:
        print("Hmm, it seems like you're feeling a bit down.")
    elif label == 1:
        print("It sounds like you're in a positive mood!")
    else:
        print("Hmm, that's weird.  My classifier predicted a value of: {0}".format(label))
