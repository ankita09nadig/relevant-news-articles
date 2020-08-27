from flask import Flask, render_template, request
from flask_mysqldb import MySQL
import yaml
import html

# The code for the main algorithm for the Search
# For the core algorithm (Positional Index) to run,
# the dependencies listed below need to be imported


import numpy as np
import os
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
from natsort import natsorted
import string
import codecs
import math

app = Flask(__name__)

db = yaml.load(open('db.yaml'))

app.config['MYSQL_HOST'] = db['mysql_host']
app.config['MYSQL_USER'] = db['mysql_user']
app.config['MYSQL_PASSWORD'] = db['mysql_password']
app.config['MYSQL_DB'] = db['mysql_db']

mysql = MySQL(app)

# poll data
poll_data_1 = {
    'question': 'How often do you do you read the news?',
    'fields': ['Every day', 'Two times a week', 'Once a week', 'Once a month', 'Never']
}
poll_data_2 = {
    'question': 'Which continent do you belong to?',
    'fields': ['Asia', 'Africa', 'Europe', 'Australia', 'North America', 'South America', 'Antarctica']
}
poll_data_3 = {
    'question': 'Gender',
    'fields': ['Male', 'Female']
}
poll_data_4 = {
    'question': 'We would first of all like to know about some of your interests. Which of the following are you '
                'interested in?',
    'fields': ["Technology", "Finance", "Sports", "Politics", "Entertainment"]
}


# asks each user to login
@app.route('/')
@app.route("/login", methods=['GET', 'POST'])
def login():
    global email, password
    if request.method == 'POST':
        user_details = request.form
        email = user_details['email']
        password = user_details['password']

        cursor = mysql.connection.cursor()
        temp = "\'" + email + "\';"
        query = "SELECT * from login WHERE email = " + temp
        cursor.execute(query)
        result = cursor.fetchone()
        if result:
            if result[2] == user_details['password']:
                return render_template('index.html', data=poll_data_1, data1=poll_data_2, data2=poll_data_3,
                                       data3=poll_data_4)
            else:
                return render_template('login_failure.html')

        else:
            cursor = mysql.connection.cursor()
            cursor.execute("INSERT INTO login(email, password) VALUES(%s, %s)",
                           (email, password))
            mysql.connection.commit()
            return render_template('index.html', data=poll_data_1, data1=poll_data_2, data2=poll_data_3,
                                   data3=poll_data_4)
        cursor.close()
    return render_template('login.html')




# The code for the main algorithm for the Search
# the dependencies listed below need to be imported

# python 3.0 and above
# libraries needed are : numpy, os, NLTK, stemmer
import numpy as np
import os
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
from natsort import natsorted
import string
import codecs
import math




datasetSize = 10067


"""
    Reading the text file and returning a single string of all the data
"""
def read_file(filename):
    with codecs.open(filename, 'r', "utf-8") as f:
        content = f.read()

    f.close()

    final_string = ""

    tokens = content.split("\n")

    # all tokens are concatenated to form a string of information
    for token in tokens:
        token = token.strip()
        final_string += token + " "

    return final_string



"""
    Tokenizing, stemming and cleaning of data
"""
def preprocessing(final_string):
    # Tokenize.
    # The tweetTokenizer libary is used to transform the words into tokens.
    tokenizer = TweetTokenizer()
    token_list = tokenizer.tokenize(final_string)

    # Remove punctuations.
    # maketrans() returns a translation table that maps each character in the
    # first argument into the character at the same position in the second argument string.
    # Then this table is passed to the translate() function.
    # When 3 arguments are passed then each character in the third argument is mapped to None.

    table = str.maketrans('', '', "\t")
    token_list = [str.translate(table) for str in token_list]

    # Replacing inverted commas with nothing
    punctuations = (string.punctuation).replace("'", "")

    trans_table = str.maketrans('', '', punctuations)
    stripped_words = [str.translate(trans_table) for str in token_list]
    token_list = [str for str in stripped_words if str]

    # Change to lowercase.
    for word in range(len(token_list)):
        token_list[word] = token_list[word].lower()

    return token_list


"""
    Computing the Core algorithm
    It has 3 major parts - Positional Index, Inverted Index and tf-idf Ranking algorithm
"""
def process(input_str):
    input_string = input_str

    orderedDocs = []

    intersect_list = set()
    input_str = input_str.split()

    # Computing Positional Index
    if len(input_str) == 1:
        input_str[0] = stemmer.stem(input_str[0])
        if input_str[0] not in pos_index.keys():
            return None
        data = pos_index[input_str[0]]
        file_list = data[1]
        file_set = set(file_list.keys())
        return file_set

    for index in range(len(input_str)):
        input_str[index] = stemmer.stem(input_str[index])

    for index in range(len(input_str) - 1):
        if (input_str[index] not in pos_index.keys()) or (input_str[index + 1] not in pos_index.keys()):
            continue
        data_current = pos_index[input_str[index]]
        data_next = pos_index[input_str[index + 1]]

        file_list_current = data_current[1]
        file_list_next = data_next[1]

        intersect_list.update(intersect(file_list_current, file_list_next))

    # Positional Index result added
    result = set(intersect_list)



    check = input_string.split()

    # The result is a set of integers representing the document IDs.
    # They are converted to string for proper output
    if result == None and len(check) == 1:
        return None

    # After the most relevant documents are found
    # We find the remaining documents which contain
    # the occurrence of either of the words that have been
    # searched using inverted index algorithm

    elif len(check) > 1:
        idfDict, docs, totalOccurDict = computeIDF(input_string)

        if result != None:
            for file in result:
                if file not in orderedDocs:
                    orderedDocs.append(file)

            if docs is None or idfDict is None:
                if orderedDocs == None:
                    return None
                else:
                    return orderedDocs

            else:
                docRankDict = {}

                input_str = input_string.split()
                for word in input_str:
                    word = stemmer.stem(word)
                    if (word not in pos_index.keys()):
                        continue

                    # getting the file and positions list of the term
                    pointer = pos_index[word]
                    file_list = pointer[1]

                    # Compute tf-idf score
                    for file in file_list.keys():
                        content = read_file(path_file + folder_name + "/article_" + str(file) + ".txt")
                        content = content.split()
                        totalWords = len(content)

                        tfScore = 1 + math.log10(len(file_list[file]) / totalWords)
                        idfScore = idfDict[word]
                        tfIdfScore = tfScore * idfScore
                        if file in docRankDict.keys():
                            docRankDict[file] += abs(tfIdfScore)
                        else:
                            docRankDict[file] = abs(tfIdfScore)

                # Sorting the documents according to tf-idf score in descending order
                sortedFiles = sorted(docRankDict.items(), key=lambda x: x[1], reverse=True)

                for file in sortedFiles:
                    if file[0] not in orderedDocs:
                        orderedDocs.append(file[0])

                return orderedDocs

        else:
            docRankDict = {}

            input_str = input_string.split()
            for word in input_str:

                word = stemmer.stem(word)
                if (word not in pos_index.keys()):
                    continue

                # getting the file and positions list of the term
                pointer = pos_index[word]
                file_list = pointer[1]

                # Compute tf-idf score
                for file in file_list.keys():
                    content = read_file(path_file + folder_name + "/article_" + str(file) + ".txt")
                    content = content.split()
                    totalWords = len(content)

                    tfScore = 1 + math.log10(len(file_list[file]) / totalWords)
                    idfScore = idfDict[word]
                    tfIdfScore = tfScore * idfScore
                    if file in docRankDict.keys():
                        docRankDict[file] += abs(tfIdfScore)
                    else:
                        docRankDict[file] = abs(tfIdfScore)


            # Sorting the documents according to tf-idf score in descending order
            sortedFiles = sorted(docRankDict.items(), key=lambda x: x[1], reverse=True)

            for file in sortedFiles:
                if file[0] not in orderedDocs:
                    orderedDocs.append(file[0])

            return orderedDocs
    else:
        docRankDict = {}
        word = stemmer.stem(input_string)

        pointer = pos_index[word]
        file_list = pointer[1]

        # Compute tf score because only one word is there in the search query
        # We don't need compute idf score.
        for file in file_list:
            content = read_file(path_file + folder_name + "/article_" + str(file) + ".txt")
            content = content.split()
            totalWords = len(content)
            tfScore = 1 + math.log10(len(file_list[file]) / totalWords)

            docRankDict[file] = tfScore

        # Sorting the documents according to tf score in descending order
        sortedFiles = sorted(docRankDict.items(), key=lambda x: x[1], reverse=True)


        for file in sortedFiles:
            if file[0] not in orderedDocs:
                orderedDocs.append(file[0])

        return orderedDocs
"""
 Used to find out documents for positional index calculation
"""
def intersect(file_list_1, file_list_2):
    intersect_list = {}
    intersect_set = set()
    f1 = set(file_list_1.keys())
    f2 = set(file_list_2.keys())

    common_files = set(f1).intersection(set(f2))

    for file in common_files:
        p1 = 0
        p2 = 0
        while p1 < len(file_list_1[file]):
            while p2 < len(file_list_2[file]):
                if file_list_1[file][p1] - file_list_2[file][p2] == -1:
                    # filePresent = False
                    if file in intersect_list.keys():
                        # filePresent = True
                        intersect_list[file] += file_list_2[file][p2]
                        intersect_set.add(file)
                    else:
                        intersect_list[file] = file_list_2[file][p2]
                        intersect_set.add(file)

                elif file_list_2[file][p2] > file_list_1[file][p1]:
                    break
                p2 += 1
            p1 += 1


    return intersect_set


"""
    Computing the IDF score for a word
"""
def computeIDF(input_string):
    docs = set()
    input_string = input_string.split()
    idfDict = {}
    totalOccurDict = {}

    # We carry out the computation of IDF value for each word.
    for word in input_string:
        word = stemmer.stem(word)
        if (word not in pos_index.keys()):
            continue
        pointer = pos_index[word]
        file_list = pointer[1]

        idfDict[word] = math.log10(datasetSize / len(file_list.keys()))

        totalOccurrences = 0
        for file in file_list.keys():
            totalOccurrences += len(file_list[file])
            docs.add(file)
        totalOccurDict[word] = totalOccurrences

    return idfDict, docs, totalOccurDict


# This code remains global so that when the file is executed the
# positional indexing is carried out first.
# Initialize the stemmer.
stemmer = PorterStemmer()

# Initialize the file number.
file_number = 0

# Initialize the dictionary.
pos_index = {}

# Initialize the file mapping (file_number -> file name).
file_map = {}

# We create the positional index for the Web Scraped folder.
folder_names = ["articles"]

# to change to current users file path of the project
path_file = "C:\Users\abhin\Desktop\RIT\Knowledge Processing Technologies - Spring 2020\TermProj\KPT_final_project\KPT\MEWS\demoapp"
for folder_name in folder_names:

    # Open files.
    # natsorted is a function that sorts the data according to meaning and not exactly lexicograhpically.
    file_names = natsorted(os.listdir(path_file + folder_name))

    # For every file.
    for file_name in file_names:

        # Read file contents.
        content = read_file(path_file + folder_name + "/" + file_name)

        # This is the list of words in order of the text.
        # 'preprocessing' function does some basic punctuation removal,
        # stopword removal etc.
        final_token_list = preprocessing(content)

        # For position and term in the tokens.
        for pos, term in enumerate(final_token_list):

            # First stem the term.
            term = stemmer.stem(term)

            # If term already exists in the positional index dictionary.
            if term in pos_index:

                # Increment total freq by 1.
                pos_index[term][0] = pos_index[term][0] + 1

                # Check if the term has already occurred in the document
                if file_number in pos_index[term][1]:
                    pos_index[term][1][file_number].append(pos)

                else:
                    pos_index[term][1][file_number] = [pos]



            # If term does not exist in the positional index dictionary

            else:

                # Initialize the list.
                pos_index[term] = []

                # Increment the frequency by 1.
                pos_index[term].append(1)

                # The postings list is initiated.
                pos_index[term].append({})

                # Add doc ID to postings list.
                pos_index[term][1][file_number] = [pos]

                # Map the file number to the file name.
        file_map[file_number] = folder_name + "/" + file_name

        # Increment the file number for next document
        file_number += 1





def question(search_val):
    input_str = search_val
    result = process(input_str)
    list_articles = []
    part_of_file = []

    # The result is a set of integers representing the document IDs.
    # They are converted to string for proper output
    if result == None:
        return []
    else:
        for file in result:
            list_articles.append(file_map[file])

        for i in range(0, len(list_articles)):
            part_of_file.append(open(path_file + list_articles[i], encoding="utf8").read().splitlines())
        return part_of_file


@app.route("/poll", methods=['GET', 'POST'])
def poll():
    vote = {'name': request.form['name'], 'Q1': request.form['q0'], 'Q2': request.form['q1'], 'Q3': request.form['q2'],
            'Q4': request.form.getlist('q3')}
    list_articles = []
    part_of_file = []
    categories = vote['Q4']

    for key in file_map:
        list_articles.append(file_map[key])

    final = []
    for i in range(0, len(list_articles)):
        part_of_file.append(open(path_file + list_articles[i], encoding="utf8").read().splitlines())

    for each_file in part_of_file:
        for category in categories:
            if category in each_file[3]:
                final.append(each_file)

    # for category in categories:
    #     if(category in part_of_file[0]):
    #         final.append()

    # part_of_file = question()
    return render_template('posts.html', list_articles=final, len=len(final))


@app.route("/autocomplete", methods=['GET', 'POST'])
def search():
    search_val = request.form['search']
    part_of_file = question(search_val)
    for p in part_of_file:
        html.unescape(p[0])
        html.unescape(p[1])
        html.unescape(p[2])

    return render_template('posts.html', list_articles=part_of_file, len=len(part_of_file))


if __name__ == '__main__':
    app.run(debug=True)
