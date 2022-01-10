import os
import glob
import sys
from operator import itemgetter  # for sort
import _pickle as cPickle
from numpy import loadtxt
import pLSA

STOP_WORDS_SET = set()


def print_topic_word_distribution(corpus, number_of_topics, topk, filepath, filepath1):
    """
    Print topic-word distribution to file and list @topk most probable words for each topic
    """
    print  (    "Writing topic-word distribution to file: " + filepath)
    V = len(corpus.vocabulary)  # size of vocabulary
    assert (topk < V)
    f = open(filepath, "w")
    f1 = open(filepath1, "w")
    pLSA_matr = []

    for k in range(number_of_topics):
        indx = []
        probs = []
        word_prob = corpus.topic_word_prob[k, :]
        word_index_prob = []
        for i in range(V):
            word_index_prob.append([i, word_prob[i]])
        word_index_prob = sorted(word_index_prob, key=itemgetter(1), reverse=True)  # sort by word count
        f.write("Topic #" + str(k) + ":\n")
        f1.write("Topic #" + str(k) + ":\n")

        for i in range(topk):
            index = word_index_prob[i][0]
            f.write(corpus.vocabulary[index] + " ")
            f1.write(str( word_index_prob[i][1])  + " ")
            indx.append(index)
            probs.append(word_index_prob[i][1])
        f.write("\n")
        f1.write("\n")
        pLSA_matr.append(list(zip(indx,probs)))
    f.close()
    print(pLSA_matr)
    return pLSA_matr


def print_document_topic_distribution(corpus, number_of_topics, topk, filepath):
    """
    Print document-topic distribution to file and list @topk most probable topics for each document
    """
    print (
    "Writing document-topic distribution to file: " + filepath)
    assert (topk <= number_of_topics)
    f = open(filepath, "w")
    D = len(corpus.documents)  # number of documents
    for d in range(D):
        topic_prob = corpus.document_topic_prob[d, :]
        topic_index_prob = []
        for i in range(number_of_topics):
            topic_index_prob.append([i, topic_prob[i]])
        topic_index_prob = sorted(topic_index_prob, key=itemgetter(1), reverse=True)
        f.write("Document #" + str(d) + ":\n")
        for i in range(topk):
            index = topic_index_prob[i][0]
            f.write("topic" + str(index) + " ")
        f.write("\n")

    f.close()


def main():
    # load stop words list from file
    stopwordsfile = open("stopwords.txt", "r")
    for word in stopwordsfile:  # a stop word in each line
        word = word.replace("\n", '')
        word = word.replace("\r\n", '')
        STOP_WORDS_SET.add(word)

    corpus = pLSA.Corpus()  # instantiate corpus
    # iterate over the files in the directory.
    document_paths = ['C:/Users/sunny/Downloads/books/bio']
    # document_paths = ['./test/']
    for document_path in document_paths:
        for document_file in glob.glob(os.path.join(document_path, '*.txt')):
            document = pLSA.Document(document_file)  # instantiate document
            document.split(STOP_WORDS_SET)  # tokenize
            corpus.add_document(document)  # push onto corpus documents list

    corpus.build_vocabulary()
    print( "Vocabulary size:" + str(len(corpus.vocabulary)))
    print ("Number of documents:" + str(len(corpus.documents)))

    number_of_topics = 10
    max_iterations = 2
    corpus.new_plsa(number_of_topics, max_iterations)

    # print corpus.document_topic_prob
    # print corpus.topic_word_prob
    # cPickle.dump(corpus, open('./models/corpus.pickle', 'w'))

    print_topic_word_distribution(corpus, number_of_topics, 10, "./topic-word.txt", "./topic-word1.txt")
    print_document_topic_distribution(corpus, number_of_topics, 10, "./document-topic.txt")
def get_main_topics(path):
    text_file = open(path, "r")
    lines = text_file.readlines()
    print(lines)
    text_file.close()
# if __name__ == "__main__":
#     main()