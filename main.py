from Parser import Parser
import util
import os
import math
from textblob import TextBlob as tb
import sys
import nltk
class VectorSpace:
    """ A algebraic model for representing text documents as vectors of identifiers. 
    A document is represented as a vector. Each dimension of the vector corresponds to a 
    separate term. If a term occurs in the document, then the value in the vector is non-zero.
    """

    #Collection of document term vectors(TF)
    TFdocumentVectors = []

    #Collection of document term vectors(TF-IDF)
    TFIDFdocumentVectors = []

    #Mapping of vector index to keyword
    vectorKeywordIndex=[]

    #IDF weights of each term
    idf=[]

    #Processed documents (Stemming & Removing Stop Words)
    docList=[]

    #remove duplicate term in the same document
    uniList=[]

    #Tidies terms
    parser=None

    #the number of documents
    length=0

    #the number of unique term in all documents
    wordlength=0

    def __init__(self, documents=[]):
        self.TFdocumentVectors=[]
        self.TFIDFdocumentVectors=[]
        self.parser = Parser()
        if(len(documents)>0):
            self.build(documents)

    def build(self,documents):
        """ Create the vector space for the passed document strings """
        self.length=len(documents)

        for i in documents:
            wordList = self.parser.tokenise(i)
            wordList = self.parser.removeStopWords(wordList)
            self.uniList.append(util.removeDuplicates(wordList))
            self.docList.append(wordList)

        self.vectorKeywordIndex = self.getVectorKeywordIndex(documents)
        for document in self.docList:
            tfidfvec,tfvec=self.makeVector(document) 
            self.TFIDFdocumentVectors.append(tfidfvec)
            self.TFdocumentVectors.append(tfvec)        

    def getVectorKeywordIndex(self, documentList):
        """ create the keyword associated to the position of the elements within the document vectors """

        flatten = lambda t: [item for sublist in t for item in sublist]
        vocabularyList = flatten(self.uniList)
        uniqueVocabularyList = util.removeDuplicates(vocabularyList)

        vectorIndex={}
        offset=0
        #Associate a position with the keywords which maps to the dimension on the vector used to represent this word
        for word in uniqueVocabularyList:
            vectorIndex[word]=offset
            offset+=1
            self.idf.append(math.log(self.length/ (1 + sum(1 for blob in self.uniList if word in blob))))

        self.wordlength=len(uniqueVocabularyList)
        return vectorIndex  #(keyword:position)


    def makeVector(self, wordString):
        """ @pre: unique(vectorIndex) """

        #Initialise vector with 0's
        vector = [0] * self.wordlength

        wordList = wordString

        for word in wordList:
            vector[self.vectorKeywordIndex[word]] += 1; #Term Count


        length = len(wordList)
        return [item*df/length for item,df in zip(vector,self.idf)],[item/length for item in vector]




    def buildQueryVector(self, termList):
        """ convert query string into a term vector """
        wordList = self.parser.tokenise(" ".join(tb(" ".join(termList)).words))
        wordList = self.parser.removeStopWords(wordList)
        query1,query2 = self.makeVector(wordList)
        return query1,query2





    def search(self,searchList):
        """ search for documents that match based on a list of terms """
        queryVector1, queryVector2 = self.buildQueryVector(searchList)
        ratings1 = [util.cosine(queryVector2, documentVector) for documentVector in self.TFdocumentVectors]
        ratings2 = [util.eudist(queryVector2, documentVector) for documentVector in self.TFdocumentVectors]
        ratings3 = [util.cosine(queryVector1, documentVector) for documentVector in self.TFIDFdocumentVectors]
        ratings4 = [util.eudist(queryVector1, documentVector) for documentVector in self.TFIDFdocumentVectors]
        return ratings1,ratings2,ratings3,ratings4

    def feedbacksearch(self,searchList,feedbackvec):
        queryVector1, queryVector2 = self.buildQueryVector(searchList)
        queryvec = [i+0.5*t for i,t in zip(queryVector1,feedbackvec)]
        ratings3 = [util.cosine(queryvec, documentVector) for documentVector in self.TFIDFdocumentVectors]
        return ratings3

if __name__ == '__main__':


    #argument parse
    arg_str = ' '.join(sys.argv[2:])

    #test data
    documents=[]
    allFiles = os.listdir('EnglishNews')
    for i in allFiles:
        file=open(os.path.join("EnglishNews", i),encoding="utf-8")
        text=" ".join(tb(file.read()).words)
        documents.append(text)
    file.close()

    vectorSpace = VectorSpace(documents)

    #link NewsIDs with scores
    result1,result2,result3,result4 = vectorSpace.search([arg_str])

    ranked1 = sorted(zip(allFiles,result1), key=lambda item: item[1],reverse=True)
    ranked2 = sorted(zip(allFiles,result2), key=lambda item: item[1],reverse=True)
    ranked3 = sorted(zip(allFiles,result3), key=lambda item: item[1],reverse=True)
    ranked4 = sorted(zip(allFiles,result4), key=lambda item: item[1],reverse=True)
    print("--------------------------------\nTF weighting + Cosine Similarity\n\nNewsID          Score\n----------      --------")
    for i in range(10):
        print("{}      {}".format(ranked1[i][0][:-4],ranked1[i][1]))
    print()

    print("--------------------------------\nTF weighting + Euclidean Distance\n\nNewsID          Score\n----------      --------")
    for i in range(10):
        print("{}      {}".format(ranked2[i][0][:-4],ranked2[i][1]))
    print()

    print("--------------------------------\nTF-IDF weighting + Cosine Similarity\n\nNewsID          Score\n----------      --------")
    for i in range(10):
        print("{}      {}".format(ranked3[i][0][:-4],ranked3[i][1]))
    print()

    print("--------------------------------\nTF-IDF weighting + Euclidean Distance\n\nNewsID          Score\n----------      --------")
    for i in range(10):
        print("{}      {}".format(ranked4[i][0][:-4],ranked4[i][1]))
    print()


#############################Relevance Feedback######################################
    
    file=open(os.path.join("EnglishNews", ranked3[0][0]),encoding="utf-8")
    text = " ".join(tb(file.read()).words)
    file.close()
    text = nltk.word_tokenize(text)
    pos_tagged = nltk.pos_tag(text)
    nnvb = [i[0] for i in pos_tagged if i[1]=='NN' or i[1]=='NNS' or i[1]=='NNP' or i[1]=='NNPS' or i[1]=='VB' or i[1]=='VBD' or i[1]=='VBG' or i[1]=='VBN' or i[1]=='VBP' or i[1]=='VBZ']
    
    wordList = vectorSpace.parser.tokenise(" ".join(nnvb))
    wordList = vectorSpace.parser.removeStopWords(wordList)
    wordList = list(set((item for item in wordList)))

    tfidfvec,tfvec=vectorSpace.makeVector(wordList) 
    result3 = vectorSpace.feedbacksearch([arg_str],tfidfvec)
    ranked3 = sorted(zip(allFiles,result3), key=lambda item: item[1],reverse=True)
    print("--------------------------------\nFeedback Queries + TF-IDF weighting + Cosine Similarity\n\nNewsID          Score\n----------      --------")
    for i in range(10):
        print("{}      {}".format(ranked3[i][0][:-4],ranked3[i][1]))
    print()