import pandas as pd
import numpy as np
import os
import codecs
import joblib
import json
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import io

import hazm

from sentence_transformers import SentenceTransformer, util

from gensim.models import FastText
from gensim.test.utils import get_tmpfile
from soupsieve import match

normalizer = None
stop_words = None
tokenize = None
lemmatizer = None


def setNormalizers():
    global normalizer
    global stop_words
    global tokenize
    global lemmatizer

    normalizer = hazm.Normalizer()
    stop_words = codecs.open('stopwords.txt', 'r', 'utf-8').readlines()
    tokenize = hazm.word_tokenize
    lemmatizer = hazm.Lemmatizer()


def clean_data(document):
    # normalization and tokenization
    tokenized = tokenize(normalizer.normalize(document))
    tokenized = [token.lower() for token in tokenized if token.lower()
                 not in stop_words]  # deleting stop words
    tokenized = [lemmatizer.lemmatize(token)
                 for token in tokenized]  # lemmatization
    return tokenized


class BooleanRecommender():
    def __init__(self, document_list):
        self.document_list = document_list.copy()
        self.boolean_df = []
        self.doc_token = []
        self.column_list = set()

    def run(self):
        try:
            self.boolean_df = np.load("models_data\\booleanVector.npy")
            with open("models_data\\booleanColumns.txt", "r", encoding="UTF-8") as file:
                self.column_list = file.read().split("\n")
        except:
            for i in self.document_list.index:
                tokens = set(clean_data("\n".join([self.document_list.loc[i, ["title"]].iloc[0], self.document_list.loc[i, [
                             "paragraphs"]].iloc[0]])))  # pre processing and getting all tokens
                # saving tokens of each document
                self.doc_token.append(list(tokens))
                # updating all tokens on each doc iteration
                self.column_list.update(tokens)

            self.column_list = list(self.column_list)

            for doc in self.doc_token:
                # creating the boolean table
                self.boolean_df.append(
                    [1 if token in doc else 0 for token in self.column_list])
            np.save("models_data\\booleanVector.npy",
                    np.array(self.boolean_df))
            with open("models_data\\booleanColumns.txt", "w", encoding="utf-8") as file:
                file.write("\n".join(self.column_list))

    def recommend(self, query, k=10):
        query_tokens = list(set(clean_data(query)))
        doc_score = []
        for doc in self.boolean_df:  # getting document score based on contained queries
            score = 0
            for token in query_tokens:
                if token in self.column_list:
                    score += doc[self.column_list.index(token)]
            doc_score.append(score)

        # getting k highest scores
        similar = np.argpartition(doc_score, -k)[-k:]

        matched_documents = self.document_list.loc[similar, :]

        return matched_documents


class TfIdfRecommender():
    def __init__(self, document_list):
        self.document_list = document_list.copy()
        # preparing the vectorizer model
        self.vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 3))
        # self.document_vectors = None
        self.vocabulary = []

    def get_sim(self, query_vector, doc_vector):
        cosine_sim = cosine_similarity([query_vector, doc_vector])
        return cosine_sim[0][1]

    def run(self):
        try:
            np.load("models_data\\tfidfVectors.npy")
            self.vocabulary = self.vectorizer.get_feature_names_out()
        except:
            self.document_list["feature"] = self.document_list.apply(lambda x: " ".join(
                clean_data("\n".join([x["title"], x["paragraphs"]]))), axis=1)
            document_vectors = self.vectorizer.fit_transform(
                self.document_list["feature"]).toarray()  # getting vector of each document
            self.vocabulary = self.vectorizer.get_feature_names_out()
            # saving attained features
            self.document_list.drop(["feature"], axis=1)
            np.save("models_data\\tfidfVectors.npy", document_vectors)

    def expanded_recommend(self, query, k=10):
        document_vectors = np.load("models_data\\tfidfVectors.npy")
        query = " ".join(clean_data(query))
        query_vector = self.vectorizer.fit_transform(
            [query]).toarray()  # getting query vector
        query_features = self.vectorizer.get_feature_names_out()
        query_vector = [query_vector[0][np.where(query_features == word)[
            0][0]] if word in query_features else 0 for word in self.vocabulary]  # expanding the vector to the feature vector
        rate = np.array([self.get_sim(query_vector, document_vectors[i]) for i in range(
            document_vectors.shape[0])], dtype=float)  # calculating cosine similarity and rating

        similars = np.argpartition(rate, -k)[-k:]
        irrelevants = np.argpartition(rate, k)[:k]
        relevant_docs = np.array([document_vectors[x] for x in similars])
        irrelevant_docs = np.array([document_vectors[x] for x in irrelevants])
        modified_query = improve_query(
            relevant_docs, irrelevant_docs, query_vector)

        better_rate = np.array([self.get_sim(modified_query, document_vectors[i]) for i in range(
            document_vectors.shape[0])], dtype=float)  # calculating cosine similarity and rating
        better_similars = np.argpartition(better_rate, -k)[-k:]
        matched_documents = self.document_list.loc[better_similars, :]
        return matched_documents

    def recommend(self, query, k=10):
        query = " ".join(clean_data(query))
        query_vector = self.vectorizer.fit_transform(
            [query]).toarray()  # getting query vector
        query_features = self.vectorizer.get_feature_names_out()
        query_vector = [query_vector[0][np.where(query_features == word)[
            0][0]] if word in query_features else 0 for word in self.vocabulary]  # expanding the vector to the feature vector
        rate = np.array([self.get_sim(query_vector, self.document_vectors[i]) for i in range(
            self.document_vectors.shape[0])], dtype=float)  # calculating cosine similarity and rating
        similars = np.argpartition(rate, -k)[-k:]

        matched_documents = self.document_list.loc[similars, :]
        return matched_documents


class TransformerRecommender():
    def __init__(self, document_list):
        self.document_list = document_list.copy()
        # achieving the transformer model
        self.model = SentenceTransformer(
            'paraphrase-multilingual-MiniLM-L12-v2')
        self.document_vectors = []

    def run(self):

        self.document_list["feature"] = self.document_list.apply(
            lambda x: " ".join(clean_data(x["title"] + " " + x["paragraphs"])), axis=1)
        self.document_list["sentence"] = self.document_list["feature"].apply(
            lambda x: hazm.sent_tokenize(x))  # tokenizing the documents by sentence
        for i in self.document_list.index:
            self.document_vectors.append(self.model.encode(self.document_list.loc[i, [
                                         "sentence"]], convert_to_tensor=True))  # encoding each document

        self.document_list.drop(["feature", "sentence"], axis=1)

    def expanded_recommend(self, query, k=10):
        query = " ".join(clean_data(query))
        query_vector = self.model.encode(hazm.sent_tokenize(
            query), convert_to_tensor=True)  # encoding the query
        # calculating cosine similarity and getting the k highest scores
        rate = list([util.cos_sim(query_vector, vector)
                    for vector in self.document_vectors])
        similars = np.argpartition(rate, -k)[-k:]
        irrelevants = np.argpartition(rate, k)[:k]
        relevant_docs = np.array([self.document_vectors[x] for x in similars])
        irrelevant_docs = np.array(
            [self.document_vectors[x] for x in irrelevants])
        modified_query = improve_query(
            relevant_docs, irrelevant_docs, query_vector)
        # calculating cosine similarity and getting the k highest scores
        better_rate = list([util.cos_sim(modified_query, vector)
                           for vector in self.document_vectors])
        better_similars = np.argpartition(better_rate, -k)[-k:]
        matched_documents = self.document_list.loc[better_similars, :]
        return matched_documents

    def recommend(self, query, k=10):
        query = " ".join(clean_data(query))
        query_vector = self.model.encode(hazm.sent_tokenize(
            query), convert_to_tensor=True)  # encoding the query
        # calculating cosine similarity and getting the k highest scores
        rate = list([util.cos_sim(query_vector, vector)
                    for vector in self.document_vectors])
        similars = np.argpartition(rate, -k)[-k:]

        matched_documents = self.document_list.loc[similars]
        return matched_documents


class EmbedRecommender():

    def __init__(self, document_list):
        self.document_list = document_list.copy()
        self.model = None
        self.document_vectors = np.array([])

    def get_vector(self, document):
        return np.mean([self.model.wv[x] for word in document["tokens"] for x in word.split()], axis=0)

    def get_sim(self, query_vector, doc_vector):
        cosine_sim = cosine_similarity([query_vector, doc_vector])
        return cosine_sim[0][1]

    def run(self):
        fname = get_tmpfile(
            "E:\\myProjects\\IR_project\\models_data\\fasttextModel")
        try:
            self.document_vectors = np.load("models_data\\fasttextVectors.npy")
            self.model = FastText.load(fname)
        except:
            self.document_list["tokens"] = self.document_list.apply(
                lambda x: (clean_data(x["title"] + x["paragraphs"])), axis=1)
            document_features = self.document_list["tokens"].apply(
                lambda x: " ".join(x))
            document_features = document_features.tolist()
            sentence_tokens = []
            for document in document_features:
                # tokenizing the documents by sentence
                sentences = hazm.sent_tokenize(document)
                for sentence in sentences:
                    # and tokenizing the sentences into tokens to achieve a list of token lists
                    sentence_tokens.append(tokenize(sentence))

            self.model = FastText(vector_size=100, window=3, min_count=1)
            self.model.build_vocab(corpus_iterable=sentence_tokens)
            total_examples = self.model.corpus_count
            self.model.train(corpus_iterable=sentence_tokens,
                             total_examples=total_examples, epochs=5)
            temp_vectors = []
            for i in self.document_list.index:
                temp_vectors.append(self.get_vector(self.document_list.loc[i]))
            self.document_list.drop(["tokens"], axis=1)
            self.document_vectors = np.asarray(temp_vectors)
            np.save("models_data\\fasttextVectors.npy", self.document_vectors)
            self.model.save(fname)

    def expanded_recommend(self, query, k=10):
        query = clean_data(query)
        query_vector = np.mean([self.model.wv[x]
                               for word in query for x in word.split()], axis=0)
        rate = np.array([self.get_sim(query_vector, vector)
                        for vector in self.document_vectors])
        similars = np.argpartition(rate, -k)[-k:]
        irrelevants = np.argpartition(rate, k)[:k]
        relevant_docs = np.array([self.document_vectors[x] for x in similars])
        irrelevant_docs = np.array(
            [self.document_vectors[x] for x in irrelevants])
        modified_query = improve_query(
            relevant_docs, irrelevant_docs, query_vector)
        better_rate = np.array([self.get_sim(modified_query, vector)
                               for vector in self.document_vectors])
        better_similars = np.argpartition(better_rate, -k)[-k:]
        matched_documents = self.document_list.loc[better_similars, :]

        return matched_documents

    def recommend(self, query, k=10):
        query = clean_data(query)
        print(type(self.model), type(self.model['کبد']))
        query_vector = np.mean([self.model[x]
                               for word in query for x in word.split()], axis=0)

        rate = np.array([self.get_sim(query_vector, vector)
                        for vector in self.document_vectors])
        similars = np.argpartition(rate, -k)[-k:]

        matched_documents = self.document_list.loc[similars, :]

        return matched_documents


class TransformerVectorizer():
    def __init__(self, document_list):
        self.document_list = document_list[document_list['categories'].map(
            lambda x: x[1]) != "تازه های سلامت"]
        self.document_list["label"] = self.document_list["categories"].apply(
            lambda x: x[1])
        # achieving the transformer model
        self.model = SentenceTransformer(
            'paraphrase-multilingual-mpnet-base-v2')
        self.document_vectors = []
        self.document_labels = []

    def run(self):
        self.document_list["feature"] = self.document_list.apply(lambda x: " ".join(
            clean_data(x["title"] + " " + x["abstract"] + " " + x["paragraphs"])), axis=1)
        self.document_list["sentence"] = self.document_list["feature"].apply(
            lambda x: hazm.sent_tokenize(x))  # tokenizing the documents by sentence
        # print(self.model.encode(self.document_list.loc[0,["sentence"]], convert_to_tensor = False)[0]) # encoding each document

        for i in self.document_list.index:
            self.document_vectors.append(self.model.encode(self.document_list.loc[i, [
                                         "sentence"]], convert_to_tensor=False)[0])  # encoding each document
        self.document_vectors = np.array(self.document_vectors)
        self.document_labels = np.array(self.document_list["label"].tolist())
        self.document_list.drop(["feature", "sentence"], axis=1)
        # return self.model

    def make_vector(self, term):
        query_vector = self.model.encode(
            hazm.sent_tokenize(term), convert_to_tensor=False)
        return query_vector

    def get_doc_by_index(self, index):
        return self.document_list.loc[index]


class Classification:
    def __init__(self, docs):
        self.transformer = TransformerVectorizer(docs)
        self.transformer.run()
        try:
            self.model = joblib.load("models_data\\classifierModel.joblib")
        except:

            X_train, X_test, y_train, y_test = train_test_split(
                self.transformer.document_vectors, self.transformer.document_labels, test_size=0.35, random_state=42)
            self.model = sklearn.linear_model.LogisticRegression(
                random_state=0).fit(X_train, y_train)
            joblib.dump(self.model, "models_data\\classifierModel.joblib")
            return self.trans_model

    def classify(self, term):
        query_vector = self.transformer.make_vector(term)
        docs = {'result': self.model.predict(
            query_vector).tolist(), 'isClassification': True}
        return docs


class Clustering:
    def __init__(self, docs):
        self.transformer = TransformerVectorizer(docs)
        self.transformer.run()
        try:
            self.model = joblib.load("models_data\\clusterModel.joblib")
        except FileNotFoundError as e:
            labels = set(docs["categories"].apply(lambda x: x[1]).tolist())
            X_train, X_test, y_train, y_test = train_test_split(
                self.transformer.document_vectors, self.transformer.document_labels, test_size=0.35, random_state=42)
            self.model = sklearn.cluster.KMeans(
                n_clusters=len(labels), random_state=40).fit(X_train)
            joblib.dump(self.model, "models_data\\clusterModel.joblib")

    def cluster(self, term):
        query_vector = self.transformer.make_vector(term)
        cluster_group = self.model.predict(query_vector)
        doc_index = np.where(self.model.labels_ == cluster_group)[0][:5]
        docs = []
        for i in doc_index:
            docs.append(self.transformer.get_doc_by_index(i))
        return docs


def improve_query(relevant_docs, irrelevant_docs, initial_query, alpha=0, beta=1, gamma=1):
    d_r = np.sum(relevant_docs, axis=0)
    d_nr = np.sum(irrelevant_docs, axis=0)
    modified_query_vector = alpha * initial_query + \
        ((beta * d_r) - (gamma * d_nr)/len(relevant_docs))

    return modified_query_vector


def getHiDoctorData():
    document_list = pd.DataFrame(
        [], columns=["title", "tags", "paragraphs", "link"])

    with os.scandir("health") as dir:
        for entity in dir:
            if entity.name.startswith("hidoctor-3") or entity.name.startswith("hidoctor-4"):
                document_list = pd.concat(
                    [document_list, pd.read_json(entity.path)], ignore_index=True)
    document_list["paragraphs"] = document_list["paragraphs"].apply(
        lambda x: "\n".join(x))
    return document_list


def getNamnakData():
    document_list = pd.DataFrame(
        [], columns=["tags", "categories", "title", "abstract", "paragraphs", "link"])
    with os.scandir("health") as dir:
        for entity in dir:
            if entity.name.startswith("namnak"):
                df = pd.read_json(entity.path)
                document_list = pd.concat(
                    [document_list, pd.read_json(entity.path)], ignore_index=True)
    document_list["paragraphs"] = document_list["paragraphs"].apply(
        lambda x: "\n".join(x))
    return document_list


class Initial:
    def __init__(self):
        setNormalizers()
        hiDoctor = getHiDoctorData()
        self.boolean = BooleanRecommender(hiDoctor)
        self.boolean.run()
        self.tfidf = TfIdfRecommender(hiDoctor)
        self.tfidf.run()
        self.transformer = TransformerRecommender(hiDoctor)
        self.transformer.run()
        self.fasttext = EmbedRecommender(hiDoctor)
        self.fasttext.run()
        self.namnak = getNamnakData()
        self.classifier = Classification(self.namnak)
        self.cluster = Clustering(self.namnak)

    def find_target(self, query, action, query_expand):
        if action == 'cluster':
            return self.cluster.cluster(query)
        elif action == 'classify':
            return self.classifier.classify(query)
        elif action == 'boolean':
            return self.boolean.recommend(query).to_dict('records')
        elif action == 'tfidf':
            return self.tfidf.expanded_recommend(query).to_dict('records') if query_expand else self.tfidf.recommend(query).to_dict('records')
        elif action == 'transformer':
            return self.transformer.expanded_recommend(query).to_dict('records') if query_expand else self.transformer.recommend(query).to_dict('records')
        elif action == 'fasttext':
            return self.fasttext.expanded_recommend(query).to_dict('records') if query_expand else self.fasttext.recommend(query).to_dict('records')

# import pandas as pd
# import numpy as np
# import os
# import codecs
# import joblib
# import json
# import sklearn
# from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.model_selection import train_test_split
# from sklearn.cluster import KMeans
# from sklearn.feature_extraction.text import TfidfVectorizer

# import hazm

# from sentence_transformers import SentenceTransformer, util

# from gensim.models import FastText
# from gensim.test.utils import get_tmpfile


# normalizer = None
# stop_words = None
# tokenize = None
# lemmatizer = None
# def setNormalizers():
#   global normalizer
#   global stop_words
#   global tokenize
#   global lemmatizer

#   normalizer = hazm.Normalizer()
#   stop_words = codecs.open('stopwords.txt','r','utf-8').readlines()
#   tokenize = hazm.word_tokenize
#   lemmatizer = hazm.Lemmatizer()

# def clean_data(document):
#   tokenized = tokenize(normalizer.normalize(document)) # normalization and tokenization
#   tokenized = [token.lower() for token in tokenized if token.lower() not in stop_words] # deleting stop words
#   tokenized = [lemmatizer.lemmatize(token) for token in tokenized] # lemmatization
#   return tokenized

# class BooleanRecommender():
#   def __init__(self, document_list):
#     self.document_list = document_list.copy()
#     self.boolean_df = []
#     self.doc_token = []
#     self.column_list = set()

#   def run(self):
#     try:
#       self.boolean_df = np.load("models_data\\booleanVector.npy")
#       with open("models_data\\booleanColumns.txt", "r", encoding="UTF-8") as file:
#         self.column_list = file.read().split("\n")
#     except:
#       for i in self.document_list.index:
#         tokens = set(clean_data("\n".join([self.document_list.loc[i, ["title"]].iloc[0], self.document_list.loc[i, ["paragraphs"]].iloc[0]]))) # pre processing and getting all tokens
#         self.doc_token.append(list(tokens)) # saving tokens of each document
#         self.column_list.update(tokens) # updating all tokens on each doc iteration

#       self.column_list = list(self.column_list)

#       for doc in self.doc_token:
#         self.boolean_df.append([1 if token in doc else 0 for token in self.column_list])  # creating the boolean table
#       np.save("models_data\\booleanVector.npy", np.array(self.boolean_df))
#       with open("models_data\\booleanColumns.txt", "w", encoding="utf-8") as file:
#         file.write("\n".join(self.column_list))

#   def recommend(self, query, k=10):
#     query_tokens = list(set(clean_data(query)))

#     doc_score = []
#     for doc in self.boolean_df: # getting document score based on contained queries
#       score = 0
#       for token in query_tokens:
#         if token in self.column_list:
#           score += doc[self.column_list.index(token)]
#       doc_score.append(score)

#     similar = np.argpartition(doc_score, -k)[-k:] # getting k highest scores

#     matched_documents = self.document_list.loc[similar, :]

#     return matched_documents


# class TfIdfRecommender():
#   def __init__(self, document_list):
#     self.document_list = document_list.copy()
#     self.vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1,3)) # preparing the vectorizer model
#     # self.document_vectors = None
#     self.vocabulary = []

#   def get_sim(self, query_vector, doc_vector):
#     cosine_sim = cosine_similarity([query_vector, doc_vector])
#     return cosine_sim[0][1]

#   def run(self):
#     try:
#       np.load("models_data\\tfidfVectors.npy")
#       self.vocabulary = self.vectorizer.get_feature_names_out()
#     except:
#       self.document_list["feature"] = self.document_list.apply(lambda x:" ".join(clean_data("\n".join([x["title"], x["paragraphs"]]))), axis = 1)
#       document_vectors = self.vectorizer.fit_transform(self.document_list["feature"]).toarray() # getting vector of each document
#       self.vocabulary = self.vectorizer.get_feature_names_out()
#       self.document_list.drop(["feature"], axis=1) # saving attained features
#       np.save("models_data\\tfidfVectors.npy", document_vectors)


#   def find_rate_for_the_query(self, query):
#     document_vectors = np.load("models_data\\tfidfVectors.npy")
#     query = " ".join(clean_data(query))
#     query_vector = self.vectorizer.fit_transform([query]).toarray() # getting query vector
#     query_features = self.vectorizer.get_feature_names_out()
#     query_vector = [query_vector[0][np.where(query_features == word)[0][0]] if word in query_features else 0 for word in self.vocabulary] # expanding the vector to the feature vector
#     rate = np.array([self.get_sim(query_vector, document_vectors[i]) for i in range(document_vectors.shape[0])], dtype=float) # calculating cosine similarity and rating
#     return rate, query_vector, document_vectors


# def recommend(ir_type, query, k=10):
#   query = clean_data(query)
#   rate, query_vector, document_vectors  = ir_type.find_rate_for_the_query(query)
#   similars = np.argpartition(rate, -k)[-k:]
#   print(similars)
#   irrelevants = np.argpartition(rate, k)[:k]
#   if isinstance(ir_type, TfIdfRecommender):
#     relevant_docs = [document_vectors[x] for x in similars]
#     irrelevant_docs = [ir_type.document_vectors[x] for x in irrelevants]
#   elif (isinstance(ir_type, TransformerRecommender) or isinstance(ir_type, EmbedRecommender)):
#     relevant_docs = np.array([ir_type.document_vectors[x] for x in similars])
#     irrelevant_docs = np.array([ir_type.document_vectors[x] for x in irrelevants])
#   modified_query = improve_query(relevant_docs, irrelevant_docs, query_vector)
#   better_rate, _ = ir_type.find_rate_for_the_query(modified_query)
#   better_similars = np.argpartition(better_rate, -k)[-k:]
#   matched_documents = ir_type.document_list.loc[better_similars, :]
#   return matched_documents


# class TransformerRecommender():
#   def __init__(self, document_list):
#     self.document_list = document_list.copy()
#     self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2') # achieving the transformer model
#     self.document_vectors = []

#   def run(self):
#     self.document_list["feature"] = self.document_list.apply(lambda x: " ".join(clean_data(x["title"] + " " + x["paragraphs"])), axis = 1)
#     self.document_list["sentence"] = self.document_list["feature"].apply(lambda x: hazm.sent_tokenize(x)) # tokenizing the documents by sentence
#     for i in self.document_list.index:
#       self.document_vectors.append(self.model.encode(self.document_list.loc[i,["sentence"]], convert_to_tensor = True)) # encoding each document

#     self.document_list.drop(["feature", "sentence"], axis=1)

#   def find_rate_for_the_query(self, query):
#     query = " ".join(clean_data(query))
#     query_vector = self.model.encode(hazm.sent_tokenize(query), convert_to_tensor = True) # encoding the query
#     rate = list([util.cos_sim(query_vector, vector) for vector in self.document_vectors]) # calculating cosine similarity and getting the k highest scores
#     return rate, query_vector, None


#   # def recommend(self, query, k=10):
#   #   rate, query_vector  = self.find_rate_for_the_query(query)
#   #   similars = np.argpartition(rate, -k)[-k:]
#   #   irrelevants = np.argpartition(rate, k)[:k]
#   #   relevant_docs = self.document_vectors[similars]
#   #   irrelevant_docs = self.document_vectors[irrelevants]
#   #   modified_query = improve_query(relevant_docs, irrelevant_docs, query_vector)
#   #   better_rate, _ = self.find_rate_for_the_query(modified_query)
#   #   better_similars = np.argpartition(better_rate, -k)[-k:]
#   #   matched_documents = self.document_list.loc[better_similars]
#   #   return matched_documents


# class EmbedRecommender():

#   def __init__(self, document_list):
#     self.document_list = document_list.copy()
#     self.model = None
#     self.document_vectors = np.array([])

#   def get_vector(self, document):
#     return np.mean([self.model.wv[x] for word in document["tokens"] for x in word.split()], axis = 0)

#   def get_sim(self, query_vector, doc_vector):
#     cosine_sim = cosine_similarity([query_vector, doc_vector])
#     return cosine_sim[0][1]

#   def run(self):
#     fname = get_tmpfile("E:\\myProjects\\IR_project\\models_data\\fasttextModel")
#     try:
#       self.document_vectors = np.load("models_data\\fasttextVectors.npy")
#       self.model = FastText.load(fname)
#     except:
#       self.document_list["tokens"] = self.document_list.apply(lambda x: (clean_data(x["title"] + x["paragraphs"])), axis=1)
#       document_features = self.document_list["tokens"].apply(lambda x: " ".join(x))
#       document_features = document_features.tolist()
#       sentence_tokens = []
#       for document in document_features:
#         sentences = hazm.sent_tokenize(document) # tokenizing the documents by sentence
#         for sentence in sentences:
#           sentence_tokens.append(tokenize(sentence))  # and tokenizing the sentences into tokens to achieve a list of token lists

#       self.model = FastText(vector_size=100, window=3, min_count=1)
#       self.model.build_vocab(corpus_iterable = sentence_tokens)
#       total_examples = self.model.corpus_count
#       self.model.train(corpus_iterable = sentence_tokens, total_examples=total_examples, epochs=5)
#       temp_vectors = []
#       for i in self.document_list.index:
#           temp_vectors.append(self.get_vector(self.document_list.loc[i]))
#       self.document_list.drop(["tokens"], axis = 1)
#       self.document_vectors = np.asarray(temp_vectors)
#       np.save("models_data\\fasttextVectors.npy", self.document_vectors)
#       self.model.save(fname)


#   def find_rate_for_the_query(self, query):
#     query_vector = np.mean([self.model.wv[x] for word in query for x in word.split()], axis = 0)
#     rate = np.array([self.get_sim(query_vector, vector) for vector in self.document_vectors])
#     return rate, query_vector, None


#   # def recommend(self, query, k=10):
#   #   rate, query_vector  = self.find_rate_for_the_query(query)
#   #   similars = np.argpartition(rate, -k)[-k:]
#   #   irrelevants = np.argpartition(rate, k)[:k]
#   #   relevant_docs = self.document_vectors[similars]
#   #   irrelevant_docs = self.document_vectors[irrelevants]
#   #   modified_query = improve_query(relevant_docs, irrelevant_docs, query_vector)
#   #   better_rate, _ = self.find_rate_for_the_query(modified_query)
#   #   better_similars = np.argpartition(better_rate, -k)[-k:]
#   #   matched_documents = self.document_list.loc[better_similars]
#   #   return matched_documents


# class TransformerVectorizer():
#   def __init__(self, document_list):
#     self.document_list = document_list[document_list['categories'].map(lambda x: x[1]) != "تازه های سلامت"]
#     self.document_list["label"] = self.document_list["categories"].apply(lambda x : x[1])
#     self.model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2') # achieving the transformer model
#     self.document_vectors = []
#     self.document_labels = []


#   def run(self):
#     self.document_list["feature"] = self.document_list.apply(lambda x: " ".join(clean_data(x["title"] + " " + x["abstract"] + " " + x["paragraphs"])), axis = 1)
#     self.document_list["sentence"] = self.document_list["feature"].apply(lambda x: hazm.sent_tokenize(x)) # tokenizing the documents by sentence

#     for i in self.document_list.index:
#       self.document_vectors.append(self.model.encode(self.document_list.loc[i,["sentence"]], convert_to_tensor = False)[0]) # encoding each document
#     self.document_vectors = np.array(self.document_vectors)
#     self.document_labels = np.array(self.document_list["label"].tolist())
#     self.document_list.drop(["feature", "sentence"], axis=1)
#     # return self.model


#   def make_vector(self, term):
#     query_vector = self.model.encode(hazm.sent_tokenize(term), convert_to_tensor = False)
#     return query_vector


#   def get_doc_by_index(self, index):
#     return self.document_list.loc[index]

# class Classification:
#     def __init__(self, docs):
#         self.transformer = TransformerVectorizer(docs)
#         self.transformer.run()
#         try:
#             self.model = joblib.load("models_data\\classifierModel.joblib")
#         except:

#             X_train, X_test, y_train, y_test = train_test_split(self.transformer.document_vectors, self.transformer.document_labels, test_size=0.35, random_state=42)
#             self.model = sklearn.linear_model.LogisticRegression(random_state=0).fit(X_train, y_train)
#             joblib.dump(self.model, "models_data\\classifierModel.joblib")
#             return self.model

#     def classify(self, term):
#       query_vector = self.transformer.make_vector(term)
#       docs = {'result' :self.model.predict(query_vector).tolist(), 'isClassification': True}
#       return docs


# class Clustering:
#   def __init__(self, docs):
#     self.transformer = TransformerVectorizer(docs)
#     self.transformer.run()
#     try:
#       self.model = joblib.load("models_data\\clusterModel.joblib")
#     except FileNotFoundError as e:
#         labels = set(docs["categories"].apply(lambda x : x[1]).tolist())
#         X_train, X_test, y_train, y_test = train_test_split(self.transformer.document_vectors, self.transformer.document_labels, test_size=0.35, random_state=42)
#         self.model = sklearn.cluster.KMeans(n_clusters=len(labels), random_state=40).fit(X_train)
#         joblib.dump(self.model, "models_data\\clusterModel.joblib")

#   def cluster(self, term):
#     query_vector = self.transformer.make_vector(term)
#     cluster_group = self.model.predict(query_vector)
#     doc_index = np.where(self.model.labels_ == cluster_group)[0][:5]
#     docs = []
#     for i in doc_index:
#       docs.append(self.transformer.get_doc_by_index(i))
#     return docs

# def improve_query(relevant_docs, irrelevant_docs, initial_query, alpha=0, beta=1, gamma=1):
#   d_r = np.sum(relevant_docs, axis=1)
#   d_nr = np.sum(irrelevant_docs, axis=1)
#   modified_query_vector = alpha * initial_query + ((beta * d_r) - (gamma * d_nr)/len(relevant_docs))

#   return modified_query_vector


# def getHiDoctorData():
#   document_list = pd.DataFrame([], columns=["title", "tags", "paragraphs", "link"])

#   with os.scandir("health") as dir:
#     for entity in dir:
#       if entity.name.startswith("hidoctor-3") or entity.name.startswith("hidoctor-4"):
#         document_list = pd.concat([document_list, pd.read_json(entity.path)], ignore_index = True)
#   document_list["paragraphs"] = document_list["paragraphs"].apply(lambda x: "\n".join(x))
#   return document_list


# def getNamnakData():
#   document_list = pd.DataFrame([], columns = ["tags", "categories", "title", "abstract", "paragraphs", "link"])
#   with os.scandir("health") as dir:
#     for entity in dir:
#       if entity.name.startswith("namnak"):
#         df = pd.read_json(entity.path)
#         document_list = pd.concat([document_list, pd.read_json(entity.path)], ignore_index = True)
#   document_list["paragraphs"] = document_list["paragraphs"].apply(lambda x: "\n".join(x))
#   return document_list


# class Initial:
#   def __init__(self):
#       setNormalizers()
#       hiDoctor = getHiDoctorData()
#       self.boolean = BooleanRecommender(hiDoctor)
#       self.boolean.run()
#       self.tfidf = TfIdfRecommender(hiDoctor)
#       self.tfidf.run()
#       self.transformer = TransformerRecommender(hiDoctor)
#       self.transformer.run()
#       self.fasttext = EmbedRecommender(hiDoctor)
#       self.fasttext.run()
#       self.namnak = getNamnakData()
#       self.classifier = Classification(self.namnak)
#       self.cluster = Clustering(self.namnak)

#   def find_target(self, query, action):
#       if action=='cluster':
#         return self.cluster.cluster(query)
#       elif action=='classify':
#         return self.classifier.classify(query)
#       elif action=='boolean':
#         return self.boolean.recommend(query).to_dict('records')
#       elif action=='tfidf':
#         return recommend(self.tfidf, query).to_dict('records')
#         # return self.tfidf.recommend(query).to_dict('records')
#       elif action=='transformer':
#         return recommend(self.transformer, query).to_dict('records')
#         # return self.transformer.recommend(query).to_dict('records')
#       elif action=='fasttext':
#         return recommend(self.fasttext, query).to_dict('records')
#         # return self.fasttext.recommend(query).to_dict('records')
