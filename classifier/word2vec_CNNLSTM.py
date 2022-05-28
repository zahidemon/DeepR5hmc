from __future__ import print_function
import _pickle as cPickle
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import LSTM, GRU, SimpleRNN
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import confusion_matrix

from sklearn import metrics

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import os
from keras.layers import Embedding
import random

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence


def twoTupleDic():
    AA_list_sort = ['A', 'U', 'C', 'G']
    AA_dict = {}
    numm = 1
    for i in AA_list_sort:
        for j in AA_list_sort:
            AA_dict[i + j] = numm
            numm += 1
    return AA_dict


def DNA2Sentence(dna, K):
    sentence = ""
    length = len(dna)
    for i in range(length - K + 1):
        sentence += dna[i: i + K] + " "
    # delete extra space
    sentence = sentence[0: len(sentence) - 1]
    return sentence


def Get_Unsupervised(fname, gname, kmer):
    f = open(fname, 'r')
    g = open(gname, 'w')
    seq = []
    k = kmer
    for i in f:
        if '>' not in i:
            i = i.strip('\n').upper()
            line = DNA2Sentence(i, k)
            seq.append(line)
            g.write(line + '\n')
    f.close()
    return seq


def createTrainTestData(str_path, nb_words=None, skip_top=0,
                        maxlen=None, test_split=0.25, seed=113,
                        start_char=1, oov_char=2, index_from=3):
    X, labels = cPickle.load(open(str_path, "rb"))
    # np.random.seed(seed)
    # np.random.shuffle(X)
    # np.random.seed(seed)
    # np.random.shuffle(labels)
    X_train = np.array(X[:int(len(X) * (1 - test_split))])
    y_train = np.array(labels[:int(len(X) * (1 - test_split))])

    X_test = np.array(X[int(len(X) * (1 - test_split)):])
    y_test = np.array(labels[int(len(X) * (1 - test_split)):])

    return (X_train, y_train), (X_test, y_test)


def getDNA_split(DNAdata, word):
    DNAlist1 = []
    # DNAlist2 = []
    counter = 0
    for DNA in DNAdata["seq"]:
        # if counter % 100 == 0:
        # print ("DNA %d of %d\r" % (counter, 2*len(DNAdata)))
        # sys.stdout.flush()

        DNA = str(DNA).upper()
        DNAlist1.append(
            DNA2Sentence(DNA, word).split(" "))  # [['ACG', 'CGT', 'GTC'],['ACG', 'CGT', 'GTC'],['ACG', 'CGT', 'GTC']]

        counter += 1
    return DNAlist1


def getWord_model(word, num_features, min_count):
    word_model = ""
    if not os.path.isfile(generated_files_folder + "//" + modelfile_string):
        sentence = LineSentence(generated_files_folder + "//2UN.txt", max_sentence_length=15000)
        print("Start Training Word2Vec model...")
        # Set values for various parameters
        num_features = int(num_features)  # Word vector dimensionality
        min_word_count = int(min_count)  # Minimum word count
        num_workers = 20  # Number of threads to run in parallel并行运行的线程数
        context = 20  # Context window size上下文窗口大小
        downsampling = 1e-3  # Downsample setting for frequent words常用词的下采样设置

        # Initialize and train the model初始化和训练模型
        print("Training Word2Vec model...")
        word_model = Word2Vec(sentence, workers=num_workers, \
                              vector_size=num_features, min_count=min_word_count, \
                              window=context, sample=downsampling, seed=1, epochs=50)
        word_model.save(generated_files_folder + "//" + modelfile_string)
        # print word_model.most_similar("CATAGT")
    else:
        print("Loading Word2Vec model...")
        word_model = Word2Vec.load(generated_files_folder + "//" + modelfile_string)
    return word_model


#########deal data, split to train and test################

os.chdir("../dataset/benchmark_dataset")
data_folder = os.getcwd()
fasta_file = data_folder + "//Dataset.fasta"
os.chdir("../../generated_files")
generated_files_folder = os.getcwd()
corpus_file = generated_files_folder + '//2UN.txt'
modelfile_string = "word2vec_model"
posnum = 230
pklfile = "data.pkl"
model_save = 'model_final.h5'
kmer = 2

# 用index来表示二氨基酸
texts = Get_Unsupervised(fasta_file, corpus_file, kmer)

word_index1 = twoTupleDic()
sequences = []
for each in texts:
    each_index_list = []
    each = each.split(' ')
    for i in each:
        each_index_list.append(word_index1[i])
    sequences.append(each_index_list)

# 生成相应的标签，正样本1，负样本0
labels = []
for i in range(0, posnum):
    labels.append(1)
for j in range(posnum, len(texts)):
    labels.append(0)

# 将训练集按照3：7分成test：train
MAX_SEQUENCE_LENGTH = 100
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
t = (data, labels)
cPickle.dump(t, open(pklfile, "wb"))
(X_train, y_train), (X_test, y_test) = createTrainTestData(pklfile, nb_words=None, test_split=0.3)

print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

#########obtain embedding_matrix###########

tokenizer = Tokenizer(num_words=None)  # 分词MAX_NB_WORDS
tokenizer.fit_on_texts(texts)
sequences2 = tokenizer.texts_to_sequences(texts)  # 受num_words影响

word_index2 = tokenizer.word_index  # 词_索引
print(len(word_index1), len(word_index2))

# word2vec
EMBEDDING_DIM = 100
getWord_model(texts, EMBEDDING_DIM, 1)
Word2VecModel = Word2Vec.load(modelfile_string)
embedding_matrix = np.zeros((len(word_index1) + 1, EMBEDDING_DIM))
for word, i in word_index1.items():
    if word.lower() in word_index2:
        embedding_vector = Word2VecModel.wv[word]
        # print(embedding_vector)
        embedding_matrix[i] = embedding_vector
print(embedding_matrix)

#########train model#######################

# k-fold
embedding_size = 100

# Convolution
# filter_length = 3
nb_filter = 64
pool_length = 2

# LSTM
lstm_output_size = 70

# Training
batch_size = 128
nb_epoch = 60

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
acc_score = []
auc_score = []
sn_score = []
sp_score = []
mcc_score = []

for i, (train, test) in enumerate(kfold.split(X_train, y_train)):
    print('\n\n%d' % i)

    model = Sequential()
    model.add(
        Embedding(len(embedding_matrix), embedding_size, weights=[embedding_matrix], input_length=100, trainable=False))
    model.add(Dropout(0.5))
    model.add(Convolution1D(filters=nb_filter,
                            kernel_size=10,
                            padding='valid',
                            activation='relu',
                            strides=1))
    model.add(MaxPooling1D(pool_size=pool_length))
    model.add(Convolution1D(filters=nb_filter,
                            kernel_size=5,
                            padding='valid',
                            activation='relu',
                            strides=1))
    model.add(MaxPooling1D(pool_size=pool_length))

    model.add(LSTM(lstm_output_size))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    print('Train...')

    # 早停法
    checkpoint = EarlyStopping(monitor='val_loss',
                               min_delta=0,
                               patience=3,
                               verbose=1, mode='auto')

    # early stopping
    model.fit(X_train[train], y_train[train], epochs=nb_epoch, batch_size=batch_size,
              validation_data=(X_train[test], y_train[test]), shuffle=True, callbacks=[checkpoint], verbose=2)
    # model.fit(X_train[train], y_train[train], epochs = nb_epoch, batch_size = batch_size, validation_data = (X_train[test], y_train[test]), shuffle = True)
    ##########################
    prd_acc = model.predict(X_train[test])
    pre_acc2 = []
    for i in prd_acc:
        pre_acc2.append(i[0])

    prd_lable = []
    for i in pre_acc2:
        if i > 0.5:
            prd_lable.append(1)
        else:
            prd_lable.append(0)
    prd_lable = np.array(prd_lable)
    obj = confusion_matrix(y_train[test], prd_lable)
    tp = obj[0][0]
    fn = obj[0][1]
    fp = obj[1][0]
    tn = obj[1][1]
    sn = tp / (tp + fn)
    sp = tn / (tn + fp)
    mcc = (tp * tn - fp * fn) / (((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5)
    sn_score.append(sn)
    sp_score.append(sp)
    mcc_score.append(mcc)
    ###########################
    pre_test_y = model.predict(X_train[test], batch_size=batch_size)
    test_auc = metrics.roc_auc_score(y_train[test], pre_test_y)
    auc_score.append(test_auc)
    print("test_auc: ", test_auc)

    score, acc = model.evaluate(X_train[test], y_train[test], batch_size=batch_size)
    acc_score.append(acc)
    print('Test score:', score)
    print('Test accuracy:', acc)
    print('***********************************************************************\n')
print('***********************print final result*****************************')
print(acc_score, auc_score)
mean_acc = np.mean(acc_score)
mean_auc = np.mean(auc_score)
mean_sn = np.mean(sn_score)
mean_sp = np.mean(sp_score)
mean_mcc = np.mean(mcc_score)
# print('mean acc:%f\tmean auc:%f'%(mean_acc,mean_auc))

line = 'acc\tsn\tsp\tmcc\tauc:\n%.2f\t%.2f\t%.2f\t%.4f\t%.4f' % (
100 * mean_acc, 100 * mean_sn, 100 * mean_sp, mean_mcc, mean_auc)
print('5-fold result\n' + line)

print('***************************save model********************************************')

model = Sequential()
model.add(Embedding(len(embedding_matrix), embedding_size, weights=[embedding_matrix], input_length=100,
                    trainable=False))  # input_length是输入维度长度，embedding_size是每个向量经过嵌入层后生成的长度
model.add(Dropout(0.5))
# nb_filter : 卷积核的数量，也是输出的维度。filter_length : 每个过滤器的长度。
model.add(Convolution1D(filters=nb_filter,
                        kernel_size=10,
                        padding='valid',
                        activation='relu',
                        strides=1))
model.add(MaxPooling1D(pool_size=pool_length))
model.add(Convolution1D(filters=nb_filter,
                        kernel_size=5,
                        padding='valid',
                        activation='relu',
                        strides=1))
model.add(MaxPooling1D(pool_size=pool_length))

model.add(LSTM(lstm_output_size))
model.add(Dense(1))
model.add(Activation('sigmoid'))
# optimizer：优化器，如Adam；loss：计算损失，这里用的是交叉熵损失；metrics: 列表，包含评估模型在训练和测试时的性能的指标，典型用法是metrics=[‘accuracy’]。如果要在多输出模型中为不同的输出指定不同的指标，可向该参数传递一个字典，例如metrics={‘output_a’: ‘accuracy’}
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# model.fit(X_train, y_train, epochs = nb_epoch, batch_size = batch_size, validation_data = (X_test, y_test), shuffle = True)
# early stopping
model.fit(X_train, y_train, epochs=nb_epoch, batch_size=batch_size, validation_data=(X_test, y_test), shuffle=True,
          callbacks=[checkpoint], verbose=2)

model.save(model_save)
