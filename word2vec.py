import string
import itertools
import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior() 

# Raw text to extract data from.
text = "queen woman. woman queen. queen royal. king man. man king. king royal."

# List of all sentences in text.
sentences = [s.lower().translate(str.maketrans('','',string.punctuation)).split() for s in text.split('.')]

# Set of all words in text.
vocabulary = {word for word in text.lower().translate(str.maketrans('','',string.punctuation)).split()}

# Size of vocabulary.
VOCAB_SIZE = len(vocabulary)

# Dictionary of the vocabulary.
word2int = {word:index for index,word in enumerate(vocabulary)}
int2word = {index:word for index,word in enumerate(vocabulary)}

# Number of words to consider as context.
CONTEXT_WINDOW = 5

# Returns a list of pairs [word,context_word] for every word in sentence.
def contexts(sentence):
    indices = [index for index,_ in enumerate(sentence)]
    nbhs = map(lambda i: sentence[max(i-CONTEXT_WINDOW,0):min(i+CONTEXT_WINDOW,len(sentence))+1],indices)
    nbs = list(map(lambda n: [[sentence[n[0]],w] for w in n[1] if sentence[n[0]]!=w],enumerate(nbhs)))
    return list(itertools.chain(*nbs))

# List of pairs [word,context_word] for every word in text.
data = list(itertools.chain(*(map(contexts,sentences))))

# Returns a OneHot vector of dimension VOCAB_SIZE of the word at the corresponding index.
def one_hot(index): 
    return np.array([(1.0 if i == index else 0.0) for i in range(VOCAB_SIZE)])

# Training examples.
x_train = np.array([one_hot(word2int[word[0]]) for word in data])
y_train = np.array([one_hot(word2int[word[1]]) for word in data])

# Placeholders for the input and output layers.
x = tf.placeholder(tf.float32,shape=(None,VOCAB_SIZE))
y = tf.placeholder(tf.float32,shape=(None,VOCAB_SIZE))

# Dimension of embeddings.
DIMENSIONALITY = 100

# Weights and biases of the input layer.
w1 = tf.Variable(tf.random_normal([VOCAB_SIZE,DIMENSIONALITY]))
b1 = tf.Variable(tf.random_normal([DIMENSIONALITY]))

# Layer of embeddings.
hidden_layer = tf.add(tf.matmul(x,w1),b1)

# Weights and biases of the hidden layer.
w2 = tf.Variable(tf.random_normal([DIMENSIONALITY,VOCAB_SIZE]))
b2 = tf.Variable(tf.random_normal([VOCAB_SIZE]))

# Layer of predictions.
output_layer = tf.nn.softmax(tf.add(tf.matmul(hidden_layer,w2),b2))

# Initializing TensorFlow session.
session = tf.Session()
session.run(tf.global_variables_initializer()) 

# Loss function.
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y,logits=output_layer))

# Learning rate for optimizer.
LEARNING_RATE = 0.1

# Optimizer.
optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)

# Number of iterations to train with.
ITERATIONS = 1000

# Training network.
for _ in range(ITERATIONS):
    session.run(optimizer,feed_dict={x:x_train,y:y_train})    
    print("Loss:",session.run(loss,feed_dict={x:x_train,y:y_train}))

# Word embeddings.
vectors = np.array(session.run(w1+b1))

# Returns the cosine similarity between vectors v1 and v2.
def cosine_similarity(v1,v2):
    return np.dot(v1,v2) / (np.linalg.norm(v1)*np.linalg.norm(v2))

# Returns a sorted list of the words that are most similar to the word 
# represented by vector. (most_similar(vectors[word2int[word]])[0] == word)
def most_similar(vector):
    sims = enumerate(map(lambda v: cosine_similarity(vector,v),vectors))
    sorted_sims = sorted(sims,key=(lambda x: x[1]),reverse=True)
    return [int2word[index] for index,_ in sorted_sims]

# Returns the word d such that a is to b as c is to d, where a, b, and c are words.
def is_to_as_is_to(a,b,c):
    v = vectors[word2int[b]] - vectors[word2int[a]] + vectors[word2int[c]]
    d = [word for word in most_similar(v) if word not in [a,b,c]][0]
    return d

# Print results.
queen = is_to_as_is_to("man","king","woman")
king = is_to_as_is_to("woman","queen","man")
print(queen)
print(king)