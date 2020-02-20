import string
import itertools
import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior() 

# Raw text to extract data from.
text = "Who's wishing that? My cousin Westmorland? No, my dear cousin, if we are marked down to die we are enough for our country to lose, and if marked down to live, the fewer the men the greater the share of honour. For the love of God, don't wish for one man more. By Jove, I'm not interested in gold, nor do I care who eats at my expense. It doesn't bother me who wears my clothes. Such outward things don't come into my ambitions. But if it is a sin to long for honour I am the most offending soul alive. No, indeed, my cousin, don't wish for another man from England. God's peace, I wouldn't lose as much honour as the share one man would take from me. No, don't wish for one more. Rather proclaim to my army, Westmorland, that anyone who doesn't have the stomach for this fight should leave now. He will be guaranteed free passage and travel money will be put in his purse. We would not like to die with any man who lacks the comradeship to die with us. This day is called the Feast of Crispian. He who outlives this day and gets home safely to reach old age will yearly on its anniversary celebrate with his neighbours and say, 'Tomorrow is Saint Crispian.' Then he will roll up his sleeve and show his scars and say 'I got these wounds on Crispin's day.' Old men are forgetful, but even if he remembers nothing else he'll remember, with embroideries, what feats he did that day. Then our names, as familiar in his mouth as household words – Harry the King, Bedford and Exeter, Warwick and Talbot, Salisbury and Gloucester – will be remembered in their toasts. This good man will teach his son, and Crispin Crispian will never pass from today until the end of the world without us being remembered: we few; we happy few; we band of brothers! The man who sheds his blood with me shall be my brother; however humble he may be, this day will elevate his status. And gentlemen in England, still lying in their beds, will think themselves accursed because they were not here, and be in awe while anyone speaks who fought with us upon Saint Crispin's day.'"

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
x = tf.placeholder(tf.float32, shape=(None,VOCAB_SIZE))
y = tf.placeholder(tf.float32, shape=(None,VOCAB_SIZE))

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
    session.run(optimizer,feed_dict={x:x_train, y:y_train})    
    print('Loss: ',session.run(loss,feed_dict={x:x_train, y:y_train}))

# Word embeddings.
vectors = np.array(session.run(w1+b1))

# Returns the cosine similarity between vectors v1 and v2.
def cosine_similarity(v1,v2):
    return np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))

# Returns an ordered list of words in vocabulary that are most similar to word,
# such that (most_similar(word))[0] = word.
def most_similar(word):
    # sims = enumerate(similarities(vectors[word2int[word]],vectors))
    sims = enumerate(map(lambda v: cosine_similarity(vectors[word2int[word]],v),vectors))
    sorted_sims = sorted(sims,key=(lambda x: x[1]),reverse=True)
    return [int2word[index] for index,_ in sorted_sims]

print(most_similar('england'))