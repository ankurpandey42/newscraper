
# coding: utf-8

# In[1]:


from pprint import pprint
from pymongo import MongoClient
from tensorflow import keras
import numpy as np
import pickle
import json
from  unidecode import unidecode_expect_nonascii, unidecode
client = MongoClient(connect=False)
db = client['newscraper']


# In[2]:


def show_schema(table='articles_cleaned'):
    from pprint import pprint
    pprint(next(db[table].find()))
    


# In[23]:


show_schema('articles')


# In[24]:



class Corpus:
    ''' Retrieves data from MongoDB'''
    def __init__(self,db_table = 'articles',field = 'title',n_words=20000):

        self.n_words = n_words
        self.field=field
        self.db_table = db_table
        self.labels = [
        'center', 'conspiracy', 'extreme left', 'extreme right', 'fake news', 'hate', 'high', 'left',
        'left-center', 'low', 'mixed', 'pro-science', 'propaganda', 'right', 'right-center', 'satire',
        'very high'
        ]
            
    def get_all_rows(self):
        ''' Retrieve target table from db '''
        print (self.n_words)
        self.articles = [_ for _ in db[self.db_table].find() if _[self.field]]
        self.n_articles = len(self.articles)
    
    
class KerasVectorizer(Corpus):
    ''' Performs vectorization and text preprocessing '''
    def __init__(self,dnn_type='seq',max_len=1000,predict_str=False):
        super().__init__()  
        if not predict_str:
            self.get_all_rows()
            self.train = True
        else:
            self.articles = predict_str
            self.train = False
        self.dnn_type = dnn_type
        self.max_len = max_len
          
        
    def clean(self,seq):
            if len(seq):
                seq = unidecode(seq)
                return ' '.join(keras.preprocessing.text.text_to_word_sequence(seq,filters='''1234567890!"#$%&()*+,-\n./—:;<=>?@[\\]^_`{|}~\t\'“”'''))
    def fit(self):
        
        ''' Fit vectorizer on corpus '''
        
        Tokenizer = keras.preprocessing.text.Tokenizer
        tokenizer = Tokenizer(self.n_words)

        print ('cleaning text')
        texts = [self.clean(entry[self.field]) for entry in self.articles]
        print('fitting vector')
        tokenizer.fit_on_texts(texts)
        pickle.dump(tokenizer, open('vector.pkl', 'wb'))
        self.corpus_vector = tokenizer
        self.lookup = {k:v for k,v in self.corpus_vector.word_index.items() if v < self.n_words}
        
        json.dump(self.lookup, open('lookup.json', 'w'))

    
    def gen_x_onehot(self):
        if self.train:
            text = [self.clean(_[self.field]) for _ in self.articles]
        else: 
            text = self.articles
        for entry in text:
            entry = keras.preprocessing.text.text_to_word_sequence(entry)
            yield [self.lookup[word] for word in entry if word in self.lookup]

                
    
    def transform_x_onehot(self):
        x = list(self.gen_x_onehot())
#         v_len = max([len(_)for _ in x])
#         print ('longest text', v_len)
#         if v_len > self.max_len:
#             v_len = self.max_len
        self.rev_lookup = {v:k for k,v in self.lookup.items()}
        v_len = self.max_len
        print ('using limit of', v_len)
        self.lens = []
        for entry in x:
            self.lens.append(len(entry))
            
            if len(entry) >= v_len:
                yield np.array(entry[-v_len:])
            else:
                yield np.array([0 for _ in range(v_len - len(entry))] + entry)
                     
    
    def transform_y(self):
        ''' Vectorizes y labels '''
        for entry in self.articles:
            yield np.array([1 if _ in entry['flags'] else 0 for _ in self.labels])
        
    def transform_x(self):
        ''' Transforms texts to the vector '''
        vector = pickle.load(open('./vector.pkl','rb'))
        
        self.lookup = json.load(open('lookup.json'))
        
        return list(self.transform_x_onehot())     
    

    def x_y(self):
        self.fit()
        print('producing x, y data')
        y = list(self.transform_y())
        
        if self.dnn_type == 'seq':
            x = list(self.transform_x_onehot())     
        elif self.dnn_type == 'bow':
            x = self.transform_x()
        return x,y

    
def prep_data():
    k_v = KerasVectorizer(max_len=1000)
    #http://www.newswhip.com/2013/12/article-length/
    x,y= k_v.x_y()
    print('data prepared')
    print(x[0].shape)
    

    return k_v, x, y


def predict_data(text):
    k_v = KerasVectorizer(max_len=1000,predict_str=[text])
    
    
    x = k_v.transform_x()
    print('data prepared')
    print(x[0].shape)
    

    return k_v, x


# In[25]:


text = '''We knew something apocalyptic was coming in Davos today. Those tactically released photographs of President Trump arriving by helicopter with his entourage were the giveaway: the silhouetted choppers strung out in extended line in the orange-yellow light above the mountains.
Why, if you’d listened carefully, you might almost have heard the strains of Wagner’s Ride of the Valkyrie wailing above the whup whup whup of those thrumming blades. And a guy with a cavalry hat and cigar in his mouth growling something about Charlie’s lack of surfing abilities, and the sweetness of the smell of napalm in the morning.

Yep, Lt Col Kilgore had arrived at the heart of the belly of the beast and the enemy was about to get a very rude awakening.

The enemy on this occasion, of course, was Davos Man. Or – if you prefer – the globalist elite which has spent the last several decades stitching up the world in its own interests: the Vampire-Squid-trained central banksters; the EU technocrats; the corporatist crony capitalists; the rent-seeking sustainability experts; the priggish, politically correct, sermonising NGOs; the controlling one world government freaks; the woke Hollywood groupies; George Soros; pretty much all the reasons that made us vote for Donald Trump or Brexit, all gathered in one very expensive Swiss ski resort.

And in the Col Kilgore role was, of course, Donald Trump.

How did he do? Did he – to quote another movie – unleash hell?

He most surely did and it was great entertainment. But more importantly than that, it was great statesmanship. Like his similarly brilliant Warsaw speech last year, Trump’s speech in Davos today establishes him as – by some margin – the most significant and inspirational and ideologically robust leader of the free world since the era of Ronald Reagan.

Let’s examine some key moments in his speech.

I’m here to represent the interests of the America people and affirm America’s friendship and partnership in building a better world.

CNN is already misrepresenting it as another of Trump’s America First speeches. But Trump’s vision is not – as his enemies mischaracterise it – one of insularity and narrow self interest. It’s about Making America Great Again in order to help the world follow by example and become great too.

Like all nations represented at this great forum, America hopes for a future which everyone can prosper and every child can grow up free from violence, poverty, and fear. Over the past year, we have made extraordinary strides in the U.S. We’re lifting up forgotten communities, creating exciting new opportunities, and helping every American find their path to the American dream. The dream of a great job, a safe home and a better life for their children.

Trump is restating here truths which ought to be obvious: that the goal of any government should be to create an environment which fosters jobs, security and prosperity. But look at what those enlightened, liberal regimes of Sweden and Germany have done to their people’s security by inviting in all those Muslim migrants; look at the harm the sustainability agenda has done to prosperity. Fifty years ago, Trump’s words might have sounded platitudinous; today they are almost daringly controversial and outspoken.

Since my election we’ve created 2.4 million jobs and that number is going up very, very substantially. Small business optimism is at an all-time high. New unemployment claims are near the lowest we’ve seen in almost half a century. African-American unemployment reached the lowest rate ever recorded in the United States and so has unemployment among Hispanic-Americans.

Free market capitalism – the thing the Davos elite work so hard to stifle with their taxes and regulations and bureaucracy and wealth redistribution and obsession with climate change – makes EVERYONE more prosperous.

I believe in America.

I am not Barack Obama.

As president of the United States I will always put America first just like the leaders of other countries should put their country first also.

This ought to be another of those “duh” statements – something so blindingly obvious that your eyes glaze over at the sheer platitudinous tedium of it all. Except it’s not, is it? It contradicts pretty much everything our global elite – from our political leaders to our MSM columnists – have been telling us these last few decades: that nationalism is something that we should feel ashamed of; that patriotism is inward looking – and probably racist; that our foreign aid budgets are at least as important – if not more so – than the taxpayers’ money we spend on domestic issues. So, thank you Donald Trump for restoring some common sense to global policy making.

We support free trade but it needs to be fair and it needs to be reciprocal because in the end unfair trade undermines us all. The United States will no longer turn a blind eye to unfair economic practices including massive intellectual property theft, industrial subsidies, and pervasive state-led economic planning.

Some of President Trump’s critics on the right say that they don’t like him because he’s a protectionist who doesn’t believe in free trade. Nope. What Trump doesn’t believe in – as he outlines here – is unfair trade. He’s talking about the tariff barriers and (often illegal) state subsidies in the EU customs union; the currency manipulations and industrial espionage and dumping which are China’s speciality. Why should the U.S. put up with this?

Only by insisting on fair and reciprocal trade can we create a system that works not just for the U.S., but for all nations.

Stick and carrot: play dirty and the U.S. will punish you; play nicely and the whole world will be a better, more prosperous place.

We have dramatically cut taxes to make America competitive. We are eliminating burdensome regulations at a record pace. We are reforming the bureaucracy to make it lean, responsive and accountable and we are insuring our laws are enforced fairly.

Possibly the most important moment in Trump’s speech. The political and economic tendency across the West has been towards bigger government, higher taxation and greater regulation – most especially in entities like the European Union, and heartily endorsed by the Social Democrat types who infest Davos. Here Trump is offering the world a different economic and political paradigm: one that will make people richer and freer.

Energy is abundant and affordable.

This is Trump’s only reference in his entire speech to one of the most controversial (and, of course, correct) decisions of his presidency: to move the U.S. away from the man-made global warming scare narrative and towards the cheap energy policy which is going to give it a massive competitive advantage over those countries  (notably the EU member states) which continue to push renewables. It’s uncharacteristically tactful of him. Then again, as Napoleon said, “Never interrupt your enemy when he is in the process of making a mistake.”

My administration is proud to have led historic efforts at the united nations security council and all around the world to unite all civilized nations in our campaign of maximum pressure to de-nuke the Korean peninsula. We continue to call on partners to confront Iran’s support for terrorists and block Iran’s path to a nuclear weapon. We’re also working with allies and partners to destroy jihad it terrorist organizations such as ISIS and very successfully so.

Note the emphasis on “allies and partners” here. Yes, U.S. is ready to undertake its traditional responsibilities as world policeman. But others must pull their weight and the U.S. is not going to be in the business of neo-con-style solo adventurism.

We must invest in our people. When people are forgotten the world becomes fractured. Only by hearing and responding to the voices of the forgotten can we create a bright future that is truly shared by all. The nation’s greatness is more than the sum of its production and a nation’s greatness is the sum of its citizens, the values, pride, love, devotion and character of the people who call that nation home.

Yes. People. You, me, our families, our friends. Real people. Individuals. As in the exact opposite of what Davos Man believes in. Davos Man hates people. He prefers abstracts.

Represented in this room are some of the remarkable citizens from all over the worlds. You are national leaders, business titans, industry giants and many of the brightest mind in many fields. Each of you has the power to change hearts transform lives and shape your country’s destinies. With this power comes an obligation however, a duty of loyalty to the people, workers, customers, who made you who you are.

Trump is treating Davos Man (and Woman) here with a respect they really don’t deserve. Again, the globalist elite have never cared about “the people, workers, customers” because they’re about top down control, about bureaucracies which suppress individual aspiration and administrations, like the EU, which are anti-democratic by design. He is firing a shot across their bows. (Which, by the way, they will completely ignore…)

Today, I am inviting all of you to become part of this incredible future we are building together. 

Trump is offering the world a choice: one that has not been properly articulated by any Western leader since the era of Margaret Thatcher and Ronald Reagan. Do you choose to follow United States down the path of freedom, liberalised markets, low taxes, minimal government, civil order, security, controlled immigration, the pursuit of an acknowledged national interest? Or do you still want to follow Davos Man down the globalist path towards more regulation, more immigration, higher taxes and general embarrassment about the achievements of Western Civilization? Your move…

Thank you and God bless you all. Thank you very much.

Thank YOU, Mr President.'''

len(text.split())


# In[26]:


get_ipython().run_cell_magic('time', '', 'k_vp, xp = predict_data(text)\n\nprint(xp[0])')


# In[27]:


get_ipython().run_cell_magic('time', '', '\ndef train_setup():\n    def val_set(x,y):\n        val_size = .15\n        val_ind = int(len(x)*val_size)\n        print (val_ind,len(x))\n\n        randomize = np.arange(len(x))\n        np.random.shuffle(randomize)\n\n\n        x = np.array(x)[randomize]\n        y = np.array(y)[randomize]\n\n        x = x[:-val_ind]\n        y = y[:-val_ind]\n        x_val = x[-val_ind:]\n        y_val = y[-val_ind:]\n        assert len(y) ==len(x)\n\n        return x,y, x_val,y_val\n    k_v,X,Y = prep_data()\n    x,y, x_val,y_val = val_set(X,Y)')


# In[28]:


i = np.random.randint(0,len(xp))
print(i)
print(xp[i])
#print([k_v.labels[n] for n,v in enumerate(y[i]) if v >0])
for word in xp[i]:
    if word:
        print (k_vp.rev_lookup[word])


# In[29]:


def dnn():

    Sequential = keras.models.Sequential
    load_model = keras.models.load_model
    Tokenizer = keras.preprocessing.text.Tokenizer
    Activation = keras.layers.Activation
    SGD = keras.optimizers.SGD
    Adam = keras.optimizers.Adam
    BatchNormalization = keras.layers.BatchNormalization
    to_categorical = keras.utils.to_categorical
    ModelCheckpoint = keras.callbacks.ModelCheckpoint
    Embedding = keras.layers.Embedding
    Reshape = keras.layers.Reshape
    Flatten = keras.layers.Flatten
    Dropout = keras.layers.Dropout
    Concatenate = keras.layers.Concatenate
    Dense = keras.layers.Dense
    Model = keras.models.Model
    Input = keras.layers.Input
    Conv2D = keras.layers.Conv2D
    MaxPool2D = keras.layers.MaxPool2D

    n_classes = 17

    def define_model_rnn():
        vector_len = x[0].shape[0]
        vocab_size = k_v.n_words
        embedding_dim = 10
        model = Sequential()
        model.add(keras.layers.Embedding(vocab_size, embedding_dim, input_shape=(vector_len,)))
        model.add(keras.layers.GRU(3, dropout=0.2, recurrent_dropout=0.2))
        model.add(Activation('relu'))
        model.add(Dense(
            n_classes,))
        model.add(Activation('sigmoid'))
        return model

    def define_model():
        vector_len = k_v.n_words
        model = Sequential()
        model.add(Dense(128, input_shape=(vector_len,)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dense(32))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dense(
            n_classes,))
        model.add(Activation('sigmoid'))
        return model

    def define_model_cnn():

        sequence_length = x.shape[1]
        vocabulary_size = k_v.n_words
        embedding_dim = 5
        filter_sizes = [3, 4, 5]
        num_filters = 512
        drop = 0.5

        epochs = 100
        batch_size = 30

        inputs = Input(shape=(sequence_length,), dtype='int32')
        embedding = Embedding(
            input_dim=vocabulary_size, output_dim=embedding_dim, input_length=sequence_length)(inputs)
        reshape = Reshape((sequence_length, embedding_dim, 1))(embedding)

        conv_0 = Conv2D(
            num_filters,
            kernel_size=(filter_sizes[0], embedding_dim),
            padding='valid',
            kernel_initializer='normal',
            activation='relu')(reshape)
        conv_1 = Conv2D(
            num_filters,
            kernel_size=(filter_sizes[1], embedding_dim),
            padding='valid',
            kernel_initializer='normal',
            activation='relu')(reshape)
        conv_2 = Conv2D(
            num_filters,
            kernel_size=(filter_sizes[2], embedding_dim),
            padding='valid',
            kernel_initializer='normal',
            activation='relu')(reshape)

        maxpool_0 = MaxPool2D(
            pool_size=(sequence_length - filter_sizes[0] + 1, 1), strides=(1, 1),
            padding='valid')(conv_0)
        maxpool_1 = MaxPool2D(
            pool_size=(sequence_length - filter_sizes[1] + 1, 1), strides=(1, 1),
            padding='valid')(conv_1)
        maxpool_2 = MaxPool2D(
            pool_size=(sequence_length - filter_sizes[2] + 1, 1), strides=(1, 1),
            padding='valid')(conv_2)

        concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])
        flatten = Flatten()(concatenated_tensor)
        dropout = Dropout(drop)(flatten)
        output = Dense(units=n_classes, activation='sigmoid')(dropout)
        model = Model(inputs=inputs, outputs=output)
        return model


    label_dict = {k: i for i, k in enumerate(k_v.labels)}

    
    
    
    print('starting training')

    def train():
        model = define_model_cnn()
        lr1 = Adam(lr=0.00005)
        lr2 = Adam(lr=0.0001)
        adam = Adam(lr=0.001)
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='auto')
#         checkpointer = ModelCheckpoint(filepath='CNN20k.h5', verbose=1, save_best_only=False)
        model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
        history = model.fit(
            np.array(x),
            np.array(y),
            epochs=1000,
            verbose=1,
            validation_data=(x_val, y_val),
            callbacks=[
                keras.callbacks.TensorBoard(log_dir='./logs/article_clean_20k_CNN_5d_SAVED', write_graph=False),
                early_stop,checkpointer
            ])

    train()
    
    

# dnn()


# In[30]:


get_ipython().run_cell_magic('time', '', "    \nload_model = keras.models.load_model\nmodel = load_model('CNN20k.h5')\n\n\n\n\nlabel_dict = {i: k for i, k in enumerate(k_vp.labels)}\n\npreds = [model.predict(np.array(text).reshape(1,-1)) for text in xp]\nnp.set_printoptions(precision=3, suppress=True)\npreds\npred_dict = {\n    label_dict[i]: round(float(p), 6) for i, p in enumerate([_ for _ in preds[0].flatten()])\n}\n\n\nfinal_output = [    {\n    label_dict[i]: round(float(p), 6) for i, p in enumerate([_ for _ in pred.flatten()])\n} for pred in preds]")


# In[31]:


import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')

res = pd.DataFrame(final_output)
res.transpose().plot(kind='barh');


# In[ ]:


from itertools import islice
rank = (_ for _ in k_v.rev_lookup.items())
[next(rank) for _ in range(18999)]
pprint([next(rank) for _ in range(1000)])

