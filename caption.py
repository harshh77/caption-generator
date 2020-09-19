import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
import re
import nltk
from nltk.corpus import stopwords
import string
import json
from time import time
import pickle
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras.preprocessing import image
from keras.models import Model, load_model
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Input, Dense, Dropout, Embedding, LSTM
from keras.layers.merge import add

#Read Text Captions
def readTextFile(path):
    with open(path) as f:
        captions=f.read()
    return captions

Captions=readTextFile("C://Users//Harsh Miglani//Desktop//ic//Flickr_Data//Flickr_Data//Flickr_TextData//Flickr8k.token.txt")
Captions=Captions.split("\n")
print(len(Captions))

Captions=Captions[:-1]

descriptions={}

for x in Captions:
    first,second=x.split('\t')
    image_name=first.split('.')[0]
    #if the image id is already present or not
    if descriptions.get(image_name) is None:
        descriptions[image_name]=[]
    descriptions[image_name].append(second)
    
descriptions["1000268201_693b08cb0e"]

IMG_PATH="C://Users//Harsh Miglani//Desktop//ic//Flickr_Data//Flickr_Data//Images/"
import cv2
image=cv2.imread(IMG_PATH+"3767841911_6678052eb6.jpg")
#cv2 reads image in BGR format
image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
plt.axis("off")
plt.imshow(image)
plt.show()

import re
def clean_text(sentence):
    sentence=sentence.lower()
    sentence=re.sub("[^a-z]+"," ",sentence)#We will use Regular Expression to replace all words which are not alphabets
    #[^a-z] means not in between a to z.
    #+ means more than one occurences of preceding characters.
    #We are replacing this with space.
    sentence=sentence.split()
    #We can remove all the words with length 1
    sentence=[s for s in sentence if len(s)>1]
    sentence=" ".join(sentence)
    return sentence

cleaned_text=clean_text("A cat is sitting over the house # 64")
print(cleaned_text)

# Clean all Captions
for key,caption_list in descriptions.items():
    for i in range(len(caption_list)):
        caption_list[i]=clean_text(caption_list[i])

descriptions["1000268201_693b08cb0e"]

with open("C://Users//Harsh Miglani//Desktop//ic//descriptions.txt",'w') as f:
    f.write(str(descriptions))
    
descriptions=None
with open("C://Users//Harsh Miglani//Desktop//ic//descriptions.txt",'r') as f:
    descriptions=f.read()
    json_acceptable_string=descriptions.replace("'","\"")
    descriptions=json.loads(json_acceptable_string)
    print(type(descriptions))
    
#Vocab
vocab=set()#Set() stores all the unique words
for key in descriptions.keys():
    [vocab.update(sentence.split()) for sentence in descriptions[key]]
print("Vocab Size :%d"%len(vocab))


#Total number of words across all the sentences.
total_words=[]
for key in descriptions.keys():
    [total_words.append(i) for des in descriptions[key] for i in des.split()]
print("Total Words %d"%len(total_words))

# Filter Words from the Vocab according to certain threshold frequncy
import collections
counter=collections.Counter(total_words)
Frequency_count=dict(counter)
print(len(Frequency_count.keys()))

# Sort this dictionary according to the freq count
sorted_frequency_count=sorted(Frequency_count.items(),reverse=True,key=lambda x:x[1]) 
#Filter
threshold=10
sorted_frequency_count=[x for x in sorted_frequency_count if x[1]>threshold]
total_words=[x[0] for x in sorted_frequency_count]

print(len(total_words))

Train_File_data=readTextFile("C://Users//Harsh Miglani//Desktop//ic//Flickr_Data//Flickr_Data//Flickr_TextData//Flickr_8k.trainImages.txt")
Test_File_data=readTextFile("C://Users//Harsh Miglani//Desktop//ic//Flickr_Data//Flickr_Data//Flickr_TextData//test.im1.txt")

train=[row.split(".")[0] for row in Train_File_data.split("\n")][:-1]
test=[row.split(".")[0] for row in Test_File_data.split("\n")[:-1]]
train[:5]
# Prepare Description for the Training Data
# Tweak - Add <s> and <e> token to our training data
train_descriptions={}
for image_id in train:
    train_descriptions[image_id]=[]
    for caption1 in descriptions[image_id]:
        caption_to_append="startseq "+caption1+" endseq"
        train_descriptions[image_id].append(caption_to_append)
        
train_descriptions["1000268201_693b08cb0e"]

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
import re
import nltk
from nltk.corpus import stopwords
import string
import json
from time import time
import pickle
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras.preprocessing import image
from PIL import Image 
from keras.models import Model, load_model
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Input, Dense, Dropout, Embedding, LSTM
from keras.layers.merge import add
#This cell must be executed before  all below cells else output will come.

model=ResNet50(weights='imagenet',input_shape=(224,224,3))
model.summary()

model_new=Model(model.input,model.layers[-2].output)

def preprocess_img(img):
    img=image.load_img(img,target_size=(224,224))
    img=image.img_to_array(img)
    img=np.expand_dims(img,axis=0)
    img=preprocess_input(img)
    return img

def encode_image(img):
    img=preprocess_img(img)
    feature_vector=model_new.predict(img)
    #print(feature_vector.shape)
    feature_vector=feature_vector.reshape((-1,))
    #print(feature_vector.shape)
    return feature_vector
    
encode_image(IMG_PATH+"1000268201_693b08cb0e.jpg")

encoding_train={}
start_time=time()
for ix,img_id in enumerate(train):
    img_path=IMG_PATH+"/"+img_id+".jpg"
    encoding_train[img_id]=encode_image(img_path)
    if ix%100==0:
        print("Encoding in Progress Time Step %d"%ix)
end_time=time()
print("Total Time taken :",(end_time-start_time))



with open("C://Users//Harsh Miglani//Desktop//ic//storage//encoded_train_images.pkl",'wb') as f:
    pickle.dump(encoding_train,f)
print(len(encoding_train))

encoding_test={}
start_time=time()
#for ix,img_id in enumerate(test):
img_path=IMG_PATH+"/3767841911_6678052eb6.jpg"
encoding_test[img_id]=encode_image(img_path)
    #if ix%100==0:
     #   print("Encoding in Progress Time Step %d"%ix)
end_time=time()
print("Total Time taken :",(end_time-start_time))
print(encoding_test)


with open("C://Users//Harsh Miglani//Desktop//ic//storage//encoded_test_images.pkl",'wb') as f:#wb is writebinary mode
    pickle.dump(encoding_test,f)#f is file pointer
print(len(encoding_test))

word_to_index={}
index_to_word={}
for i,word in enumerate(total_words):
    word_to_index[word]=i+1#i+1 ,because we will reserve the index zero
    index_to_word[i+1]=word
    
word_to_index["startseq"]=1846
word_to_index["endseq"]=1847
index_to_word[1846]="startseq"
index_to_word[1847]="endseq"

print(len(word_to_index))

vocab_size=len(word_to_index)+1
print("Vocab_Size : ",vocab_size)

max_len=0
for key in train_descriptions.keys():
    for caption in train_descriptions[key]:
        max_len=max(max_len,len(caption.split()))
print(max_len)

def data_generator(train_descriptions,encoding_train,word_to_index,max_len,batch_size):
    #batch_size is how many training examples we should have in 1 batch.
    #Our data has 2 parts-:1)Image 2)Partial Captions.
    #We are going to build partial captions
    X1,X2,y=[],[],[]
    n=0
    while True:
        for key,caption_list in train_descriptions.items():
            n+=1
            photo=encoding_train[key]
            for caption in caption_list:
                sequence=[word_to_index[word] for word in caption.split() if word in word_to_index]
                #for all unknown words(not present in word_to_index) ,we are going to ignore it 
                for i in range(1,len(sequence)):
                    xi=sequence[0:i]
                    yi=sequence[i]
                    #Later ,we will do padding which will ensure every xi is of same length
                    #using pad_sequences function available in keras,It accepts a 2D list and returns a 2D Matrix.
                    #[[xi]]--->Here xi is a 2D matrix.
                    #0 denote padding word
                    #padding='post' means we are adding zeros after Sequence of words.
                    xi=pad_sequences([xi],maxlen=max_len,value=0,padding='post')[0]#Since we have only one example,so we are extracting only first one..
                    yi=to_categorical([yi],num_classes=vocab_size)[0]#yi should be one hot vector.
                    #xi and yi --->one training point
                    #In our mini batch,we are going to append these values.
                    X1.append(photo)
                    X2.append(xi)
                    y.append(yi)
                    if n==batch_size:
                        yield[[np.array(X1),np.array(X2)],np.array(y)]
                        #We are not using return function as it is a Generator 
                        #and Generator remembers the state where the function was in the previous call 
                        #For next batch When control comes back again to this Generator Function:-
                        X1,X2,y=[],[],[]
                        n=0
                        
f=open("C://Users//Harsh Miglani//Desktop//ic//glove.6B.50d.txt",encoding='utf-8')#For windows,we will specify encoding='utf-8'
for line in f:
    values=line.split()
    print(values)
    break
import numpy as np

embedding_index={}#It will store word-vector for every word.
for line in f:
    values=line.split()
    word=values[0]
    word_embedding=np.array(values[1:],dtype='float')
    embedding_index[word]=word_embedding

    #Whenever we pass data to RNN/LSTM layer,that data must pass through embedding layer.
    #Either we can train as we go or we can preinitialize this layer like using Glove6B50D.txt,
    #But in our work,we don't need these 6 Billion words.
    #So ,We will make a matrix of (vocab_size,50).
    #For each word in vocab,we will have 50 dimensional vector.
    #How to construct this matrix from already trained GloveVectors.
f.close()

embedding_index["dog"]

def get_embedding_matrix():
    embedding_dimension=50
    Matrix=np.zeros((vocab_size,embedding_dimension))
    for word,index in word_to_index.items():
        embedding_vector=embedding_index.get(word)
        if embedding_vector is not None:
            Matrix[index]=embedding_vector
    return Matrix
        
embedding_matrix=get_embedding_matrix()
print(type(embedding_matrix))

print(embedding_matrix[8])
print(embedding_matrix.shape)

input_img_features=Input(shape=(2048,))
input_img1=Dropout(0.3)(input_img_features)
input_img2=Dense(256,activation='relu')(input_img1)

# Captions as Input
input_captions=Input(shape=(max_len,))
input_caption1=Embedding(input_dim=vocab_size,output_dim=50,mask_zero=True)(input_captions)
input_caption2=Dropout(0.5)(input_caption1)
input_caption3=LSTM(256)(input_caption2)

decoder1=add([input_img2,input_caption3]) 
decoder2=Dense(256,activation='relu')(decoder1)
outputs=Dense(vocab_size,activation='softmax')(decoder2)
# Combined Model
model=Model(inputs=[input_img_features,input_captions],outputs=outputs)

model.summary()

# Important Thing - Embedding Layer
model.layers[2].set_weights([embedding_matrix])
model.layers[2].trainable=False

model.compile(loss='categorical_crossentropy',optimizer='adam')

epochs=20
number_pics_per_batch=3
steps=len(train_descriptions)//number_pics_per_batch

def train_model():
    for i in range(epochs):
        generator=data_generator(train_descriptions,encoding_train,word_to_index,max_len,number_pics_per_batch)
        model.fit_generator(generator,epochs=1,steps_per_epoch=steps,verbose=1)
        model.save('C://Users//Harsh Miglani//Desktop//ic//model_weights//model_'+str(i)+'.h5')
        
train_model()

model=load_model('C://Users//Harsh Miglani//Desktop//ic//model_weights//model_9.h5')#These are the model weights after 10 epoch.

def predict_caption(photo):
    in_text="startseq"
    for i in range(max_len):
        sequence=[word_to_index[w] for w in in_text.split() if w in word_to_index]
        sequence=pad_sequences([sequence],maxlen=max_len,padding='post')
        y_predicted=model.predict([photo,sequence])
        y_predicted=y_predicted.argmax()#Word with max prob always - Greedy Sampling
        word=index_to_word[y_predicted]
        in_text+=(' '+word)
        if word=='endseq':
            break
        final_caption=in_text.split()[1:-1]
        final_caption=' '.join(final_caption)
    return final_caption

# Pick Some Random Images and See Results
# for i in range(95):
#     index=np.random.randint(0,1000)
#     all_image_names=list(encoding_test.keys())
#     image_name=all_image_names[index]
#     photo_2048=encoding_test[image_name].reshape((1,2048))
#     #Because,To our model ,we are feeding Batch_size cross 2048,(It has two axes).
#     #We are generating predictions for one image at a time.
#     i=plt.imread(IMG_PATH+image_name+".jpg")
#     caption=predict_caption(photo_2048)
#     print(caption)
#     plt.axis("off")
#     plt.imshow(i)
#     plt.show()
    
all_image_names=list(encoding_test.keys())
image_name=all_image_names[0]
photo_2048=encoding_test[image_name].reshape((1,2048))
i=plt.imread(IMG_PATH+"/3767841911_6678052eb6.jpg")
caption=predict_caption(photo_2048)
print(caption)
plt.axis("off")
plt.imshow(i)
plt.show()