#student 1: eli haddad   ID: 207931536
#student 2: samer najjar ID:207477522
import pandas as pd
import re
import numpy
train_file = pd.read_csv('C:/Users/win10/OneDrive/Desktop/machine_learning/HW/data_train2.csv',names=['Email', 'Label'])#load files
test_file = pd.read_csv('C:/Users/win10/OneDrive/Desktop/machine_learning/HW/data_test2.csv',names=['Email', 'Label'])
train_file['Email'] = train_file['Email'].str.replace('\W', ' ') #delete puncuation marks and change all letters to small
train_file['Email'] = train_file['Email'].str.lower()
train_file['Email'] = train_file['Email'].str.split()
train_file=train_file.dropna(axis = 0, how ='any') #delete line if there is a null value
test_file=test_file.dropna(axis = 0, how ='any')
dectionary = []
for Email in train_file['Email']:
    for word in Email:
        dectionary.append(word) #add all words to dectionary
dectionary = list(set(dectionary))
num_words_email = {dif: [0] * len(train_file['Email']) for dif in dectionary} 
for i, Email in enumerate(train_file['Email']):
    for word in Email:
       num_words_email[word][i] += 1
counter = pd.DataFrame(num_words_email)
new_train = pd.concat([train_file, counter], axis=1)
spam_list = new_train[new_train['Label'] == '1'] #a list of spam emails
non_spam_list = new_train[new_train['Label'] == '0'] # a list of non spam emails
spam_percent = len(spam_list) / len(new_train) # probabily of having a spam word form all words
non_spam_percent = len(non_spam_list) / len(new_train) # probabily of having a non spam words
wordinmail = spam_list['Email'].apply(len) #words in a spam email
spamcounter = wordinmail.sum() #add them
wordinmail_notspam = non_spam_list['Email'].apply(len) #words in a non spam email 
nonspamcounter = wordinmail_notspam.sum() # add them
total_num_words = len(dectionary) #total number of words in dectionary
pspam = {dif:0 for dif in dectionary} #parameters for spam
pnonspam = {dif:0 for dif in dectionary} #parameters for non spam
for word in dectionary: # we used smoothing set equal to 1
    ifspam = spam_list[word].sum()  
    pifspam = (ifspam + 1) / (spamcounter + total_num_words)
    pspam[word] = pifspam
    ifnotspam = non_spam_list[word].sum() 
    pifnotspam = (ifnotspam + 1) / (nonspamcounter + total_num_words)
    pnonspam[word] = pifnotspam
def classify(mail): # function to see if an email is spam or not
   mail = re.sub('\W', ' ', mail)
   mail = mail.lower().split()
   prob_spam = spam_percent #checking the probability of the email being spam depending on every word in it 
   prob_non_spam = non_spam_percent #and the chance of that word being in a spam email or not
   for word in mail:
        if word in pnonspam:
            prob_non_spam *= pnonspam[word]
        if word in pspam:
            prob_spam *= pspam[word]
   if prob_non_spam >= prob_spam:#if not spam probability is higher return 0
      return '0'
   elif prob_spam > prob_non_spam: #else return 1
      return '1'
accuracy=train_file.shape[0]

accuracy_counter = sum(1 for column in test_file['Email'])
for line in test_file.iterrows(): 
   line = line[1]
   if line['Label'] == classify(line['Email']): #check if we got the right result
    accuracy_counter += 1 
accuracy=accuracy_counter/accuracy
print("*********************************************")
print('the accuracy is :', accuracy) 
print("*********************************************")
    

    
