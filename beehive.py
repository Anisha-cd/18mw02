import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import statistics
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
# Importing The Dataset
dataset = pd.read_csv("E:\\sem 3\\sd lab\\DATA SET\\bee_data.csv")
print(dataset)
#checking of missing values
X=[]
dataset.isnull().sum()/len(dataset)*100
# saving missing values in a variable
a =dataset.isnull().sum()/len(dataset)*100
print(a)
# saving column names in a variable
variables =dataset.columns
#variable = [ ]
for i in range(0,9):
    if a[i]<=20:   #setting the threshold as 20%
            X.append(variables[i])
    print(X)    
#Label encoding
from sklearn.preprocessing import LabelEncoder
number=LabelEncoder()
dataset['subspecies']=number.fit_transform(dataset['subspecies'].astype(str))
print(dataset['subspecies'])
dataset['location']=number.fit_transform(dataset['location'].astype(str))
print(dataset['location'])
dataset['health']=number.fit_transform(dataset['health'].astype(str))
print(dataset['health'])


X = dataset.iloc[:, 5:7].values
print(X)
y=dataset['health']
#y = dataset.iloc[:, 6].values
print(y)

#splitting of training and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.50, random_state = 0)
print(X_train, X_test, y_train, y_test)
#Applying the model
model1 =DecisionTreeClassifier()
model2 = KNeighborsClassifier()
model3= LogisticRegression()

model1.fit(X_train,y_train)
score1=model1.score(X_test,y_test)

model2.fit(X_train,y_train)
score2=model2.score(X_test,y_test)

model3.fit(X_train,y_train)
score3=model3.score(X_test,y_test)

#predicting 
pred1=model1.predict(X_test)
#print(pred1)

pred2=model2.predict(X_test)
#print(pred2)

pred3=model3.predict(X_test)
#print(pred3)

final_pred = np.array([])
for i in range(0,len(X_test)):
    final_pred = np.append(final_pred,statistics.mode([pred1[i], pred2[i], pred3[i]]))
print(final_pred)
#applying Voting Classifier:Hard
from sklearn.ensemble import VotingClassifier
model1 = DecisionTreeClassifier(random_state=1)
model1.fit(X_train,y_train)
y_pred = model1.predict(X_test)
score1= model1.score(X_test,y_test)
print(score1)

model2 = KNeighborsClassifier()
model2.fit(X_train,y_train)
y_pred = model2.predict(X_test)
score2= model2.score(X_test,y_test)
print(score2)

model3 = LogisticRegression(random_state=1)
model3.fit(X_train,y_train)
y_pred = model3.predict(X_test)
score3= model3.score(X_test,y_test)
print(score3)

model = VotingClassifier(estimators=[('dt', model1),('knn',model2),('lr',model3)],voting='hard')
print(model)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
score4= model.score(X_test,y_test)
print(score4)

#from sklearn.utils.multiclass import unique_labels
#Eavluating the Model
def plot_confusion_matrix(y_test, y_pred, classes,normalize=False,title=None,cmap=plt.cm.Blues):
   if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
   cm = confusion_matrix(y_test, y_pred)
    # Only use the labels that appear in the data
   classes = [0,1,2,3,4,5]
   print(classes)
   if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
   else:
        print('Confusion matrix, without normalization')

   print(cm)

   fig, ax = plt.subplots()
   im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
   ax.figure.colorbar(im, ax=ax)
   # We want to show all ticks...
   ax.set(xticks=np.arange(cm.shape[1]),
          yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
   plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
   fmt = '.2f' if normalize else 'd'
   thresh = cm.max() / 2.
   for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
   fig.tight_layout()
   return ax
np.set_printoptions(precision=2)



# Plot non-normalized confusion matrix
plot_confusion_matrix(y_test, y_pred, classes=y,title='Confusion matrix, without normalization')
#fig1=plt.gcf()
#plt.savefig('beehive-withoutnormalizedcm.png')
#plt.show()
#Plot normalized confusion matrix
plot_confusion_matrix(y_test, y_pred, classes=y, normalize=True,title='Normalized confusion matrix')
fig2=plt.gcf()
fig2.savefig('beehivecm.png')
plt.show()
print('Accuracy')
print(accuracy_score(y_test, y_pred)) 

clsf_report = pd.DataFrame(classification_report(y_true = y_test, y_pred = y_pred, output_dict=True)).transpose()
clsf_report.to_csv('classification_report.csv', index= True)
print("Classification Report")
print(classification_report(y_test,y_pred))