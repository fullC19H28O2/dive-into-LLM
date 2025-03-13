from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC 
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score


iris = datasets.load_iris()
X=iris.data
y=iris.target
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42) 

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

svm_model=SVC(kernel='linear')
svm_model.fit(X_train,y_train)

y_pred = svm_model.predict(X_test)

print("Confusion Matrix:")
print(confusion_matrix(y_test,y_pred))


print("\nClassificatiom Report:")
print(classification_report(y_test,y_pred))


print("\nAccuracy Score:" )
print(accuracy_score(y_test,y_pred))
