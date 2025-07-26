import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris

data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
df.head()

df.isnull().sum()


X = df.drop('target', axis=1)
y = df['target']


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)


from sklearn.tree import plot_tree
plt.figure(figsize=(20,10))
plot_tree(model, feature_names=X.columns, class_names=str(np.unique(y)), filled=True)
plt.show()


y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score 
print("Accuracy Score:", accuracy_score(y_test, y_pred))


from sklearn.metrics import classification_report
print("Classification Report:", classification_report(y_test, y_pred))

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',xticklabels=['Predicted: 0', 'Predicted: 1'],
           yticklabels=['Actual: 0', 'Actual: 1'])


plt.xlabel("Predicted Labels")
plt.ylabel("Actual Labels")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()