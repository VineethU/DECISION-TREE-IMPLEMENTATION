import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target


print("Missing values:\n", df.isnull().sum())


X = df.drop('target', axis=1)
y = df['target']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=10)


model = DecisionTreeClassifier(max_depth=3, criterion='entropy', random_state=10)
model.fit(X_train, y_train)


plt.figure(figsize=(18,8))
plot_tree(model, 
          feature_names=X.columns, 
          class_names=data.target_names,
          filled=True)
plt.title("Modified Decision Tree Visualization")
plt.show()


y_pred = model.predict(X_test)


print("Accuracy Score:", accuracy_score(y_test, y_pred))


print("Classification Report:\n", classification_report(y_test, y_pred, target_names=data.target_names))


cm = confusion_matrix(y_test, y_pred)


plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
            xticklabels=data.target_names,
            yticklabels=data.target_names)
plt.xlabel("Predicted Labels")
plt.ylabel("Actual Labels")
plt.title("Modified Confusion Matrix")
plt.tight_layout()
plt.show()
