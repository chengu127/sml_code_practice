#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 19:53:58 2024

@author: chengu
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
#import data
iris = load_iris()
X, y = iris.data, iris.target

# split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Verify the sizes to ensure they match
print(f'X_train size: {X_train.shape}')  # Should be (120, 4)
print(f'y_train size: {y_train.shape}')  # Should be (120,)
print(f'X_test size: {X_test.shape}')    # Should be (30, 4)
print(f'y_test size: {y_test.shape}')    # Should be (30,)

X_train, X_val, y_train, y_val = train_test_split(X_train,y_train, test_size= 0.25, random_state=42)
print(f'X_train size: {X_train.shape}')  # Should be (120, 4)
print(f'y_train size: {y_train.shape}')  # Should be (120,)
print(f'X_test size: {X_val.shape}')    # Should be (30, 4)
print(f'y_test size: {y_val.shape}')   
#train the model
knn =KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)

#validate the model, evaluate model's performance on the validation set
y_val_pred = knn.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f"validation accuracy:{val_accuracy:.2f}")

#tuning the hyperpermater
best_k =1
best_accuracy =0
for k in range (1,11):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,y_train)
    y_val_pred = knn.predict(X_val)
    val_accuracy = accuracy_score(y_val,y_val_pred)
    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        best_k =k
        
print(f"best k:{best_k},Validation accuracy:{best_accuracy:.2f}")

#final model training
X_train_final = np.vstack((X_train,X_val))
y_train_final = np.hstack((y_train,y_val))

#train the final model
knn_final = KNeighborsClassifier(best_k)
knn_final.fit(X_train_final,y_train_final)

#evaluate the final model
y_test_pred = knn_final.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"test accuracy:{test_accuracy:.2f}")