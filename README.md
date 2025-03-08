This is the code for Multi-layer perceptron model to predict Rheumatoid arthritis(RA) severity with three labels mild, moderate and severe which achieved training accuracy of 56.34%



import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\kotha\OneDrive\Desktop\biomarkers.csv")

cols_to_convert = ["TNF-Alpha", "IL-6", "IL-10", "IL-17", "CD4+"]
df[cols_to_convert] = df[cols_to_convert].apply(pd.to_numeric, errors='coerce')

def classify_severity(row):
    if ((row["TNF-Alpha"] > 10 and row["IL-6"] > 40) or 
        (row["IL-10"] > 25 and row["IL-17"] > 20) or 
        (row["CD4+"] > 6) or 
        (row["TNF-Alpha"] > 15 or row["IL-6"] > 50)):
        return "Severe"
    
    elif ((row["TNF-Alpha"] > 5 and row["IL-6"] > 10) or 
          (row["IL-10"] > 10 and row["IL-17"] > 8) or 
          (row["CD4+"] > 3) or 
          (row["TNF-Alpha"] > 8 or row["IL-6"] > 25)):  
        return "Moderate"

    else:
        return "Mild"

df["Severity"] = df.apply(classify_severity, axis=1)

df.to_csv(r"C:\Users\kotha\OneDrive\Desktop\RA_severity_dataset.csv", index=False)

df = pd.read_csv(r"C:\Users\kotha\OneDrive\Desktop\RA_severity_dataset.csv")

X = df[["TNF-Alpha", "IL-6", "IL-10", "IL-17", "CD4+"]]
y = df["Severity"]

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)  

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),  
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(3, activation='softmax')  
])

optimizer = Adam(learning_rate=0.005)  
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=25, batch_size=64, validation_data=(X_test, y_test))

train_loss = history.history['loss'][-1]
val_loss = history.history['val_loss'][-1]
train_accuracy = history.history['accuracy'][-1]
val_accuracy = history.history['val_accuracy'][-1]
test_loss, test_accuracy = model.evaluate(X_test, y_test)

print(f"Training Loss: {train_loss:.4f}")
print(f"Validation Loss: {val_loss:.4f}")
print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

model.save("RA_Severity_model.h5")

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()

plt.show()
