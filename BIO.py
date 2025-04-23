import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import hashlib
import json
import datetime
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score
from tkinter import messagebox

file_path = r"C:\Users\shari\Downloads\RA_severity_dataset_without_severity.csv"
df = pd.read_csv(file_path)

def convert_IL6(mbei_value):
    return 5 * mbei_value  

def convert_IL17(log2_gcrma_value):
    return 2 ** log2_gcrma_value * 1.2  

def convert_TNF(log2_gcrma_value):
    return 2 ** log2_gcrma_value * 1.5  

def convert_IL10(rma_value):
    return 3.5 * rma_value  

df["IL-6"] = df["IL-6"].apply(convert_IL6)
df["IL-17"] = df["IL-17"].apply(convert_IL17)
df["TNF-Alpha"] = df["TNF-Alpha"].apply(convert_TNF)
df["IL-10"] = df["IL-10"].apply(convert_IL10)

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

df.to_csv("RA_severity_dataset_with_serum_levels.csv", index=False)

biomarkers = ["IL-6", "IL-17", "TNF-Alpha", "IL-10", "CD4+"]
target_column = "Severity"

if not os.path.exists("xgboost_RA_severity_model.json"):
    label_encoder = LabelEncoder()
    df[target_column] = label_encoder.fit_transform(df[target_column])
    
    scaler = MinMaxScaler()
    df[biomarkers] = scaler.fit_transform(df[biomarkers])
    
    X = df[biomarkers]
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBClassifier(
        use_label_encoder=False,
        tree_method="hist",
        objective="multi:softmax",
        num_class=3,
        eval_metric="mlogloss"
    )
    model.fit(
        X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=True
    )

    if not os.path.exists("xgboost_RA_severity_model.json"):
        label_encoder = LabelEncoder()
        df[target_column] = label_encoder.fit_transform(df[target_column])
    
        scaler = MinMaxScaler()
        df[biomarkers] = scaler.fit_transform(df[biomarkers])
    
        X = df[biomarkers]
        y = df[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = xgb.XGBClassifier(
            use_label_encoder=False,
            tree_method="hist",
            objective="multi:softmax",
                num_class=3,
        eval_metric="mlogloss"
    )
    model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=True)
    evals_result = model.evals_result()

    plt.figure(figsize=(8, 5))
    plt.plot(evals_result["validation_0"]["mlogloss"], label="Training Loss", color="blue")
    plt.plot(evals_result["validation_1"]["mlogloss"], label="Validation Loss", color="red", linestyle="dashed")
    plt.xlabel("Epoch")
    plt.ylabel("Log Loss")
    plt.title("XGBoost Loss Curve")
    plt.legend()
    plt.grid()
    plt.show()
    
    model.save_model("xgboost_RA_severity_model.json")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(label_encoder, "label_encoder.pkl")  
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"XGBoost Test Accuracy: {accuracy:.4f}")

else:
    model = xgb.XGBClassifier()
    model.load_model("xgboost_RA_severity_model.json")
    scaler = joblib.load("scaler.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    
class SimpleBlockchain:
    def _init_(self):
        self.chain = []
        self.create_genesis_block()
        self.load_chain()
        
    def create_genesis_block(self):
        if not self.chain:
            genesis_data = "Genesis Block"
            self.add_block(genesis_data)
    
    def add_block(self, data):

        formatted_data = {
            "metadata": {
                "version": "1.0",
                "hospital": "RA Clinic Network",
                "data_type": "patient_record"
            },
            "patient_data": data  
        }
    
 
        hash = self.calculate_hash(index, previous_hash, timestamp, formatted_data)

        
        block = {
            'index': index,
            'timestamp': timestamp,
            'data': data,
            'previous_hash': previous_hash,
            'hash': hash
        }
        
        self.chain.append(block)
        self.save_chain()
    
    def calculate_hash(self, index, previous_hash, timestamp, data):
        value = f"{index}{previous_hash}{timestamp}{json.dumps(data)}"
        return hashlib.sha256(value.encode()).hexdigest()
    
    def save_chain(self):
        with open('patient_records.chain', 'w') as f:
            json.dump(self.chain, f, indent=4, sort_keys=True)
    
    def load_chain(self):
        if os.path.exists('patient_records.chain'):
            with open('patient_records.chain', 'r') as f:
                self.chain = json.load(f)

AUTHORIZED_USERS = {
    "doctor1": "password123",
    "doctor2": "secure456"
}

TREATMENTS = {
    "Mild": ["NSAIDs (e.g., Ibuprofen)", "Physical Therapy"],
    "Moderate": ["DMARDs (e.g., Methotrexate)", "Low-dose Steroids"],
    "Severe": ["Biologics (e.g., TNF inhibitors)", "Surgical Consultation"]
}

import tkinter as tk
from tkinter import ttk
import numpy as np
import xgboost as xgb
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

model = xgb.XGBClassifier()
model.load_model("xgboost_RA_severity_model.json")

scaler = joblib.load("scaler.pkl")

biomarkers = ["IL-6", "IL-17", "TNF-Alpha", "IL-10", "CD4+"]

df = pd.read_csv("RA_severity_dataset_with_serum_levels.csv")
label_encoder = LabelEncoder()
df["Severity"] = label_encoder.fit_transform(df["Severity"])

class RAApp:
    def _init_(self):
        self.blockchain = SimpleBlockchain()
        self.current_user = None

        self.login_window = tk.Tk()
        self.login_window.title("Medical Login")
        self.login_window.configure(bg="#87CEEB")
        
        ttk.Label(self.login_window, text="Username:", background="#87CEEB").grid(row=0, padx=10, pady=5)
        self.username = ttk.Entry(self.login_window)
        self.username.grid(row=0, column=1, padx=10)
        
        ttk.Label(self.login_window, text="Password:", background="#87CEEB").grid(row=1, padx=10, pady=5)
        self.password = ttk.Entry(self.login_window, show="*")
        self.password.grid(row=1, column=1, padx=10)
        
        ttk.Button(self.login_window, text="Login", command=self.check_login).grid(row=2, columnspan=2, pady=10)
        
        self.login_window.mainloop()
    
    def check_login(self):
        user = self.username.get()
        pwd = self.password.get()
        
        if AUTHORIZED_USERS.get(user) == pwd:
            self.current_user = user
            self.login_window.destroy()
            self.create_main_app()
        else:
            messagebox.showerror("Error", "Invalid credentials")
    
    def create_main_app(self):
        self.root = tk.Tk()
        self.root.title(f"RA Severity Prediction - {self.current_user}")
        self.root.geometry("800x600")
        self.root.configure(bg="#87CEEB")
        
        self.entry_vars = {}
        for i, bio in enumerate(biomarkers):
            label = tk.Label(self.root, text=bio, fg="black", bg="#87CEEB", font=("Arial", 12, "bold"))
            label.place(x=50, y=50 + i * 50)
            
            self.entry_vars[bio] = tk.StringVar()
            entry = ttk.Entry(self.root, textvariable=self.entry_vars[bio], width=10)
            entry.place(x=200, y=50 + i * 50)

        predict_button = tk.Button(self.root, text="Predict Severity", command=self.on_predict, 
                                 fg="white", bg="blue", font=("Arial", 12, "bold"))
        predict_button.place(x=350, y=300)
        
        self.result_label = tk.Label(self.root, text="", font=("Arial", 14, "bold"), 
                                   bg="light yellow", fg="dark blue", wraplength=400)
        self.result_label.place(x=200, y=350)
        
        self.history = tk.Listbox(self.root, height=8)
        self.history.place(x=50, y=450, width=700)
        self.load_history()
        
        self.root.mainloop()
    
    def on_predict(self):
        try:
            user_inputs = np.array([[float(self.entry_vars[bio].get()) for bio in biomarkers]])

            scaled_inputs = scaler.transform(user_inputs)
            prediction = model.predict(scaled_inputs)[0]
            severity = label_encoder.inverse_transform([prediction])[0]

            patient_data = {
                "values": {bio: self.entry_vars[bio].get() for bio in biomarkers},
                "severity": severity,
                "doctor": self.current_user,
                "timestamp": str(datetime.datetime.now())
            }
            self.blockchain.add_block(patient_data)
            
            treatment = "\nâ€¢ ".join(TREATMENTS[severity])
            self.result_label.config(
                text=f"Predicted Severity: {severity}\n\nSuggested Treatments:\nâ€¢ {treatment}"
            )
            
            self.history.insert(0, f"{datetime.date.today()} - {severity}")
            
        except Exception as e:
            self.result_label.config(text="Invalid input! Please check all values.", fg="red")
    
    def load_history(self):
        if os.path.exists('patient_records.chain'):
            with open('patient_records.chain', 'r') as f:
                chain = json.load(f)
                for block in reversed(chain[1:]): 
                    entry = f"{block['timestamp'][:10]} - {block['data']['severity']}"
                    self.history.insert(0, entry)


if _name_ == "_main_":
    RAApp()