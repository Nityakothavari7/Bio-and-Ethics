## PREDICTION OF RHEUMATOID ARTHRITIS SEVERITY USING BIOMARKERS AND BLOCKCHAIN

## Problem Statement
Autoimmune diseases like rheumatoid arthritis (RA) involve multiple biomarkers like cytokines and immune cell markers. RA diagnosis and treatment strategies often depend on subjective clinical evaluations, leading to treatment delays. So, there’s a need to develop a model to predict RA severity and to suggest treatment strategies.

## Proposed Solution
To develop a Graphical User Interface (GUI) which allows the users to input the biomarker levels which can predict the severity of RA using a Machine learning algorithm and suggest treatments. We will be integrating Blockchain technology into our project for data privacy.

## Dataset Acquisition
The datasets were taken from the NCBI’s Gene Expression Omnibus(GEO) and hence their respective accession IDs are given here

1. GSM1068616 - IL-10
2. GSM993402 - IL-17
3. GSM993406 - TNF-Alpha 
4. GSM258773 - CD4+
5. GSM211514 - IL-6
Link for the dataset is https://www.ncbi.nlm.nih.gov/geo/
search with IDs Given above

## Methodology
1. Pre-processing using MinMaxScaler in Python(20K samples  taken in each biomarker)
2. Conversion of log2 normalised gene expression data to  serum  levels.
3. Application of threshold values for each of the biomarkers to detect Rheumatoid Arthritis 
4. Xgboost (gradient boosting) algorithm used to predict RA)
5. Implemented GUI using tkinter to input user serum levels of biomarkers
6. Use of Blockchain technology to secure user data




