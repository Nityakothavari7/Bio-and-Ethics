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
# Python is used for pre-processing, training the model, building GUI and Blockchain integration 
  1. CSV file containing rheumatology biomarers are taken as input.
   Features:
   Cytokine Biomarkers:
   IL-6,
   IL-17,
   TNF-alpha,
   IL-10 and
   Imunne Cellmarker:
   CD4+.

   The taken input values are raw biomarker values which are then standardized using conversion functions.
   Severity labels are given as follows for predicting the severity. The system predicts the person has

     1. Severe RA if TNF-alpha is greater than 10 and IL-6 is greater than 40 or IL-10 is greater than 25 and IL-17 is greater than 20 or CD+4 is greater than 6.
     2. Moderate RA is if TNF-alpha is greater than 5 and IL-6 is greater than 10 or IL-10 is greater than 10 and IL-17  is greater than 8 or CD+4 is greater than 3.
     3. If the above statement are not satisfied then the system predicts that the preson has mild RA.

  2. Machine Learning Implementation

  In python, using the XG-Boost(v1.7.6) algorithm pre-processing and training is performed.
MinMax Scaler is used in pre-processing. The data is split to 80-20 for traninig and testing respectively. 

   










