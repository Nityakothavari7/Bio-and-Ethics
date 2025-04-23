## PREDICTION OF RHEUMATOID ARTHRITIS SEVERITY USING BIOMARKERS AND BLOCKCHAIN

## PROBLEM STATEMENT
Autoimmune diseases like rheumatoid arthritis (RA) involve multiple biomarkers like cytokines and immune cell markers. RA diagnosis and treatment strategies often depend on subjective clinical evaluations, leading to treatment delays. So, there’s a need to develop a model to predict RA severity and to suggest treatment strategies.

## PROPOSED SOLUTION
To develop a Graphical User Interface (GUI) which allows the users to input the biomarker levels which can predict the severity of RA using a Machine learning algorithm and suggest treatments. We will be integrating Blockchain technology into our project for data privacy.

## DATASET ACQUISITION
The datasets were taken from the NCBI’s Gene Expression Omnibus(GEO) and hence their respective accession IDs are given here

1. GSM1068616 - IL-10
2. GSM993402 - IL-17
3. GSM993406 - TNF-Alpha 
4. GSM258773 - CD4+
5. GSM211514 - IL-6

Link for the dataset is https://www.ncbi.nlm.nih.gov/geo/
search with IDs Given above

## METHODOLOGY
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
  Testing is done by plotting the validation loss and training loss. The trained model is stored in the device its run by in xgboost_RA_severity_model.json as a .json file. The Feature scaler is saved as scaler.pk1 and the Label encounter as label_encoder.pk1.

## BUILDING GUI 

  	1. Using Tkinter in python we built the Graphical User Interface (GUI) for entering biomarker levels(5) and get prediction using the trained model.
 	2. Tkinter is a standard GUI library for Python which provides a fast and easy way to create desktop applications.
  	3. The pre-trained model was loaded for making predictions.The application window was given a sky blue background.
  	4. 5 Input fields were created and a “prediction severity” button was added to get prediction.
  	5. This GUI takes input biomarker levels, scales them, makes a prediction and displays the result.
  	6. This GUI is useful for lab researchers, clinicians and doctors to analyze patients with RA severity.

## BLOCKCHAIN INTEGRATION

 	 Data Bundling
    - Each time a user inputs biomarker levels and gets a severity prediction, the data is bundled into a "block" along with a timestamp.
  	Blockchain Structure
    - These blocks are added to a secure, linked digital ledger (the "blockchain").
	Immutability
    - Once added, the blocks cannot be changed, ensuring tamper-proof data.
	Data Integrity
    - This prevents alteration of past medical records, ensuring accurate and trustworthy information.
	Transparency & Security
    - Blockchain guarantees full transparency, security, and integrity, critical for sensitive health data tracking.
## ETHICAL STANDARDS:

The ethical standards taken into consideration when dealing with AI based systems are:

	a. The system is designed with minimum complexity and more user friendliness so that can serve the purpose respecting human rights and enhancing their capabilities it by serving their needs. 
	b. User trust is gained by providing transparency ensuring of how the system takes decisions. The model counts on moral human society and human rights.
	c. Given to its transparency user might think that the system is not safe for use but our first intention is to be private. Our system is Tamper-evident which is by the blockchain integration.

## Libraries used in python:
	pandas for data handling
	numpy for math operations
	xgboost for machine learning
	jonlib for saving/loading models
	hashlib for blockchain security 
	tkinter for creating the user interface
