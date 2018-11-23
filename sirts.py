
#!/usr/bin/env python

# -------------------------------------------------------------------
# Predict acetylated lysine residues using machine learning 
#
#   The program will read the peptides from a CSV file and use 80% of the 
#   peptides to train the model. The remaining 20% will be used to validate 
#   the model. The protein sequence to be analyzed will be read from Entrez
#   and split into peptides 13 amino acids in length, surrounding each lysine
#   residuue. Predictions will be made on these peptides and the results are
#   printed to the screen. 
#  
# Input:    1. A CSV file containing the peptides used for training
#           2. A protein ID  
#
# The sample SIRT1 and SIRT2 peptides were derived from PMID 23995836 
#           SupplData1-ncomms3327-s2.xlsx
#
# Output:   1. Training model description
#           2. Protein sequence in FASTA format 
#           3. A list of potential K-acetylated peptides 
#
# Background reading:
#   https://towardsdatascience.com/hands-on-introduction-to-scikit-learn-sklearn-f3df652ff8f2
#   https://jakevdp.github.io/PythonDataScienceHandbook/05.02-introducing-scikit-learn.html
#
# Used with Python 2.71 or 3.67
#
# Version 1.1 (11/14/2018)
#   - Added the confusion matrix output
# -------------------------------------------------------------------

import sys
import numpy as np
import pandas as pd
from Bio import Entrez
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier


__author__ = "William A. Coetzee"
__copyright__ = "Copyright 2018, William A. Coetzee"
__license__ = "Apache License v2.0"
__version__ = "1.0"

# -------------------------------------------------------------------
# User-defined para,meters 
#   pep_file is the list of peptides used to train the model 
#   Entrez.email is needed to look up protein sequences 
#   protein_ID is the accession number of the protein to use
# 
# pep_file = "D:\\Programming\\Python\\sirts\\peptides.csv"       # home computer
# pep_file = "D:\\NYUMC\\Programming\\Python\\sirts\\peptides.csv"  # work laptop
pep_file = '/media/william/ntfs_2/Programming/Python/sirts/peptides.csv'       # linux home
Entrez.email = "add your email address here"
# protein_ID = "NP_000516.3"  # Kir6.2
# protein_ID = "NP_005682.2"  # SUR2A
protein_ID = "NP_001276674.1"   # GAPDHs

#  There is generally no need to edit below this line 
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# Lookup dictionary to convert an amino acid to a number
# There are 23 possible letters, including a stop codon
# -------------------------------------------------------------------
aa2n = {'A': '10000000000000000000000',
        'I': '01000000000000000000000',
        'L': '00100000000000000000000',
        'V': '00010000000000000000000',
        'G': '00001000000000000000000',
        'P': '00000100000000000000000',
        'D': '00000010000000000000000',
        'E': '00000001000000000000000',
        'H': '00000000100000000000000',
        'K': '00000000010000000000000',
        'R': '00000000001000000000000',
        'F': '00000000000100000000000',
        'W': '00000000000010000000000',
        'Y': '00000000000001000000000',
        'N': '00000000000000100000000',
        'Q': '00000000000000010000000',
        'S': '00000000000000001000000',
        'T': '00000000000000000100000',
        'C': '00000000000000000010000',
        'M': '00000000000000000001000',
        'X': '00000000000000000000100',
        'Z': '00000000000000000000010',
        '*': '00000000000000000000001'}


def Peptide2Matrix(peptide):
    # ------------------------------------------------------
    # Convert a peptide to a matrix.
    # Tt is assumed that all peptides are 13 aa in length
    # ------------------------------------------------------
    a = np.zeros((13, 23), dtype=int)
    irow = 0
    col = []
    for aa in peptide:
        col = aa2n[aa].index('1') - 2
        # col = str(map(aa2n.get, aa)).index('1') - 2
        a[irow, col] = 1
        irow += 1
    return a

def main():
    # ------------------------------------------------------
    # Read peptides into a pandas DataFrame
    # ------------------------------------------------------
    sirts = pd.read_csv(pep_file, engine='python')
    n_peptides = sirts.shape[0]
    classes = ['Neither', 'SIRT1', 'SIRT2']

    # Represent peptides numerically. <type 'numpy.ndarray'>
    data = np.zeros((n_peptides, 13, 23), dtype=int)
    row = 0
    for pept in sirts['Sequence']:
        data[row] = Peptide2Matrix(pept)
        row += 1

    # flatten the peptide matrix data. <type 'numpy.ndarray'>
    data = data.reshape(n_peptides, -1)

    # ------------------------------------------------------
    # Construct the features matrix with domentions: [n_samples, n_features]
    # ------------------------------------------------------
    # Train with 80% of the peptides
    n = int(n_peptides * 0.8)
    X = data[0:n]
    # Construct the target as a one dimensional array, with length n_samples
    y = sirts.target[0:n]

    # ------------------------------------------------------
    # Choose the model
    # ------------------------------------------------------
    # Try linear SVC first. If it does not work, try KNeighbors Classifier
    # model = svm.SVC(gamma=0.001)
    model = KNeighborsClassifier(n_neighbors=3, 
                                weights='distance', 
                                algorithm='kd_tree',
                                leaf_size=30,
                                p=2)
    print('Using model: \n', model)
    # Fit the model to the data
    X = data
    y = sirts.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    model.fit(X_train, y_train)

    # ------------------------------------------------------
    # Test the model accuracy
    # ------------------------------------------------------
    y_model = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_model)
    print('Training accuracy: %.2f%%' % (100 * test_acc))

    # Print the xonfusion matrix
    cm = pd.crosstab(y_test, y_model, rownames=['True'], colnames=['Predicted'], margins=True)
    print("\nConfusion matrix: %s \n %s" % (classes, cm))

    # ------------------------------------------------------
    # Read protein sequence in FASTA format
    # ------------------------------------------------------
    handle = Entrez.efetch(db="protein", id=protein_ID, rettype="fasta", retmode="text")
    protein = handle.read()  # <type 'str'>
    handle.close()
    print('\n', protein)
    # drop the first (descriptor) line and leave only the amino acids
    p = protein.split('\n', 1)[1]
    # remove all '/n' characters
    p = p.replace('\n', '')

    # ------------------------------------------------------
    # Extract all lysine-containing peptides as 13 mers
    # ------------------------------------------------------
    lysine = 'K'
    un_pep = []
    un_lys = []

    i = 0
    for aa in p:
        k_Ac_pep = []
        if(aa == lysine):
            # print("Lysine: %s" % str(i + 1))
            if(i - 7 < 0):
                padding = 13 - len(p[0:i + 7])
                for j in range(padding):
                    k_Ac_pep.append('X')
                k_Ac_pep.append(p[0:i + 7])
            else:
                k_Ac_pep.append(p[i - 6:i + 7])
            # print(i-6, i+6)
            k_Ac_pep = ''.join(k_Ac_pep)
            un_lys.append(i)
            un_pep.append(k_Ac_pep)
            # print(k_Ac_pep)
        i += 1

    # ------------------------------------------------------
    # Represent peptides numerically. <type 'numpy.ndarray'>
    # ------------------------------------------------------
    data2 = np.zeros((len(un_pep), 13, 23), dtype=int)
    row = 0
    for pept in un_pep:
        data2[row] = Peptide2Matrix(pept)
        row += 1

    # flatten the peptide matrix data. <type 'numpy.ndarray'>
    data2 = data2.reshape(len(un_pep), -1)

    # ------------------------------------------------------
    # Predict the acetylation sites
    # ------------------------------------------------------
    un_pred = model.predict(data2)
    un_prob = model.predict_proba(data2)
    # print(un_pred)


    # ------------------------------------------------------
    # Report data
    # ------------------------------------------------------
    print("Prediction of %s " % protein_ID)
    i = 0
    for temp in un_pred:
        if un_pred[i] > 0:
            score = un_prob[i][un_pred[i]]*100
            sirt_type = classes[un_pred[i]]
            print("K%3i in %s by %s (%.1f%%)" % (1+un_lys[i], un_pep[i], sirt_type, score))
        i += 1

    print('Model training accuracy: %.2f%%' % (100 * test_acc))


# ------------------------------------------------------
#   Main 
# ------------------------------------------------------
if __name__ == '__main__':
    print('Python version: ' + sys.version + '\n')
    main()

