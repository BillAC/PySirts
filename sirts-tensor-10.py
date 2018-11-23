
#!/usr/bin/env python

# -------------------------------------------------------------------
# Background reading:
#   https://www.tensorflow.org/install/pip
#   https://www.tensorflow.org/tutorials/keras/basic_classification
#
# Use with python 3.7
#   virtualenv --system-site-packages -p python3 ./venv
#   .\venv\Scripts\activate
#   deactivate
# -------------------------------------------------------------------

import sys
import numpy as np
import pandas as pd
from Bio import Entrez
import tensorflow as tf
from tensorflow import keras


__author__ = "William A. Coetzee"
__copyright__ = "Copyright 2018, William A. Coetzee"
__license__ = "Apache License v2.0"
__version__ = "1.0"


# -------------------------------------------------------------------
# User-defined parameters
# -------------------------------------------------------------------
pep_file = "D:\\Programming\\Python\\sirts\\peptides.csv"       # home computer
# pep_file = "D:\\NYUMC\\Programming\\Python\\sirts\\peptides.csv"  # work laptop
protein_ID = "NP_000516.3"  # Kir6.2
# protein_ID = "NP_005682.2"  # SUR2A
# protein_ID = "NP_001276674.1"   # GAPDH
Entrez.email = "add your email address here"
confidence = 80
# -------------------------------------------------------------------
# Do not edit below these lines
# -------------------------------------------------------------------


# Lookup dictionary to convert an amino acid to a number
#   There are 23 possible letters, including a stop codon
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
    # Read training/testing peptides into a pandas DataFrame
    # ------------------------------------------------------

    sirts = pd.read_csv(pep_file)
    n_peptides = sirts.shape[0]

    # Train with 80% of the peptides and test with the remaining 20%
    frac = int(n_peptides * 0.8)
    sirts = sirts.sample(n=n_peptides)
    sirts = sirts.reset_index()

    # Represent peptides numerically. <type 'numpy.ndarray'>
    data = np.zeros((n_peptides, 13, 23), dtype=int)
    row = 0
    for pept in sirts['Sequence']:
        data[row] = Peptide2Matrix(pept)
        row += 1

    tr_peptides = data[0:frac]
    tr_labels = sirts['target'][0:frac]
    tst_peptides = data[frac:]
    tst_labels = sirts['target'][frac:]

    tst_labels = tst_labels.reset_index()
    tst_labels = tst_labels.drop('index', axis=1)
    # classes = ['SIRT1', 'SIRT2', 'Neither']

    # ------------------------------------------------------
    # Machine learning and validation
    # ------------------------------------------------------

    # Build the model
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(13, 23)),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(3, activation=tf.nn.softmax)
    ])

    # Compile the model
    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    training = model.fit(tr_peptides, tr_labels, epochs=12)

    # Evaluate accuracy
    test_loss, test_acc = model.evaluate(tst_peptides, tst_labels)

    # Make predictions
    predictions = model.predict(tst_peptides)

    # n = 28
    # predictions[n]
    # np.argmax(predictions[n])
    # tst_labels['target'][n]

    # Generate ROC (Receiver operating characteristic) curve
    #   https://stackoverflow.com/questions/34564830/roc-curve-with-sklearn-python

    # ------------------------------------------------------
    # Read protein sequence, generate 13 peptides and predict
    # ------------------------------------------------------

    # Read protein sequence in FASTA format

    handle = Entrez.efetch(db="protein", id=protein_ID, rettype="fasta", retmode="text")
    protein = handle.read()  # <type 'str'>
    handle.close()
    # print(protein)
    # drop the first (descriptor) line and leave only the amino acids
    p = protein.split('\n', 1)[1]
    # remove all '/n' characters
    p = p.replace('\n', '')
    #p = list(enumerate(p))

    # Extract all lysine-containing peptides as 13 mers
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

    # Represent peptides numerically. <type 'numpy.ndarray'>
    data2 = np.zeros((len(un_pep), 13, 23), dtype=int)
    row = 0
    for pept in un_pep:
        data2[row] = Peptide2Matrix(pept)
        row += 1

    # Predict the acetylation sites
    un_pred = model.predict(data2)
    # print(un_pred)

    # Report data
    #print("\n\n\nPrediction of %s (1 = SIRT1, 2 = SIRT2, 0 = Neither)" % protein_ID)
    print("\nPrediction of %s " % protein_ID)
    i = 0
    for temp in un_pred:
        k_ac = np.argmax(predictions[i])
        perc = 100 * predictions[i][k_ac]
        if k_ac > 0:
            if k_ac == 1:
                s = 'SIRT1'
            else:
                s = 'SIRT2'
            if float(perc) > confidence:
                print("K%4i in %s : %s (%.1f%%)" % (1 + un_lys[i], un_pep[i], s, float(perc)))
        i += 1
    print('Training accuracy: %.2f%%' % (100 * test_acc))


if __name__ == '__main__':
    print('Python version: ' + sys.version + '\n')
    main()
