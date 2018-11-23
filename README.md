# PySirts
Predict K-acetylated peptides using machine learning algorithms
------------------------------------------------------------------------------

These scripts explore machine learning as a tool to predict whether specific lysine residues are subject to acetylation (K-acetylation).  The focus of these scripts are the cytosolic sirtuins (SIRT1 and SIRT2).  

The training peptides used are from Ruad et al (PMID: 23995836; full reference below). Specifically, the file used are from the Supplementary Data 1 Excel file (“Annotated microarray peptides and their deacetylation by human Sirt1-Sirt7”.) 

The sklearn’s KneighborsClassifier is used in sirts.py, which obtains an accuracy of ~57% when tested with a fraction of the peptides in the positive training dataset. It accurately predicts (some of the) known K-acetylation sites  in GAPDH, which is not achieved by some  other K-acetylation prediction web sites.  Sirts.py does a poor job at distinguishing between SIRT1 and SIRT2, as is evident when examining the confusion matrix. 

The script in ‘sirts-tensor-10.py’ uses Google’s Tensor Flow algorithms. The output is extremely inconsistent and this is very much a work in progress. 
 

References 
-------------
Rauh D1, Fischer F, Gertz M, Lakshminarasimhan M, Bergbrede T, Aladini F, Kambach C, Becker CF, Zerweck J, Schutkowski M, Steegborn C. An acetylome peptide microarray reveals specificities and deacetylation substrates for all human sirtuin isoforms. Nat Commun. 2013;4:2327. doi: 10.1038/ncomms3327.