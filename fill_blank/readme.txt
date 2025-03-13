We are going to use pytorch pretrained Bert embedding i.e. bert-large-uncased

We use BertForMaskedLM, the blank in the question '_' is considered as [MASK]. Then model will decide which [MASK] to use to fit the context of the sentence.

How to Run:

1) open the .ipynb file in google colab and run all cells.
2) Make sure to upload all the files (train,test,dev) in colab local directory.


