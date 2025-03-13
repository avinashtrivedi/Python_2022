## Requirements

- Conda with Python 3.6 (https://www.anaconda.com/download)

## Install, Windows instructions

1. Open 'Anaconda Prompt'

> Ensure the 'conda' command can be found

```SH
conda --version # Displays 'conda 4.4.11'
```

If your have a lower version. Update conda:

```SH
conda update -n base conda
```

2. Install dependencies:

```SH
conda install -c conda-forge spacy
```

Install spacy's English language model:

```SH
python3 -m spacy download en
```

For more information about Anaconda, please check: https://conda.io/docs/user-guide/getting-started.html

## Run the notebook


```SH
cd sentence-tokenizer-v2
jupyter notebook
```

Open: 'sentence-tokenizer-nouns.ipynb'

Click on: `Cell > Run all`. The excel file will be recreated