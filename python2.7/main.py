import pandas as pd
import re

input_filename  = 'data/source/find-keywords.xlsx'
output_filename = 'data/find-keywords-nouns.xlsx'

df = pd.read_excel(input_filename)
# print('Article count: ', len(df))
# For testing
# df = df.head(10)
# print df.head(10) 

import spacy
nlp = spacy.load("en")
# Use python -m spacy download en

def tokenize_text(texts):
  docs = [doc for doc in nlp.pipe(texts, batch_size=500,nthreads=4)]
  return docs

def to_text(tokens):
  for token in tokens:
    print(token.orth_)
  return next(map(lambda token: token.orth_, tokens), '')

def filter_first_punct(noun_chunks):
  noun_chunks = list(noun_chunks)
  if len(noun_chunks) > 0:
    print('ROOT', noun_chunks[0].sent[noun_chunks[0].start])
  return []

def get_nouns(sentences):
  for docs in sentences:
    print docs.noun_chunks
  return [to_text(docs.noun_chunks) for docs in sentences]

df['docs'] = tokenize_text(df['Title'].astype(str,errors='ignore'))

def print_tokens(article_docs):
  # print('Domain:       ', url)
  print('Title:     ', article_docs)
  print('-------------')
  print('Words:     ', list(map(lambda word: word, article_docs)))
  print('Lemma:     ', list(map(lambda word: word.lemma_, article_docs)))
  print('Types:     ', list(map(lambda word: word.pos_, article_docs)))
  print('Tags:      ', list(map(lambda word: word.tag_, article_docs)))
  print('>')
  print('Nouns:     ', list(filter(lambda word: word.pos_ == 'NOUN' or word.tag_ == 'NNP' or word.tag_ == 'NNPS', article_docs)))
  print('Nouns sentences (chunks):     ', get_nouns(article_docs.sents))
  print('Noun chunks:', list(article_docs.noun_chunks))
  print('Noun chunks +1 words:', list(filter(lambda chunk: len(str(chunk).split(' ')) >= 2, list(article_docs.noun_chunks))))
  print('Verbs:     ', list(filter(lambda word: word.pos_ == 'VERB', article_docs)))
  print('Verbs Lemma:', list(map(lambda word: word.lemma_, filter(lambda word: word.pos_ == 'VERB', article_docs))))
  print('Adjectives:', list(filter(lambda word: word.pos_ == 'ADJ', article_docs)))
  print('Adjs Lemma:', list(map(lambda word: word.lemma_, filter(lambda word: word.pos_ == 'ADJ', article_docs))))
  print('Adverbs:   ', list(filter(lambda word: word.pos_ == 'ADV', article_docs)))
  print('Adverbs Lemma:', list(map(lambda word: word.lemma_, filter(lambda word: word.pos_ == 'ADV', article_docs))))
  print('Superlatives:', list(filter(lambda word: word.tag_ == 'JJS' or word.tag_ == 'RBS', article_docs)))
  print('Entities:  ', list(map(lambda entity: (entity, entity.label_), article_docs.ents)))
    
def df_url_docs(id, docs_field = 'docs'):
  return df[docs_field][id]

# Loops through every article and applies f() to it. 
# Then applies token_extractor() to convert from a Token to a string.
# Finally, concatenates the tokens of a single type with commas
def filter_bad_excel_strings(tokens_string):
    tokens_string = re.sub('[\000-\010]|[\013-\014]|[\016-\037]', '', tokens_string)
    if tokens_string.startswith("="):
        return tokens_string[1:]
    elif tokens_string.startswith("- "):
        return tokens_string[2:]
    else:
        return tokens_string

def map_articles(token_extractor, f, articles):
    def map_article(article):
        tokens_string = ",".join(map(token_extractor, f(article)))
        # Replace excel invalid chars
        tokens_strings = filter_bad_excel_strings(tokens_string)
        return tokens_strings
    return list(map(map_article, articles))

def make_excel_df(docs_column_name = 'docs'):
    df_excel = pd.DataFrame()
    docs = df[docs_column_name]
    df_excel['Title'] = df['Title']
    df_excel['Nouns'] = map_articles(lambda token: token.orth_, 
                                     lambda sentence : filter(lambda word: word.pos_ == 'NOUN' or word.tag_ == 'NNP' or word.tag_ == 'NNPS', sentence), 
                                     docs)
    df_excel['Noun Chunks (1)'] = map_articles(lambda token: token, 
                                     lambda doc : get_nouns(doc.sents), 
                                     docs)
    df_excel['Noun Chunks (2)'] = map_articles(
                                     lambda chunk: str(chunk),
                                     lambda doc : list(doc.noun_chunks), 
                                     docs)
    df_excel['Noun Chunks (3) +1 words'] = map_articles(
                                     lambda chunk: str(chunk),
                                     lambda doc : list(filter(lambda chunk: len(str(chunk).split(' ')) >= 2, list(doc.noun_chunks))), 
                                     docs)
    #df_excel['Noun Chunks +1 words (3)'] = map_articles(lambda token: token, 
    #                                 list(filter(lambda chunk: len(str(chunk).split(' ')) >= 2, list(article_docs.noun_chunks))))
    df_excel['Verbs'] = map_articles(lambda token: token.orth_, 
                                     lambda sentence : filter(lambda word: word.pos_ == 'VERB', sentence), 
                                     docs)
    df_excel['Verbs Lemma'] = map_articles(lambda token: token.lemma_, 
                                     lambda sentence : filter(lambda word: word.pos_ == 'VERB', sentence), 
                                     docs)
    df_excel['Adjectives'] = map_articles(lambda token: token.orth_, 
                                     lambda sentence : filter(lambda word: word.pos_ == 'ADJ', sentence), 
                                     docs)
    df_excel['Adjectives Lemma'] = map_articles(lambda token: token.lemma_, 
                                     lambda sentence : filter(lambda word: word.pos_ == 'ADJ', sentence), 
                                     docs)
    df_excel['Adverbs'] = map_articles(lambda token: token.orth_, 
                                     lambda sentence : filter(lambda word: word.pos_ == 'ADV', sentence), 
                                     docs)
    df_excel['Adverbs Lemma'] = map_articles(lambda token: token.lemma_, 
                                     lambda sentence : filter(lambda word: word.pos_ == 'ADV', sentence), 
                                     docs)
    df_excel['Superlatives'] = map_articles(lambda token: token.orth_, 
                                     lambda sentence : filter(lambda word: word.tag_ == 'JJS' or word.tag_ == 'RBS', sentence), 
                                     docs)
    df_excel['Superlatives Lemma'] = map_articles(lambda token: token.orth_, 
                                     lambda sentence : filter(lambda word: word.tag_ == 'JJS' or word.tag_ == 'RBS', sentence), 
                                     docs)
    df_excel['Entities'] = map_articles(lambda ent: ent.orth_, 
                                     lambda sentence : sentence.ents, 
                                     docs)
    return df_excel

df_excel_titles = make_excel_df()

writer = pd.ExcelWriter(output_filename)
df_excel_titles.to_excel(writer,'Titles')
writer.save()