import math

def main():
	with open(r"D:\OneDrive - NITT\Custom_Download\1b_benchmark.train.tokens","r",encoding='utf-8') as data:
		training_data = data.readlines();
#		print(training_data)
	train_words, train_wordfreq = wordlist(training_data)
	train_unigram, train_words = unigram_prob(train_words, train_wordfreq)
	train_bigram, train_bilist, train_bi = bigram_prob(train_words, train_wordfreq)
	train_trigram, train_words = trigram_prob(train_words, train_wordfreq, train_bi)
#perplexity for train
	train_p1,  train_p2,  train_p3 = perplexity(train_words,train_unigram, train_bigram, train_trigram)
	print("Train Perplexities\nunigram: ", train_p1, "\nbigram: ", train_p2, "\ntrigram: ", train_p3)
	print("lambda1 = 0.1, lambda2 = 0.3, lambda3 = 0.6 --> ",smoothing(0.1, 0.3, 0.6, train_p1, train_p2, train_p3))
	print("lambda1 = 0.33, lambda2 = 0.34, lambda3 = 0.33 --> ",smoothing(0.33, 0.34, 0.33, train_p1, train_p2, train_p3))
	print("lambda1 = 0.1, lambda2 = 0.1, lambda3 = 0.8 --> ", smoothing(0.1, 0.1,0.8, train_p1, train_p2, train_p3))
	print("lambda1 = 0.1, lambda2 = 0.8, lambda3 = 0.1 --> ", smoothing(0.1, 0.8, 0.1, train_p1, train_p2, train_p3))
	print("lambda1 = 0.8, lambda2 = 0.1, lambda3 = 0.1 --> ", smoothing(0.8, 0.1,0.1, train_p1, train_p2, train_p3))	
	

#input: all train/dev/test data
#output: list of all words and dictionary of unique words and their counts

def wordlist(training_data):
	wordfreq = {"UNK": 0}
	lines = []
	words = []
	count = 0
	#words.append("<START>")
	for line in training_data:
		w = line.split();	#w is each an array of each line
		j = 1
		for word in w:		#word is every word in each line
			words.append(word)	#words - array of all words
			count = count + 1
			if j == len(w):	
				words.append("<STOP>")
	#			words.append("<START>")
				count = count + 2
			j += 1

		lines.append(w);	#lines - array of lines
	#words.pop(-1)
	for token in words:
	#make dictionary of unique words + their counts
		if token not in wordfreq:
			wordfreq[token] = 1
		else:
				x = wordfreq.get(token)
				x = x + 1
				wordfreq[token] = x

	delete = [key for key in wordfreq if wordfreq[key] < 3]
	keys = list(wordfreq.keys())
	for wor in wordfreq:
		if wordfreq.get(wor) < 3:
			i = wordfreq.get(wor)
			wordfreq["UNK"] = i + wordfreq.get("UNK") 	
	for k in delete: 
		if k != "UNK":
			del wordfreq[k]

	#new list of all words but with words < 3 replaced with UNK
	newlistofwords = []
	for word in words:
		if wordfreq.get(word) == None:
			newlistofwords.append("UNK")
		else:
			newlistofwords.append(word)
	words = newlistofwords

	index = 0
	while index != len(words) - 1:
		if words[index] not in wordfreq:
			words[index] = "UNK"
		index += 1		
	
	return words, wordfreq

#input: list of all words and dictionary of unique tokens
#output: unigram, bigram, trigram and their probability dictionaries
def unigram_prob(words, wordfreq):
	#UNIGRAM
	add = 0
	unigram = {}
	for word in wordfreq:
		uni = wordfreq[word] / (len(words))
		unigram[word] = uni
		add = add + uni
#	print(add)
	return unigram, words

def bigram_prob(words, wordfreq):
	#BIGRAM
	bi = {}	#pairs and their counts
	bigram = {}	#pairs and their probabilities
	bilist = []	#list of all pairs
	track = 0
	while track != len(words):
#	if track == 0:
#		bigram["<START>", words[track]] = wordfreq[track] / wordfreq["<START>"] 
#		track += 1
#		continue

#	if words[track] == "<STOP>":
#		track = track + 1
#		continue
		if track + 1 != len(words):
			list = (words[track], words[track + 1])
			bilist.append(list)
		track = track + 1
		if list in bi:
			bi[list] += 1
		elif list not in bi:
			bi[list] = 1
	#tabs right?
	for pair in bi:
		b = bi[pair] / wordfreq[pair[0]]
		bigram[pair] = b	

# 	print(sum(bigram.values()))
	return bigram, bilist, bi

def trigram_prob(words, wordfreq, bi):
	#TRIGRAM

	tri = {}	#triplets and their counts
	trigram = {}	#triplets and their probabilities
	trilist = []
	track = 0
	while track != len(words):
#	if track == 0:
#		trigram[words[0]] = unigram[words[0]]
#	elif track == 1:
#		trigram[words[0], words[1]] = bigram[words[0], words[1]]
#	if words[track] == "<STOP>" or (track + 1 != len(words) and words[track + 1] == "<STOP>"):
#		track = track + 1
#		continue
		if track + 1 != len(words) and track + 2 != len(words):
			list = (words[track], words[track + 1], words[track + 2])
			trilist.append(list)
		track = track + 1
		if list in tri:
			tri[list] += 1
		elif list not in tri:
			tri[list] = 1
	for triplet in tri:
		t = tri[triplet] / bi[triplet[0], triplet[1]]
		trigram[triplet] = t

	return trigram, words


#input: uni/bi/trigram dictionaries
#output: perplexity values for each

def perplexity(words, unigram, bigram, trigram):
#UNIGRAM PERPLEXITY
	unisum = 0
	exp = 0
# 	print(trigram)
	for x in words:
		exp += math.log(unigram[x],2)
		if x == "<STOP>":
			unisum += exp
			exp = 0
	unisum = -1*unisum/len(words)
	uni_perplexity = math.pow(2, unisum)
# 	print(unisum)

#BIGRAM PERPLEXITY
	bisum = 0
#	exp2 = math.log(bigram[("<STOP>", words[0])])
	exp2 = 0
	for x in range(len(words) - 1):
		if bigram[(words[x], words[x+1])] == 0:
			bigram[(words[x], words[x+1])] = smoothing(0.3, 0.1, 0.6, unigram[words[x+1]], 0.0,0.0)
		exp2 += math.log(bigram[(words[x], words[x + 1])], 2)
	# exp2 = log base 2 of count of the bigram's probability
		if words[x + 1] == "<STOP>":
			bisum += exp2
			x += 1
			exp2 = 0
	# add the probabilities per sentence	
	
	bisum = -1*bisum/len(words)
	bi_perplexity = math.pow(2, bisum)
	#print(bisum)

#TRIGRAM PERPLEXITY
	trisum = 0
	exp3 = 0
	for x in range(len(words) - 2):
		if trigram[(words[x], words[x+1], words[x+2])] == 0:
			trigram[(words[x], words[x + 1], words[x + 2])] = smoothing(0.3, 0.1, 0.6, unigram[words[x+1]], bigram[(words[x+1], words[x + 2])], 0.0)
		exp3 += math.log(trigram[(words[x], words[x+1], words[x+2])], 2)
		if words[x + 2] == "<STOP>":
			trisum += exp3
			x += 2
			exp3 = 0
	trisum = trisum/len(words) * -1
	tri_perplexity = math.pow(2, trisum)
	#print(trisum)

	return uni_perplexity, bi_perplexity, tri_perplexity
#SMOOTHING

def smoothing(lambda1, lambda2, lambda3, uni_perp, bi_perp, tri_perp):
	return lambda1 * uni_perp + lambda2 * bi_perp + lambda3 * tri_perp 

# devtest
# inputs: training dictionary, test or dev list of tokens
# output: dictionary of test/dev tokens: trained probability
def devtest(train_dict, test_list):
    test_dict = {}
    for i in test_list:
        if i in train_dict:
            test_dict[i] = train_dict[i]
        else:
            test_dict[i] = 0
    return test_dict

if __name__ == '__main__':
    main()
