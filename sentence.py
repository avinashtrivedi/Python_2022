def sentence_type (a_string):
    return 'bad ending' if a_string[-1] not in ('.','!','?') else 'good ending'

def main():
    print(sentence_type('hi there'))
    print(sentence_type('hi there!'))
    print(sentence_type('hi there how are you.'))
main()