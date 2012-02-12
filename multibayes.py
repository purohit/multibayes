from __future__ import division
import re
from nltk import word_tokenize
from nltk.corpus import stopwords
from collections import Counter, defaultdict
from math import factorial, log

class MultinomialBayes():
    """ Naive Bayes is too naive... word frequencies matter.  """
    punctuations = re.compile(r"^[.,!?;]*$")
    examples = []
    counters = defaultdict(Counter)
    labels = Counter()

    def __init__(self, examples=[]):
        """ Examples are tuples, like: ((sentence, label), (sentence, label)...) """
        for example, label in examples:
            self.train(example, label)

    @classmethod
    def lcm(cls, a, b):
        """ Least-common multiple, Euclid style. """
        gcd, tmp = a,b
        while tmp != 0:
            gcd,tmp = tmp, gcd % tmp
        return a*b/gcd

    def train(self, example, label):
        """ Trains itself on a single example, label pair """
        tokens = self.smart_tokenize(example)
        self.examples.append((tokens, label))
        self.counters[label].update(tokens)
        self.labels[label] += 1

    def classify(self, example):
        """
        Classifies a example based on training data passed into the initialization.
        We need to compare P(document|class1)P(class1)/P(document) > P(document|class2)P(class2)/P(document)
        = P(document|class1)P(class1) > P(document|class2)P(class2)
        Uses Laplace smoothing and log-likelihood to prevent underflow/0-mangling.
        """
        example_counter = Counter(self.smart_tokenize(example))
        unique_words_in_example = len(example_counter.keys())

        if unique_words_in_example == 0:
            raise ValueError("Can't classify an empty document.")

        num_docs = sum(self.labels.values())

        likelihoods = Counter()
        for label, label_freq in self.labels.items():
            num_words_for_label = sum(self.counters[label].values())
            fractions = [] # This is a list of [(num, denom)] for all multinomial terms
            for term, freq in example_counter.items():
                num = self.counters[label][term]**freq
                denom = factorial(freq) * (num_words_for_label)**freq
                fractions.append((num, denom))
            lcm = reduce(self.lcm, [denom for num, denom in fractions] + [num_docs])
            # Laplace smoothing, log-likehood
            fractions = [log((num + 1)/(denom + unique_words_in_example)) for num, denom in fractions]
            n_factorial = log((factorial(unique_words_in_example) * lcm + 1) / (lcm + unique_words_in_example))
            prob_class = log(label_freq/num_docs)
            log_likelihood_of_class = sum(fractions) + n_factorial + prob_class
            likelihoods[label] = log_likelihood_of_class

        # You could change this to (5) for the top 5 likely classes
        label, likelihood = likelihoods.most_common(1)[0]
        return label

    @classmethod
    def smart_tokenize(cls, sentence):
        """ Removes stopwords, replaces emoticons with flags, and tokenizes the string """
        return cls.strip_nonwords(cls.remove_stopwords(word_tokenize(cls.emoticons_to_flags(sentence))))

    @classmethod
    def strip_nonwords(cls, tokens):
        """ Given a list of tokens, remove ones that are merely punctuation """
        return [t for t in tokens if not cls.punctuations.match(t)]

    @classmethod
    def remove_stopwords(cls, tokens):
        """ Given a list of tokens, removes those that are stopwords """
        return [t for t in tokens if t not in stopwords.words('english')]

    @classmethod
    def emoticons_to_flags(cls, sentence):
        """ Replaces emoticons in a sentence with the emo_happy, emo_sad flags. Use before tokenization.  """
        happy = set((">:]",":-)",":)",":o)",":]",":3",":c)",":>","=]","8)","=)",":}",":^)"))
        sad = set((">:[",":-(",":(",":-c",":c",":-<",":<",":-[",":[",":{",">.>","<.<",">.<"))
        for emoticon in happy:
            sentence = sentence.replace(emoticon, "emo_happy")
        for emoticon in sad:
            sentence = sentence.replace(emoticon, "emo_sad")
        return sentence

# Example usage
if __name__ == '__main__':
    examples = (("Biscuit is a happy, happy dog", "positive"),
                ("Biscuit is a terribly sad dog", "negative"),
                ("Biscuit is sad constantly. :(", "negative"),
                ("Did you know biscuit's depression results from over-eating?", "negative"),
                ("When biscuit sees the sun shine, he grins widely", "positive"),
                ("That's not to say some days aren't gloomy. Clouds make him so sad.", "negative"),
                ("But when the clouds disappear, his face brightens. He wags his tail, and is happy", "positive"),
                ("I love biscuit :-)!", "positive"))
    m = MultinomialBayes(examples)
    print m.classify("Biscuit is so happy")
    print m.classify("Biscuit is a sad puppy :(")
