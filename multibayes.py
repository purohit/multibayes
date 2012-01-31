from __future__ import division
import pdb
import nltk
import pprint
from collections import Counter, defaultdict
from math import factorial, log10

class MultinomialBayes():
    """
    Naive Bayes is too naive... word frequencies matter.
    """
    examples = []
    counters = defaultdict(Counter)
    labels = Counter()

    def __init__(self, examples):
        """
        Takes examples which is an iterable of tuples, like: ((sentence, label), (sentence, label)...)
        """
        for example, label in examples:
            tokens = self.smart_tokenize(example)
            self.examples.append((tokens, label))
            self.counters[label].update(tokens)
            self.labels[label] += 1

    @classmethod
    def lcm(cls, a, b):
        """ Least-common multiple, Euclid style. """
        gcd, tmp = a,b
        while tmp != 0:
            gcd,tmp = tmp, gcd % tmp
        return a*b/gcd

    def classify(self, example):
        """
        Classifies a example based on training data passed into the initialization.
        Note that we don't have to do a full Bayesian comparision. Bayes rule is:
        P(a|b) = P(b|a)P(a)/P(b). So if we're asking if P(class1|document) > P(class2|document),
        we just need to compare P(document|class1)P(class1)/P(document) > P(document|class2)P(class2)/P(document)
        = P(document|class1)P(class1) > P(document|class2)P(class2)
        Also performs Laplace smoothing for the product.
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
            fractions = [log10((num + 1)/(denom + unique_words_in_example)) for num, denom in fractions]
            n_factorial = log10((factorial(unique_words_in_example) * lcm + 1) / (lcm + unique_words_in_example))
            prob_class = log10(label_freq/num_docs)
            log_likelihood_of_class = sum(fractions) + n_factorial + prob_class
            likelihoods[label] = log_likelihood_of_class

        print likelihoods
        return likelihoods

    @classmethod
    def smart_tokenize(cls, sentence):
        return nltk.wordpunct_tokenize(cls.emoticons_to_sentinels(sentence))

    @classmethod
    def emoticons_to_sentinels(cls, sentence):
        """
        Replaces emoticons in a sentence with the emo_happy, emo_sad labels
        """
        happy = set((">:]",":-)",":)",":o)",":]",":3",":c)",":>","=]","8)","=)",":}",":^)"))
        sad = set((">:[",":-(",":(",":-c",":c",":-<",":<",":-[",":[",":{",">.>","<.<",">.<"))
        for emoticon in happy:
            sentence = sentence.replace(emoticon, "emo_happy")
        for emoticon in sad:
            sentence = sentence.replace(emoticon, "emo_sad")
        return sentence

examples = (("Biscuit is a happy, happy dog", "positive"),
            ("Biscuit is a terribly sad dog", "negative"),
            ("Biscuit is sad constantly. :(", "negative"),
            ("Did you know biscuit's depression results from over-eating?", "negative"),
            ("When biscuit sees the sun shine, he grins widely", "positive"),
            ("That's not to say some days aren't gloomy. Clouds make him so sad.", "negative"),
            ("But when the clouds disappear, his face brightens. He wags his tail, and is happy", "positive"),
            ("I love biscuit :-)!", "positive"))

if __name__ == '__main__':
    m = MultinomialBayes(examples)
    m.classify("Biscuit is so happy")

