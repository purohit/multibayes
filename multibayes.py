from __future__ import division
import pdb
import nltk
import pprint
from collections import Counter, defaultdict
from math import factorial, log10

class MultinomialBayes():
    """
    We don't use straight up Naive Bayes because it's too naive... word frequencies matter.
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

        print self.examples
        print self.counters
        print self.labels

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
        num_docs = sum(self.labels.values())
        # TODO: Laplace smooth this initial probability; only smoothing the terms for now
        prod = factorial(unique_words_in_example)
        print "{} unique words in example, against {} docs".format(unique_words_in_example, num_docs)

        if unique_words_in_example == 0:
            raise ValueError("Can't classify an empty document.")

        for label, label_freq in self.labels.items():
            prob = prod
            num_words_for_label = sum(self.counters[label].values())
            for term, freq in example_counter.items():
                # TODO: Transfer this to the log domain
                prob_of_term_given_class = ((self.counters[label][term]/num_words_for_label)**freq + 1)/(factorial(freq) + unique_words_in_example + 1)
                prob = prob * prob_of_term_given_class
                print "total prob is {:f}, after applying P({}|{}) = {:f}".format(prob, term, label, prob_of_term_given_class)

            prob = prob * (label_freq/num_docs)
            print "Probability that example is class {0} is {1:f}".format(label, prob)

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

