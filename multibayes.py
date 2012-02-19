from __future__ import division
import re
from nltk import word_tokenize
from nltk.corpus import stopwords
from collections import Counter, defaultdict
from math import factorial, log

class memoized(object):
   """Decorator that caches a function's return value each time it is called.
   If called later with the same arguments, the cached value is returned, and
   not re-evaluated.
   """
   def __init__(self, func):
      self.func = func
      self.cache = {}
   def __call__(self, *args):
      try:
         return self.cache[args]
      except KeyError:
         value = self.func(*args)
         self.cache[args] = value
         return value
      except TypeError:
         # uncachable -- for instance, passing a list as an argument.
         # Better to not cache than to blow up entirely.
         return self.func(*args)
   def __repr__(self):
      """Return the function's docstring."""
      return self.func.__doc__
   def __get__(self, obj, objtype):
      """Support instance methods."""
      return functools.partial(self.__call__, obj)

class MultinomialBayesException(Exception):
    pass

class MultinomialBayes():
    """ Naive Bayes is too naive... word frequencies matter.  """
    punctuations = re.compile(r"^[.,!?;]*$")
    delimiters = re.compile(r"(\.|\,)+")

    examples = []
    counters = defaultdict(Counter)
    labels = Counter()

    def __init__(self, examples=[]):
        """ Examples are tuples, like: ((sentence, label), ...) """
        for example, label in examples:
            self.train(example, label)

    @classmethod
    def gcd(cls, a, b):
        """Return greatest common divisor using Euclid's Algorithm."""
        while b:
            a, b = b, a % b
        return a

    @classmethod
    @memoized
    def lcm(cls, a, b):
        """Return lowest common multiple."""
        return a * b // cls.gcd(a, b)

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
            raise MultinomialBayesException("Can't classify an empty document")

        num_docs = sum(self.labels.values())

        likelihoods = Counter()
        #print example_counter
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
        return likelihoods.most_common(3)

    @classmethod
    def smart_tokenize(cls, sentence):
        """ Removes stopwords, replaces emoticons with flags, and tokenizes the string """
        return cls.strip_nonwords(cls.remove_stopwords(cls.lowercase_words(word_tokenize(cls.delimiters_to_spaces(cls.emoticons_to_flags(sentence))))))

    @classmethod
    def strip_nonwords(cls, tokens):
        """ Given a list of tokens, remove ones that are merely punctuation """
        return [t for t in tokens if not cls.punctuations.match(t)]

    @classmethod
    def remove_stopwords(cls, tokens):
        """ Given a list of tokens, removes those that are stopwords """
        return [t for t in tokens if t not in stopwords.words('english')]

    @classmethod
    def lowercase_words(cls, tokens):
        return [t.lower() for t in tokens]

    @classmethod
    def delimiters_to_spaces(cls, sentence):
        return cls.delimiters.sub(" ", sentence)

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
