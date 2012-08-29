A multinomial Bayes classifier for English text. Has a few features like
removing stopwords, and changing emoticons to happy/negative flags.

Short, and sweet.

Here:
    from multibayes import MultinomialBayes
    training_set = (('some sentence example', 'label one'), ('another sentence', 'label two'))
    m = MultinomialBayes(training_set)

    print m.classify('a test example')
    # Returns labels for the test example, in order of decreasing loglikeliness
    # [('label one', -2.5494451709255714), ('label two', -3.242592351485517)]
