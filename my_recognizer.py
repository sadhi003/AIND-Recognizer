import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

    :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
    :param test_set: SinglesData object
    :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
    """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # implement recognizer 
    X_lengths = test_set.get_all_Xlengths()
    for X, lengths in X_lengths.values():
      score_dict = {} 
      #initialize maximum score as infinity
      max_score = float("-inf")
      best_guess = None
      for word, model in models.items():
        try:
          
          log_l = model.score(X,lengths)
          if log_l > max_score:
            max_score = log_l
            best_guess = word
        except:
          None
        score_dict[word] = log_l
      
      guesses.append(best_guess)
      probabilities.append(score_dict)
    
    return probabilities, guesses
