import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose
        self.n_components = range(self.min_n_components, self.max_n_components + 1)

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


##################   SelectorBIC #########################

class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """


    def bic_score(self, num_states):        
        """
         BIC Equation:  BIC = -2 * log L + p * log N 
            L : is the likelihood of the fitted model
            p : is the number of parameters
            N : is the number of data points
                    
        Notes:
          -2 * log L    -> decreases with higher "p"
          p * log N     -> increases with higher "p"
          N > e^2 = 7.4 -> BIC applies larger "penalty term" in this case
            
        """
        
        model = self.base_model(num_states)
        log_likelihood = model.score(self.X, self.lengths)
        number_of_parameters = num_states ** 2 + 2 * num_states * model.n_features - 1
        score = -2 * log_likelihood + number_of_parameters * np.log(num_states)
        return score, model

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores
        

        try:
            
            best_score = float("Inf") 
            best_model = None

            for num_states in range(self.min_n_components, self.max_n_components + 1):
                score, model = self.bic_score(num_states)
                if score < best_score:
                    best_score, best_model = score, model
            return best_model

        except:
            return self.base_model(self.n_constant)

        
        

################# Selector DIC #########################

class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def dic_score(self, num_states):
        """
          DIC Equation:
            DIC = log(P(X(i)) - 1/(M - 1) * sum(log(P(X(all but i))
            i <- current_word
            DIC = log likelihood of the data belonging to model
                - avg of anti-log likelihood of data X and model M
                = log(P(original word)) - average(log(P(other words)))
               
            where anti-log likelihood means likelihood of data X and model M belonging
            to competing categories where log(P(X(i))) is the log-likelihood of the fitted
            model for the current word.
        
          Note:
            - log likelihood of the data belonging to model
            - anti_log_likelihood of data X vs model M
        """
        
        model = self.base_model(num_states)
        logs_likelihood = []
        
        for word, (X, lengths) in self.hwords.items():
            
            # likelihood of current word
            if word == self.this_word:
                current_word_likelihood = model.score(self.X, self.lengths)
                
            # if word != self.this_word:
            # likelihood of remaining words
            else:
                logs_likelihood.append(model.score(X, lengths))
             
        score = current_word_likelihood - np.mean(logs_likelihood)
        
        return score, model





    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        try:
            best_score = float("-Inf")
            best_model = None
            
            for num_states in range(self.min_n_components, self.max_n_components+1):
                score, model = self.dic_score(num_states)
                if score > best_score:
                    best_score = score
                    best_model = model
            return best_model   

        except:
            return self.base_model(self.n_constant)


#############  Selector CV #################################


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''
    def cv_score(self, num_states):
        """
        Calculate the average log likelihood of cross-validation folds using the KFold class
        :return: tuple of the mean likelihood and the model with the respective score
        
        CV Equation:
        
        
        """
        fold_scores = []
        split_method = KFold(n_splits = 3, shuffle = True, random_state = 1)
        
        
        for train_idx, test_idx in split_method.split(self.sequences):
            # Training sequences split using KFold are recombined
            self.X, self.lengths = combine_sequences(train_idx, self.sequences)
            # Get test sequences
            test_X, test_length = combine_sequences(test_idx, self.sequences)
            # Record each model score
            model = self.base_model(num_states)
            fold_scores.append(model.score(test_X, test_length))
            
        # Compute mean of all fold scores
        score = np.mean(fold_scores)
            
        return score, model

    

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection using CV
        # check based on cross validation

        try:
            best_score = float("Inf")
            best_model = None
            
            for num_states in range(self.min_n_components, self.max_n_components+1):
                score, model = self.cv_score(num_states)
                if score < best_score:
                    best_score = score
                    best_model = model
            return best_model
        except:
            return self.base_model(self.n_constant)







