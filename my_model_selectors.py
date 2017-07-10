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


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores
        # BIC score = -2logL + plogN(penalize part)
        # where L -> likelihood of model that is fitted
        # N -> data points 
        # p -> number of parameters

        bic_scores = []
        try:
            for n in self.n_components:
                model = self.base_model(n)
                log_l = model.score(self.X, self.lengths)
                p = n ** 2 + 2 * n * model.n_features - 1
                bic_score = -2 * log_l + p * math.log(n)
                bic_scores.append(bic_score)
        except Exception as e:
            pass

        states = self.n_components[np.argmax(bic_scores)] if bic_scores else self.n_constant
        return self.base_model(states)
        


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        # DIC score = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
        
        dic_scores = []
        logs_l = []
        try:
            for n_component in self.n_components:
                model = self.base_model(n_component)
                logs_l.append(model.score(self.X, self.lengths))
            sum_logs_l = sum(logs_l)
            m = len(self.n_components)
            for log_l in logs_l:     
                other_words_likelihood = (sum_logs_l - log_l) / (m - 1)
                dic_scores.append(log_l - other_words_likelihood)
        except Exception as e:
            pass

        states = self.n_components[np.argmax(dic_scores)] if dic_scores else self.n_constant
        return self.base_model(states)

class TestMyCode(ModelSelector):

    def select(self):
        print ("this is the test code for method overriding")
        return None


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection using CV
        # check based on cross validation

        split_cnt = 3 # 3 fold
        kf = KFold(n_splits=split_cnt)
        
        max_logl = float("-inf")
        max_logl_n = self.min_n_components
        
        for n in range(self.min_n_components, self.max_n_components):
            try:
                avg_logL = 0
                for train_index, test_index in kf.split(self.sequences):
                    cv_train_param_X, cv_train_param_lengths = combine_sequences(train_index, self.sequences)
                    model = self.base_model(n, param_X=cv_train_param_X, param_lengths=cv_train_param_lengths)
                    cv_test_param_X, cv_test_param_lengths = combine_sequences(test_index, self.sequences)
                    logL = model.score(cv_test_param_X, cv_test_param_lengths)
                    avg_logL = avg_logL + logL/split_cnt
                if avg_logL > max_logl:
                    max_log_l = avg_logL
                    max_logl_n = n
            except:
                if self.verbose:
                    print("failure on {} with {} states".format(self.this_word, n))
                
        return self.base_model(max_logl_n)


'''        
        mean_scores = []
        split_method = KFold()
        try:
            for n_component in self.n_components:
                model = self.base_model(n_component)
                # save score on each fold
                fold_scores = []
                for _, test_idx in split_method.split(self.sequences):
                    test_X, test_length = combine_sequences(test_idx, self.sequences)
                    fold_scores.append(model.score(test_X, test_length))

                # get mean value from each score
                mean_scores.append(np.mean(fold_scores))
        except Exception as e:
            pass

        states = self.n_components[np.argmax(mean_scores)] if mean_scores else self.n_constant
        return self.base_model(states)
'''





