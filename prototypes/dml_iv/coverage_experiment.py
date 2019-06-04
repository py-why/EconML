import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, RobustScaler, QuantileTransformer
from sklearn.pipeline import Pipeline
import scipy.special
from dml_iv import DMLIV, GenericDMLIV
from dr_iv import DRIV, ProjectedDRIV
from dml_ate_iv import DMLATEIV
import numpy as np
import pandas as pd
import locale
from utilities import SubsetWrapper
import statsmodels.api as sm
from utilities import RegWrapper
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, LogisticRegressionCV, LassoCV
from sklearn import metrics
from xgboost import XGBClassifier, XGBRegressor
from xgb_utilities import XGBWrapper
from utilities import SelectiveLasso
from econml.utilities import hstack
from joblib.parallel import Parallel, delayed


# A wrapper of statsmodel linear regression, wrapped in a sklearn interface.
# We can use statsmodel for all hypothesis testing capabilities
class StatsModelLinearRegression:
    def __init__(self):
        return
    def fit(self, X, y):
        self.model = sm.OLS(y, X).fit()
        return self
    def predict(self, X):
        return self.model.predict(exog=X)
    def summary(self):
        return self.model.summary()
    @property
    def coef_(self):
        return self.model._results.params
    @property
    def intercept_(self):
        return 0

# For DMLIV we also need a model for E[T | X, Z]. We use a classifier since T is binary
# Because Z is also binary, we could have also done a more complex model_T_XZ, where we split
# the data based on Z=1 and Z=0 and fit a separate sub-model for each case.
#model_T_XZ = lambda: model_clf()
class SeparateModel:
    def __init__(self, model0, model1):
        self.model0 = model0
        self.model1 = model1
        return
    def fit(self, XZ, T):
        Z0 = (XZ[:, -1] == 0)
        Z1 = (XZ[:, -1] == 1)
        self.model0.fit(XZ[Z0, :-1], T[Z0])
        self.model1.fit(XZ[Z1, :-1], T[Z1])
        return self
    def predict(self, XZ):
        Z0 = (XZ[:, -1] == 0)
        Z1 = (XZ[:, -1] == 1)
        t_pred = np.zeros(XZ.shape[0])
        if np.sum(Z0) > 0:
                t_pred[Z0] = self.model0.predict(XZ[Z0, :-1])
        if np.sum(Z1) > 0:
                t_pred[Z1] = self.model1.predict(XZ[Z1, :-1])
                
        return t_pred
    @property
    def coef_(self):
        return np.concatenate((self.model0.coef_, self.model1.coef_))

X_colnames = {
    'days_visited_exp_pre': 'day_count_pre',  # How many days did they visit TripAdvisor attractions pages in the pre-period
    'days_visited_free_pre': 'day_count_pre',  # How many days did they visit TripAdvisor through free channels (e.g. domain direct) in the pre-period
    'days_visited_fs_pre': 'day_count_pre',  # How many days did they visit TripAdvisor fs pages in the pre-period    
    'days_visited_hs_pre': 'day_count_pre',  # How many days did they visit TripAdvisor hotels pages in the pre-period
    'days_visited_rs_pre': 'day_count_pre',  # How many days did they visit TripAdvisor restaurant pages in the pre-period
    'days_visited_vrs_pre': 'day_count_pre',  # How many days did they visit TripAdvisor vrs pages in the pre-period
    'is_existing_member': 'binary', #Binary indicator of whether they are existing member
    'locale_en_US': 'binary',  # User's locale
    'os_type': 'os',  # User's operating system
    'revenue_pre': 'revenue',  # Revenue in the pre-period
}

treat_colnames = {
    'treatment': 'binary',  # Did they receive the Google One-Tap experiment? [This is the instrument]
    'is_member': 'is_member'  # Did they become a member during the experiment period (through any means)? [This is the treatment of interest]
}

outcome_colnames = {
    'days_visited': 'days_visited',  # How many days did they visit TripAdvisor in the experimental period
}


def gen_data(data_type, n):
    gen_func = {'day_count_pre': lambda: np.random.randint(0, 29 , n),  # Pre-experiment period was 28 days
                'day_count_post': lambda: np.random.randint(0, 15, n),  # Experiment ran for 14 days
                'os': lambda: np.random.choice(['osx', 'windows', 'linux'], n),
                'locale': lambda: np.random.choice(list(locale.locale_alias.keys()), n),
                'count': lambda: np.random.lognormal(1, 1, n).astype('int'),
                'binary': lambda: np.random.binomial(1, .5, size=(n,)),
                ##'days_visited': lambda: 
                'revenue': lambda: np.round(np.random.lognormal(0, 3, n), 2)
                
            }
    
    return gen_func[data_type]() if data_type else None


def dgp_binary(X, n, true_fn):
    ##X = np.random.uniform(-1, 1, size=(n, d))
    Z = np.random.binomial(1, .5, size=(n,))
    nu = np.random.uniform(0, 10, size=(n,))
    coef_Z = 0.2
    C = np.random.binomial(1, coef_Z*scipy.special.expit(0.1*(X[:, 0] + nu))) # Compliers when recomended
    C0 = np.random.binomial(1, .1*np.ones(X.shape[0])) # Non-compliers when not recommended 
    T = C * Z + C0 * (1 - Z)
    y = true_fn(X) * (T + 0.2*nu)  + (0.1*X[:, 0] + 0.1*np.random.uniform(0, 1, size=(n,)))
    return y, T, Z

def exp(n):

    COV_CLIP = 10/n

    X_data = {colname: gen_data(datatype, n) for colname, datatype in X_colnames.items()}
    X_data=pd.DataFrame({**X_data})
    # Turn strings into categories for numeric mapping
    X_data['os_type'] = X_data.os_type.astype('category').cat.codes
    X_pre=X_data.values.astype('float')

    true_fn = lambda X: (.8+.5*X[:,0] - 3*X[:, 6])

    y, T, Z = dgp_binary(X_pre, n, true_fn)
    X = QuantileTransformer(subsample=n//10).fit_transform(X_pre)

    true_ate = np.mean(true_fn(X_pre))
    print("True ATE: {:.3f}".format(true_ate))
    print("New members: in treatment = {:f}, in control = {:f}".format(T[Z == 1].sum()/Z.sum(), T[Z == 0].sum()/(1-Z).sum()))
    print("Z treatment proportion: {:.5f}".format(np.mean(Z)))


    # ### Defining some generic regressors and classifiers

    # This a generic non-parametric regressor
    # model = lambda: GradientBoostingRegressor(n_estimators=20, max_depth=3, min_samples_leaf=20,
    #                                        n_iter_no_change=5, min_impurity_decrease=.001, tol=0.001)
    #model = lambda: XGBWrapper(XGBRegressor(gamma=0.001, n_estimators=50, min_child_weight=50, n_jobs=10),
    #                        early_stopping_rounds=5, eval_metric='rmse', binary=False)

    # model = lambda: RandomForestRegressor(n_estimators=100)
    # model = lambda: Lasso(alpha=0.0001) #CV(cv=5)
    # model = lambda: GradientBoostingRegressor(n_estimators=60)
    # model = lambda: LinearRegression(n_jobs=-1)
    model = lambda: LassoCV(cv=5, n_jobs=-1)

    # This is a generic non-parametric classifier. We have to wrap it with the RegWrapper, because
    # we want to use predict_proba and not predict. The RegWrapper calls predict_proba of the
    # underlying model whenever predict is called.
    # model_clf = lambda: RegWrapper(GradientBoostingClassifier(n_estimators=20, max_depth=3, min_samples_leaf=20,
    #                                        n_iter_no_change=5, min_impurity_decrease=.001, tol=0.001))
    # model_clf = lambda: RegWrapper(XGBWrapper(XGBClassifier(gamma=0.001, n_estimators=50, min_child_weight=50, n_jobs=10),
    #                                        early_stopping_rounds=5, eval_metric='logloss', binary=True))
    # model_clf = lambda: RandomForestClassifier(n_estimators=100)
    # model_clf = lambda: RegWrapper(GradientBoostingClassifier(n_estimators=60))
    # model_clf = lambda: RegWrapper(LogisticRegression(C=10, penalty='l1', solver='liblinear'))
    model_clf = lambda: RegWrapper(LogisticRegressionCV(n_jobs=-1, cv=5, scoring='neg_log_loss'))

    model_clf_dummy = lambda: RegWrapper(DummyClassifier(strategy='prior'))

    # We need to specify models to be used for each of these residualizations
    model_Y_X = lambda: model() # model for E[Y | X]
    model_T_X = lambda: model_clf() # model for E[T | X]. We use a classifier since T is binary

    # model_Z_X = lambda: model_clf() # model for E[Z | X]. We use a classifier since Z is binary
    model_Z_X = lambda: model_clf_dummy() # model for E[Z | X]. We use a classifier since Z is binary

    # E[T | X, Z]
    model_T_XZ = lambda: SeparateModel(model_clf(), model_clf())

    # E[TZ | X]
    model_TZ_X = lambda: model_clf()


    # We fit DMLATEIV with these models and then we call effect() to get the ATE.
    # n_splits determines the number of splits to be used for cross-fitting.


    # # Algorithm 2 - Current Method

    # In[121]:

    dmlateiv_obj = DMLATEIV(model_Y_X(), model_T_X(), model_Z_X(),
                    n_splits=10, # n_splits determines the number of splits to be used for cross-fitting.
                    binary_instrument=True, # a flag whether to stratify cross-fitting by instrument
                    binary_treatment=True # a flag whether to stratify cross-fitting by treatment
                    )

    dmlateiv_obj.fit(y, T, X, Z)


    ta_effect = dmlateiv_obj.effect()
    ta_effect_conf = dmlateiv_obj.normal_effect_interval(lower=2.5, upper=97.5)
    
    print("{:.3f}, ({:.3f}, {:3f})".format(ta_effect, ta_effect_conf[0], ta_effect_conf[1]))


    # # Algorithm 3 - DRIV ATE

    driv_model_effect = lambda: Pipeline([('poly', PolynomialFeatures(degree=0, include_bias=True)),
                                                    ('reg', StatsModelLinearRegression())])

    dmliv_featurizer = lambda: PolynomialFeatures(degree=1, include_bias=True)
    dmliv_model_effect = lambda: SelectiveLasso(np.arange(1, X.shape[1]+1), LassoCV(cv=5, n_jobs=-1))
    prel_model_effect = DMLIV(model_Y_X(), model_T_X(), model_T_XZ(),
                            dmliv_model_effect(), dmliv_featurizer(), n_splits=1)
    #dmliv_model_effect = lambda: model()
    #prel_model_effect = GenericDMLIV(model_Y_X(), model_T_X(), model_T_XZ(), 
    #                                 dmliv_model_effect(),
    #                                 n_splits=1)
    dr_cate = DRIV(model_Y_X(), model_T_X(), model_Z_X(), # same as in DMLATEIV
                            prel_model_effect, # preliminary model for CATE, must support fit(y, T, X, Z) and effect(X)
                            model_TZ_X(), # model for E[T * Z | X]
                            driv_model_effect(), # model for final stage of fitting theta(X)
                            cov_clip=COV_CLIP, # covariance clipping to avoid large values in final regression from weak instruments
                            n_splits=10, # number of splits to use for cross-fitting
                            binary_instrument=True, # a flag whether to stratify cross-fitting by instrument
                            binary_treatment=True # a flag whether to stratify cross-fitting by treatment
                        )
    dr_cate.fit(y, T, X, Z)
    dr_effect = dr_cate.effect_model.named_steps['reg'].coef_[0]
    dr_effect_conf = dr_cate.effect_model.named_steps['reg'].model.conf_int(alpha=0.05)[0]
    print("{:.3f}, ({:.3f}, {:3f})".format(dr_effect, dr_effect_conf[0], dr_effect_conf[1]))
    return true_ate, ta_effect, ta_effect_conf[0], ta_effect_conf[1], dr_effect, dr_effect_conf[0], dr_effect_conf[1]



if __name__=="__main__":
    np.random.seed(123)
    n_samples = 100000
    n_exp = 100
    res = np.array(Parallel(n_jobs=-1, verbose=3)(
            delayed(exp)(n_samples) for _ in range(n_exp)))
    np.save('coverage_results.npy', res)
    print("Coverage DMLATE: {:.3f}".format(np.mean((res[:, 0] >= res[:, 2]) & (res[:, 0] <= res[:, 3]))))
    print("Coverage DRIV: {:.3f}".format(np.mean((res[:, 0] >= res[:, 5]) & (res[:, 0] <= res[:, 6]))))