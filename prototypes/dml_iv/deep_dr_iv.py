import numpy as np
from sklearn.model_selection import KFold
from econml.utilities import hstack
from dr_iv import DRIV, ProjectedDRIV, IntentToTreatDRIV
import keras
import keras.layers as L
from keras.models import Model, clone_model

class _KerasModel:
    """
    A model that fits data using a Keras model

    Parameters
    ----------
    h: Model
        The Keras model that takes input X and returns a prediction Y
    """

    def __init__(self, h,
                 optimizer='adam',
                 training_options={ "epochs": 30,
                                    "batch_size": 32,
                                    "validation_split": 0.1,
                                    "callbacks": [keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)]}):
        self._h = clone_model(h)
        self._h.set_weights(h.get_weights())
        self._optimizer = optimizer
        self._training_options = training_options

    def fit(self, X, Y, sample_weight=None):
        if sample_weight is not None:
            sample_weight = X.shape[0] * sample_weight / np.linalg.norm(sample_weight, ord=1)
        self._h.compile(self._optimizer, loss='mse')
        self._h.fit([X], Y, sample_weight=sample_weight, **self._training_options)

    def predict(self, X):
        return self._h.predict([X]).flatten()

    def __deepcopy__(self, memo):
        h_clone = clone_model(self._h)
        h_clone.set_weights(self._h.get_weights())
        return _KerasModel(h_clone, self._optimizer, self._training_options)

class DeepDRIV(DRIV):
    """
    DRIV with a Deep neural net as a final CATE model
    """

    def __init__(self, model_Y_X, model_T_X, model_Z_X,
                 prel_model_effect, model_TZ_X,
                 h,
                 optimizer='adam',
                 training_options={ "epochs": 30,
                                    "batch_size": 32,
                                    "validation_split": 0.1,
                                    "callbacks": [keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)]},
                 cov_clip=.1,
                 n_splits=3,
                 binary_instrument=False, binary_treatment=False,
                 opt_reweighted=False):
        """
        Parameters
        ----------
        model_Y_X : model to predict E[Y | X]
        model_T_X : model to predict E[T | X]. In alt_fit, this model is also
            used to predict E[T | X, Z]
        model_Z_X : model to predict E[Z | X]
        prel_model_effect : model that estimates a preliminary version of the CATE
            (e.g. via DMLIV or other method)
        model_TZ_X : model to estimate E[T * Z | X]
        h : Model
            Keras model that takes X as an input and returns a layer of dimension d_y by d_t
        optimizer : keras optimizer
        training_options : dictionary of keras training options
        cov_clip : clipping of the covariate for regions with low "overlap",
            so as to reduce variance
        n_splits : number of splits to use in cross-fitting
        binary_instrument : whether to stratify cross-fitting splits by instrument
        binary_treatment : whether to stratify cross-fitting splits by treatment
        opt_reweighted : whether to reweight the samples to minimize variance. If True then
            model_effect.fit must accept sample_weight as a kw argument (WeightWrapper from
            utilities can be used for any linear model to enable sample_weights). If True then
            assumes the model_effect is flexible enough to fit the true CATE model. Otherwise,
            it method will return a biased projection to the model_effect space, biased
            to give more weight on parts of the feature space where the instrument is strong.
        """
        super(DeepDRIV, self).__init__(model_Y_X, model_T_X, model_Z_X,
                                    prel_model_effect, model_TZ_X,
                                   _KerasModel(h, optimizer=optimizer, training_options=training_options),
                                   cov_clip=cov_clip,
                                   n_splits=n_splits,
                                   binary_instrument=binary_instrument, binary_treatment=binary_treatment,
                                   opt_reweighted=opt_reweighted)
        return

class DeepProjectedDRIV(ProjectedDRIV):
    """
    ProjectedDRIV with deep net as final CATE model
    """

    def __init__(self, model_Y_X, model_T_X, model_T_XZ,
                 prel_model_effect, model_TZ_X,
                 h,
                 optimizer='adam',
                 training_options={ "epochs": 30,
                                    "batch_size": 32,
                                    "validation_split": 0.1,
                                    "callbacks": [keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)]},
                 cov_clip=.1,
                 n_splits=3,
                 binary_instrument=False, binary_treatment=False,
                 opt_reweighted=False):
        """
        Parameters
        ----------
        model_Y_X : model to predict E[Y | X]
        model_T_X : model to predict E[T | X]. In alt_fit, this model is also
            used to predict E[T | X, Z]
        model_T_XZ : model to predict E[T | X, Z]
        model_theta : model that estimates a preliminary version of the CATE
            (e.g. via DMLIV or other method)
        model_TZ_X : model to estimate cov[T, E[T|X,Z] | X] = E[(T-E[T|X]) * (E[T|X,Z] - E[T|X]) | X].
        h : Model
            Keras model that takes X as an input and returns a layer of dimension d_y by d_t
        optimizer : keras optimizer
        training_options : dictionary of keras training options
        cov_clip : clipping of the covariate for regions with low "overlap",
            so as to reduce variance
        n_splits : number of splits to use in cross-fitting
        binary_instrument : whether to stratify cross-fitting splits by instrument
        binary_treatment : whether to stratify cross-fitting splits by treatment
        opt_reweighted : whether to reweight the samples to minimize variance. If True then
            model_effect.fit must accept sample_weight as a kw argument (WeightWrapper from
            utilities can be used for any linear model to enable sample_weights). If True then
            assumes the model_effect is flexible enough to fit the true CATE model. Otherwise,
            it method will return a biased projection to the model_effect space, biased
            to give more weight on parts of the feature space where the instrument is strong.
        """
        super(DeepProjectedDRIV, self).__init__(model_Y_X, model_T_X, model_T_XZ,
                                            prel_model_effect, model_TZ_X,
                                            _KerasModel(h, optimizer=optimizer, training_options=training_options),
                                            cov_clip=cov_clip,
                                            n_splits=n_splits,
                                            binary_instrument=binary_instrument, binary_treatment=binary_treatment,
                                            opt_reweighted=opt_reweighted)
        return

class DeepIntentToTreatDRIV(IntentToTreatDRIV):
    """
    Implements the DRIV algorithm for the intent-to-treat A/B test setting
    """

    def __init__(self, model_Y_X, model_T_XZ,
                 h,
                 optimizer='adam',
                 training_options={ "epochs": 30,
                                    "batch_size": 32,
                                    "validation_split": 0.1,
                                    "callbacks": [keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)]},
                 final_model_effect=None,
                 cov_clip=.1,
                 n_splits=3,
                 opt_reweighted=False):
        """
        Parameters
        ----------
        model_Y_X : model to predict E[Y | X]
        model_T_XZ : model to predict E[T | X, Z]
        h : Model
            Keras model that takes X as an input and returns a layer of dimension d_y by d_t
        optimizer : keras optimizer
        training_options : dictionary of keras training options
        final_model_effect : a final model for the CATE and projections. If None, then
            flexible_model_effect is also used as a final model
        cov_clip : clipping of the covariate for regions with low "overlap",
            so as to reduce variance
        n_splits : number of splits to use in cross-fitting
        opt_reweighted : whether to reweight the samples to minimize variance. If True then
            final_model_effect.fit must accept sample_weight as a kw argument (WeightWrapper from
            utilities can be used for any linear model to enable sample_weights). If True then
            assumes the final_model_effect is flexible enough to fit the true CATE model. Otherwise,
            it method will return a biased projection to the model_effect space, biased
            to give more weight on parts of the feature space where the instrument is strong.
        """
        flexible_model_effect = _KerasModel(h, optimizer=optimizer, training_options=training_options)
        super(DeepIntentToTreatDRIV, self).__init__(model_Y_X, model_T_XZ,
                                                    flexible_model_effect,
                                                    final_model_effect=final_model_effect,
                                                    cov_clip=cov_clip,
                                                    n_splits=n_splits,
                                                    opt_reweighted=opt_reweighted)
        return
