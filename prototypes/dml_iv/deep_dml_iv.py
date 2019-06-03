
import numpy as np
from sklearn.model_selection import KFold
from econml.utilities import hstack
from dml_iv import _BaseDMLIV
import keras
import keras.layers as L
from keras.models import Model, clone_model

class DeepDMLIV(_BaseDMLIV):
    """
    A child of the _BaseDMLIV class that specifies a deep neural network effect model
    where the treatment effect is linear in some featurization of the variable X.
    """

    def __init__(self, model_Y_X, model_T_X, model_T_XZ, h,
                 optimizer='adam',
                 training_options={ "epochs": 30,
                                    "batch_size": 32,
                                    "validation_split": 0.1,
                                    "callbacks": [keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)]},
                 n_splits=2, binary_instrument=False, binary_treatment=False):
        """
        Parameters
        ----------
        model_Y_X : arbitrary model to predict E[Y | X]
        model_T_X : arbitrary model to predict E[T | X]
        model_T_XZ : arbitrary model to predict E[T | X, Z]
        h : Model
            Keras model that takes X as an input and returns a layer of dimension d_y by d_t
        optimizer : keras optimizer
        training_options : dictionary of keras training options
        n_splits : number of splits to use in cross-fitting
        binary_instrument : whether to stratify cross-fitting splits by instrument
        binary_treatment : whether to stratify cross-fitting splits by treatment
        """
        class ModelEffect:
            """
            A wrapper class that takes as input X, T, y and estimates an effect model of the form
            $y= \\theta(X) \\cdot T + \\epsilon$
            """

            def __init__(self, h):
                """
                Parameters
                ----------
                h : Keras model mapping X to Theta(X)
                """
                self._h = clone_model(h)
                self._h.set_weights(h.get_weights())

            def fit(self, Y, T, X):
                """
                Parameters
                ----------
                y : outcome
                T : treatment
                X : features
                """
                d_x, d_t, d_y = [np.shape(arr)[1:] for arr in (X, T, Y)]
                self.d_t = d_t  # keep track in case we need to reshape output by dropping singleton dimensions
                self.d_y = d_y  # keep track in case we need to reshape output by dropping singleton dimensions
                d_x, d_t, d_y = [1 if not d else d[0] for d in (d_x, d_t, d_y)]
                x_in, t_in = [L.Input((d,)) for d in (d_x, d_t)]
                # reshape in case we get fewer dimensions than expected from h (e.g. a scalar)
                h_out = L.Reshape((d_y, d_t))(self._h(x_in))
                y_out = L.Dot([2, 1])([h_out, t_in])
                self.theta = Model([x_in], self._h(x_in))
                model = Model([x_in, t_in], y_out)
                model.compile(optimizer, loss='mse')
                model.fit([X, T], Y, **training_options)
                return self

            def predict(self, X):
                """
                Parameters
                ----------
                X : features
                """

                # HACK: DRIV doesn't expect a treatment dimension, so pretend we got a vector even if we really had a one-column array
                #       Once multiple treatments are supported, we'll need to fix this
                self.d_t = ()

                return self.theta.predict([X]).reshape((-1,)+self.d_y+self.d_t)

        super(DeepDMLIV, self).__init__(model_Y_X, model_T_X, model_T_XZ,
                         ModelEffect(h), n_splits=n_splits,
                         binary_instrument=binary_instrument,
                         binary_treatment=binary_treatment)

