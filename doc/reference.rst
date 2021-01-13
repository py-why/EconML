Public Module Reference
=======================

CATE Estimators
---------------

Double Machine Learning (DML)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: _autosummary

    econml.dml.DML
    econml.dml.LinearDML
    econml.dml.SparseLinearDML
    econml.dml.CausalForestDML
    econml.dml.NonParamDML

Doubly Robust (DR)
^^^^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: _autosummary

    econml.drlearner.DRLearner
    econml.drlearner.LinearDRLearner
    econml.drlearner.SparseLinearDRLearner
    econml.drlearner.ForestDRLearner

Meta-Learners
^^^^^^^^^^^^^

.. autosummary::
    :toctree: _autosummary

    econml.metalearners.XLearner
    econml.metalearners.TLearner
    econml.metalearners.SLearner
    econml.metalearners.DomainAdaptationLearner

Orthogonal Random Forest (ORF)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: _autosummary

    econml.ortho_forest.DMLOrthoForest
    econml.ortho_forest.DROrthoForest

Instrumental Variable CATE Estimators
-------------------------------------

Double Machine Learning (DML) IV
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: _autosummary

    econml.ortho_iv.DMLATEIV
    econml.ortho_iv.ProjectedDMLATEIV
    econml.ortho_iv.DMLIV
    econml.ortho_iv.NonParamDMLIV

Doubly Robust (DR) IV
^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: _autosummary

    econml.ortho_iv.IntentToTreatDRIV
    econml.ortho_iv.LinearIntentToTreatDRIV

DeepIV
^^^^^^

.. autosummary::
    :toctree: _autosummary

    econml.deepiv.DeepIV
    econml.deepiv.mog_loss_model
    econml.deepiv.mog_loss_model
    econml.deepiv.mog_sample_model
    econml.deepiv.response_loss_model

Two Stage Least Squares (2SLS)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: _autosummary

    econml.two_stage_least_squares.NonparametricTwoStageLeastSquares
    econml.two_stage_least_squares.HermiteFeatures
    econml.two_stage_least_squares.DPolynomialFeatures


CATE Interpreters
-----------------

.. autosummary::
    :toctree: _autosummary

    econml.cate_interpreter.SingleTreeCateInterpreter
    econml.cate_interpreter.SingleTreePolicyInterpreter

CATE Scorers
------------

.. autosummary::
    :toctree: _autosummary
    
    econml.score.RScorer
    econml.score.EnsembleCateEstimator


Generalized Random Forests
--------------------------

.. autosummary::
    :toctree: _autosummary

    econml.grf.CausalForest
    econml.grf.CausalIVForest
    econml.grf.RegressionForest
    econml.grf.MultiOutputGRF
    econml.grf.LinearMomentGRFCriterion
    econml.grf.LinearMomentGRFCriterionMSE
    econml.grf._base_grf.BaseGRF
    econml.grf._base_grftree.GRFTree


Scikit-Learn Extensions
-----------------------

Linear Model Extensions
^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: _autosummary

    econml.sklearn_extensions.linear_model.DebiasedLasso
    econml.sklearn_extensions.linear_model.MultiOutputDebiasedLasso
    econml.sklearn_extensions.linear_model.SelectiveRegularization
    econml.sklearn_extensions.linear_model.StatsModelsLinearRegression
    econml.sklearn_extensions.linear_model.StatsModelsRLM
    econml.sklearn_extensions.linear_model.WeightedLasso
    econml.sklearn_extensions.linear_model.WeightedLassoCV
    econml.sklearn_extensions.linear_model.WeightedMultiTaskLassoCV
    econml.sklearn_extensions.linear_model.WeightedLassoCVWrapper

Model Selection Extensions
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: _autosummary

    econml.sklearn_extensions.model_selection.GridSearchCVList
    econml.sklearn_extensions.model_selection.WeightedKFold
    econml.sklearn_extensions.model_selection.WeightedStratifiedKFold

Inference
---------

.. autosummary::
    :toctree: _autosummary

    econml.bootstrap
    econml.inference

Utilities
---------

.. autosummary::
    :toctree: _autosummary
    
    econml.utilities

Private Module Reference
========================

.. autosummary::
    :toctree: _autosummary

    econml._ortho_learner
    econml._cate_estimator
    econml._causal_tree
    econml.dml._rlearner

