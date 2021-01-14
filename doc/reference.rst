Public Module Reference
=======================

CATE Estimators
---------------

.. _dml_api:

Double Machine Learning (DML)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: _autosummary

    econml.dml.DML
    econml.dml.LinearDML
    econml.dml.SparseLinearDML
    econml.dml.CausalForestDML
    econml.dml.NonParamDML

.. _dr_api:

Doubly Robust (DR)
^^^^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: _autosummary

    econml.drlearner.DRLearner
    econml.drlearner.LinearDRLearner
    econml.drlearner.SparseLinearDRLearner
    econml.drlearner.ForestDRLearner

.. _metalearners_api:

Meta-Learners
^^^^^^^^^^^^^

.. autosummary::
    :toctree: _autosummary

    econml.metalearners.XLearner
    econml.metalearners.TLearner
    econml.metalearners.SLearner
    econml.metalearners.DomainAdaptationLearner

.. _orf_api:

Orthogonal Random Forest (ORF)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: _autosummary

    econml.orf.DMLOrthoForest
    econml.orf.DROrthoForest

Instrumental Variable CATE Estimators
-------------------------------------

.. _dmliv_api:

Double Machine Learning (DML) IV
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: _autosummary

    econml.ortho_iv.DMLATEIV
    econml.ortho_iv.ProjectedDMLATEIV
    econml.ortho_iv.DMLIV
    econml.ortho_iv.NonParamDMLIV

.. _driv_api:

Doubly Robust (DR) IV
^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: _autosummary

    econml.ortho_iv.IntentToTreatDRIV
    econml.ortho_iv.LinearIntentToTreatDRIV

.. _deepiv_api:

DeepIV
^^^^^^

.. autosummary::
    :toctree: _autosummary

    econml.deepiv.DeepIV
    econml.deepiv.mog_loss_model
    econml.deepiv.mog_loss_model
    econml.deepiv.mog_sample_model
    econml.deepiv.response_loss_model

.. _tsls_api:

Two Stage Least Squares (2SLS)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: _autosummary

    econml.two_stage_least_squares.NonparametricTwoStageLeastSquares
    econml.two_stage_least_squares.HermiteFeatures
    econml.two_stage_least_squares.DPolynomialFeatures


.. _interpreters_api:

CATE Interpreters
-----------------

.. autosummary::
    :toctree: _autosummary

    econml.cate_interpreter.SingleTreeCateInterpreter
    econml.cate_interpreter.SingleTreePolicyInterpreter

.. _scorers_api:

CATE Scorers
------------

.. autosummary::
    :toctree: _autosummary
    
    econml.score.RScorer
    econml.score.EnsembleCateEstimator


.. _grf_api:

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


Integration with AzureML AutoML
-------------------------------

.. autosummary::
    :toctree: _autosummary

    econml.automated_ml.setAutomatedMLWorkspace
    econml.automated_ml.addAutomatedML
    econml.automated_ml.AutomatedMLModel
    econml.automated_ml.AutomatedMLMixin
    econml.automated_ml.EconAutoMLConfig

Scikit-Learn Extensions
-----------------------

.. _sklearn_linear_api:

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

.. _sklearn_model_api:

Model Selection Extensions
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: _autosummary

    econml.sklearn_extensions.model_selection.GridSearchCVList
    econml.sklearn_extensions.model_selection.WeightedKFold
    econml.sklearn_extensions.model_selection.WeightedStratifiedKFold


.. _inference_api:

Inference
---------

.. autosummary::
    :toctree: _autosummary

    econml.bootstrap
    econml.inference


.. _utilities_api:

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
    econml._shap
    econml.cate_interpreter._tree_exporter
    econml.cate_interpreter._interpreters
    econml.dml._rlearner
    econml.orf._causal_tree
    econml.orf._ortho_forest
