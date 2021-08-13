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

    econml.dr.DRLearner
    econml.dr.LinearDRLearner
    econml.dr.SparseLinearDRLearner
    econml.dr.ForestDRLearner

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

    econml.iv.dml.OrthoIV
    econml.iv.dml.DMLIV
    econml.iv.dml.NonParamDMLIV

.. _driv_api:

Doubly Robust (DR) IV
^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: _autosummary

    econml.iv.dr.DRIV
    econml.iv.dr.LinearDRIV
    econml.iv.dr.SparseLinearDRIV
    econml.iv.dr.ForestDRIV
    econml.iv.dr.IntentToTreatDRIV
    econml.iv.dr.LinearIntentToTreatDRIV

.. _deepiv_api:

DeepIV
^^^^^^

.. autosummary::
    :toctree: _autosummary

    econml.iv.nnet.DeepIV

.. _tsls_api:

Sieve Methods
^^^^^^^^^^^^^

.. autosummary::
    :toctree: _autosummary

    econml.iv.sieve.SieveTSLS
    econml.iv.sieve.HermiteFeatures
    econml.iv.sieve.DPolynomialFeatures

.. _dynamic_api:

Estimators for Dynamic Treatment Regimes
----------------------------------------

.. _dynamicdml_api:

Dynamic Double Machine Learning
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: _autosummary

    econml.dynamic.dml.DynamicDML

.. _policy_api:

Policy Learning
---------------

.. autosummary::
    :toctree: _autosummary

    econml.policy.DRPolicyForest
    econml.policy.DRPolicyTree
    econml.policy.PolicyForest
    econml.policy.PolicyTree

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


.. Integration with AzureML AutoML
.. -------------------------------

.. .. autosummary::
..     :toctree: _autosummary

..     econml.automated_ml

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

Inference Results
^^^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: _autosummary

    econml.inference.NormalInferenceResults
    econml.inference.EmpiricalInferenceResults
    econml.inference.PopulationSummaryResults

Inference Methods
^^^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: _autosummary

    econml.inference.BootstrapInference
    econml.inference.GenericModelFinalInference
    econml.inference.GenericSingleTreatmentModelFinalInference
    econml.inference.LinearModelFinalInference
    econml.inference.StatsModelsInference
    econml.inference.GenericModelFinalInferenceDiscrete
    econml.inference.LinearModelFinalInferenceDiscrete
    econml.inference.StatsModelsInferenceDiscrete

.. _solutions_api:

Solutions
---------

Causal Analysis
^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: _autosummary

    econml.solutions.causal_analysis.CausalAnalysis

.. _dowhy_api:

Integration with DoWhy
----------------------

.. autosummary::
    :toctree: _autosummary
    
    econml.dowhy.DoWhyWrapper


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
    econml.dml._rlearner
    econml.inference._bootstrap
