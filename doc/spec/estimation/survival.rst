.. _survivaluserguide:

========================
Survival HTE Estimation
========================

What is it?
===========

This section documents the censored-outcome estimators for single-event survival
targets that were added for this project. The implemented estimand is the
restricted mean survival time (RMST) contrast up to a user-specified horizon
:math:`\tau`.

The implementation follows the migration from the R reference code in
``original/learners/survival_cut.R`` and ``original/learners/hte_learners.R``.
The main design choices in this project are:

* nuisance quantities are estimated out of sample
* learner predictions on the training sample are also out of sample
* survival-specific nuisances are handled through ``econml.censor``
* second-stage HTE estimation is handled through cross-fitted learner classes


What is the estimand?
=====================

The main estimand in the single-event survival setting is the conditional RMST
treatment effect up to horizon :math:`\tau`:

.. math::

    \psi_{0}^{\mathrm{RMST}}(\tau, X)
    =
    E_{0}\{\min(T^{a=1}, \tau) \mid X\}
    -
    E_{0}\{\min(T^{a=0}, \tau) \mid X\}.

Equivalently, if :math:`\mathrm{RMST}_{0}(\tau \mid a, X)` denotes the arm-
specific conditional restricted mean survival time, then

.. math::

    \psi_{0}^{\mathrm{RMST}}(\tau, X)
    =
    \mathrm{RMST}_{0}(\tau \mid 1, X)
    -
    \mathrm{RMST}_{0}(\tau \mid 0, X).

The transformations :func:`~econml.censor.ipcw_cut_rmst`,
:func:`~econml.censor.bj_cut_rmst`,
:func:`~econml.censor.aipcw_cut_rmst`, and
:func:`~econml.censor.uif_diff_rmst` are all designed so that downstream
continuous-outcome learners target this same conditional RMST contrast.


What are the core building blocks?
==================================

Single-cause survival estimation in the package is organized into four layers:

* nuisance estimation in :mod:`econml.censor`
* RMST pseudo-outcome transformations in :mod:`econml.censor`
* direct survival learners in :mod:`econml.metalearners`
* forest-based estimators in :mod:`econml.grf`

The survival nuisance functions are:

* :math:`G(t \mid A, X)`: censoring survival
* :math:`S(t \mid A, X)`: event survival
* :math:`e(X) = P(A=1 \mid X)`: treatment propensity, when needed for UIF-style scores

These nuisance quantities are combined into the following transformations:

* :func:`~econml.censor.ipcw_cut_rmst`
* :func:`~econml.censor.bj_cut_rmst`
* :func:`~econml.censor.aipcw_cut_rmst`
* :func:`~econml.censor.uif_diff_rmst`


How is the event variable defined?
==================================

In the single-event survival setting, the structured outcome ``Y`` uses two
fields: ``time`` and ``event``. The event coding is:

* ``event == 0``: censored observation
* ``event != 0``: failure event

This event coding is used consistently by the nuisance functions:

* :math:`G(t \mid A, X)` is the censoring survival function, so its event
  indicator is ``event == 0``
* :math:`S(t \mid A, X)` is the failure-event survival function, so its event
  indicator is ``event != 0``

The same convention is used in the simulation and bone marrow example
notebooks for the single-cause survival endpoints.


Which estimators are available?
===============================

Direct survival learners
------------------------

The following classes operate directly on a structured survival outcome object
``Y`` with fields ``event`` and ``time``:

* :class:`~econml.metalearners.SurvivalTLearner`
* :class:`~econml.metalearners.SurvivalSLearner`
* :class:`~econml.grf.CausalSurvivalForest`

Pseudo-outcome learners
-----------------------

After constructing an RMST pseudo-outcome, the following cross-fitted learners
can be used:

* :class:`~econml.metalearners.TLearner`
* :class:`~econml.metalearners.SLearner`
* :class:`~econml.metalearners.XLearner`
* :class:`~econml.metalearners.IPTWLearner`
* :class:`~econml.metalearners.AIPTWLearner`
* :class:`~econml.metalearners.MCLearner`
* :class:`~econml.metalearners.MCEALearner`
* :class:`~econml.metalearners.ULearner`
* :class:`~econml.metalearners.RALearner`
* :class:`~econml.metalearners.RLearner`
* :class:`~econml.metalearners.IFLearner`
* :class:`~econml.grf.GRFCausalForest`


Recommended workflow
====================

The intended workflow is:

1. Estimate nuisance functions with
   :func:`~econml.censor.fit_nuisance_survival_crossfit` or
   :func:`~econml.censor.fit_nuisance_survival`.
2. Construct the target pseudo-outcome with one of the RMST transformations.
3. Fit a cross-fitted learner on ``(Y^*, T, X)``.

The public ``econml.metalearners`` names listed here resolve to the
cross-fitted censored-outcome implementations, so training-sample predictions
are returned as fold-held-out out-of-sample estimates rather than in-sample
fits.

This mirrors the two-stage orthogonal-learning structure used throughout
EconML, except that the censored-outcome transformation is made explicit as a
separate survival-specific step.


Notebook examples
=================

The following notebooks demonstrate the survival workflow:

* `Survival HTE Examples.ipynb <https://github.com/py-why/EconML/blob/main/notebooks/Survival%20HTE%20Examples.ipynb>`_
* `Censored Outcomes - Bone Marrow Transplant.ipynb <https://github.com/py-why/EconML/blob/main/notebooks/Solutions/Censored%20Outcomes%20-%20Bone%20Marrow%20Transplant.ipynb>`_


API summary
===========

Utilities and transformations
-----------------------------

.. autosummary::
    :toctree: ../../_autosummary

    econml.censor.fit_nuisance_survival
    econml.censor.fit_nuisance_survival_crossfit
    econml.censor.NuisanceResult
    econml.censor.CrossFitNuisanceResult
    econml.censor.ipcw_cut_rmst
    econml.censor.bj_cut_rmst
    econml.censor.aipcw_cut_rmst
    econml.censor.uif_diff_rmst

Learners
--------

.. autosummary::
    :toctree: ../../_autosummary

    econml.metalearners.SurvivalTLearner
    econml.metalearners.SurvivalSLearner
    econml.metalearners.TLearner
    econml.metalearners.SLearner
    econml.metalearners.XLearner
    econml.metalearners.IPTWLearner
    econml.metalearners.AIPTWLearner
    econml.metalearners.MCLearner
    econml.metalearners.MCEALearner
    econml.metalearners.ULearner
    econml.metalearners.RALearner
    econml.metalearners.RLearner
    econml.metalearners.IFLearner
    econml.grf.CausalSurvivalForest
    econml.grf.GRFCausalForest
    econml.grf.causal_forest
