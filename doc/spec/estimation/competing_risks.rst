.. _competingrisksuserguide:

================================
Competing Risks HTE Estimation
================================

What is it?
===========

This section documents the censored-outcome estimators for competing-risks
targets that were added for this project. The primary estimand is the
restricted mean time lost (RMTL) for a target cause :math:`j` up to a horizon
:math:`\tau`. The package also includes separable direct and indirect effect
transformations for the competing-risks setting.

The implementation follows the migration from the R reference code in
``original/learners/competing_cut.R`` and ``original/learners/hte_learners.R``.
As in the survival setting, nuisance quantities and learner predictions are
computed out of sample.


What are the estimands?
=======================

For a target cause :math:`j`, the primary total-effect estimand is the
conditional RMTL treatment effect up to horizon :math:`\tau`:

.. math::

    \psi_{j,0}^{\mathrm{RMTL}}(\tau, X)
    =
    E_{0}[\{\tau - \min(T^{a=1}, \tau)\} I(J^{a=1}=j) \mid X]
    -
    E_{0}[\{\tau - \min(T^{a=0}, \tau)\} I(J^{a=0}=j) \mid X].

Equivalently, if :math:`\mathrm{RMTL}_{j,0}(\tau \mid a, X)` denotes the
arm-specific conditional restricted mean time lost for cause :math:`j`, then

.. math::

    \psi_{j,0}^{\mathrm{RMTL}}(\tau, X)
    =
    \mathrm{RMTL}_{j,0}(\tau \mid 1, X)
    -
    \mathrm{RMTL}_{j,0}(\tau \mid 0, X).

The total-effect transformations
:func:`~econml.censor.ipcw_cut_rmtlj`,
:func:`~econml.censor.bj_cut_rmtlj`,
:func:`~econml.censor.aipcw_cut_rmtlj`, and
:func:`~econml.censor.uif_diff_rmtlj` all target this same conditional RMTL
contrast.

The notebook and API also expose separable direct and indirect effects. These
are defined by decomposing treatment into :math:`A_1` and :math:`A_2`, where
:math:`A_1` acts directly on the event of interest and :math:`A_2` acts
indirectly through the competing event.

The separable direct RMTL effect is

.. math::

    \psi_{j,0}^{\mathrm{RMTL}, \mathrm{sep-D}}(\tau, a_2, X)
    =
    \mathrm{RMTL}_{j,0}(\tau \mid 1, a_2, X)
    -
    \mathrm{RMTL}_{j,0}(\tau \mid 0, a_2, X),

and the separable indirect RMTL effect is

.. math::

    \psi_{j,0}^{\mathrm{RMTL}, \mathrm{sep-I}}(\tau, a_1, X)
    =
    \mathrm{RMTL}_{j,0}(\tau \mid a_1, 1, X)
    -
    \mathrm{RMTL}_{j,0}(\tau \mid a_1, 0, X).

The current notebook workflows and transformation helpers with suffix
``astar1`` use the case :math:`a^{\star}=1`, i.e. direct effects at
:math:`a_2=1` and indirect effects at :math:`a_1=1`.


What are the nuisance quantities?
=================================

For a target cause :math:`j`, the competing-risks workflow uses:

* :math:`G(t \mid A, X)`: censoring survival
* :math:`S(t \mid A, X)`: overall event survival
* :math:`S_j(t \mid A, X)`: cause-:math:`j` survival component
* :math:`\bar{S}_j(t \mid A, X)`: competing-cause survival component
* :math:`e(X) = P(A=1 \mid X)`: treatment propensity, when needed for UIF scores

These are estimated through
:func:`~econml.censor.fit_nuisance_competing_crossfit`.


How is the event variable defined?
==================================

In the competing-risks setting, the structured outcome ``Y`` again uses the
fields ``time`` and ``event``, but the event field is multi-valued:

* ``event == 0``: censored observation
* ``event == j``: target cause :math:`j`
* ``event != 0``: any failure event
* ``event != 0`` and ``event != j``: a competing failure cause

For the common case ``cause=1``, this becomes:

* ``event == 0``: censored
* ``event == 1``: target failure cause
* ``event > 1``: competing failure cause

This coding is used by the nuisance functions as follows:

* :math:`G(t \mid A, X)` uses ``event == 0``
* :math:`S(t \mid A, X)` uses ``event != 0``
* :math:`S_j(t \mid A, X)` uses ``event == j``
* :math:`\bar{S}_j(t \mid A, X)` uses ``event != 0`` and ``event != j``

The same convention is used in the simulation and bone marrow example
notebooks for relapse-vs-NRM analyses.


Which transformations are available?
====================================

Total-effect transformations
----------------------------

* :func:`~econml.censor.ipcw_cut_rmtlj`
* :func:`~econml.censor.bj_cut_rmtlj`
* :func:`~econml.censor.aipcw_cut_rmtlj`
* :func:`~econml.censor.uif_diff_rmtlj`

Separable-effect transformations
--------------------------------

* :func:`~econml.censor.aipcw_cut_rmtlj_sep_direct_astar1`
* :func:`~econml.censor.aipcw_cut_rmtlj_sep_indirect_astar1`
* :func:`~econml.censor.uif_diff_rmtlj_sep_direct_astar1`
* :func:`~econml.censor.uif_diff_rmtlj_sep_indirect_astar1`


Which estimators are available?
===============================

Direct competing-risks learners
-------------------------------

The following classes operate directly on a structured competing-risks outcome
object ``Y`` with fields ``event`` and ``time``:

* :class:`~econml.metalearners.CompetingRisksTLearner`
* :class:`~econml.metalearners.CompetingRisksSLearner`

The following dedicated direct learners target the separable ``astar1``
estimands directly:

* :class:`~econml.metalearners.SeparableDirectAstar1TLearner`
* :class:`~econml.metalearners.SeparableDirectAstar1SLearner`
* :class:`~econml.metalearners.SeparableIndirectAstar1TLearner`
* :class:`~econml.metalearners.SeparableIndirectAstar1SLearner`

Pseudo-outcome learners
-----------------------

After constructing an RMTL pseudo-outcome, the following cross-fitted learners
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

1. Estimate competing-risks nuisance functions with
   :func:`~econml.censor.fit_nuisance_competing_crossfit`.
2. Construct a total-effect or separable-effect pseudo-outcome.
3. Fit a direct competing-risks learner or a generic cross-fitted learner on
   the pseudo-outcome.

This gives a unified way to compare direct competing-risks learners against
transformed-outcome learners in the same notebook workflow.

The public ``econml.metalearners`` names listed here resolve to the
cross-fitted censored-outcome implementations, so training-sample predictions
are returned as fold-held-out out-of-sample estimates rather than in-sample
fits.


Notebook examples
=================

The following notebooks demonstrate the competing-risks workflow:

* `Competing Risks HTE Examples.ipynb <https://github.com/py-why/EconML/blob/main/notebooks/Competing%20Risks%20HTE%20Examples.ipynb>`_
* `Censored Outcomes - Bone Marrow Transplant.ipynb <https://github.com/py-why/EconML/blob/main/notebooks/Solutions/Censored%20Outcomes%20-%20Bone%20Marrow%20Transplant.ipynb>`_


API summary
===========

Utilities and transformations
-----------------------------

.. autosummary::
    :toctree: ../../_autosummary

    econml.censor.fit_nuisance_competing_crossfit
    econml.censor.CrossFitNuisanceResult
    econml.censor.ipcw_cut_rmtlj
    econml.censor.bj_cut_rmtlj
    econml.censor.aipcw_cut_rmtlj
    econml.censor.aipcw_cut_rmtlj_sep_direct_astar1
    econml.censor.aipcw_cut_rmtlj_sep_indirect_astar1
    econml.censor.uif_diff_rmtlj
    econml.censor.uif_diff_rmtlj_sep_direct_astar1
    econml.censor.uif_diff_rmtlj_sep_indirect_astar1

Learners
--------

.. autosummary::
    :toctree: ../../_autosummary

    econml.metalearners.CompetingRisksTLearner
    econml.metalearners.CompetingRisksSLearner
    econml.metalearners.SeparableDirectAstar1TLearner
    econml.metalearners.SeparableDirectAstar1SLearner
    econml.metalearners.SeparableIndirectAstar1TLearner
    econml.metalearners.SeparableIndirectAstar1SLearner
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
    econml.grf.GRFCausalForest
    econml.grf.causal_forest
