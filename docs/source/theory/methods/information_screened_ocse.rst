==========================
Information-Screened oCSE
==========================

``method="information_screened"`` uses Information-LASSO as a proposal stage,
adds a conditional-information rescue pass, and then runs standard oCSE only
inside the retained candidate set.

The screen is deliberately conservative. Its purpose is not to estimate the
final parent set directly; it is to reduce the candidate universe while
preserving variables needed by the downstream oCSE refinement.

Method
======

For a target variable :math:`Y`, let:

- :math:`C` be the external lagged-candidate set;
- :math:`H_Y` be the lagged target-history conditioning set;
- :math:`S_0` be the Information-LASSO proposal;
- :math:`\widehat{W}` be the retained working set.

The proposal stage first computes

.. math::

   S_0 = \operatorname{InfoLasso}(C \rightarrow Y).

Candidates outside :math:`S_0` are then ranked by conditional mutual
information

.. math::

   r_j = \widehat{I}(X_j; Y \mid H_Y \cup S_0).

The target of this rescue pass is a parent-superset working set rather than an
exact graph estimate. In particular, the screening objective is

.. math::

   \Pr\!\left(\operatorname{Pa}(Y) \subseteq \widehat{W}\right) \rightarrow 1,

while keeping :math:`|\widehat{W}|` substantially smaller than :math:`|C|`.
The final graph decision remains the responsibility of standard oCSE:

.. math::

   \widehat{\operatorname{Pa}}(Y)
   = \operatorname{oCSE}(Y; \widehat{W}, H_Y).

Retention Is a Soft Target
==========================

``screen_retention`` is a target fraction, not a hard cap. Let
:math:`p = |C|` and

.. math::

   B_0 = \left\lceil \text{screen\_retention} \cdot p \right\rceil.

The implementation never truncates the Information-LASSO proposal. When there
are candidates outside :math:`S_0`, it also keeps at least one rescue slot so
that the screened pathway cannot silently collapse back to the proposal-only
path. The realized target size is therefore

.. math::

   B = \min\!\left(p,\; \max(B_0, |S_0| + 1)\right)

when :math:`|S_0| < p`, and :math:`B=p` otherwise.

This asymmetry is intentional. A false positive at the screening stage only
increases downstream computation, whereas a false negative removes a candidate
that restricted oCSE can no longer recover. Consequently, preserving the
proposal and leaving a rescue channel open takes priority over enforcing an
exact candidate count.

The resulting working set is

.. math::

   \widehat{W}
   = S_0 \cup \operatorname{Top}_{B-|S_0|}\{r_j : j \notin S_0\}.

Interpretation
==============

A useful way to view the method is as a two-level search:

.. code-block:: none

   all external lagged candidates
              |
              v
      Information-LASSO proposal
              |
              v
   conditional rescue given target history + proposal
              |
              v
      conservative working set
              |
              v
       standard oCSE refinement
              |
              v
          causal graph

The screening stage therefore trades a small amount of extra candidate
retention for protection against proposal-stage misses. The downstream oCSE
stage still performs the significance-based causal selection.

Current Scope
=============

Standard-path conditioning is corrected independently of screening; see
:doc:`standard_oce`. Both full and restricted searches must use that same
baseline for a meaningful comparison. Existing positional arguments retain
their order; pass the new retention control by name.

Evaluation distinguishes three quantities:

* True-parent recall: the fraction of generating external parent-lag pairs in
  the screen. Parent-complete target rate separately measures whether every
  generating parent of a target survives, excluding targets with no parents.
* Full-oCSE-support coverage: the fraction of the finite-sample full-oCSE
  selection in the screen. This is an agreement diagnostic, not ground truth.
* Final graph precision/recall/F1 after restricted refinement, plus total CMI
  evaluations and runtime including screening.

Measure these over sample sizes and realized retention fractions. A
proposal-only ablation removes rescue but changes candidate count; a
budget-matched marginal-fill ablation retains the proposal and fills to the
same size without conditioning on the proposal. Report both to distinguish
the value of conditional rescue from simply keeping more candidates.

The current implementation does not claim a finite-sample sure-screening
theorem and does not guarantee that every causal parent is retained for every
finite dataset. The relevant empirical quantity is therefore the
retention/coverage frontier: how much the candidate universe can be reduced
before parent or full-oCSE-support coverage begins to degrade.

Likewise, the rescue score is univariate conditional CMI for each excluded
candidate given :math:`H_Y \cup S_0`. Higher-order synergy searches are outside
the present scope because they would increase screening cost and change the
computational objective of this pathway.
