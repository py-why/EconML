Machine Learning Based Estimation of Heterogeneous Treatment Effects
====================================================================

One of the biggest promises of machine learning is the automation of decision making in a multitude of application domains. 
A core problem that arises in most data-driven personalized decision scenarios is the estimation of heterogeneous treatment 
effects: what is the effect of an intervention on an outcome of interest as a function of a set of observable characteristics
of the treated sample? For instance, this problem arises in personalized pricing, where the goal is to estimate the effect of a
price discount on the demand as a function of characteristics of the consumer. Similarly it arises in medical trials where the 
goal is to estimate the effect of a drug treatment on the clinical response of a patient as a function of patient 
characteristics. In many such settings we have an abundance of observational data, where the treatment was chosen via 
some unknown policy and the ability to run A/B tests is limited. 

The EconML package implements recent techniques in the literature at the intersection of econometrics and machine
learning that tackle the problem of heterogeneous treatment effect estimation via machine learning based approaches.
These novel methods offer large flexibility in modeling the effect heterogeneity (via techniques such as random forests,
boosting, lasso and neural nets), while at the same time leverage techniques from causal inference and econometrics to
preserve the causal interpretation of the learned model and many times also offer statistical validity via the construction
of valid confidence intervals. 

It implements techniques from recent academic works, several of which produced in-house by 
the ALICE project of Microsoft Research, and many others from leading groups in the field.
Examples include Double Machine Learning (see e.g. [Chernozhukov2016]_, [Chernozhukov2017]_,
[Mackey2017]_, [Nie2017]_, [Chernozhukov2018]_, [Foster2019]_), Causal Forests (see e.g. [Wager2018]_, [Athey2019]_
[Oprescu2019]_),
Deep Instrumental Variables (see e.g. [Hartford2017]_), Non-parametric Instrumental Variables [Newey2003]_,
meta-learners (see e.g. [Kunzel2017]_).
The library brings together all these diverse techniques under a common
python API.


Motivating Examples
===================

Recommendation A/B testing
-----------------------------

Interpret experiments with imperfect compliance

Question: A travel website would like to know whether joining a membership program
causes users to spend more time engaging with the website. 

Problem: They can’t look directly at existing data, comparing members and non-members,
because the customers who chose to become members are likely already more engaged than other users.
Nor can they run a direct A/B test because they can’t force users to sign up for membership. 

Solution: The company had run an earlier experiment to test the value of a new,
faster sign-up process. EconML’s DRIV estimator uses this experimental nudge towards membership
as an instrument that generates random variation in the likelihood of membership. 
The DRIV model adjusts for the fact that not every customer who was offered the easier sign-up
became a member and returns the effect of membership rather than the effect of receiving the quick sign-up.

You can find the jupyter notebook 
`here <https://github.com/microsoft/EconML/blob/main/notebooks/CustomerScenarios/Case%20Study%20-%20Recommendation%20AB%20Testing%20at%20An%20Online%20Travel%20Company.ipynb>`_

Customer Segmentation
----------------------

Estimate individualized responses to incentives

Question: A media subscription service would like to offer targeted discounts
through a personalized pricing plan. 

Problem: They observe many features of their customers,
but are not sure which customers will respond most to a lower price. 

Solution: EconML’s DML estimator uses price variations in existing data, 
along with a rich set of user features, to estimate heterogeneous price sensitivities
that vary with multiple customer features. 
The tree interpreter provides a presentation-ready summary of the key features
that explain the biggest differences in responsiveness to a discount.

You can find the jupyter notebook 
`here <https://github.com/microsoft/EconML/blob/main/notebooks/CustomerScenarios/Case%20Study%20-%20Customer%20Segmentation%20at%20An%20Online%20Media%20Company.ipynb>`_

Multi-investment Attribution
-----------------------------
Distinguish the effects of multiple outreach efforts

Question: A startup would like to know the most effective approach for recruiting new customers: 
price discounts, technical support to ease adoption, or a combination of the two. 

Problem: The risk of losing customers makes experiments across outreach efforts too expensive. 
So far, customers have been offered incentives strategically, 
for example larger businesses are more likely to get technical support. 

Solution: EconML’s Doubly Robust Learner model jointly estimates the effects of multiple discrete treatments. 
The model uses flexible functions of observed customer features to filter out confounding correlations
in existing data and deliver the causal effect of each effort on revenue.

You can find the jupyter notebook 
`here <https://github.com/microsoft/EconML/blob/main/notebooks/CustomerScenarios/Case%20Study%20-%20Multi-investment%20Attribution%20at%20A%20Software%20Company.ipynb>`_