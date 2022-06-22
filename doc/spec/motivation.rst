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

Customer Targeting
------------------

An important problem in modern business analytics is building automated tools to prioritize customer
acquisition and personalize customer interactions to increase sales and revenue. Typically businesses
will offer personalized incentives to customers to increase spend or increase the level of
engagement via more human resources. Any such personalized intervention corresponds to a monetary
investment and the main question that business analytics are called to answer is: what is the return
on investment (ROI)? 

Analyzing the ROI is inherently a treatment effect question: what was the effect of any investment
on a particular customer on its spend? Understanding how these return on investment varies across
customers can enable more targeted investment policies and increased ROI via better targeting. Using historical
data from deployed investments, and estimating the heterogeneous treatment effect via any of
the proposed methods, business analysts can learn in an automated manner, data-driven
customer targeting and prioritization policies.

Personalized Pricing
--------------------

Personalized discounts have become very widespread in the digital economy. To set the optimal
personalized discount policy a business needs to understand what is the effect
of a drop in price on the demand of a customer for a product as a function of customer
characteristics. The estimation of such personalized demand elasticities can also be
phrased in the language of heterogeneous treatment effects, where the treatment 
is the price (or typically log of price) on the demand (or typically log of demand)
as a function of observable features of the customer. Hence, estimation of heterogeneous
treatment effects can lead to optimal pricing policies.


Stratification in Clinical Trials
----------------------------------------

Which patients should be selected for a clinical trial? If we want to demonstrate
that a clinical treatment has an effect on at least some subset of a population, then
fully randomized clinical trials are inappropriate as they will solely estimate
average effects. Using heterogeneous treatment effect techniques, we can use
observational data to come up with estimates of these effects and identify
good candidate patients for a clinical trial that our model estimates have high
treatment effects.

Learning Click-Through-Rates
----------------------------

In the design of a page layout and more importantly in ad placement, it is important
to understand the click-through-rate of page components (e.g. ads) on different positions
of a page. Even though the modern approach is to run multiple A/B tests, when such
page components involve revenue considerations (such as ad placement), then observational
data can help guide correct A/B tests to run. Heterogeneous treatment effect estimation
can provide estimates of the click-through-rate of page components from
observational data. In this setting, the treatment is simply whether the component is
placed on that page position and the response is whether the user clicked on it.


More Use Cases
----------------------------

You can find more use cases on the Microsoft EconML project page `here <https://www.microsoft.com/en-us/research/project/econml/use-cases/>`_: 