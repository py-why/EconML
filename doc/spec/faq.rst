Frequently Asked Questions (FAQ)
====================================================================

When should I use EconML?
------------------

EconML is designed to answer causal questions: what will happen in response to some change in behavior, 
prices, or conditions? These questions require different methods than forecasting questions: 
what will happen next if everything continues as it has been?


What are the advantages of EconML?
------------------

EconML offers the broadest range of cutting-edge AI models designed specifically to answer causal questions. 
The EconML models also build on familiar Python packages, allowing users to easily select the best model for their question. 
Finally, EconML includes custom interpreters to create presentation-ready output.


How do I know if the results make sense?
------------------

Try comparing the consistency of your estimates across multiple models, including some that make
stronger structural assumptions like linear relationships and some that do not. Pay attention to the 
standard errors as well as the point estimates—imprecise estimates should be interpreted accordingly. 
While researchers can introduce bias by narrowly fishing for estimates that match their prior, it is also important
to use your expertise to evaluate results. If you estimate that a 5% decrease in price generates
an implausible 5000% increase in sales you should carefully review your code!

I'm getting causal estimates that don't make sense. What next?
------------------
First carefully check your code for errors and try several causal models. 
If your estimates are consistent, but implausible, you may have a confounding variable that hasn’t been measured in your data.
Think carefully about the source of the data you are using: was there something unusual going on 
during the period when the data were collected (for example a holiday or an economic downturn)?
Is there something unusual about your sample (for example, all men with pre-existing heart conditions)?


What if I don't have a good instrument, can't run an experiment, and don't observe all confounders?
------------------
In this case, no statistical approach can perfectly isolate the causal effect of the treatment on the outcome. 
DML, OrthoForest, or MetaLearners, all including all the confounders you can observe, 
will deliver the best approximation of the causal effect that minimizes the bias from confounders. 
Be aware of some remaining bias when using these estimates.


How can I test whether I'm identifying the causal effect?
------------------
You are identifying a valid causal effect if and only if the underlying assumptions of the causal model
assumed by the estimation routine are correct. Those are often hard to test (though the DoWhy package may help).
Having made those assumptions, the EconML package allows you to fit the best causal model you can.
Many models will store a final stage fit metric that can be used to validate how well the causal model predicts out of sample, 
which is a good diagnostic as to the quality of your model.


How do I give feedback?
------------------

This project welcomes contributions and suggestions. Most contributions require you to agree to
a Contributor License Agreement (CLA) declaring that you have the right to, and actually do, 
grant us the rights to use your contribution. For details, visit https://cla.microsoft.com.


When you submit a pull request, a CLA-bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., label, comment). 
Simply follow the instructions provided by the bot. You will only need to do this once across all repos using our CLA.


This project has adopted the Microsoft Open Source Code of Conduct. 
For more information see the Code of Conduct FAQ or contact opencode@microsoft.com with any additional questions or comments.







