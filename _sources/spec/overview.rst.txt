Overview
=========

EconML is a Python package that applies the power of machine learning techniques to estimate individualized causal responses from observational or experimental data. The suite of estimation methods provided in EconML represents the latest advances in causal machine learning. By incorporating individual machine learning steps into interpretable causal models, these methods improve the reliability of what-if predictions and make causal analysis quicker and easier for a broad set of users.

EconML is open source software developed by the `ALICE <https://www.microsoft.com/en-us/research/project/alice/>`__ team at Microsoft Research.

.. raw:: html

    <p></p>
    <div class="ms-grid " style = "text-align: left; box-sizing: border-box; display: block; margin-left: auto; margin-right: auto; max-width: 1600px; position: relative; padding-left: 0; padding-right: 0; width: 100%;">
            <div class="ms-row" style = "text-align: left; box-sizing: border-box; -webkit-box-align: stretch; align-items: stretch; display: flex; flex-wrap: wrap; margin-left: 3px; margin-right: 3px;">
                    <div class="m-col-8-24 x-hidden-focus" style = "text-align: left; box-sizing: border-box; float: left; margin: 0; padding-left: 1vw; padding-right: 1vw; position: relative; width: 33.33333%;">
                    <p style="text-align:center;"><img loading="lazy" class="size-full wp-image-656358 aligncenter x-hidden-focus" src="../imgFlexible.png" alt="Flexible icon" width="92" height="92"></p><p style="text-align: center"><b>Flexible</b></p><p class="x-hidden-focus">Allows for flexible model forms that do not impose strong assumptions, including models of heterogenous responses to treatment.</p><p>	</p></div>
            <div class="m-col-8-24" style = "text-align: left; box-sizing: border-box; float: left; margin: 0; padding-left: 1vw; padding-right: 1vw; position: relative; width: 33.33333%;">
            <p style="text-align:center;"><img loading="lazy" class="size-full wp-image-656355 aligncenter" src="../imgUnified.png" alt="Unified icon" width="92" height="92"></p><p style="text-align: center"><b>Unified</b></p><p>Broad set of methods representing latest advances in the econometrics and machine learning literature within a unified API.</p><p>	</p></div>
            <div class="m-col-8-24" style = "text-align: left; box-sizing: border-box; float: left; margin: 0; padding-left: 1vw; padding-right: 1vw; position: relative; width: 33.33333%;">
            <p style="text-align:center;"><img loading="lazy" class="size-full wp-image-656352 aligncenter" src="../imgFamiliar.png" alt="Familiar icon" width="92" height="92"></p><p style="text-align: center"><b>Familiar Interface</b></p><p class="x-hidden-focus">Built on standard Python packages for machine learning and data analysis.</p><p>	</p></div>
        <p></p>		</div>
        </div>

**Why causality?**

Decision-makers need estimates of causal impacts to answer what-if questions about shifts in policy - such as changes in product pricing for businesses or new treatments for health professionals.

**Why not just a vanilla machine learning solution?**

Most current machine learning tools are designed to forecast what will happen next under the present strategy, but cannot be interpreted to predict the effects of particular changes in behavior. 

**Why causal machine learning/EconML?**

Existing solutions to answer what-if questions are expensive. Decision-makers can engage in active experimentation like A/B testing or employ highly trained economists who use traditional statistical models to infer causal effects from previously collected data. 
