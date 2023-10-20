Federated Learning in Microsoft EconML Library
==============================================
.. contents::
    :local:
    :depth: 2

Overview
--------

Federated Learning in the Microsoft EconML Library empowers collaborative model training across decentralized edge devices or data silos while preserving privacy and data security. Unlike traditional centralized machine learning, Federated Learning allows models to be trained locally on individual devices or servers without sharing raw data with a central server.

Motivation for Incorporating Federated Learning into the EconML Library
-----------------------------------------------------------------------

1. **Privacy Preservation**: Federated Learning enables organizations to build machine learning models without centralizing or sharing sensitive data.

2. **Data Decentralization**: It's well-suited for scenarios where data is decentralized across various devices or locations.

3. **Efficient Model Training**: Federated Learning leverages the computational power of distributed nodes, enabling the development of machine learning models in resource-constrained environments.

4. **Regulatory Compliance**: It assists in complying with data privacy regulations by keeping data localized and reducing exposure to compliance risks.

Federated Learning with EconML
------------------------------

Introducing `FederatedEstimator`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the EconML Library, we provide the `FederatedEstimator` class, a powerful tool for conducting Federated Learning using LinearDML estimators. The `FederatedEstimator` class allows you to aggregate models trained on local data sources in a privacy-preserving and efficient manner. This is invaluable for scenarios where sensitive data remains on local devices, such as smartphones, IoT devices, or remote servers.

`FederatedEstimator` Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- `estimators`: A list of `LinearDML` estimators. These are individual LinearDML models trained on different data sources or devices.

`FederatedEstimator` Attributes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- `estimators`: A list of `LinearDML` estimators provided during initialization.
- `model_final_`: The aggregated model obtained by combining models from the `estimators`.

Example Usage
~~~~~~~~~~~~~
.. code:: python

    # This is a Python code block
    from econml.dml import LinearDML
    from econml.sklearn_extensions.linear_model import StatsModelsLinearRegression
    from your_library import FederatedEstimator  # Replace with the actual import path

    # Create individual LinearDML estimators
    estimator1 = LinearDML(...)
    estimator2 = LinearDML(...)
    # Add more estimators as needed

    # Create a FederatedEstimator by providing a list of estimators
    federated_estimator = FederatedEstimator([estimator1, estimator2, ...])

    # Access the aggregated model
    aggregated_model = federated_estimator.model_final_

Choices of combining Final Results in Federated Learning
--------------------------------------------------------

The choice of aggregation method depends on your specific requirements. Options include centralized aggregation, federated aggregation, secure multi-party computation (SMPC), and differential privacy.

By using FederatedEstimator, you can efficiently aggregate models trained on decentralized data sources to achieve collaborative model training without centralizing or sharing sensitive data.

This is just one example of how EconML empowers practitioners and researchers to harness the potential of decentralized, privacy-preserving machine learning across various domains and use cases.

For more information on the EconML Library, including detailed documentation and usage examples, please refer to the official documentation.

Other federated Learning Options
--------------------------------

1. **Option 1: Evaluating with Apache Spark**: Use Apache Spark for distributed evaluation.

2. **Option 2: Azure Apache Spark Cluster Provisioning**: Set up an Azure Apache Spark cluster for local testing using Azure Databricks.