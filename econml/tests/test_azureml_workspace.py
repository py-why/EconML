# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import unittest
import pytest
from azureml.core import Workspace
from azureml.core.authentication import AzureCliAuthentication
import os


@pytest.mark.automl
class TestAutoML(unittest.TestCase):

    # can't easily test output, but can at least test that we can all export_graphviz, render, and plot
    def test_can_create_workspace(self):
        subscription_id = os.getenv("SUBSCRIPTION_ID")
        resource_group = os.getenv("RESOURCE_GROUP")
        workspace_name = os.getenv("WORKSPACE_NAME")

        cli = AzureCliAuthentication()

        wkspc = Workspace(subscription_id, resource_group, workspace_name, auth=cli)

    def test_failure(self):
        self.assertFalse(True)