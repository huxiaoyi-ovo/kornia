# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright 2018 Kornia Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import pytest
import torch

from kornia.feature.adalam import adalam as adalam_module
from kornia.feature.lightglue import LightGlue


def test_lightglue_init_does_not_print_without_weights(capsys):
    LightGlue(features=None, n_layers=1, input_dim=32, descriptor_dim=32, num_heads=4)
    assert capsys.readouterr().out == ""


def test_unknown_adalam_config_warns_without_printing(monkeypatch, capsys):
    class StubFilter:
        def __init__(self, config):
            assert "unrecognized_option" not in config

        def match_and_filter(self, *args, **kwargs):
            return torch.empty((0, 2), dtype=torch.long), torch.empty(0)

    monkeypatch.setattr(adalam_module, "AdalamFilter", StubFilter)
    descriptors = torch.ones(1, 128)
    lafs = torch.tensor([[[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]])

    with pytest.warns(UserWarning, match="unrecognized key"):
        distances, indices = adalam_module.match_adalam(
            descriptors, descriptors, lafs, lafs, config={"unrecognized_option": True}
        )

    assert distances.numel() == 0
    assert indices.shape == (0, 2)
    assert capsys.readouterr().out == ""
