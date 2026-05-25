import pytest
import torch
import torch.nn as nn

from model.classifier import SkinLesionClassifier, build_model


@pytest.fixture(scope="module")
def model() -> SkinLesionClassifier:
    m = build_model(num_classes=8, pretrained=False)
    m.eval()
    return m


@pytest.fixture(scope="module")
def dummy_input() -> torch.Tensor:
    return torch.randn(1, 3, 384, 384)


# --- build_model ---

def test_build_model_returns_nn_module():
    assert isinstance(build_model(num_classes=8, pretrained=False), nn.Module)


def test_build_model_is_skin_lesion_classifier():
    assert isinstance(build_model(num_classes=8, pretrained=False), SkinLesionClassifier)


# --- output shape ---

def test_output_shape_single(model, dummy_input):
    with torch.no_grad():
        out = model(dummy_input)
    assert out.shape == (1, 8)


def test_output_shape_batch(model):
    x = torch.randn(4, 3, 384, 384)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (4, 8)


def test_custom_num_classes_output():
    m = build_model(num_classes=3, pretrained=False)
    m.eval()
    with torch.no_grad():
        out = m(torch.randn(1, 3, 384, 384))
    assert out.shape == (1, 3)


# --- correctness ---

def test_output_no_nan(model, dummy_input):
    with torch.no_grad():
        out = model(dummy_input)
    assert not torch.isnan(out).any()


def test_output_no_inf(model, dummy_input):
    with torch.no_grad():
        out = model(dummy_input)
    assert not torch.isinf(out).any()


# --- head structure ---

def test_head_is_sequential():
    m = SkinLesionClassifier(num_classes=8, pretrained=False)
    assert isinstance(m.head, nn.Sequential)


def test_head_first_layer_batchnorm():
    m = SkinLesionClassifier(num_classes=8, pretrained=False)
    assert isinstance(list(m.head.children())[0], nn.BatchNorm1d)


def test_head_first_linear_dims():
    m = SkinLesionClassifier(num_classes=8, pretrained=False)
    layers = [layer for layer in m.head.children() if isinstance(layer, nn.Linear)]
    assert layers[0].in_features == 1792
    assert layers[0].out_features == 512


def test_head_last_linear_out_features():
    m = SkinLesionClassifier(num_classes=8, pretrained=False)
    layers = [layer for layer in m.head.children() if isinstance(layer, nn.Linear)]
    assert layers[-1].out_features == 8


def test_head_last_linear_out_features_custom():
    m = SkinLesionClassifier(num_classes=5, pretrained=False)
    layers = [layer for layer in m.head.children() if isinstance(layer, nn.Linear)]
    assert layers[-1].out_features == 5


# --- backbone ---

def test_backbone_feature_dim():
    # EfficientNet-B4 always produces 1792-dimensional features
    m = SkinLesionClassifier(num_classes=8, pretrained=False)
    assert m.backbone.num_features == 1792


# --- eval mode ---

def test_eval_mode_no_gradient(model, dummy_input):
    with torch.no_grad():
        out = model(dummy_input)
    assert out.requires_grad is False
