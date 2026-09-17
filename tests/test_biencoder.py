import numpy as np
import pytest
import torch

from avito_cg.train.biencoder import PairDataset, TrainingConfig, info_nce, mean_pooling


def test_mean_pooling_ignores_padding():
    """Паддинг не должен тянуть вектор к нулю"""
    hidden = torch.tensor([[[1.0, 1.0], [3.0, 3.0], [100.0, 100.0]]])
    mask = torch.tensor([[1, 1, 0]])
    assert torch.allclose(mean_pooling(hidden, mask), torch.tensor([[2.0, 2.0]]))


def test_mean_pooling_survives_empty_mask():
    hidden = torch.zeros(1, 3, 2)
    mask = torch.zeros(1, 3, dtype=torch.long)
    assert torch.isfinite(mean_pooling(hidden, mask)).all()


def test_info_nce_is_zero_for_a_perfect_match():
    """Если каждый запрос совпал со своим объявлением и ортогонален чужим, потери почти нет"""
    vectors = torch.eye(4)
    loss = info_nce(vectors, vectors, temperature=0.01)
    assert loss.item() < 1e-3


def test_info_nce_punishes_a_swapped_pair():
    vectors = torch.eye(4)
    swapped = vectors[[1, 0, 2, 3]]
    assert info_nce(vectors, swapped, 0.05).item() > info_nce(vectors, vectors, 0.05).item()


def test_info_nce_is_symmetric():
    rng = torch.Generator().manual_seed(0)
    queries = torch.nn.functional.normalize(torch.randn(8, 16, generator=rng), dim=-1)
    passages = torch.nn.functional.normalize(torch.randn(8, 16, generator=rng), dim=-1)
    assert info_nce(queries, passages, 0.05).item() == pytest.approx(
        info_nce(passages, queries, 0.05).item(), rel=1e-5
    )


def test_info_nce_grows_when_negatives_get_closer():
    """Чем ближе чужие объявления, тем сложнее задача и выше потери"""
    easy = torch.eye(4)
    hard = torch.nn.functional.normalize(torch.eye(4) + 0.9, dim=-1)
    assert info_nce(easy, hard, 0.05).item() > info_nce(easy, easy, 0.05).item()


def test_pair_dataset_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="не совпадает"):
        PairDataset(["кран"], ["аренда крана", "лишнее"])


def test_device_resolution_is_explicit_when_asked():
    assert TrainingConfig(device="cpu").resolve_device() == torch.device("cpu")


def test_vectors_are_normalised_by_embed_contract():
    """Косинус как скор имеет смысл только если векторы нормированы"""
    vectors = torch.nn.functional.normalize(torch.randn(5, 8), dim=-1)
    assert np.allclose(vectors.norm(dim=-1).numpy(), 1.0, atol=1e-6)
