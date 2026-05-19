import torch

from models.embeddings import DiffusionStepEmbedding, SinusoidalTimeEmbedding


def test_sinusoidal_output_shape():
    emb = SinusoidalTimeEmbedding(L=64, d_model=64)
    t = torch.rand(4, 72)
    out = emb(t)
    assert out.shape == (4, 72, 64)


def test_sinusoidal_no_nan():
    emb = SinusoidalTimeEmbedding(L=64, d_model=64)
    t = torch.rand(4, 72)
    out = emb(t)
    assert not torch.isnan(out).any()


def test_sinusoidal_different_times_differ():
    emb = SinusoidalTimeEmbedding(L=64, d_model=64)
    t1 = torch.zeros(1, 72)
    t2 = torch.ones(1, 72)
    assert not torch.allclose(emb(t1), emb(t2))


def test_step_embed_output_shape():
    emb = DiffusionStepEmbedding(T_diff=1000, L=64, d_model=64)
    s = torch.randint(0, 1000, (4,))
    out = emb(s)
    assert out.shape == (4, 1, 64)


def test_step_embed_no_nan():
    emb = DiffusionStepEmbedding(T_diff=1000, L=64, d_model=64)
    s = torch.randint(0, 1000, (4,))
    assert not torch.isnan(emb(s)).any()
