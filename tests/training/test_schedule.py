import torch

from training.schedule import DDPMSchedule


def test_alpha_bars_shape():
    schedule = DDPMSchedule(T=1000)
    assert schedule.alpha_bars.shape == (1000,)


def test_alpha_bars_decreasing():
    schedule = DDPMSchedule(T=1000)
    assert (schedule.alpha_bars[1:] <= schedule.alpha_bars[:-1]).all()


def test_q_sample_shapes():
    schedule = DDPMSchedule(T=1000)
    x0 = torch.randn(4, 72, 39)
    dm = torch.ones(4, 72, 39)
    s = torch.randint(0, 1000, (4,))
    x_noisy, eps = schedule.q_sample(x0, s, dm)
    assert x_noisy.shape == x0.shape
    assert eps.shape == x0.shape


def test_unobserved_positions_stay_zero():
    schedule = DDPMSchedule(T=1000)
    x0 = torch.randn(2, 72, 39)
    dm = torch.zeros(2, 72, 39)
    dm[:, :10, :] = 1.0  # only first 10 real
    x0 = x0 * dm
    s = torch.tensor([100, 500])
    x_noisy, _ = schedule.q_sample(x0, s, dm)
    assert x_noisy[:, 10:, :].abs().sum().item() == 0.0


def test_high_noise_step_corrupts():
    schedule = DDPMSchedule(T=1000)
    torch.manual_seed(0)
    x0 = torch.ones(1, 10, 5)
    dm = torch.ones(1, 10, 5)
    s = torch.tensor([999])  # max noise
    x_noisy, _ = schedule.q_sample(x0, s, dm)
    assert not torch.allclose(x_noisy, x0)
