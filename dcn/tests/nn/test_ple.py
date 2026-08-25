import torch
import polars as pl

from dcn.nn.ple import PiecewiseLinearEncoder


def test_piecewise_linear_encoder():
    df = pl.DataFrame(
        {
            "f1": [0.0, 1.0, 2.0, 3.0, 4.0],
            "f2": [10.0, 10.0, 10.0, 20.0, 20.0],
        }
    )

    encoder = PiecewiseLinearEncoder.from_dataset(df, n_bins=4)

    assert isinstance(encoder.n_bins, list)
    assert len(encoder.n_bins) == 2
    assert all(1 <= x <= 4 for x in encoder.n_bins)

    x = df.to_torch().to(torch.float32)
    out = encoder(x)

    assert out.shape[0] == len(df)
    assert out.ndim == 2
    assert out.dtype == torch.float32
    assert out.shape[1] == sum(encoder.n_bins)

    bins = PiecewiseLinearEncoder.compute_bins(x, n_bins=4)
    assert isinstance(bins, list)
    assert len(bins) == x.shape[1]
    assert all(isinstance(b, torch.Tensor) for b in bins)
    assert all(len(b) >= 2 for b in bins)

    df_single_bin = pl.DataFrame(
        {
            "f1": [0.0, 0.0, 1.0, 1.0],
            "f2": [5.0, 6.0, 7.0, 8.0],
        }
    )
    encoder_single_bin = PiecewiseLinearEncoder.from_dataset(df_single_bin, n_bins=4)
    x_single_bin = df_single_bin.to_torch().to(torch.float32)
    out_single_bin = encoder_single_bin(x_single_bin)

    assert out_single_bin.shape[0] == len(df_single_bin)
    assert out_single_bin.ndim == 2
    assert out_single_bin.dtype == torch.float32
    assert out_single_bin.shape[1] == sum(encoder_single_bin.n_bins)


def test_a_constant_feature_gets_a_constant_column():
    df = pl.DataFrame(
        {
            "varying": [0.0, 1.0, 2.0, 3.0],
            "constant": [7.0, 7.0, 7.0, 7.0],
        }
    )

    encoder = PiecewiseLinearEncoder.from_dataset(df, n_bins=4)
    out = encoder(df.to_torch().to(torch.float32))

    assert encoder.n_bins[-1] == 1
    assert out.shape == (len(df), sum(encoder.n_bins))
    assert torch.all(out[:, -1] == out[0, -1])
