import torch

from dcn.semantic import (
    ResidualCodebooks,
    fit_residual_kmeans,
    fit_residual_kmeans_with_diagnostics,
)


def _three_clusters(points_per_cluster: int = 60) -> torch.Tensor:
    generator = torch.Generator().manual_seed(0)
    centers = torch.tensor([[10.0, 0.0], [0.0, 10.0], [-10.0, -10.0]])
    noise = torch.randn(3, points_per_cluster, 2, generator=generator) * 0.05
    return (centers[:, None, :] + noise).reshape(-1, 2)


def _reconstruction_error(
    codebooks: ResidualCodebooks, embeddings: torch.Tensor
) -> float:
    reconstruction = codebooks.decode(codebooks.encode(embeddings))
    return float((embeddings - reconstruction).pow(2).sum())


class TestFitResidualKmeans:
    def test_convergent_fit_stops_before_the_iteration_cap(self) -> None:
        embeddings = _three_clusters()

        fit = fit_residual_kmeans_with_diagnostics(
            embeddings,
            num_levels=2,
            num_codes=3,
            max_iterations=300,
            relative_inertia_tolerance=1e-4,
            assignment_early_stopping=True,
        )

        assert fit.codebooks.num_levels == 2
        assert torch.equal(fit.codes, fit.codebooks.encode(embeddings))
        assert all(level.iterations_run < 300 for level in fit.diagnostics.levels)
        assert all(
            level.stop_reason in {"assignments_stable", "relative_inertia"}
            for level in fit.diagnostics.levels
        )
        assert all(
            level.final_inertia <= level.initial_inertia
            for level in fit.diagnostics.levels
        )

    def test_convergent_fit_reports_the_iteration_cap(self) -> None:
        generator = torch.Generator().manual_seed(9)
        embeddings = torch.randn(200, 8, generator=generator)

        fit = fit_residual_kmeans_with_diagnostics(
            embeddings,
            num_levels=1,
            num_codes=8,
            max_iterations=1,
            relative_inertia_tolerance=0.0,
            assignment_early_stopping=False,
        )

        level = fit.diagnostics.levels[0]
        assert level.iterations_run == 1
        assert level.stop_reason == "max_iterations"

    def test_legacy_wrapper_keeps_fixed_iteration_semantics(self) -> None:
        generator = torch.Generator().manual_seed(11)
        embeddings = torch.randn(120, 6, generator=generator)

        legacy = fit_residual_kmeans(
            embeddings,
            num_levels=2,
            num_codes=7,
            num_iterations=4,
            seed=5,
        )
        explicit = fit_residual_kmeans_with_diagnostics(
            embeddings,
            num_levels=2,
            num_codes=7,
            max_iterations=4,
            relative_inertia_tolerance=None,
            assignment_early_stopping=False,
            seed=5,
        )

        assert torch.equal(legacy.centroids, explicit.codebooks.centroids)
        assert all(
            level.iterations_run == 4 and level.stop_reason == "max_iterations"
            for level in explicit.diagnostics.levels
        )

    def test_relative_inertia_stop_never_accepts_an_increase(self) -> None:
        generator = torch.Generator().manual_seed(12)
        embeddings = torch.randn(200, 8, generator=generator)

        fit = fit_residual_kmeans_with_diagnostics(
            embeddings,
            num_levels=3,
            num_codes=9,
            max_iterations=300,
            relative_inertia_tolerance=1e-4,
            assignment_early_stopping=False,
        )

        for level in fit.diagnostics.levels:
            if level.stop_reason == "relative_inertia":
                assert level.final_relative_inertia_improvement is not None
                assert level.final_relative_inertia_improvement >= 0

    def test_separated_clusters_get_one_code_each(self) -> None:
        embeddings = _three_clusters()

        codebooks = fit_residual_kmeans(embeddings, num_levels=1, num_codes=3)
        codes = codebooks.encode(embeddings)

        assert codes.shape == (len(embeddings), 1)
        by_cluster = codes.reshape(3, -1, 1)
        assert all(len(set(cluster.flatten().tolist())) == 1 for cluster in by_cluster)
        assert len(set(codes.flatten().tolist())) == 3

    def test_more_levels_reconstruct_better(self) -> None:
        generator = torch.Generator().manual_seed(1)
        embeddings = torch.randn(200, 8, generator=generator)

        errors = [
            _reconstruction_error(
                fit_residual_kmeans(embeddings, num_levels=levels, num_codes=4),
                embeddings,
            )
            for levels in (1, 2, 3)
        ]

        assert errors[0] > errors[1] > errors[2]

    def test_codes_stay_inside_the_codebook(self) -> None:
        generator = torch.Generator().manual_seed(2)
        embeddings = torch.randn(50, 4, generator=generator)

        codes = fit_residual_kmeans(embeddings, num_levels=3, num_codes=5).encode(
            embeddings
        )

        assert codes.shape == (50, 3)
        assert codes.min() >= 0 and codes.max() < 5

    def test_the_same_seed_gives_the_same_codebooks(self) -> None:
        generator = torch.Generator().manual_seed(3)
        embeddings = torch.randn(80, 6, generator=generator)

        first = fit_residual_kmeans(embeddings, num_levels=2, num_codes=4, seed=7)
        second = fit_residual_kmeans(embeddings, num_levels=2, num_codes=4, seed=7)

        assert torch.equal(first.centroids, second.centroids)

    def test_fewer_points_than_codes_still_fits(self) -> None:
        embeddings = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])

        codebooks = fit_residual_kmeans(embeddings, num_levels=1, num_codes=8)

        assert codebooks.centroids.shape == (1, 8, 2)
        assert _reconstruction_error(codebooks, embeddings) < 1e-8


class TestResidualCodebooks:
    def test_decode_sums_the_per_level_centroids(self) -> None:
        centroids = torch.tensor(
            [
                [[1.0, 0.0], [2.0, 0.0]],
                [[0.0, 10.0], [0.0, 20.0]],
            ]
        )
        codebooks = ResidualCodebooks(centroids)

        decoded = codebooks.decode(torch.tensor([[0, 1], [1, 0]]))

        assert torch.equal(decoded, torch.tensor([[1.0, 20.0], [2.0, 10.0]]))

    def test_round_trip_through_a_file(self, tmp_path) -> None:
        generator = torch.Generator().manual_seed(4)
        embeddings = torch.randn(40, 5, generator=generator)
        codebooks = fit_residual_kmeans(embeddings, num_levels=2, num_codes=3)

        path = tmp_path / "codebooks.pt"
        codebooks.save(path)

        assert torch.equal(
            ResidualCodebooks.load(path).encode(embeddings),
            codebooks.encode(embeddings),
        )
