import torch

from dcn.semantic import ResidualQuantizer, RqVae, fit_residual_kmeans


def _embeddings(num_rows: int = 128, dim: int = 6) -> torch.Tensor:
    generator = torch.Generator().manual_seed(0)
    return torch.randn(num_rows, dim, generator=generator)


def _rq_vae(dim: int = 6, latent_dim: int = 4) -> RqVae:
    torch.manual_seed(0)
    return RqVae(
        encoder=torch.nn.Linear(dim, latent_dim),
        decoder=torch.nn.Linear(latent_dim, dim),
        quantizer=ResidualQuantizer(num_levels=2, num_codes=4, dim=latent_dim),
    )


class TestResidualQuantizer:
    def test_codes_match_the_codebooks_it_was_initialised_from(self) -> None:
        latents = _embeddings(dim=4)
        codebooks = fit_residual_kmeans(latents, num_levels=2, num_codes=4)
        quantizer = ResidualQuantizer(num_levels=2, num_codes=4, dim=4)

        quantizer.initialize_from(codebooks)

        assert torch.equal(quantizer(latents).codes, codebooks.encode(latents))

    def test_each_level_is_trained_against_what_the_ones_above_it_left(
        self,
    ) -> None:
        quantizer = ResidualQuantizer(num_levels=2, num_codes=1, dim=2)
        with torch.no_grad():
            quantizer.centroids[0, 0] = torch.tensor([1.0, 0.0])
            quantizer.centroids[1, 0] = torch.tensor([0.0, 0.1])
        latents = torch.tensor([[1.2, 0.3]])

        quantizer(latents).codebook_loss.backward()

        def direction(vector: torch.Tensor) -> torch.Tensor:
            return vector / vector.norm()

        # The coarse centroid chases the latent itself...
        assert torch.allclose(
            direction(quantizer.centroids.grad[0, 0]),
            direction(torch.tensor([1.0, 0.0]) - latents[0]),
            atol=1e-5,
        )
        # ...the fine one only what the coarse level missed.
        assert torch.allclose(
            direction(quantizer.centroids.grad[1, 0]),
            direction(
                torch.tensor([0.0, 0.1]) - (latents[0] - torch.tensor([1.0, 0.0]))
            ),
            atol=1e-5,
        )

    def test_gradients_reach_the_input_through_the_quantizer(self) -> None:
        quantizer = ResidualQuantizer(num_levels=2, num_codes=4, dim=4)
        latents = _embeddings(dim=4).requires_grad_()

        quantizer(latents).quantized.sum().backward()

        assert latents.grad is not None and latents.grad.abs().sum() > 0


class TestRqVae:
    def test_training_reduces_the_reconstruction_error(self) -> None:
        embeddings = _embeddings()
        model = _rq_vae()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, fused=True)

        first = float(model(embeddings)["reconstruction_loss"])
        for _ in range(200):
            optimizer.zero_grad()
            model(embeddings)["loss"].backward()
            optimizer.step()
        last = float(model(embeddings)["reconstruction_loss"])

        assert last < first

    def test_codes_stay_inside_the_codebooks(self) -> None:
        codes = _rq_vae().codes(_embeddings())

        assert codes.shape == (128, 2)
        assert codes.min() >= 0 and codes.max() < 4

    def test_a_kmeans_initialised_run_starts_from_the_kmeans_codes(
        self,
    ) -> None:
        embeddings = _embeddings()
        model = _rq_vae()

        model.initialize_codebooks(embeddings)

        latents = model.encoder(embeddings)
        codebooks = model.quantizer.codebooks()
        assert torch.equal(model.codes(embeddings), codebooks.encode(latents))
