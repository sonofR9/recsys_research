from pathlib import Path


class GlobalConfig:
    """Global configuration for the project.

    User must call initialize() with base_path before using.
    """

    def __init__(self):
        self._base_path: Path | None = None
        self._cpu_attention: bool = False

    def initialize(self, base_path: Path) -> None:
        self._base_path = base_path

    def set_cpu_attention(self, enabled: bool) -> None:
        self._cpu_attention = enabled

    @property
    def cpu_attention(self) -> bool:
        return self._cpu_attention

    @property
    def base_path(self) -> Path:
        if self._base_path is None:
            raise RuntimeError(
                "Global config not initialized. Call config.initialize(base_path) first."
            )
        return self._base_path

    @property
    def counters_path(self) -> Path:
        return self.base_path / "counters"

    @property
    def candgen_path(self) -> Path:
        return self.base_path / "candgen"

    @property
    def preprocessed_path(self) -> Path:
        return self.base_path / "preprocessed"

    @property
    def dataset_path(self) -> Path:
        return self.preprocessed_path / "dataset"

    @property
    def splitted_path(self) -> Path:
        return self.dataset_path / "splitted"

    @property
    def tmp_path(self) -> Path:
        return self.base_path / "tmp"

    @property
    def checkpoints_path(self) -> Path:
        return self.base_path / "checkpoints"

    @property
    def logs_path(self) -> Path:
        return self.base_path / "logs"

    @property
    def predictions_path(self) -> Path:
        return self.base_path / "predictions"

    def datasets_path(self, dataset_name: str) -> Path:
        return self.base_path / "datasets" / dataset_name


# Global config instance - must be initialized before use
config = GlobalConfig()
