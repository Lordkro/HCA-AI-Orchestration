"""Tests for configuration constants — HARDWARE_TIERS and HARDWARE_BACKENDS."""

from hca.core.config import HARDWARE_BACKENDS, HARDWARE_TIERS


class TestHardwareTiers:
    def test_expected_tiers_present(self) -> None:
        expected = {"high", "medium", "low", "minimal", "tiny"}
        assert set(HARDWARE_TIERS) == expected

    def test_each_tier_has_required_keys(self) -> None:
        for name, tier in HARDWARE_TIERS.items():
            assert "vram" in tier, f"{name} missing vram"
            assert "default_model" in tier, f"{name} missing default_model"
            assert "coder_model" in tier, f"{name} missing coder_model"
            assert "num_ctx" in tier, f"{name} missing num_ctx"
            assert isinstance(tier["vram"], str)
            assert isinstance(tier["num_ctx"], int)


class TestHardwareBackends:
    def test_expected_backends_present(self) -> None:
        expected = {"cpu", "nvidia", "rocm", "metal"}
        assert set(HARDWARE_BACKENDS) == expected

    def test_each_backend_has_required_keys(self) -> None:
        for name, backend in HARDWARE_BACKENDS.items():
            assert "label" in backend, f"{name} missing label"
            assert "image" in backend, f"{name} missing image"
            assert "tag" in backend, f"{name} missing tag"
            assert "compose_profile" in backend, f"{name} missing compose_profile"
            assert "note" in backend, f"{name} missing note"

    def test_metal_backend_values(self) -> None:
        metal = HARDWARE_BACKENDS["metal"]
        assert metal["label"] == "Apple Metal"
        assert metal["compose_profile"] == "metal"
        assert metal["image"] == "ollama/ollama"

    def test_compose_profile_matches_key(self) -> None:
        for name, backend in HARDWARE_BACKENDS.items():
            assert backend["compose_profile"] == name, (
                f"{name} compose_profile should match its key"
            )
