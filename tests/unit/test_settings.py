"""
Unit tests for app/config/settings.py -- Module 2.

Tests cover:
    - Settings loads with defaults (no .env required)
    - Type validation rejects wrong types
    - LLM_API_KEY masked by SecretStr (never appears as plain text)
    - validate_startup_config passes on valid configuration
    - validate_startup_config catches every semantic error condition
    - Production-only rules are not enforced in development
    - get_settings_summary never exposes the real API key value
    - lru_cache behaviour and cache_clear() for test isolation

These are unit tests -- no infrastructure (ChromaDB, Redis) required.
All tests run without a .env file by passing fields directly to Settings().
"""

import pytest
from pydantic import SecretStr, ValidationError

from app.config.settings import (
    Settings,
    get_settings,
    get_settings_summary,
    validate_startup_config,
)
from app.models.exceptions import ConfigurationError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_settings(**overrides: object) -> Settings:
    """
    Build a Settings instance with safe test defaults.

    Accepts keyword overrides for any field. All other fields use
    their production-safe defaults. This avoids repeating boilerplate
    in every test.

    The LLM_API_KEY default here is "test-key" so production rules
    (which check for "changeme") pass by default unless a test
    specifically overrides this.
    """
    defaults: dict[str, object] = {
        "APP_ENV": "development",
        "LLM_API_KEY": SecretStr("test-key"),
    }
    defaults.update(overrides)
    return Settings(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 1. Settings loading
# ---------------------------------------------------------------------------


class TestSettingsLoading:
    """Settings can be instantiated with defaults."""

    def test_loads_with_all_defaults(self) -> None:
        """Settings() can be instantiated without any env vars set."""
        s = Settings()  # type: ignore[call-arg]
        assert s.APP_ENV == "development"
        assert s.APP_PORT == 8000
        assert s.CHROMA_HOST == "localhost"
        assert s.CHROMA_PORT == 8001
        assert s.CHROMA_COLLECTION_NAME == "documents"
        assert s.REDIS_HOST == "localhost"
        assert s.REDIS_PORT == 6379
        assert s.EMBEDDING_MODEL_NAME == "all-MiniLM-L6-v2"
        assert s.CHUNK_SIZE_TOKENS == 512
        assert s.CHUNK_OVERLAP_TOKENS == 50
        assert s.RETRIEVAL_TOP_K == 5
        assert s.RETRIEVAL_CONFIDENCE_THRESHOLD == 0.35
        assert s.LLM_PROVIDER == "openai"
        assert s.LLM_MODEL_NAME == "gpt-4o"
        assert s.LLM_TIMEOUT_SECONDS == 30
        assert s.MAX_UPLOAD_SIZE_MB == 50
        assert s.UPLOAD_TIMEOUT_SECONDS == 120

    def test_llm_api_key_default_is_changeme(self) -> None:
        """LLM_API_KEY defaults to 'changeme' so the app starts without config."""
        s = Settings()  # type: ignore[call-arg]
        assert s.LLM_API_KEY.get_secret_value() == "changeme"

    def test_app_env_accepted_values(self) -> None:
        """APP_ENV accepts exactly development, staging, production."""
        for env in ("development", "staging", "production"):
            s = make_settings(APP_ENV=env)
            assert s.APP_ENV == env

    def test_app_env_rejects_invalid_value(self) -> None:
        """APP_ENV rejects values outside the allowed Literal."""
        with pytest.raises(ValidationError) as exc_info:
            make_settings(APP_ENV="test")
        assert "APP_ENV" in str(exc_info.value)

    def test_log_level_rejects_invalid_value(self) -> None:
        """LOG_LEVEL rejects arbitrary strings."""
        with pytest.raises(ValidationError):
            make_settings(LOG_LEVEL="VERBOSE")

    def test_app_port_must_be_integer(self) -> None:
        """APP_PORT rejects non-integer values."""
        with pytest.raises(ValidationError):
            make_settings(APP_PORT="not-a-port")

    def test_retrieval_threshold_must_be_float(self) -> None:
        """RETRIEVAL_CONFIDENCE_THRESHOLD rejects non-numeric values."""
        with pytest.raises(ValidationError):
            make_settings(RETRIEVAL_CONFIDENCE_THRESHOLD="high")


# ---------------------------------------------------------------------------
# 2. SecretStr masking -- security-critical tests
# ---------------------------------------------------------------------------


class TestSecretStrMasking:
    """LLM_API_KEY is never exposed as plain text."""

    def test_str_representation_is_masked(self) -> None:
        """str(settings.LLM_API_KEY) returns masked value, not the real key."""
        s = make_settings(LLM_API_KEY=SecretStr("sk-real-secret-key-abc123"))
        masked = str(s.LLM_API_KEY)
        assert masked == "**********"
        assert "sk-real-secret-key-abc123" not in masked

    def test_repr_is_masked(self) -> None:
        """repr(settings.LLM_API_KEY) does not expose the real key."""
        s = make_settings(LLM_API_KEY=SecretStr("sk-real-secret-key-abc123"))
        r = repr(s.LLM_API_KEY)
        assert "sk-real-secret-key-abc123" not in r

    def test_model_dump_masks_secret(self) -> None:
        """settings.model_dump() serialises LLM_API_KEY as masked value."""
        s = make_settings(LLM_API_KEY=SecretStr("sk-real-secret-key-abc123"))
        dumped = s.model_dump()
        # pydantic v2 serialises SecretStr as the masked representation
        api_key_value = dumped["LLM_API_KEY"]
        assert str(api_key_value) == "**********"
        assert "sk-real-secret-key-abc123" not in str(api_key_value)

    def test_get_secret_value_returns_real_key(self) -> None:
        """get_secret_value() is the only way to access the real key."""
        real_key = "sk-real-secret-key-abc123"
        s = make_settings(LLM_API_KEY=SecretStr(real_key))
        assert s.LLM_API_KEY.get_secret_value() == real_key

    def test_settings_summary_masks_api_key(self) -> None:
        """get_settings_summary() never exposes the real API key."""
        s = make_settings(LLM_API_KEY=SecretStr("sk-real-secret-key-abc123"))
        summary = get_settings_summary(s)
        assert summary["LLM_API_KEY"] == "**********"
        assert "sk-real-secret-key-abc123" not in str(summary)


# ---------------------------------------------------------------------------
# 3. validate_startup_config -- universal rules (all environments)
# ---------------------------------------------------------------------------


class TestStartupValidationUniversalRules:
    """Rules that apply in all environments."""

    def test_valid_config_passes(self) -> None:
        """A well-formed development config raises no errors."""
        s = make_settings()
        validate_startup_config(s)  # must not raise

    def test_chunk_overlap_equal_to_chunk_size_raises(self) -> None:
        """Overlap == chunk size would produce infinite/empty chunks."""
        s = make_settings(CHUNK_SIZE_TOKENS=512, CHUNK_OVERLAP_TOKENS=512)
        with pytest.raises(ConfigurationError) as exc_info:
            validate_startup_config(s)
        assert "CHUNK_OVERLAP_TOKENS" in str(exc_info.value)
        assert "CHUNK_SIZE_TOKENS" in str(exc_info.value)

    def test_chunk_overlap_greater_than_chunk_size_raises(self) -> None:
        """Overlap > chunk size is also invalid."""
        s = make_settings(CHUNK_SIZE_TOKENS=256, CHUNK_OVERLAP_TOKENS=300)
        with pytest.raises(ConfigurationError):
            validate_startup_config(s)

    def test_chunk_overlap_less_than_chunk_size_passes(self) -> None:
        """Overlap strictly less than chunk size is valid."""
        s = make_settings(CHUNK_SIZE_TOKENS=512, CHUNK_OVERLAP_TOKENS=50)
        validate_startup_config(s)  # must not raise

    def test_confidence_threshold_of_zero_raises(self) -> None:
        """Threshold of 0.0 accepts all chunks regardless of relevance."""
        s = make_settings(RETRIEVAL_CONFIDENCE_THRESHOLD=0.0)
        with pytest.raises(ConfigurationError) as exc_info:
            validate_startup_config(s)
        assert "RETRIEVAL_CONFIDENCE_THRESHOLD" in str(exc_info.value)

    def test_confidence_threshold_of_one_raises(self) -> None:
        """Threshold of 1.0 matches nothing in practice."""
        s = make_settings(RETRIEVAL_CONFIDENCE_THRESHOLD=1.0)
        with pytest.raises(ConfigurationError):
            validate_startup_config(s)

    def test_confidence_threshold_valid_range_passes(self) -> None:
        """Thresholds in (0.0, 1.0) are valid."""
        for value in (0.01, 0.35, 0.5, 0.99):
            s = make_settings(RETRIEVAL_CONFIDENCE_THRESHOLD=value)
            validate_startup_config(s)  # must not raise

    def test_retrieval_top_k_zero_raises(self) -> None:
        """top_k of 0 would return no results on every query."""
        s = make_settings(RETRIEVAL_TOP_K=0)
        with pytest.raises(ConfigurationError) as exc_info:
            validate_startup_config(s)
        assert "RETRIEVAL_TOP_K" in str(exc_info.value)

    def test_retrieval_top_k_above_20_raises(self) -> None:
        """top_k above 20 produces context windows too large for most models."""
        s = make_settings(RETRIEVAL_TOP_K=21)
        with pytest.raises(ConfigurationError):
            validate_startup_config(s)

    def test_retrieval_top_k_valid_range_passes(self) -> None:
        """top_k values 1-20 are all valid."""
        for value in (1, 5, 10, 20):
            s = make_settings(RETRIEVAL_TOP_K=value)
            validate_startup_config(s)  # must not raise

    def test_embedding_batch_size_zero_raises(self) -> None:
        """Batch size of 0 means nothing is embedded per call."""
        s = make_settings(EMBEDDING_BATCH_SIZE=0)
        with pytest.raises(ConfigurationError) as exc_info:
            validate_startup_config(s)
        assert "EMBEDDING_BATCH_SIZE" in str(exc_info.value)

    def test_llm_timeout_zero_raises(self) -> None:
        """A timeout of 0 seconds means every LLM call fails immediately."""
        s = make_settings(LLM_TIMEOUT_SECONDS=0)
        with pytest.raises(ConfigurationError) as exc_info:
            validate_startup_config(s)
        assert "LLM_TIMEOUT_SECONDS" in str(exc_info.value)

    def test_max_upload_size_zero_raises(self) -> None:
        """A max upload size of 0MB rejects every file."""
        s = make_settings(MAX_UPLOAD_SIZE_MB=0)
        with pytest.raises(ConfigurationError) as exc_info:
            validate_startup_config(s)
        assert "MAX_UPLOAD_SIZE_MB" in str(exc_info.value)

    def test_upload_timeout_zero_raises(self) -> None:
        """An upload timeout of 0 seconds means every upload times out."""
        s = make_settings(UPLOAD_TIMEOUT_SECONDS=0)
        with pytest.raises(ConfigurationError) as exc_info:
            validate_startup_config(s)
        assert "UPLOAD_TIMEOUT_SECONDS" in str(exc_info.value)

    def test_multiple_errors_reported_together(self) -> None:
        """All validation errors are collected and reported in one exception."""
        s = make_settings(
            CHUNK_SIZE_TOKENS=100,
            CHUNK_OVERLAP_TOKENS=100,  # violates rule 1
            RETRIEVAL_CONFIDENCE_THRESHOLD=0.0,  # violates rule 2
            RETRIEVAL_TOP_K=0,  # violates rule 3
        )
        with pytest.raises(ConfigurationError) as exc_info:
            validate_startup_config(s)
        error_message = str(exc_info.value)
        # All three violations must appear in the single error message
        assert "CHUNK_OVERLAP_TOKENS" in error_message
        assert "RETRIEVAL_CONFIDENCE_THRESHOLD" in error_message
        assert "RETRIEVAL_TOP_K" in error_message
        assert "3 configuration error(s)" in error_message


# ---------------------------------------------------------------------------
# 4. validate_startup_config -- production-only rules
# ---------------------------------------------------------------------------


class TestStartupValidationProductionRules:
    """Rules that are only enforced when APP_ENV == 'production'."""

    def test_changeme_key_raises_in_production(self) -> None:
        """LLM_API_KEY='changeme' must be caught in production."""
        s = make_settings(
            APP_ENV="production",
            LLM_API_KEY=SecretStr("changeme"),
            LOG_LEVEL="INFO",
        )
        with pytest.raises(ConfigurationError) as exc_info:
            validate_startup_config(s)
        error_message = str(exc_info.value)
        assert "LLM_API_KEY" in error_message
        assert "changeme" in error_message

    def test_changeme_key_allowed_in_development(self) -> None:
        """LLM_API_KEY='changeme' is acceptable in development."""
        s = make_settings(
            APP_ENV="development",
            LLM_API_KEY=SecretStr("changeme"),
        )
        validate_startup_config(s)  # must not raise

    def test_changeme_key_allowed_in_staging(self) -> None:
        """LLM_API_KEY='changeme' is acceptable in staging."""
        s = make_settings(
            APP_ENV="staging",
            LLM_API_KEY=SecretStr("changeme"),
        )
        validate_startup_config(s)  # must not raise

    def test_real_key_passes_in_production(self) -> None:
        """A real API key passes production validation."""
        s = make_settings(
            APP_ENV="production",
            LLM_API_KEY=SecretStr("sk-real-key-abc123"),
            LOG_LEVEL="INFO",
        )
        validate_startup_config(s)  # must not raise

    def test_debug_log_level_raises_in_production(self) -> None:
        """LOG_LEVEL=DEBUG is not acceptable in production."""
        s = make_settings(
            APP_ENV="production",
            LLM_API_KEY=SecretStr("sk-real-key-abc123"),
            LOG_LEVEL="DEBUG",
        )
        with pytest.raises(ConfigurationError) as exc_info:
            validate_startup_config(s)
        assert "LOG_LEVEL" in str(exc_info.value)
        assert "DEBUG" in str(exc_info.value)

    def test_debug_log_level_allowed_in_development(self) -> None:
        """LOG_LEVEL=DEBUG is fine in development."""
        s = make_settings(APP_ENV="development", LOG_LEVEL="DEBUG")
        validate_startup_config(s)  # must not raise

    def test_production_error_message_does_not_expose_key_value(self) -> None:
        """
        Error messages about LLM_API_KEY never include the real key value.

        The placeholder "changeme" is the input and can appear in the message.
        But if someone sets a real key and some other production rule fails,
        the real key must not leak into the error message.
        """
        # A real key + DEBUG log in production triggers the LOG_LEVEL error.
        # The error message should name the field, not show the key value.
        real_key = "sk-very-secret-production-key"
        s = make_settings(
            APP_ENV="production",
            LLM_API_KEY=SecretStr(real_key),
            LOG_LEVEL="DEBUG",
        )
        with pytest.raises(ConfigurationError) as exc_info:
            validate_startup_config(s)
        assert real_key not in str(exc_info.value)


# ---------------------------------------------------------------------------
# 5. get_settings_summary
# ---------------------------------------------------------------------------


class TestGetSettingsSummary:
    """Summary output is safe and complete."""

    def test_summary_contains_all_settings_keys(self) -> None:
        """Summary includes a key for every setting."""
        s = make_settings()
        summary = get_settings_summary(s)
        expected_keys = {
            "APP_ENV", "APP_HOST", "APP_PORT", "LOG_LEVEL",
            "CHROMA_HOST", "CHROMA_PORT", "CHROMA_COLLECTION_NAME",
            "REDIS_HOST", "REDIS_PORT", "REDIS_CACHE_TTL_SECONDS",
            "REDIS_EMPTY_RESULT_TTL_SECONDS", "EMBEDDING_MODEL_NAME",
            "EMBEDDING_BATCH_SIZE", "CHUNK_SIZE_TOKENS", "CHUNK_OVERLAP_TOKENS",
            "RETRIEVAL_TOP_K", "RETRIEVAL_CONFIDENCE_THRESHOLD",
            "LLM_PROVIDER", "LLM_MODEL_NAME", "LLM_API_KEY",
            "LLM_TIMEOUT_SECONDS", "MAX_UPLOAD_SIZE_MB", "UPLOAD_TIMEOUT_SECONDS",
        }
        assert expected_keys == set(summary.keys())

    def test_summary_api_key_is_always_masked(self) -> None:
        """LLM_API_KEY is always '**********' in the summary."""
        for key_value in ("changeme", "sk-real-key-abc123", "sk-ant-xyz"):
            s = make_settings(LLM_API_KEY=SecretStr(key_value))
            summary = get_settings_summary(s)
            assert summary["LLM_API_KEY"] == "**********"
            assert key_value not in str(summary)

    def test_summary_reflects_actual_values(self) -> None:
        """Summary values match the actual settings (for non-secret fields)."""
        s = make_settings(
            APP_ENV="staging",
            CHROMA_HOST="chroma-prod",
            RETRIEVAL_TOP_K=10,
        )
        summary = get_settings_summary(s)
        assert summary["APP_ENV"] == "staging"
        assert summary["CHROMA_HOST"] == "chroma-prod"
        assert summary["RETRIEVAL_TOP_K"] == 10


# ---------------------------------------------------------------------------
# 6. get_settings singleton and cache_clear
# ---------------------------------------------------------------------------


class TestGetSettingsSingleton:
    """get_settings() returns the same instance; cache_clear() resets it."""

    def test_get_settings_returns_settings_instance(self) -> None:
        """get_settings() returns a Settings instance."""
        s = get_settings()
        assert isinstance(s, Settings)

    def test_get_settings_is_cached(self) -> None:
        """Calling get_settings() twice returns the identical object."""
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2

    def test_cache_clear_allows_new_instance(self) -> None:
        """After cache_clear(), get_settings() returns a new instance."""
        s1 = get_settings()
        get_settings.cache_clear()
        s2 = get_settings()
        # Both are valid Settings instances
        assert isinstance(s1, Settings)
        assert isinstance(s2, Settings)
        # They are not the same object (cache was cleared)
        assert s1 is not s2
