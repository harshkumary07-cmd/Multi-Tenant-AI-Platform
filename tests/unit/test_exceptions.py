"""
Unit tests for app/models/exceptions.py -- Module 2.

Tests cover:
    - All exceptions inherit from PlatformError
    - PlatformError inherits from Exception
    - Each exception carries the correct error_code
    - Exception messages are stored and accessible
    - __str__ format is consistent
"""

import pytest

from app.models.exceptions import (
    CacheError,
    ConfigurationError,
    CorruptFileError,
    CSVParseError,
    EmbeddingFailedError,
    EmptyDocumentError,
    FileTooLargeError,
    InvalidFileTypeError,
    LLMProviderError,
    LLMTimeoutError,
    NoRelevantChunksError,
    PlatformError,
    UnauthorizedError,
    UserAlreadyExistsError,
    UserNotFoundError,
    VectorStoreError,
)


class TestPlatformErrorBase:
    """PlatformError is the base for all domain exceptions."""

    def test_platform_error_is_exception(self) -> None:
        """PlatformError inherits from Exception."""
        err = PlatformError("something went wrong")
        assert isinstance(err, Exception)

    def test_platform_error_stores_message(self) -> None:
        """The message attribute is set on instantiation."""
        err = PlatformError("test message")
        assert err.message == "test message"

    def test_platform_error_str_format(self) -> None:
        """__str__ includes error_code and message."""
        err = PlatformError("test message")
        result = str(err)
        assert "PLATFORM_ERROR" in result
        assert "test message" in result

    def test_platform_error_can_be_raised_and_caught(self) -> None:
        """PlatformError can be raised and caught normally."""
        with pytest.raises(PlatformError):
            raise PlatformError("raised")


class TestAllExceptionsInheritFromPlatformError:
    """Every domain exception is a PlatformError."""

    @pytest.mark.parametrize(
        "exception_class",
        [
            ConfigurationError,
            VectorStoreError,
            InvalidFileTypeError,
            FileTooLargeError,
            CorruptFileError,
            CSVParseError,
            EmptyDocumentError,
            EmbeddingFailedError,
            LLMTimeoutError,
            LLMProviderError,
            NoRelevantChunksError,
            CacheError,
            UnauthorizedError,
            UserAlreadyExistsError,
            UserNotFoundError,
        ],
    )
    def test_inherits_from_platform_error(
        self, exception_class: type[PlatformError]
    ) -> None:
        """Each exception class is a subclass of PlatformError."""
        err = exception_class("test")
        assert isinstance(err, PlatformError)
        assert isinstance(err, Exception)

    @pytest.mark.parametrize(
        "exception_class",
        [
            ConfigurationError,
            VectorStoreError,
            InvalidFileTypeError,
            FileTooLargeError,
            CorruptFileError,
            CSVParseError,
            EmptyDocumentError,
            EmbeddingFailedError,
            LLMTimeoutError,
            LLMProviderError,
            NoRelevantChunksError,
            CacheError,
            UnauthorizedError,
            UserAlreadyExistsError,
            UserNotFoundError,
        ],
    )
    def test_can_be_caught_as_platform_error(
        self, exception_class: type[PlatformError]
    ) -> None:
        """Every specific exception can be caught by catching PlatformError."""
        with pytest.raises(PlatformError):
            raise exception_class("test")


class TestErrorCodes:
    """Each exception has a unique, correctly named error_code."""

    @pytest.mark.parametrize(
        ("exception_class", "expected_code"),
        [
            (ConfigurationError, "CONFIGURATION_ERROR"),
            (VectorStoreError, "VECTOR_STORE_UNAVAILABLE"),
            (InvalidFileTypeError, "INVALID_FILE_TYPE"),
            (FileTooLargeError, "FILE_TOO_LARGE"),
            (CorruptFileError, "CORRUPT_FILE"),
            (CSVParseError, "CSV_PARSE_ERROR"),
            (EmptyDocumentError, "EMPTY_DOCUMENT"),
            (EmbeddingFailedError, "EMBEDDING_FAILED"),
            (LLMTimeoutError, "LLM_TIMEOUT"),
            (LLMProviderError, "LLM_PROVIDER_ERROR"),
            (NoRelevantChunksError, "NO_RELEVANT_CHUNKS"),
            (CacheError, "CACHE_ERROR"),
            (UnauthorizedError, "UNAUTHORIZED"),
            (UserAlreadyExistsError, "USER_ALREADY_EXISTS"),
            (UserNotFoundError, "USER_NOT_FOUND"),
        ],
    )
    def test_error_code_is_correct(
        self, exception_class: type[PlatformError], expected_code: str
    ) -> None:
        """Each exception class reports its own error_code correctly."""
        err = exception_class("test message")
        assert err.error_code == expected_code

    @pytest.mark.parametrize(
        ("exception_class", "expected_code"),
        [
            (ConfigurationError, "CONFIGURATION_ERROR"),
            (VectorStoreError, "VECTOR_STORE_UNAVAILABLE"),
            (InvalidFileTypeError, "INVALID_FILE_TYPE"),
        ],
    )
    def test_error_code_appears_in_str(
        self, exception_class: type[PlatformError], expected_code: str
    ) -> None:
        """The error_code is included in the string representation."""
        err = exception_class("something failed")
        assert expected_code in str(err)


class TestConfigurationError:
    """ConfigurationError is raised during startup validation."""

    def test_carries_full_message(self) -> None:
        """The message passed to ConfigurationError is stored verbatim."""
        msg = "LLM_API_KEY is set to placeholder value 'changeme'"
        err = ConfigurationError(msg)
        assert err.message == msg

    def test_error_code_is_configuration_error(self) -> None:
        """ConfigurationError.error_code is 'CONFIGURATION_ERROR'."""
        err = ConfigurationError("any message")
        assert err.error_code == "CONFIGURATION_ERROR"

    def test_is_raised_correctly(self) -> None:
        """ConfigurationError can be raised and caught by its own type."""
        with pytest.raises(ConfigurationError) as exc_info:
            raise ConfigurationError("startup failed")
        assert "startup failed" in str(exc_info.value)

    def test_is_also_caught_as_platform_error(self) -> None:
        """ConfigurationError is also caught by PlatformError handlers."""
        with pytest.raises(PlatformError):
            raise ConfigurationError("caught as parent")
