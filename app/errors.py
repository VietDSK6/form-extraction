class ServiceError(Exception):
    def __init__(self, status_code: int, code: str, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


class OpenAINotConfiguredError(ServiceError):
    def __init__(self) -> None:
        super().__init__(
            status_code=503,
            code="openai_not_configured",
            message="OPENAI_API_KEY is not configured",
        )


class PromptTooLargeError(ServiceError):
    def __init__(self, maximum: int) -> None:
        super().__init__(
            status_code=413,
            code="prompt_too_large",
            message=f"Compact prompt exceeds the configured limit of {maximum} characters",
        )


class UpstreamServiceError(ServiceError):
    pass

