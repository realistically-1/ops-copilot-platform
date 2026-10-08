class ApplicationError(Exception):
    code = "application_error"
    status_code = 500
    public_message = "The request could not be completed."

    def __init__(self, message: str | None = None) -> None:
        super().__init__(message or self.public_message)


class ConfigurationError(ApplicationError):
    code = "configuration_error"
    status_code = 503
    public_message = "The diagnosis service is not configured."


class LLMError(ApplicationError):
    code = "llm_error"
    status_code = 502
    public_message = "The model provider could not complete the request."


class LLMTimeoutError(LLMError):
    code = "llm_timeout"
    status_code = 504
    public_message = "The model provider timed out."


class LLMResponseError(LLMError):
    code = "invalid_model_output"
    public_message = "The model returned an invalid structured response."


class AgentLimitError(ApplicationError):
    code = "agent_step_limit"
    status_code = 422
    public_message = "Diagnosis exceeded the configured execution limit."


class LogBackendError(ApplicationError):
    code = "log_backend_error"
    status_code = 502
    public_message = "Log data is temporarily unavailable."


class CatalogError(ApplicationError):
    code = "catalog_error"
    status_code = 502
    public_message = "Service metadata is temporarily unavailable."


class NotFoundError(ApplicationError):
    code = "not_found"
    status_code = 404
    public_message = "The requested resource was not found."


class DomainValidationError(ApplicationError):
    code = "invalid_operation"
    status_code = 422
    public_message = "The operation is invalid for the supplied context."


class ExecutionConflictError(ApplicationError):
    code = "execution_conflict"
    status_code = 409
    public_message = "The execution cannot transition from its current state."


class StoreCapacityError(ApplicationError):
    code = "diagnosis_capacity"
    status_code = 503
    public_message = "The in-memory diagnosis store is full of active executions."
