from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Query, Request, Response

from app.api.dependencies import DiagnosisServiceDependency
from app.api.schemas.diagnosis import DiagnosisListResponse, DiagnosisRequest, DiagnosisResponse
from app.api.schemas.errors import ErrorResponse
from app.models.execution import DiagnosisExecution, ExecutionFilter

router = APIRouter(prefix="/api/v1", tags=["diagnosis"])


@router.post(
    "/diagnosis",
    response_model=DiagnosisResponse,
    responses={
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        502: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
        504: {"model": ErrorResponse},
    },
)
async def diagnose(
    body: DiagnosisRequest,
    service: DiagnosisServiceDependency,
    request: Request,
    response: Response,
) -> DiagnosisResponse:
    def created(diagnosis_id: UUID) -> None:
        request.state.diagnosis_id = diagnosis_id

    result = await service.diagnose(body, on_created=created)
    response.headers["Location"] = f"/api/v1/diagnoses/{result.diagnosis_id}"
    return result


@router.get(
    "/diagnoses/{diagnosis_id}",
    response_model=DiagnosisExecution,
    responses={404: {"model": ErrorResponse}},
)
def get_diagnosis(diagnosis_id: UUID, service: DiagnosisServiceDependency) -> DiagnosisExecution:
    return service.get(diagnosis_id)


@router.get("/diagnoses", response_model=DiagnosisListResponse)
def list_diagnoses(
    service: DiagnosisServiceDependency,
    filters: Annotated[ExecutionFilter, Query()],
) -> DiagnosisListResponse:
    return DiagnosisListResponse.from_page(service.list(filters))
