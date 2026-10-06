import json
from types import SimpleNamespace

import pytest
from httpx import ASGITransport, AsyncClient
from pydantic import ValidationError

import app.service as service_module
from app.config import Settings
from app.main import create_app
from app.service import ExtractionService


class FakeCompletions:
    def __init__(self, payload=None, error=None):
        self.payload = payload or {"extractions": []}
        self.error = error
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(
                        content=json.dumps(self.payload, ensure_ascii=False),
                        refusal=None,
                    ),
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=321,
                completion_tokens=45,
                total_tokens=366,
            ),
        )


def build_settings(**overrides):
    values = {
        "app_env": "test",
        "openai_api_key": "test-openai-key",
        "openai_model": "gpt-4o-mini-2024-07-18",
        "internal_api_key": None,
        "allow_public_extraction": False,
        "cors_allowed_origins": "",
    }
    values.update(overrides)
    return Settings(**values)


def build_app(payload=None, settings=None, error=None):
    resolved_settings = settings or build_settings()
    completions = FakeCompletions(payload=payload, error=error)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    service = ExtractionService(resolved_settings, client=client)
    return create_app(resolved_settings, service), completions


async def request(app, method, path, **kwargs):
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        return await client.request(method, path, **kwargs)


def basic_payload():
    return {
        "transcript": "Tôi tên Nguyễn Văn An",
        "fields": [
            {
                "field_code": "hoTen",
                "label": "Họ và tên",
                "type": "text",
                "required": True,
            }
        ],
    }


@pytest.mark.asyncio
async def test_service_starts_without_openai_key_and_reports_not_ready():
    settings = Settings(app_env="development", openai_api_key=None)
    app = create_app(settings)

    live_response = await request(app, "GET", "/health/live")
    ready_response = await request(app, "GET", "/health/ready")
    extract_response = await request(
        app,
        "POST",
        "/api/v1/openai/extract-form",
        json=basic_payload(),
    )

    assert live_response.status_code == 200
    assert live_response.json() == {"status": "ok"}
    assert ready_response.status_code == 503
    assert ready_response.json()["openai_configured"] is False
    assert extract_response.status_code == 503
    assert extract_response.json()["error"]["code"] == "openai_not_configured"
    assert extract_response.headers["X-Request-ID"]


def test_production_requires_internal_api_key():
    with pytest.raises(ValidationError, match="INTERNAL_API_KEY is required"):
        Settings(
            app_env="production",
            openai_api_key="test-openai-key",
            internal_api_key=None,
            allow_public_extraction=False,
        )


@pytest.mark.asyncio
async def test_production_rejects_missing_or_invalid_internal_key():
    settings = build_settings(
        app_env="production",
        internal_api_key="internal-secret",
    )
    app, _ = build_app(settings=settings)

    missing = await request(
        app,
        "POST",
        "/api/v1/openai/extract-form",
        json=basic_payload(),
    )
    invalid = await request(
        app,
        "POST",
        "/api/v1/openai/extract-form",
        json=basic_payload(),
        headers={"X-Internal-API-Key": "wrong"},
    )

    assert missing.status_code == 401
    assert invalid.status_code == 401
    assert missing.json()["error"]["code"] == "unauthorized"


@pytest.mark.asyncio
async def test_public_mode_allows_browser_request_without_internal_key():
    settings = build_settings(
        app_env="production",
        internal_api_key=None,
        allow_public_extraction=True,
        cors_allowed_origins="https://miniapp.example",
    )
    app, _ = build_app(settings=settings)

    preflight = await request(
        app,
        "OPTIONS",
        "/api/v1/openai/extract-form",
        headers={
            "Origin": "https://miniapp.example",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "authorization,content-type",
        },
    )
    response = await request(
        app,
        "POST",
        "/api/v1/openai/extract-form",
        json=basic_payload(),
        headers={"Origin": "https://miniapp.example"},
    )

    assert preflight.status_code == 200
    assert preflight.headers["access-control-allow-origin"] == "https://miniapp.example"
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "https://miniapp.example"


@pytest.mark.asyncio
async def test_extracts_all_supported_types_and_reports_usage():
    transcript = (
        "Tôi tên Nguyễn Văn An. Nội dung đề nghị là sửa đèn đường. "
        "Có 12 hộ, ngày khảo sát 03 tháng 10 năm 2026. "
        "Giới tính nam, chọn ưu tiên cao và khẩn cấp, liên hệ qua điện thoại và trực tiếp, đồng ý. "
        "Danh sách gồm giấy đề nghị và căn cước."
    )
    model_payload = {
        "extractions": [
            {"field_code": "hoTen", "value": "Nguyễn Văn An", "status": "extracted", "evidence": "tên Nguyễn Văn An"},
            {"field_code": "noiDung", "value": "Sửa đèn đường", "status": "extracted", "evidence": "sửa đèn đường"},
            {"field_code": "soHo", "value": 12, "status": "extracted", "evidence": "12 hộ"},
            {"field_code": "ngayKhaoSat", "value": "2026-10-03", "status": "extracted", "evidence": "03 tháng 10 năm 2026"},
            {"field_code": "gioiTinh", "value": "M", "status": "extracted", "evidence": "Giới tính nam"},
            {"field_code": "mucDo", "value": "high", "status": "extracted", "evidence": "ưu tiên cao"},
            {"field_code": "nhan", "value": ["priority", "urgent"], "status": "extracted", "evidence": "ưu tiên cao và khẩn cấp"},
            {"field_code": "kenhLienHe", "value": ["phone", "direct"], "status": "extracted", "evidence": "qua điện thoại và trực tiếp"},
            {"field_code": "xacNhan", "value": True, "status": "extracted", "evidence": "đồng ý"},
            {"field_code": "taiLieu", "value": ["giấy đề nghị", "căn cước"], "status": "extracted", "evidence": "giấy đề nghị và căn cước"},
        ]
    }
    app, completions = build_app(payload=model_payload)
    payload = {
        "transcript": transcript,
        "reference_date": "2026-10-03",
        "current_values": {"ghiChuCu": "Đã có dữ liệu"},
        "fields": [
            {"field_code": "hoTen", "label": "Họ tên", "type": "text", "required": True},
            {"field_code": "noiDung", "label": "Nội dung", "type": "textarea", "required": True},
            {"field_code": "soHo", "label": "Số hộ", "type": "number", "required": True},
            {"field_code": "ngayKhaoSat", "label": "Ngày khảo sát", "type": "date", "required": True},
            {"field_code": "gioiTinh", "label": "Giới tính", "type": "select", "options": [{"value": "M", "label": "Nam"}, {"value": "F", "label": "Nữ"}]},
            {"field_code": "mucDo", "label": "Mức độ", "type": "radio", "options": [{"value": "high", "label": "Cao"}, {"value": "low", "label": "Thấp"}]},
            {"field_code": "nhan", "label": "Nhãn", "type": "multiselect", "options": [{"value": "priority", "label": "Ưu tiên"}, {"value": "urgent", "label": "Khẩn cấp"}]},
            {"field_code": "kenhLienHe", "label": "Kênh liên hệ", "type": "checkbox", "options": [{"value": "phone", "label": "Điện thoại"}, {"value": "direct", "label": "Trực tiếp"}]},
            {"field_code": "xacNhan", "label": "Xác nhận", "type": "switch"},
            {"field_code": "taiLieu", "label": "Tài liệu", "type": "list"},
            {"field_code": "ghiChuCu", "label": "Ghi chú cũ", "type": "text", "required": True},
        ],
    }

    response = await request(
        app,
        "POST",
        "/api/v1/openai/extract-form",
        json=payload,
    )

    assert response.status_code == 200
    body = response.json()
    assert len(body["extractions"]) == 10
    assert all(item["status"] == "extracted" for item in body["extractions"])
    assert body["missing_required"] == []
    assert body["skipped_existing"] == ["ghiChuCu"]
    assert body["warnings"] == []
    assert body["usage"]["input_tokens"] == 321
    assert body["usage"]["output_tokens"] == 45
    assert body["usage"]["total_tokens"] == 366
    assert body["usage"]["model"] == "gpt-4o-mini-2024-07-18"
    assert len(completions.calls) == 1
    sent_fields = json.loads(completions.calls[0]["messages"][1]["content"])["fields"]
    assert all(field[0] != "ghiChuCu" for field in sent_fields)
    assert completions.calls[0]["response_format"]["type"] == "json_schema"


@pytest.mark.asyncio
async def test_invalid_values_and_evidence_become_ambiguous():
    model_payload = {
        "extractions": [
            {"field_code": "ngay", "value": "03/10/2026", "status": "extracted", "evidence": "ngày ba tháng mười"},
            {"field_code": "soLuong", "value": "mười hai", "status": "extracted", "evidence": "mười hai hộ"},
            {"field_code": "mucDo", "value": "unknown", "status": "extracted", "evidence": "mức độ cao"},
            {"field_code": "hoTen", "value": "Nguyễn Văn An", "status": "extracted", "evidence": "đoạn không tồn tại"},
            {"field_code": "fieldLa", "value": "x", "status": "extracted", "evidence": "x"},
        ]
    }
    app, _ = build_app(payload=model_payload)
    payload = {
        "transcript": "Nguyễn Văn An, ngày ba tháng mười, mười hai hộ, mức độ cao",
        "fields": [
            {"field_code": "ngay", "label": "Ngày", "type": "date", "required": True},
            {"field_code": "soLuong", "label": "Số lượng", "type": "number", "required": True},
            {"field_code": "mucDo", "label": "Mức độ", "type": "select", "required": True, "options": [{"value": "high", "label": "Cao"}]},
            {"field_code": "hoTen", "label": "Họ tên", "type": "text", "required": True},
        ],
    }

    response = await request(app, "POST", "/api/v1/openai/extract-form", json=payload)

    assert response.status_code == 200
    body = response.json()
    items = {item["field_code"]: item for item in body["extractions"]}
    assert items["ngay"]["status"] == "ambiguous"
    assert items["ngay"]["value"] is None
    assert items["soLuong"]["status"] == "ambiguous"
    assert items["mucDo"]["status"] == "ambiguous"
    assert items["hoTen"]["status"] == "ambiguous"
    assert items["hoTen"]["value"] == "Nguyễn Văn An"
    assert body["missing_required"] == ["ngay", "soLuong", "mucDo", "hoTen"]
    assert {warning["code"] for warning in body["warnings"]} == {
        "invalid_date",
        "invalid_number",
        "invalid_option",
        "invalid_evidence",
        "unknown_field",
    }


@pytest.mark.asyncio
async def test_duplicate_model_extraction_is_ambiguous():
    model_payload = {
        "extractions": [
            {"field_code": "hoTen", "value": "Nguyễn Văn An", "status": "extracted", "evidence": "Nguyễn Văn An"},
            {"field_code": "hoTen", "value": "Nguyễn Văn B", "status": "extracted", "evidence": "Nguyễn Văn B"},
        ]
    }
    app, _ = build_app(payload=model_payload)
    payload = basic_payload()
    payload["transcript"] = "Nguyễn Văn An hay Nguyễn Văn B"

    response = await request(app, "POST", "/api/v1/openai/extract-form", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["extractions"] == [
        {"field_code": "hoTen", "value": None, "status": "ambiguous", "evidence": None}
    ]
    assert body["missing_required"] == ["hoTen"]
    assert body["warnings"][0]["code"] == "duplicate_extraction"


@pytest.mark.asyncio
async def test_request_validation_rejects_blank_transcript_and_duplicate_codes():
    app, _ = build_app()
    payload = basic_payload()
    payload["transcript"] = "   "
    payload["fields"].append(dict(payload["fields"][0]))

    response = await request(app, "POST", "/api/v1/openai/extract-form", json=payload)

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


@pytest.mark.asyncio
async def test_prompt_limit_returns_413_without_calling_openai():
    settings = build_settings(max_prompt_chars=5_000)
    app, completions = build_app(settings=settings)
    fields = []
    for index in range(20):
        fields.append(
            {
                "field_code": f"field{index}",
                "label": f"Nhãn rất dài {index} " + ("x" * 150),
                "type": "text",
                "extraction_hint": "y" * 500,
            }
        )
    payload = {
        "transcript": "Nội dung hợp lệ",
        "fields": fields,
    }

    response = await request(app, "POST", "/api/v1/openai/extract-form", json=payload)

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "prompt_too_large"
    assert completions.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("exception_name", "expected_status", "expected_code"),
    [
        ("APITimeoutError", 504, "openai_timeout"),
        ("RateLimitError", 503, "openai_rate_limited"),
        ("APIConnectionError", 502, "openai_connection_error"),
        ("APIStatusError", 502, "openai_upstream_error"),
    ],
)
async def test_maps_openai_errors(monkeypatch, exception_name, expected_status, expected_code):
    class FakeOpenAIError(Exception):
        status_code = 500

    monkeypatch.setattr(service_module, exception_name, FakeOpenAIError)
    app, _ = build_app(error=FakeOpenAIError("upstream failed"))

    response = await request(
        app,
        "POST",
        "/api/v1/openai/extract-form",
        json=basic_payload(),
    )

    assert response.status_code == expected_status
    assert response.json()["error"]["code"] == expected_code


@pytest.mark.asyncio
async def test_all_existing_values_skip_openai_call():
    app, completions = build_app()
    payload = basic_payload()
    payload["current_values"] = {"hoTen": "Đã nhập"}

    response = await request(app, "POST", "/api/v1/openai/extract-form", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["extractions"] == []
    assert body["missing_required"] == []
    assert body["skipped_existing"] == ["hoTen"]
    assert body["usage"]["total_tokens"] == 0
    assert completions.calls == []


@pytest.mark.asyncio
async def test_multi_turn_completion_skips_existing_and_fills_missing_required():
    fields = [
        {
            "field_code": "displayName",
            "label": "Tên hiển thị",
            "type": "text",
            "required": True,
        },
        {
            "field_code": "favoriteColor",
            "label": "Màu yêu thích",
            "type": "select",
            "required": True,
            "options": [
                {"value": "blue", "label": "Xanh"},
                {"value": "red", "label": "Đỏ"},
            ],
        },
    ]
    app, completions = build_app(
        payload={
            "extractions": [
                {
                    "field_code": "displayName",
                    "value": "An",
                    "status": "extracted",
                    "evidence": "Tên hiển thị là An",
                }
            ]
        }
    )

    first_turn = await request(
        app,
        "POST",
        "/api/v1/openai/extract-form",
        json={
            "transcript": "Tên hiển thị là An",
            "fields": fields,
            "current_values": {},
        },
    )

    assert first_turn.status_code == 200
    assert first_turn.json()["missing_required"] == ["favoriteColor"]

    completions.payload = {
        "extractions": [
            {
                "field_code": "favoriteColor",
                "value": "blue",
                "status": "extracted",
                "evidence": "Màu yêu thích là xanh",
            }
        ]
    }
    second_turn = await request(
        app,
        "POST",
        "/api/v1/openai/extract-form",
        json={
            "transcript": "Màu yêu thích là xanh",
            "fields": fields,
            "current_values": {"displayName": "An"},
        },
    )

    assert second_turn.status_code == 200
    body = second_turn.json()
    assert body["skipped_existing"] == ["displayName"]
    assert body["missing_required"] == []
    assert body["extractions"][0]["field_code"] == "favoriteColor"
    sent_fields = json.loads(completions.calls[-1]["messages"][1]["content"])["fields"]
    assert all(field[0] != "displayName" for field in sent_fields)
