# Form Extraction API

FastAPI độc lập để trích xuất nhiều field từ một transcript tiếng Việt theo
schema biểu mẫu. Service dùng OpenAI Chat Completions Structured Outputs, sau
đó hậu kiểm field code, kiểu dữ liệu, option và evidence trước khi trả kết quả.

Service không thay thế STT. Mini App tiếp tục dùng Zipformer để lấy transcript
rồi gửi transcript trực tiếp sang API này trong bản demo không có quyền sửa
backend.

## Yêu cầu

- Python 3.11 trở lên.
- `uv` (khuyến nghị) hoặc `pip`.
- `OPENAI_API_KEY` khi muốn gọi extraction thật.

## Chạy local

```bash
cd ai-service/api_form_extraction
cp .env.example .env
uv sync --extra dev
uv run uvicorn app.main:app --reload
```

Khi chưa có `OPENAI_API_KEY`, service vẫn khởi động:

```bash
curl http://127.0.0.1:8000/health/live
curl -i http://127.0.0.1:8000/health/ready
```

Kết quả mong đợi là `/health/live` trả `200`, còn `/health/ready` trả `503`
với `openai_configured: false`.

## Cấu hình key

Điền vào `.env` và khởi động lại service:

```dotenv
OPENAI_API_KEY=sk-...
```

Không commit file `.env`.

Khi Mini App gọi trực tiếp trong môi trường demo:

```dotenv
ALLOW_PUBLIC_EXTRACTION=true
CORS_ALLOWED_ORIGINS=*
```

Nếu đã biết chính xác origin của bản triển khai, thay `*` bằng danh sách origin
phân tách bằng dấu phẩy. Chế độ public không làm lộ `OPENAI_API_KEY`, nhưng bất
kỳ ai biết URL endpoint đều có thể gửi request; nên đặt spending limit cho key
OpenRouter/OpenAI và chỉ dùng chế độ này cho demo.

Khi có backend proxy đáng tin cậy, tắt public mode và cấu hình:

```dotenv
APP_ENV=production
INTERNAL_API_KEY=<chuoi-ngau-nhien-dai>
ALLOW_PUBLIC_EXTRACTION=false
```

Backend proxy gọi endpoint extraction với header:

```http
X-Internal-API-Key: <chuoi-ngau-nhien-dai>
```

## API

### `GET /health/live`

Liveness check, luôn trả `200` khi process hoạt động.

### `GET /health/ready`

Trả `200` khi OpenAI client đã được cấu hình; trả `503` nếu thiếu key.

### `POST /api/v1/openai/extract-form`

Ví dụ:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/openai/extract-form \
  -H 'Content-Type: application/json' \
  -d '{
    "transcript": "Tôi tên Nguyễn Văn An, sinh ngày 12 tháng 8 năm 1998 và giới tính nam",
    "reference_date": "2026-10-03",
    "current_values": {},
    "fields": [
      {
        "field_code": "hoTen",
        "label": "Họ và tên",
        "type": "text",
        "required": true
      },
      {
        "field_code": "ngaySinh",
        "label": "Ngày sinh",
        "type": "date",
        "required": true
      },
      {
        "field_code": "gioiTinh",
        "label": "Giới tính",
        "type": "select",
        "required": true,
        "options": [
          {"value": "M", "label": "Nam"},
          {"value": "F", "label": "Nữ"}
        ]
      }
    ]
  }'
```

Response rút gọn:

```json
{
  "request_id": "2e9fbd62-ccdd-4cc2-a285-d65756f8a8f4",
  "extractions": [
    {
      "field_code": "hoTen",
      "value": "Nguyễn Văn An",
      "status": "extracted",
      "evidence": "tên Nguyễn Văn An"
    },
    {
      "field_code": "ngaySinh",
      "value": "1998-08-12",
      "status": "extracted",
      "evidence": "sinh ngày 12 tháng 8 năm 1998"
    }
  ],
  "missing_required": [],
  "skipped_existing": [],
  "warnings": [],
  "usage": {
    "model": "gpt-4o-mini-2024-07-18",
    "input_tokens": 850,
    "output_tokens": 120,
    "total_tokens": 970,
    "latency_ms": 642.5
  }
}
```

Các field đã có giá trị không được gửi cho mô hình và xuất hiện trong
`skipped_existing`. Field bắt buộc không được trích xuất thành công xuất hiện
trong `missing_required`. Client chỉ nên tự áp dụng item có
`status: "extracted"`; item `ambiguous` cần người dùng xác nhận.

Giới hạn request:

- Transcript: 10.000 ký tự.
- Tối đa 60 field.
- Tối đa 100 option mỗi field.
- `extraction_hint`: 500 ký tự.
- Prompt compact hoàn chỉnh: mặc định 50.000 ký tự.

## Kiểm thử

Test suite mock OpenAI client và không cần API key:

```bash
uv run --extra dev pytest
```

## Docker

```bash
docker build -t api-form-extraction .
docker run --rm -p 8000:8000 --env-file .env api-form-extraction
```

## Theo dõi chi phí

Response có `usage.input_tokens` và `usage.output_tokens`. Chi phí mỗi request
được tính theo giá model tại thời điểm sử dụng:

```text
chi phí = input_tokens × giá input/token
        + output_tokens × giá output/token
```

Không log transcript, giá trị field, `OPENAI_API_KEY` hoặc
`INTERNAL_API_KEY`.
