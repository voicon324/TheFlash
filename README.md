# VNPT AI - The Builder Track 2

## Giới thiệu

Đây là submission cho cuộc thi VNPT AI - Age of AInicorns - Track 2 The Builder.

## Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                     Input: private_test.json                    │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Data Preprocessing                          │
│  - Load JSON questions                                          │
│  - Extract context (if embedded)                                │
│  - Parse choices (A, B, C, D, ...)                              │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                     LLM Inference (Small Model)                 │
│  - Format prompt with question + choices                        │
│  - Add context if available                                     │
│  - Call VNPT AI Small Model API                                 │
│  - Parse response to extract answer (A/B/C/D)                   │
│  - Measure inference time per sample                            │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Output Generation                           │
│  - submission.csv: qid,answer                                   │
│  - submission_time.csv: qid,answer,time                         │
└─────────────────────────────────────────────────────────────────┘
```

## Cấu trúc thư mục

```
.
├── Dockerfile              # Docker configuration (CUDA 12.2)
├── requirements.txt        # Python dependencies
├── inference.sh            # Entry script for Docker
├── predict.py              # Main prediction pipeline
├── api-keys.json           # API credentials
├── pipeline/               # Core pipeline modules
│   ├── config.py           # Configuration
│   ├── data_loader.py      # Data loading utilities
│   ├── inference.py        # LLM inference logic
│   ├── embedding.py        # Embedding utilities
│   ├── search.py           # Vector search
│   ├── rag.py              # RAG implementation
│   └── ...
└── README.md
```

## Data Processing

1. **Load JSON**: Đọc file test từ `/code/private_test.json`
2. **Parse Questions**: Trích xuất câu hỏi, các lựa chọn, và context (nếu có)
3. **Context Extraction**: Nhận diện và tách context từ câu hỏi (với pattern "Đoạn thông tin:", "[1] Tiêu đề:", etc.)

## Resource Initialization

Pipeline không yêu cầu khởi tạo Vector Database hoặc Indexing trước. Tất cả được xử lý real-time qua API.

### API Credentials

API credentials được load từ file `api-keys.json`:
```json
[
  {
    "llmApiName": "LLM small",
    "authorization": "Bearer <token>",
    "tokenId": "<id>",
    "tokenKey": "<key>"
  }
]
```

## Hướng dẫn Build và Run

### 1. Build Docker Image

```bash
sudo docker build -t team_submission .
```

### 2. Run Docker Container

```bash
sudo docker run --gpus all \
  -v /path/to/private_test.json:/code/private_test.json \
  -v /path/to/api-keys.json:/code/api-keys.json \
  -v /path/to/output:/code/output \
  team_submission
```

**Giải thích:**
- `/path/to/private_test.json`: File test data (sẽ được BTC mount vào)
- `/path/to/api-keys.json`: File API credentials
- `/path/to/output`: Thư mục output trên máy host

### 3. Output

Sau khi chạy xong, file kết quả sẽ được lưu vào thư mục output:
- `output/submission.csv`: File kết quả với format `qid,answer`
- `output/submission_time.csv`: File kết quả với timing `qid,answer,time`

## Liên hệ

Team: TheFlash
