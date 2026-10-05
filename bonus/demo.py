"""bonus/demo.py — Execution demo running 5 representative query scenarios.

Exits with code 0 upon successfully demonstrating:
1. Pure vector episodic retrieval.
2. Profile-guided personalization recommendation.
3. Fresh activity awareness from streaming velocity.
4. Paraphrase conceptual matching.
5. Mixed hybrid search combining profile and episodic memory.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bonus.agent import HybridMemoryAgent


def main() -> int:
    print("=== Initializing HybridMemoryAgent ===")
    agent = HybridMemoryAgent()

    user_id = "u_001"

    # Seed realistic user episodic memories
    sample_memories = [
        "Đã hoàn thành chương 3 về triển khai cụm Kubernetes (k8s) và thiết lập Ingress Controller.",
        "Ghi chú kỹ thuật: Phương pháp tự động co giãn Pod theo chỉ số CPU và RAM trong môi trường Cloud.",
        "Tài liệu phân tích kiến trúc Zero Trust và quản lý định danh người dùng trong Cloud Security.",
        "Hướng dẫn tối ưu hóa câu truy vấn cơ sở dữ liệu PostgreSQL cho hệ thống xử lý giao dịch cao.",
        "Tổng hợp bài học khi cấu hình hệ thống giám sát Prometheus và Grafana cho hạ tầng Microservices.",
    ]

    print(f"Ingesting {len(sample_memories)} episodic memories for user {user_id}...")
    for text in sample_memories:
        agent.remember(text=text, user_id=user_id)
    print("Ingestion complete.\n")

    test_queries = [
        (
            "Query 1 (Vector Hit / Đơn giản)",
            "Tôi đã đọc gì về Kubernetes?",
        ),
        (
            "Query 2 (Cần Profile Context)",
            "Recommend đọc gì tiếp",
        ),
        (
            "Query 3 (Cần Fresh Activity)",
            "Tôi đang quan tâm gì gần đây?",
        ),
        (
            "Query 4 (Paraphrase / Vector Wins)",
            "Tài liệu về tự động mở rộng hạ tầng?",
        ),
        (
            "Query 5 (Mixed / Hybrid + Profile)",
            "Cho tôi summary cloud security",
        ),
    ]

    for label, query in test_queries:
        print(f"--------------------------------------------------")
        print(f"[{label}]")
        print(f"User Query: '{query}'")
        context = agent.recall(query=query, user_id=user_id)
        print("Assembled Context:")
        print(context)
        print()

    print("=== All 5 query scenarios executed successfully ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
