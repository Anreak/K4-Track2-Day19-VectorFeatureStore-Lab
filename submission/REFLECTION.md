# Reflection — Lab 19

Tên: Nguyễn Quang Hữu
Cohort: A20-K4
Path đã chạy: lite

---

## Câu hỏi (≤ 200 chữ)

> Trên golden set 50 queries, mode nào thắng ở loại query nào (exact / paraphrase / mixed), và tại sao? Khi nào bạn không dùng hybrid (i.e. khi nào pure BM25 hoặc pure vector là lựa chọn đúng)?

Trên tập golden 50 queries:
- Nhóm exact: BM25 và Hybrid cùng dẫn đầu (96.7%) nhờ khả năng so khớp chính xác từ vựng kỹ thuật.
- Nhóm paraphrase: Vector nắm bắt ngữ nghĩa trừu tượng tốt hơn từ khóa thô khi người dùng diễn đạt lại.
- Nhóm mixed: Hybrid chiến thắng tuyệt đối (100.0%) nhờ Reciprocal Rank Fusion kết hợp hài hòa độ chính xác từ vựng và độ phủ ngữ nghĩa.

Không dùng hybrid trong các trường hợp:
1. Cần độ trễ cực thấp (dưới 5ms): Pure BM25 trên CPU nhanh hơn nhiều và không tốn chi phí suy luận embedding.
2. Tác vụ đặc thù: Pure BM25 cho tra cứu chính xác mã lỗi, ID sản phẩm, tên hàm; Pure Vector cho tìm kiếm ý niệm trừu tượng hoặc xuyên ngôn ngữ.

Note: Hybrid Search đòi hỏi chi phí tính toán gấp đôi cho hai retriever kèm bước hợp nhất thứ hạng, do đó không nên áp dụng tràn lan cho truy vấn đơn giản.

---

## Điều ngạc nhiên nhất khi làm lab này

Hiện tượng post-filtering làm sụt giảm độ phủ (recall drop) từ 100% xuống 28.6% khi điều kiện lọc quá hẹp, và khoảng chênh rò rỉ thông tin lên tới 0.477 của target encoding trên session_id nếu thiếu point-in-time join.

---

## Bonus challenge

- [x] Đã làm bonus (xem bonus/)
- [ ] Pair work với: Tự thực hiện
