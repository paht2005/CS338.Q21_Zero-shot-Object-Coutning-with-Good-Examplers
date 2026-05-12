---
created: 2026-05-12T17:49:43.990Z
title: Rewrite main.tex Introduction and Related Work sections
area: docs
files:
  - Rich_Prompt_Guided_Open_World_Detection_For_Zero_Shot_Object_Counting_MAPR2026/main/main.tex:86-500
---

## Problem

Ưu tiên cao — cần làm trước.

Phần từ abstract trở xuống (Introduction và Related Work) cần được viết lại hoàn toàn vì:

1. **Phong cách viết liệt kê, thiếu liền mạch**: các đoạn văn hiện tại liệt kê ý như bullet points mở rộng thay vì viết narrative mạch lạc. Paper cần có tính kết nối giữa các đoạn và các mục.
2. **Thiếu flow dẫn dắt**: chuyển tiếp giữa các section và subsection đột ngột, thiếu câu bridge dẫn reader từ ý này sang ý khác.
3. **Cần đúng với claim của abstract**: abstract claim rằng paper propose Rich Prompts để cải thiện exemplar quality cho zero-shot object counting — phần Introduction và Related Work phải build up motivation cho đúng contribution này.

## Yêu cầu cụ thể

### Section I — Introduction
Viết lại theo 4 phần rõ ràng nhưng liền mạch (không dùng bold headers nếu không cần):
1. Giới thiệu bài toán (zero-shot object counting, vì sao quan trọng)
2. Thách thức tồn đọng / gặp phải (open-vocab grounding, exemplar quality, speed)
3. Giải pháp khắc phục (Rich Prompts + CLIP re-ranking + YOLO-World)
4. Summary / contributions (tóm tắt đóng góp)

### Section II — Related Work
Viết lại có tính narrative: dẫn dắt từ các hướng nghiên cứu liên quan (class-specific counting → few-shot → zero-shot) một cách rõ ràng hơn, chỉ ra hạn chế của từng hướng, từ đó tự nhiên dẫn đến khoảng trống mà paper này lấp đầy.

## Solution

Giữ nguyên phần từ đầu file đến hết `\begin{abstract}...\end{abstract}` (không sửa abstract).
Viết lại toàn bộ `\section{Introduction}` và `\section{Related Work}` (và subsections) với văn phong học thuật IEEE, liền mạch, có câu chuyển tiếp, đúng với claim của abstract.
