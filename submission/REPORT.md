# Lab 21 — Evaluation Report

**Họ tên**: Hoàng Mạnh Dũng
**MSSV**: 2A202601213
**Ngày**: 21/08/2026
**Tier**: `T4`
**Base model**: `unsloth/Qwen3.5-4B`
**GPU thực tế**: `Tesla T4 — 14.6 GB VRAM`

> Mọi số liệu trong báo cáo này được lấy từ kết quả chạy NB1–NB5. Phần đánh giá cuối được chạy ở chế độ full evaluation, không sử dụng `EVAL_LIMIT=8`.

---

## 1. Setup

|                    |                                                         |
| ------------------ | ------------------------------------------------------- |
| Dataset            | 250 ticket chăm sóc khách hàng tiếng Việt → JSON triage |
| Train / val        | 225 / 25, seed = 42                                     |
| `max_length`       | 1024 — p95 đo được là 98 token                          |
| `MASK_MODE`        | `assistant-only`                                        |
| Epochs / max_steps | 2 epochs / 30 optimizer steps                           |

**Template có giữ khối ****`<think>`**** không?** Có.

Kết quả kiểm tra chat template ở NB1 cho thấy template của Qwen3.5-4B vẫn giữ được khối `<think>...</think>`, vì vậy cấu trúc reasoning không bị chat template tự động làm mất. Tuy nhiên dataset triage hiện tại chủ yếu yêu cầu model sinh JSON ngắn chứ không huấn luyện reasoning trace dài.

Thống kê token cho 250 mẫu cho kết quả:

* Mean: 93.1 token
* p50: 93 token
* p95: 98 token
* p99: 100 token
* Max: 101 token
* `suggested_max_length`: 256

Tier T4 hiện đặt `max_length=1024`, lớn hơn nhiều so với p95=98. Tôi giữ cấu hình mặc định 1024 để đảm bảo thí nghiệm nhất quán với cấu hình của lab, nhưng về mặt tối ưu tài nguyên thì `max_length=256` đã đủ cho gần như toàn bộ dataset và có thể giảm lượng compute không cần thiết.

---

## 2. Mask proof (NB1)

|                              |        |
| ---------------------------- | ------ |
| `supervised_fraction`        | 0.4149 |
| Câu trả lời nằm trong loss   | `true` |
| Câu hỏi KHÔNG nằm trong loss | `true` |

Đoạn được tính loss:

```text
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

NB1 cho thấy có 39/94 token được supervise, tương đương khoảng 41.49%. Đây là điều mong muốn khi sử dụng `assistant-only`: phần system prompt và câu hỏi của user bị mask khỏi loss, trong khi phần câu trả lời của assistant được dùng làm tín hiệu huấn luyện.

Nếu không mask phần câu hỏi, model sẽ bị tối ưu cả trên những token của input thay vì tập trung vào việc học output JSON cần thiết. Kết quả `answer_is_supervised=true` và `question_is_masked=true` chứng minh pipeline masking đang hoạt động đúng.

---

## 3. Ba baseline và fine-tune

| Run                         |     target | regression | format | latency (ms) |
| --------------------------- | ---------: | ---------: | -----: | -----------: |
| (a) base + naive prompt     |     0.0000 |     0.7578 | 0.0000 |       3276.2 |
| (b) base + optimized prompt |     0.7650 |     0.7578 | 1.0000 |       1012.4 |
| (c) LoRA fine-tune          | **0.9700** | **0.6111** | 1.0000 |       1415.8 |

**(b) có thật sự mạnh hơn (a) không?** Có.

Baseline (a) gần như thất bại hoàn toàn trên task target: `target=0.000` và `format=0.000`. Khi sử dụng optimized prompt, base model tăng lên `target=0.765` và đạt `format=1.000`. Điều này cho thấy prompt engineering có tác động rất lớn và baseline để so sánh với fine-tuning phải là baseline (b), không phải baseline (a).

Tôi **không sửa ****`OPTIMIZED_PROMPT`** sau khi xem kết quả fine-tune. SHA của optimized prompt trong kết quả là `719e74d3b6232053`, trùng với prompt được cung cấp trong source code. Việc đóng băng baseline trước khi train giúp tránh tình trạng cố tình làm yếu prompt để fine-tune trông tốt hơn.

Fine-tune đạt `target=0.970`, cao hơn optimized prompt `0.765`, tức tăng **0.205 hay 20.5 điểm phần trăm** trên nhiệm vụ triage. Tuy nhiên, sự cải thiện này đi kèm với suy giảm đáng kể ở regression, được phân tích tại phần 5.

---

## 4. Giải phẫu cấu hình sai (NB4)

| Run         | vị trí      |    r |  trainable |   LR | train loss (NB4) | **target (NB5)** |     s |  VRAM GB |
| ----------- | ----------- | ---: | ---------: | ---: | ---------------: | ---------------: | ----: | -------: |
| `correct`   | text-linear |   16 | 32,464,896 | 1e-4 |           0.6255 |       **0.9700** | 923.8 |    12.01 |
| `attn_only` | q,v         |  283 | 32,456,704 | 1e-4 |       **0.5373** |       **0.9700** | 783.7 |    12.02 |
| `wrong_lr`  | text-linear |   16 | 32,464,896 | 1e-5 |           1.5704 |       **0.0000** | 926.1 |    12.01 |
| `qlora`     | text-linear |   16 | 32,464,896 | 1e-4 |           0.7058 |       **0.9400** | 995.4 | **7.09** |

### 4.1 — `attn_only` so với `correct`

`attn_only` và `correct` có số lượng trainable parameters gần như bằng nhau: khoảng 32.46 triệu parameters. `attn_only` phải tăng rank lên `r=283` để match ngân sách tham số của cấu hình `correct`, trong khi `correct` sử dụng `r=16` nhưng đặt LoRA trên nhiều linear layer hơn.

Trên tập target full, hai cấu hình **hoà nhau ở 0.970**. Tuy nhiên nếu chỉ nhìn train loss, `attn_only=0.5373` thấp hơn `correct=0.6255`, do đó người đánh giá chỉ dựa trên loss có thể kết luận sai rằng `attn_only` tốt hơn. Target evaluation cho thấy loss thấp hơn không chuyển thành accuracy cao hơn.

Kết quả này cũng cho thấy rank không thể được xem là đòn bẩy độc lập. Tăng rank của q,v lên rất lớn giúp `attn_only` có cùng ngân sách tham số và đạt cùng target score, nhưng không chứng minh rank lớn tốt hơn placement rộng hơn. Trong bài toán triage JSON hẹp này, tôi chưa có bằng chứng rằng `text-linear` tốt hơn `q,v` về target accuracy; điều có thể khẳng định là **train loss không đủ để xếp hạng hai cấu hình**.

### 4.2 — `wrong_lr`

`wrong_lr` chỉ thay đổi learning rate từ `1e-4` xuống `1e-5`, trong khi placement, rank và số lượng trainable parameters vẫn giống `correct`. Tuy nhiên đường loss khác rất rõ. Với `correct`, loss log giảm nhanh từ khoảng 2.163 xuống 1.381, sau đó xuống 0.1385 và cuối training ở vùng rất thấp. Ngược lại, `wrong_lr` giảm chậm hơn nhiều: từ khoảng 2.163 → 2.066 → 1.606 → 1.326 → 1.141 → 1.119 ở các checkpoint log.

Kết quả task cuối cùng còn rõ hơn: `wrong_lr` đạt `target=0.000` và `format=0.000`, trong khi `correct` đạt `target=0.970` và `format=1.000`. Điều này chứng minh learning rate quá nhỏ khiến adapter không học đủ trong ngân sách 30 optimizer steps.

Nếu chỉ nhìn loss mà không biết learning rate, tôi có thể kết luận nhầm rằng dataset, model hoặc loss mask có vấn đề. Thực tế, chỉ một thay đổi `1e-4 → 1e-5` đã đủ làm fine-tuning gần như thất bại hoàn toàn trên task.

### 4.3 — QLoRA

QLoRA giảm peak VRAM từ khoảng `12.01 GB` xuống `7.09 GB`, tiết kiệm khoảng **4.92 GB**, tương đương khoảng **41% VRAM**. Đây là mức tiết kiệm rất đáng kể đối với GPU T4 chỉ có khoảng 14.6 GB VRAM.

Đổi lại, target score giảm từ `0.970` xuống `0.940`, tức giảm khoảng **3 điểm phần trăm**. QLoRA cũng có thời gian train khoảng 995.4 giây, lâu hơn `correct` khoảng 923.8 giây, và latency target khoảng 1875.7 ms so với 1415.8 ms của `correct`.

Số đo của tôi vì vậy chỉ **ủng hộ một phần** khuyến nghị không dùng QLoRA cho model này. Nếu ưu tiên chất lượng và đủ VRAM, fp16 LoRA `correct` tốt hơn. Nhưng nếu VRAM là giới hạn chính, việc tiết kiệm khoảng 41% bộ nhớ với mức giảm target chỉ 3 điểm phần trăm vẫn là một trade-off có thể chấp nhận trong một số hệ thống. Vì vậy không nên kết luận QLoRA luôn không dùng được; quyết định phải phụ thuộc vào giới hạn phần cứng và yêu cầu accuracy thực tế.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: `FAILED`

`target Δ = +0.205`
`regression Δ = -0.147`
`valid_trace_rate = 0.00`

Fine-tuning đem lại một cải thiện rất lớn trên nhiệm vụ chuyên biệt: target tăng từ `0.765` của base model với optimized prompt lên `0.970`, tương đương tăng 20.5 điểm phần trăm. Nếu chỉ đo target task, đây là một kết quả rất tốt và có thể khiến tôi kết luận model nên được deploy ngay. Tuy nhiên, regression score giảm từ `0.7578` xuống `0.6111`, tức giảm khoảng 0.147. Regression gate của bài chỉ cho phép mức suy giảm tối đa 0.020, vì vậy cấu hình này bị đánh `FAILED`.

Điều này cho thấy fine-tuning đã làm model chuyên môn hoá mạnh cho ticket triage nhưng đồng thời làm suy giảm một phần năng lực chung. Đây là biểu hiện của forgetting khi model được tối ưu quá tập trung vào một tập dữ liệu chuyên biệt. Vì vậy một model có target accuracy cao chưa chắc là một model tốt để triển khai nếu nó vẫn phải xử lý các yêu cầu ngoài domain.

`valid_trace_rate=0.00` cũng cho thấy output của cấu hình hiện tại không tạo reasoning trace hợp lệ. Tuy nhiên task chính ở đây yêu cầu JSON triage và dataset không được thiết kế để tối ưu reasoning trace, nên đây không phải nguyên nhân trực tiếp khiến regression gate thất bại.

Hướng xử lý tiếp theo hợp lý nhất là thêm khoảng **1–5% replay data** từ dữ liệu general-capability vào quá trình fine-tuning, giữ nguyên tập eval, sau đó train và chạy NB5 lại để xem có giữ được target gain mà giảm mức regression hay không.

---

## 6. Định tính — bắt buộc có cả ca THUA

**Lưu ý:** artifact NB2 hiện tại chỉ lưu metric tổng hợp của baseline (b), không lưu prediction theo từng ticket. Vì vậy tôi không tự tạo hoặc đoán prediction của baseline (b). Trong bảng dưới, “FT thắng” được dùng cho trường hợp fine-tune dự đoán đúng cả 4/4 trường; “FT thua” là trường hợp fine-tune vẫn sai ít nhất 1/4 trường so với ground truth.

| #   | Ticket (rút gọn)                                     | Nhãn đúng                                                      | (b) prompt                                      | (c) fine-tune                                                           | Nhận xét               |
| --- | ---------------------------------------------------- | -------------------------------------------------------------- | ----------------------------------------------- | ----------------------------------------------------------------------- | ---------------------- |
| 1   | Shipper không gọi — ốp lưng điện thoại               | `van_chuyen / thap / ốp lưng điện thoại / tich_cuc`            | Không có per-case prediction trong artifact NB2 | Đúng 4/4 trường (`ft_score=1.00`)                                       | ✅ **FT thắng**         |
| 2   | Hỏi giá ốp lưng điện thoại                           | `hoi_thong_tin / trung_binh / ốp lưng điện thoại / trung_tinh` | Không có per-case prediction trong artifact NB2 | Đúng 4/4 trường (`ft_score=1.00`)                                       | ✅ **FT thắng**         |
| 3   | Bình giữ nhiệt — chưa thấy tiền, “khi nào tiện”      | `hoan_tien / thap / bình giữ nhiệt / tich_cuc`                 | Không có per-case prediction trong artifact NB2 | Dự đoán `urgency=trung_binh`, các trường còn lại đúng (`ft_score=0.75`) | ❌ **FT thua 1 trường** |
| 4   | Nồi chiên không dầu — thiếu phụ kiện, “khi nào tiện” | `san_pham_loi / thap / nồi chiên không dầu / trung_tinh`       | Không có per-case prediction trong artifact NB2 | Dự đoán `urgency=trung_binh`, các trường còn lại đúng (`ft_score=0.75`) | ❌ **FT thua 1 trường** |
| 5   | Áo khoác gió — bị lỗi, “khi nào tiện”                | `san_pham_loi / thap / áo khoác gió / tich_cuc`                | Không có per-case prediction trong artifact NB2 | Dự đoán `urgency=trung_binh`, các trường còn lại đúng (`ft_score=0.75`) | ❌ **FT thua 1 trường** |

### Có mẫu chung nào ở các ca FT thua không?

Có. Cả ba trường hợp tệ nhất đều có ground-truth `urgency=thap`, nhưng model fine-tune dự đoán thành `urgency=trung_binh`. Hai ticket có cụm từ trực tiếp **“Khi nào tiện”**, vốn thể hiện mức khẩn cấp thấp, nhưng model vẫn nâng urgency lên trung bình. Điều này cho thấy lỗi còn lại của model không nằm chủ yếu ở `intent` hay `product`, mà tập trung vào cách diễn giải các tín hiệu ngôn ngữ mềm về mức độ khẩn cấp.

Đây là một gợi ý rõ ràng cho bước cải thiện dataset: bổ sung thêm các hard examples cho `urgency=thap`, đặc biệt các câu như “khi nào tiện”, “không gấp”, “hỏi cho biết”, và các cách diễn đạt tương tự. Đồng thời cần tránh làm mất khả năng generalization bằng cách kết hợp các ví dụ chuyên biệt này với replay data tổng quát.

---

## 7. Kết luận & điều tôi học được

### Kết luận

Tôi **chưa nên deploy phiên bản fine-tune hiện tại** ở trạng thái hiện tại nếu model vẫn được kỳ vọng xử lý cả nhiệm vụ triage và các yêu cầu general-purpose khác. Lý do là mặc dù LoRA cải thiện target score rất mạnh từ `0.765` lên `0.970`, regression score lại giảm từ `0.7578` xuống `0.6111`. Mức giảm khoảng 14.7 điểm phần trăm vượt xa tolerance 2 điểm của regression gate. Vì vậy kết quả `FAILED` phản ánh một vấn đề triển khai thật chứ không phải lỗi của bài lab.

Nếu hệ thống được cô lập hoàn toàn và chỉ dùng model cho ticket triage thì accuracy 97% là một tín hiệu rất tích cực. Tuy nhiên với tiêu chí của lab, tôi vẫn cần khắc phục regression trước khi quyết định deploy. Hướng tiếp theo là thêm 1–5% replay data mang năng lực tổng quát, train lại với cùng tập eval và kiểm tra xem target gain có được giữ lại hay không.

Trong các đòn bẩy được thử trực tiếp, **learning rate là yếu tố có ảnh hưởng rõ nhất**. Chỉ giảm LR từ `1e-4` xuống `1e-5` khiến target giảm từ 0.97 về 0.00. Loss mask lại là điều kiện nền tảng: nếu mask sai thì toàn bộ thí nghiệm sau đó không còn đáng tin. Placement chưa cho thấy ưu thế rõ ràng vì `correct` và `attn_only` cùng đạt 0.97 khi ngân sách parameter được match. Cuối cùng, kết quả regression cho thấy **data mixture** sẽ là đòn bẩy cần thử tiếp theo, đặc biệt là replay data để giảm forgetting.

### Ba điều tôi học được

1. **Fine-tuning tốt hơn trên target task không đồng nghĩa model tổng thể tốt hơn.** LoRA tăng target từ 76.5% lên 97%, nhưng general-capability regression giảm khoảng 14.7 điểm, đủ để model không vượt qua deployment gate.

2. **Train loss không phải metric để chọn model.** `attn_only` có train loss 0.5373 thấp hơn `correct` 0.6255 nhưng hai model cùng đạt target 0.970. Model phải được đánh giá trên task metric thực tế, không được kết luận chỉ dựa trên loss.

3. **Learning rate và giới hạn phần cứng tạo ra trade-off rất lớn.** `wrong_lr` chỉ thay một con số nhưng target giảm về 0. QLoRA thì giảm VRAM khoảng 41% nhưng mất khoảng 3 điểm phần trăm target, cho thấy lựa chọn fine-tuning phải đồng thời cân nhắc accuracy, VRAM, latency và chi phí.

### Nếu có thêm 2 giờ nữa, tôi sẽ thử:

Tôi sẽ giữ nguyên eval set và cấu hình `correct`, thêm khoảng **1–5% replay data general-purpose** vào training set rồi train lại. Sau đó tôi sẽ chạy lại NB5 để kiểm tra xem regression score có trở lại gần baseline mà vẫn giữ target score cao hay không. Nếu còn thời gian, tôi cũng sẽ thử giảm `max_length` từ 1024 xuống 256 vì p95 chỉ là 98 token, nhằm giảm compute dư thừa mà không làm mất dữ liệu.

---

## Phụ lục — thưởng đã làm

* [ ] B1 NB6 merge + hot-swap
* [ ] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
* [ ] B3 reasoning-trace collapse (hai `MASK_MODE`, kèm `valid_trace_rate`)
* [ ] B4 quét rank có kiểm soát
* [x] B5 HuggingFace Hub — https://huggingface.co/Hoangdung04/Day21-LoRA-2A202601213
