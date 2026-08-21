# Reflection — Lab 21

*Ngắn gọn, thành thật. Phần này chấm theo độ cụ thể, không theo độ dài.*

## 1. Điều gì làm bạn ngạc nhiên nhất?

Điều làm tôi ngạc nhiên nhất là fine-tuning giúp target accuracy tăng từ 76.5% lên 97%, nhưng regression lại giảm từ 75.78% xuống 61.11%. Trước đó tôi nghĩ model làm tốt hơn nhiệm vụ được fine-tune thì có nghĩa là model đã tốt hơn, nhưng thực tế nó có thể giỏi hơn một nhiệm vụ và đồng thời kém đi ở năng lực khác.

## 2. Bạn mất nhiều thời gian nhất ở đâu? Nó có phải chỗ bạn dự đoán không?

Tôi mất nhiều thời gian nhất ở NB4, khoảng 48 phút, vì phải train ba cấu hình đối chứng: `attn_only`, `wrong_lr` và `qlora`. Ban đầu tôi không nghĩ phần so sánh cấu hình lại lâu hơn việc train model chính ở NB3.

## 3. Trước lab này bạn tin điều gì về fine-tuning mà giờ bạn không còn tin?

Trước lab tôi nghĩ fine-tuning chủ yếu là đưa dữ liệu vào rồi train, và loss càng thấp thì model càng tốt. Sau lab tôi hiểu rằng train loss thấp chưa chắc cho model tốt hơn trên dữ liệu đánh giá. Ví dụ `attn_only` có train loss thấp hơn `correct`, nhưng cả hai đều đạt target accuracy 97%.

## 4. Bạn dùng AI assistant vào việc gì trong lab? Chỗ nào nó sai?

Tôi dùng AI assistant để giải thích fine-tuning, đọc log, xử lý lỗi test, hướng dẫn chạy từng notebook và phân tích kết quả NB1–NB5. AI assistant có lúc ước lượng thời gian chạy NB5 khoảng 10–20 phút, nhưng thực tế NB5 smoke chỉ mất khoảng 4.5 phút. Vì vậy các dự đoán về thời gian chạy chỉ nên dùng để tham khảo, còn kết quả thực tế phải dựa vào log.

## 5. Nếu ngày mai phải fine-tune cho một khách hàng thật, bước đầu tiên bạn làm là gì?

Bước đầu tiên tôi sẽ xác định rõ task, dữ liệu và metric đánh giá trước khi train. Tôi cũng sẽ tạo một baseline mạnh bằng prompt engineering và chuẩn bị cả target eval lẫn regression eval. Sau lab này, tôi sẽ không quyết định deploy chỉ vì accuracy của task fine-tune tăng mà phải kiểm tra model có làm suy giảm những năng lực quan trọng khác hay không.