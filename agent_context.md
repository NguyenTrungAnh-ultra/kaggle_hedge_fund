+làm chính xác tuyệt đối, không tự bịa thuật toán dưới bất kì lí do gì bao gôm cả vì để code chạy được; nếu thuật toán gặp vấn đề, dừng tiến trình, báo với người dùng

+Triết lí là quyển sách "advanced in financial machine learning" của Marcos Lopez de Prado

+luôn quét qua file trước khi phân tích, tôi có thêm vài chi tiết ảnh hướng lớn đến kết quả

+luôn dùng sys và os.path() để tránh lỗi đường dẫn

+trong xử lí dữ liệu, áp dụng các kĩ thuật tính toán song song, gpu, async, etc. để tăng tốc độ tính toán

+cho một module là một class riêng, ưu tiên khả năng debug, cleancode.

+luôn cập nhật các bước đã làm vào README.md, nếu bước nào bị thay đổi thì thay đổi luôn trong readme

+chú ý đến việc sắp xếp các file đầu ra vào folder thích hợp.

+thứ tự ưu tiên: chính xác tuyệt đối (luôn đứng đầu); tốc độ và clean (ngang nhau, ưu tiên thứ 2)

\*\*\*\* TUYỆT ĐỐI KHÔNG ĐỂ LEAK DATA BẰNG BẤT CỨ GIÁ NÀO

+luôn xóa các cột "is*nan*\*" (hoặc gắt cờ) nếu chúng toàn 0 hoặc toàn 1 để tránh nạp các feature vô dụng.

"thiết bị: i5 12400f, 32gb ddr4, RTX 5060"
