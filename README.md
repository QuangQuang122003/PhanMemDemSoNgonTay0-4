# Hướng dẫn sử dụng

## Yêu cầu

Đảm bảo bạn đã cài đặt các thư viện sau:
- streamlit
- cv2
- numpy
- keras
- pickle
- PIL

Bạn cũng cần có mô hình `model_bbox_regression_and_classification_VGG16.h5` trong thư mục làm việc của mình 
(vì trong ứng dụng này tôi sử dụng model model_bbox_regression_and_classification_VGG16.h5, bạn có thể sử dụng mô hình khác tùy thích)

## Chạy ứng dụng
Để chạy ứng dụng, mở terminal và điều hướng đến thư mục chứa tệp `app.py`. Sau đó, chạy lệnh sau:
streamlit run app.py
