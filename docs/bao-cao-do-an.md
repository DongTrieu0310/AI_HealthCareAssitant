# Báo cáo đồ án: Trợ lý Sức khỏe AI

**Hệ thống hỗ trợ sàng lọc nguy cơ bệnh mãn tính bằng học máy**

Đánh giá nguy cơ ba bệnh: tim mạch, đái tháo đường, tăng huyết áp.

---

## 1. Đặt vấn đề và phạm vi

Ba bệnh mãn tính phổ biến — tim mạch, đái tháo đường và tăng huyết áp — thường
tiến triển âm thầm và chỉ được phát hiện khi đã có biến chứng. Việc sàng lọc
sớm dựa trên các chỉ số đo được đơn giản (huyết áp, đường huyết, cân nặng,
tuổi, thói quen sinh hoạt) có giá trị thực tiễn, đặc biệt ở tuyến cơ sở.

**Mục tiêu.** Xây dựng một hệ thống ước tính nguy cơ mắc ba bệnh trên từ các chỉ
số nhập vào, phân loại mức nguy cơ, đưa ra khuyến nghị chung và cho phép theo
dõi nhiều lần đo của cùng một bệnh nhân theo thời gian.

**Phạm vi — hệ thống này KHÔNG:**

- không chẩn đoán bệnh;
- không kê đơn, không đề xuất liều thuốc, không đưa phác đồ điều trị;
- không thay thế nhân viên y tế.

Đây là công cụ **hỗ trợ sàng lọc và hỗ trợ quyết định**. Mọi kết quả là ước
tính xác suất của mô hình, không phải kết luận y khoa. Nguyên tắc này được thể
hiện trực tiếp trong sản phẩm: cảnh báo hiển thị ở cuối mỗi lần đánh giá, và
hệ thống chỉ đưa ra khuyến nghị theo dõi, không có chức năng nào liên quan tới
kê đơn hay điều trị.

---

## 2. Dữ liệu

| Bộ dữ liệu | Số bản ghi | Huấn luyện / Kiểm thử | Số đặc trưng dùng | Tỉ lệ ca dương |
|---|---:|---:|---:|---:|
| Tim mạch (cardio) | 70.000 | 56.000 / 14.000 | 11 | 50,0% |
| Đái tháo đường (diabetes) | 768 | 614 / 154 | 8 | 35,1% |
| Tăng huyết áp (hypertension) | 4.240 | 3.392 / 848 | 12 | 31,0% |

Dữ liệu được tách theo tỉ lệ 80/20. Tập kiểm thử **không tham gia huấn luyện**
và được dùng cho toàn bộ số liệu đánh giá ở mục 4.

Tiền xử lý: làm sạch giá trị bất thường, chuẩn hoá kiểu dữ liệu, mã hoá biến
phân loại, tách tập huấn luyện/kiểm thử. Kết quả lưu thành các tệp CSV riêng để
mọi bước sau đều dùng chung một bản tách cố định (đảm bảo tái lập được).

**Chênh lệch quy mô dữ liệu là một hạn chế cần lưu ý:** bộ đái tháo đường chỉ có
768 bản ghi, tập kiểm thử 154 ca — mọi chỉ số đánh giá trên bộ này có sai số
lớn hơn nhiều so với hai bộ còn lại.

---

## 3. Kiến trúc hệ thống

Hệ thống gồm **hai luồng tách rời**, chỉ gặp nhau ở tệp mô hình đã huấn luyện.

### Luồng 1 — Huấn luyện (ngoại tuyến, chạy một lần)

```
Dữ liệu y tế thô → Tiền xử lý → Huấn luyện + tinh chỉnh + chọn ngưỡng
                                        ↓
                          Mô hình đã lưu (.pkl) + ngưỡng
```

Do người phát triển chạy tay. Sản phẩm không phải là màn hình nào, mà là tệp mô
hình nằm sẵn trên đĩa.

### Luồng 2 — Ứng dụng (trực tuyến, chạy mỗi lần sử dụng)

```
Người dùng nhập chỉ số
        ↓
   [ Dự đoán ] → [ Phân mức nguy cơ ] → [ Khuyến nghị ]
        ↑                                      ↓
  Mô hình đã lưu                        Kết quả hiển thị
                                               ↓  (bấm lưu)
                                      Hồ sơ bệnh nhân (SQLite)
                                               ↓
                                      Lịch sử + biểu đồ
```

Ứng dụng **không huấn luyện lại gì cả** — chỉ nạp mô hình có sẵn rồi tính toán,
mất vài giây mỗi lần.

### Các tầng trong ứng dụng

| Tầng | Vai trò |
|---|---|
| Giao diện (Streamlit) | Nhập liệu, kiểm tra hợp lệ, hiển thị kết quả |
| Prediction Layer | Nạp 3 mô hình + ngưỡng, trả xác suất cho từng bệnh |
| Decision Engine | Quy xác suất thành mức nguy cơ, chọn bệnh ưu tiên |
| Recommendation Engine | Sinh khuyến nghị theo bệnh và theo mức nguy cơ |
| Storage | Lưu bệnh nhân và các lần đo vào SQLite |

---

## 4. Mô hình và kết quả đánh giá

Cả ba bệnh dùng **Random Forest**. Ngưỡng quyết định của từng mô hình được chọn
riêng thay vì mặc định 0,5 — ưu tiên **Recall** (giảm bỏ sót ca bệnh), vì trong
bài toán sàng lọc, bỏ sót một ca bệnh nguy hiểm hơn là báo động nhầm.

Kết quả trên tập kiểm thử:

| Mô hình | Ngưỡng | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Tim mạch | 0,40 | 0,722 | 0,696 | **0,786** | 0,739 | 0,799 |
| Đái tháo đường | 0,35 | 0,786 | 0,648 | **0,852** | 0,736 | 0,875 |
| Tăng huyết áp | 0,35 | 0,900 | 0,789 | **0,924** | 0,851 | 0,954 |

Nhận xét:

- **Tăng huyết áp cho kết quả tốt nhất** (AUC 0,954). Dễ hiểu: nhãn bệnh gắn
  chặt với chính hai đặc trưng huyết áp có trong đầu vào.
- **Đái tháo đường có AUC khá (0,875) nhưng Precision thấp (0,648)** — cứ 10 ca
  bị báo nguy cơ thì khoảng 3–4 ca là báo nhầm. Đây là cái giá phải trả khi hạ
  ngưỡng xuống 0,35 để đạt Recall 0,852.
- **Tim mạch yếu nhất** (AUC 0,799, Accuracy 0,722). Bộ dữ liệu này chỉ có các
  chỉ số rất cơ bản, không đủ thông tin để phân biệt tốt hơn. Không nên trình
  bày con số này như một thành công.

### Tiền xử lý phải giống hệt lúc huấn luyện

Mô hình tim mạch và đái tháo đường được huấn luyện trên dữ liệu đã chuẩn hoá
bằng `StandardScaler`; mô hình tăng huyết áp dùng đơn vị gốc. Do đó khi dự đoán
cho một bệnh nhân mới, hai mô hình đầu bắt buộc phải áp dụng đúng bộ scaler đã
học từ tập huấn luyện.

Đây từng là một lỗi thực sự của hệ thống: scaler không được lưu lại, ứng dụng
đưa thẳng giá trị thô (huyết áp 140, chiều cao 170) vào mô hình, khiến xác suất
gần như không đổi giữa các bệnh nhân — một người 25 tuổi khoẻ mạnh vẫn nhận
khoảng 64% nguy cơ tim mạch. Lỗi đã được sửa: scaler và imputer được dựng lại
đúng quy trình gốc (`src/preprocessing/export_transformers.py`, đối chiếu khớp
tới sai số 1e-16 với tập kiểm thử đã lưu) và được áp dụng trong tầng dự đoán.
Kết quả sau khi sửa:

| Ca thử | Tim mạch | Đái tháo đường | Tăng huyết áp |
|---|---:|---:|---:|
| Nữ 25 tuổi, khoẻ mạnh, HA 110/70 | 12,6% | 3,0% | 0,3% |
| Nam 45 tuổi, HA 132/85 | 51,4% | 39,0% | 14,0% |
| Nam 70 tuổi, HA 185/115, hút thuốc | 76,7% | 82,0% | 90,3% |

Bài kiểm thử `tests/test_prediction_scaling.py` chốt lại hành vi này để lỗi
không tái diễn.

### Phân mức nguy cơ hiển thị

Cần phân biệt **hai loại ngưỡng** trong hệ thống:

1. *Ngưỡng quyết định của mô hình* (0,35–0,40): dùng để quy xác suất thành nhãn
   nhị phân khi đánh giá mô hình.
2. *Ngưỡng phân mức hiển thị* trong Decision Engine: `< 0,30` → THẤP;
   `0,30 – 0,70` → TRUNG BÌNH; `≥ 0,70` → CAO.

Mức nguy cơ tổng thể lấy theo mức cao nhất trong ba bệnh; bệnh ưu tiên là bệnh
có xác suất cao nhất.

---

## 5. Chức năng của ứng dụng

- **Nhập chỉ số** trong một biểu mẫu ba cột (thông tin cơ bản · chỉ số đo được ·
  lối sống và tiền sử), kèm mục "chỉ số chuyên sâu" thu gọn cho các giá trị xét
  nghiệm. Các trường dùng chung giữa ba mô hình (tuổi, giới tính, huyết áp,
  đường huyết) chỉ nhập một lần; BMI tự tính từ chiều cao và cân nặng. Tổng số
  ô nhập giảm từ 30 xuống 15 ô chính. Hệ thống kiểm tra hợp lệ trước khi chạy
  mô hình (ví dụ: huyết áp tâm thu phải cao hơn tâm trương).
- **Kết quả đánh giá**: xác suất từng bệnh, mức nguy cơ, bệnh cần ưu tiên, số
  bệnh ở mức cao/trung bình.
- **Khuyến nghị** theo từng bệnh và theo mức nguy cơ, kèm cảnh báo giới hạn sử
  dụng.
- **Hồ sơ bệnh nhân**: thêm/chọn/xoá bệnh nhân; mỗi lần bấm lưu là một lần đo
  được ghi lại kèm thời điểm.
- **Lịch sử theo dõi**: chỉ số mới nhất và mức thay đổi so với lần trước, biểu
  đồ xu hướng (huyết áp; đường huyết – cân nặng – nhịp tim; nguy cơ dự đoán theo
  thời gian), bảng đầy đủ và xuất CSV.
- **Toàn bộ giao diện bằng tiếng Việt**; các khoá dùng trong logic vẫn giữ
  nguyên tiếng Anh, chỉ dịch ở lớp hiển thị.
- **Trình bày kết quả trực quan**: dải tóm tắt nguy cơ tổng thể và ba thẻ bệnh
  có màu theo mức (xanh / vàng / đỏ) kèm thanh tỉ lệ, thay cho danh sách chữ.

Dữ liệu bệnh nhân lưu trong một tệp SQLite cục bộ (`data/patient_records.db`),
gồm hai bảng: `patients` và `measurements` (quan hệ một–nhiều).

---

## 6. Đánh giá Trustworthy AI

### 6.1 Khả năng giải thích (SHAP)

Đặc trưng có ảnh hưởng lớn nhất tới dự đoán của từng mô hình (giá trị SHAP
tuyệt đối trung bình):

| Mô hình | Đặc trưng đứng đầu | Mean \|SHAP\| | Đặc trưng thứ hai |
|---|---|---:|---|
| Tim mạch | `ap_hi` (huyết áp tâm thu) | 0,1458 | `age` |
| Đái tháo đường | `SkinThickness` | 0,1341 | `Glucose` (0,1179) |
| Tăng huyết áp | `sysBP` (huyết áp tâm thu) | 0,2009 | `diaBP` (0,1220) |

Kết quả phù hợp với y văn ở hai mô hình tim mạch và tăng huyết áp: huyết áp tâm
thu là yếu tố chi phối. Riêng mô hình đái tháo đường, việc `SkinThickness` vượt
lên trên `Glucose` là **một dấu hiệu đáng nghi ngờ** — nhiều khả năng do bộ
Pima có lượng lớn giá trị 0 được điền thay cho dữ liệu thiếu, khiến mô hình học
phải một quy luật giả. Đây là điểm cần kiểm tra lại, không nên diễn giải như
một phát hiện y khoa.

### 6.2 Tính công bằng

Chênh lệch hiệu năng giữa các nhóm (gap = |nhóm cao nhất − nhóm thấp nhất|):

| Phân tích | Accuracy gap | Recall gap |
|---|---:|---:|
| Tăng huyết áp theo giới tính | 0,032 | 0,070 |
| Tăng huyết áp theo nhóm tuổi | 0,042 | 0,104 |
| Đái tháo đường theo nhóm tuổi | 0,179 | **0,304** |

Mô hình tăng huyết áp tương đối đồng đều giữa các nhóm. Ngược lại, **mô hình
đái tháo đường lệch nặng theo nhóm tuổi**: chênh lệch Recall tới 0,304 nghĩa là
có nhóm tuổi bị bỏ sót ca bệnh nhiều hơn hẳn nhóm khác. Với tập kiểm thử chỉ 154
ca, con số này vừa đáng lo vừa kém tin cậy — cần thêm dữ liệu để kết luận.

Lưu ý phương pháp: chênh lệch hiệu năng **chưa đủ** để kết luận có phân biệt đối
xử; nó là tín hiệu cần theo dõi và phân tích tiếp.

### 6.3 Độ bền vững

Thêm nhiễu ngẫu nhiên vào dữ liệu đầu vào và đo mức suy giảm:

| Mức nhiễu | F1 giảm (đái tháo đường) | F1 giảm (tăng huyết áp) |
|---:|---:|---:|
| 1% | 0,000 | 0,006 |
| 5% | 0,016 | 0,012 |
| 10% | 0,006 | 0,019 |

Hai mô hình khá ổn định: nhiễu 10% chỉ làm F1 giảm dưới 0,02. Điều này hợp lý
với bản chất của Random Forest (tập hợp nhiều cây, ít nhạy với dao động nhỏ).

### 6.4 Quyền riêng tư và trách nhiệm giải trình

- Dữ liệu y tế là thông tin nhạy cảm; bộ cardio gốc còn chứa cột định danh trực
  tiếp (`id`) — khi triển khai thật bắt buộc phải loại bỏ.
- Hệ thống nêu rõ mô hình nào phụ trách bệnh nào, mức nguy cơ được phân loại
  tường minh, khuyến nghị tách bạch khỏi chẩn đoán, và quyết định cuối cùng
  thuộc về nhân viên y tế.

---

## 7. Hạn chế (trình bày trung thực)

1. **SHAP là giải thích ở mức mô hình, không phải cho từng bệnh nhân.** Giá trị
   SHAP được tính sẵn ngoại tuyến trên một mẫu của tập kiểm thử; giao diện chỉ
   đọc lại kết quả có sẵn, nên không giải thích được cho ca vừa nhập.
2. **Chưa phân tích công bằng và độ bền vững cho mô hình tim mạch** — mới làm
   trên hai mô hình còn lại.
3. **Mô hình tim mạch có hiệu năng khiêm tốn** (AUC 0,799) do dữ liệu đầu vào
   hạn chế.
4. **Bộ đái tháo đường quá nhỏ** (154 ca kiểm thử), mọi chỉ số trên bộ này có
   khoảng tin cậy rộng.
5. **Dữ liệu bệnh nhân lưu cục bộ, chưa có xác thực người dùng** — phù hợp với
   bản thử nghiệm, chưa đủ điều kiện triển khai thật với dữ liệu y tế thật.
6. **Chưa có giám sát vận hành và nhật ký kiểm toán** sau khi triển khai.
7. **Hệ thống không có trợ lý hội thoại.** Mục hỏi đáp trước đây chỉ dò từ khoá
   và trả về câu trả lời viết sẵn nên đã được gỡ bỏ; AI của đồ án nằm ở ba mô
   hình Random Forest, không phải ở một chatbot.

---

## 8. Hướng phát triển

- Tính SHAP trực tiếp cho từng ca vừa nhập, để giải thích "vì sao ca này bị xếp
  nguy cơ cao".
- Xử lý lại giá trị thiếu của bộ Pima (các số 0 vô lý) rồi huấn luyện lại mô
  hình đái tháo đường, kiểm tra xem `SkinThickness` còn đứng đầu không.
- Bổ sung phân tích công bằng và độ bền vững cho mô hình tim mạch.
- Nếu cần trợ lý hội thoại thật thì nối vào một mô hình ngôn ngữ kèm ràng buộc
  an toàn y tế, thay vì mô phỏng bằng dò từ khoá.
- Chuyển lưu trữ sang máy chủ có xác thực, thêm cảnh báo tự động khi chỉ số của
  bệnh nhân xấu đi qua nhiều lần đo.

---

## 9. Kết luận

Đồ án xây dựng được một hệ thống hoàn chỉnh từ dữ liệu thô tới ứng dụng sử dụng
được: ba mô hình Random Forest với ngưỡng tối ưu theo hướng ưu tiên phát hiện ca
bệnh, một quy trình dự đoán – phân mức – khuyến nghị rõ ràng, giao diện tiếng
Việt và khả năng theo dõi bệnh nhân qua nhiều lần đo.

Mô hình tăng huyết áp đạt chất lượng tốt (AUC 0,954); hai mô hình còn lại ở mức
chấp nhận được cho mục đích sàng lọc nhưng chưa thể dùng cho quyết định lâm
sàng. Các đánh giá Trustworthy AI đã chỉ ra hai vấn đề cụ thể cần xử lý tiếp:
chênh lệch công bằng theo nhóm tuổi ở mô hình đái tháo đường, và dấu hiệu mô
hình học phải quy luật giả từ dữ liệu thiếu.

Quá trình kiểm thử cũng phát hiện và sửa một lỗi nghiêm trọng ở khâu suy luận
(thiếu bước chuẩn hoá dữ liệu), kèm bài kiểm thử chống tái diễn.

Giá trị của đồ án nằm ở chỗ hệ thống được đặt đúng vị trí của nó: **hỗ trợ sàng
lọc, không thay thế chẩn đoán.**

---

## Phụ lục — Cách chạy

```bash
pip install streamlit pandas scikit-learn joblib shap matplotlib
streamlit run src/ui/app.py
```

Kiểm tra khâu tiền xử lý khi dự đoán:

```bash
python tests/test_prediction_scaling.py
```

Ứng dụng dùng mô hình đã huấn luyện sẵn trong `data/models/`, không cần huấn
luyện lại. Tệp cơ sở dữ liệu bệnh nhân được tạo tự động ở lần lưu đầu tiên.

Hai mục "Trí tuệ nhân tạo đáng tin cậy" và "Câu hỏi và câu trả lời bảo vệ" đang
được ẩn bằng hai cờ ở đầu `src/ui/app.py`; đổi thành `True` để hiện lại khi cần
trình bày.

---

*Toàn bộ số liệu trong báo cáo được tính lại trực tiếp từ mô hình và tập kiểm
thử trong repo, không phải số ước lượng.*
