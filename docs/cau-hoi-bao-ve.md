# Bộ câu hỏi bảo vệ đồ án và gợi ý trả lời

Trợ lý Sức khỏe AI — sàng lọc nguy cơ tim mạch, đái tháo đường, tăng huyết áp.

> Nguyên tắc chung khi trả lời: **nói số cụ thể**, **thừa nhận hạn chế trước khi
> bị chỉ ra**, và luôn kéo về đúng phạm vi — hỗ trợ sàng lọc, không chẩn đoán.

---

## Số liệu cần thuộc

| Mô hình | Ngưỡng | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Tăng huyết áp | 0,35 | 0,900 | 0,789 | 0,924 | 0,851 | 0,954 |
| Đái tháo đường | 0,35 | 0,786 | 0,648 | 0,852 | 0,736 | 0,875¹ |
| Tim mạch | 0,40 | 0,722 | 0,696 | 0,786 | 0,739 | 0,799 |

¹ AUC thật khoảng 0,806 sau khi loại rò rỉ nhãn — xem câu 2.3.

| Dữ liệu | Tổng | Train / Test | Đặc trưng | Tỉ lệ dương |
|---|---:|---:|---:|---:|
| Tim mạch | 70.000 | 56.000 / 14.000 | 11 | 50,0% |
| Đái tháo đường | 768 | 614 / 154 | 8 | 35,1% |
| Tăng huyết áp | 4.240 | 3.392 / 848 | 12 | 31,0% |

Ngưỡng phân mức hiển thị: `< 0,30` THẤP · `0,30–0,70` TRUNG BÌNH · `≥ 0,70` CAO.

---

## 1. Bài toán và phạm vi

**1.1. Đề tài giải quyết vấn đề gì?**
Ba bệnh mãn tính phổ biến thường tiến triển âm thầm, chỉ phát hiện khi đã có
biến chứng. Hệ thống ước tính nguy cơ mắc ba bệnh từ những chỉ số đo được đơn
giản — huyết áp, đường huyết, cân nặng, tuổi, thói quen — để gợi ý người dùng
đi khám sớm. Nó phục vụ sàng lọc, không phải chẩn đoán.

**1.2. Vì sao không để AI tự chẩn đoán?**
Vì mô hình chỉ học tương quan thống kê trên dữ liệu quá khứ, không khám lâm
sàng, không xét nghiệm, không biết bệnh sử đầy đủ. Hệ thống được thiết kế theo
nguyên tắc human-in-the-loop: AI đưa ra ước tính nguy cơ, con người ra quyết
định y khoa. Điều này thể hiện ngay trong sản phẩm — có cảnh báo ở mỗi lần đánh
giá và không có bất kỳ chức năng nào liên quan tới kê đơn hay điều trị.

**1.3. Ai là người dùng mục tiêu?**
Nhân viên y tế tuyến cơ sở hoặc người dân tự theo dõi sức khỏe định kỳ. Giao
diện được thiết kế đơn giản, tiếng Việt, gom nhóm chỉ số, và các trường trùng
nhau chỉ phải nhập một lần.

**1.4. Sản phẩm khác gì một bảng tra nguy cơ giấy?**
Ba điểm: mô hình học từ dữ liệu thay vì dùng công thức cố định; hệ thống lưu
được nhiều lần đo của cùng một bệnh nhân và vẽ xu hướng theo thời gian; và có
đánh giá Trustworthy AI đi kèm (công bằng, độ bền vững, khả năng giải thích).

**1.5. Nếu hệ thống dự đoán sai thì ai chịu trách nhiệm?**
Hệ thống là công cụ hỗ trợ quyết định, không phải thiết bị y tế. Trách nhiệm y
khoa thuộc về người ra quyết định cuối cùng là nhân viên y tế. Về phía nhóm
phát triển, trách nhiệm là công bố trung thực năng lực và giới hạn của mô hình
— đó là lý do báo cáo nêu rõ cả những chỗ mô hình còn yếu.

---

## 2. Dữ liệu

**2.1. Dữ liệu lấy từ đâu, có bao nhiêu?**
Ba bộ dữ liệu công khai: Cardiovascular Disease (70.000 bản ghi), Pima Indians
Diabetes (768), Framingham Heart Study (4.240). Tách 80/20 với `random_state=42`
và `stratify` theo nhãn, lưu ra CSV cố định để mọi bước sau dùng chung một bản
tách — bảo đảm tái lập được.

**2.2. Vì sao ba bộ dữ liệu chứ không phải một?**
Không có bộ dữ liệu công khai nào chứa đủ nhãn cho cả ba bệnh. Mỗi bệnh dùng bộ
dữ liệu chuyên biệt của nó, rồi hệ thống ánh xạ các chỉ số người dùng nhập sang
đúng đặc trưng của từng mô hình. Đây cũng là một hạn chế — xem câu 2.4.

**2.3. ⚠️ Vì sao SkinThickness lại quan trọng hơn Glucose trong mô hình đái tháo đường?**
*(Câu hỏi khó nhất về dữ liệu — chuẩn bị kỹ.)*
Vì dữ liệu bị **rò rỉ nhãn**. Tệp Pima nhóm em dùng là bản đã điền sẵn giá trị
thiếu, và người điền đã dùng chính nhãn bệnh: `Insulin = 102,5` thì 100% là
không mắc (236 ca), `Insulin = 169,5` thì 100% là mắc (138 ca); `SkinThickness
= 27` chỉ 4,3% mắc, `= 32` tới 85,7% mắc. Nghĩa là mô hình có thể đọc ra nhãn
từ chính giá trị điền. Bước tiền xử lý có lọc hai giá trị Insulin nhưng bỏ sót
SkinThickness.
Nhóm em đã đo mức ảnh hưởng: không lọc gì thì AUC 0,945; chỉ lọc Insulin như
hiện tại 0,875; lọc cả SkinThickness còn **0,806**. Vậy AUC thật khoảng 0,806.
Đây là hạn chế nhóm em tự phát hiện khi phân tích SHAP, và hướng khắc phục là
huấn luyện lại trên Pima gốc, tự điền giá trị thiếu bằng thống kê chỉ tính từ
tập huấn luyện.

**2.4. Dữ liệu nước ngoài thì áp cho người Việt có hợp lý không?**
Không hoàn toàn. Pima là phụ nữ thổ dân Mỹ từ 21 tuổi, Framingham là dân số Mỹ
32–70 tuổi. Phân bố chỉ số và tỉ lệ mắc bệnh ở Việt Nam có thể khác. Vì vậy
nhóm em định vị hệ thống là công cụ sàng lọc tham khảo, và hướng phát triển là
hiệu chỉnh lại trên dữ liệu Việt Nam khi có. Ngoài ra ứng dụng cho nhập tuổi
1–120 trong khi mô hình tăng huyết áp chỉ từng thấy 32–70 tuổi — ngoài khoảng
đó là ngoại suy, nên cần thêm cảnh báo.

**2.5. Dữ liệu có được làm sạch không?**
Có nhưng chưa đủ. Bộ tim mạch còn chứa giá trị vô lý: huyết áp tâm thu từ −150
tới 16.020 mmHg, 228 dòng nằm ngoài khoảng 60–250; 53 dòng chiều cao dưới 120
hoặc trên 220 cm. Bước tiền xử lý mới chỉ bỏ trùng lặp và đổi tuổi từ ngày sang
năm. Đây là một phần lý do mô hình tim mạch yếu nhất, và là việc cần làm tiếp.

**2.6. Có kiểm tra mất cân bằng dữ liệu không?**
Có. Tim mạch cân bằng (50/50), đái tháo đường 35% dương, tăng huyết áp 31%
dương. Mức lệch này chưa nghiêm trọng nên nhóm em không dùng SMOTE, mà xử lý
bằng cách hạ ngưỡng quyết định để ưu tiên Recall — xem câu 3.3.

---

## 3. Mô hình và chỉ số

**3.1. Vì sao chọn Random Forest?**
Ba lý do: dữ liệu dạng bảng với vài chục đặc trưng — đúng địa hình mạnh của mô
hình cây; Random Forest ít nhạy với nhiễu và outlier hơn một cây đơn, phù hợp
dữ liệu y tế; và nó cho ra feature importance cùng SHAP để giải thích được kết
quả, điều rất quan trọng trong y tế.

**3.2. Sao không dùng Deep Learning?**
Với dữ liệu bảng cỡ vài nghìn tới vài chục nghìn dòng, mạng nơ-ron thường không
tốt hơn ensemble cây mà lại khó giải thích và cần nhiều dữ liệu hơn. Trong bài
toán y tế, khả năng giải thích là yêu cầu bắt buộc chứ không phải tuỳ chọn.

**3.3. Vì sao ngưỡng là 0,35 và 0,40 chứ không phải 0,5?**
Vì đây là bài toán sàng lọc: bỏ sót một ca bệnh (âm tính giả) nguy hiểm hơn báo
động nhầm (dương tính giả). Hạ ngưỡng làm Recall tăng — đạt 0,924 với tăng
huyết áp và 0,852 với đái tháo đường. Cái giá phải trả là Precision giảm, ví dụ
đái tháo đường chỉ còn 0,648. Nhóm em chấp nhận đánh đổi đó và nói rõ trong báo
cáo.

**3.4. Precision 0,648 nghĩa là gì? Có chấp nhận được không?**
Nghĩa là trong 10 người bị hệ thống báo nguy cơ, khoảng 3–4 người thực tế không
mắc. Với một công cụ sàng lọc thì hệ quả là những người đó được khuyên đi khám
— chi phí thấp. Nếu đây là công cụ chẩn đoán thì con số này không chấp nhận
được, và đó chính là lý do hệ thống không được định vị như vậy.

**3.5. Có hai loại ngưỡng, khác nhau thế nào?**
Ngưỡng của mô hình (0,35–0,40) dùng để quy xác suất thành nhãn nhị phân khi
đánh giá. Ngưỡng phân mức hiển thị trong Decision Engine (0,30 và 0,70) dùng để
xếp kết quả thành THẤP / TRUNG BÌNH / CAO cho người dùng đọc. Hai cái phục vụ
hai mục đích khác nhau.

**3.6. Xác suất 70% nghĩa là bệnh nhân chắc chắn mắc bệnh 70%?**
Không. Đó là đầu ra của mô hình trên dữ liệu đầu vào, phản ánh "trong dữ liệu
huấn luyện, những người có chỉ số tương tự thì khoảng 70% có nhãn mắc bệnh".
Nó không phải xác suất lâm sàng. Thêm nữa, xác suất của Random Forest chưa được
hiệu chỉnh (calibration), nên con số nên đọc theo hướng so sánh cao/thấp hơn là
theo trị tuyệt đối. Hiệu chỉnh xác suất là một hướng phát triển nhóm em đã ghi
nhận.

**3.7. Vì sao mô hình tăng huyết áp tốt hơn hẳn hai mô hình kia?**
Vì nhãn tăng huyết áp gắn rất chặt với hai đặc trưng huyết áp có sẵn trong đầu
vào — SHAP cũng cho thấy `sysBP` và `diaBP` chi phối. Nói cách khác bài toán đó
dễ hơn. Ngược lại, bộ tim mạch chỉ có các chỉ số rất cơ bản nên AUC chỉ 0,799 —
nhóm em không trình bày con số này như một thành công.

**3.8. Đánh giá trên tập nào? Có bị học vẹt không?**
Toàn bộ chỉ số đo trên tập kiểm thử 20% không tham gia huấn luyện. Scaler và
imputer cũng chỉ fit trên tập huấn luyện rồi transform tập kiểm thử, không nhìn
dữ liệu test. Điểm chưa hoàn hảo là nhóm em dùng một lần chia cố định thay vì
cross-validation, nên với bộ đái tháo đường 154 ca kiểm thử thì sai số còn rộng.

---

## 4. Hệ thống và kỹ thuật

**4.1. Kiến trúc hệ thống gồm những gì?**
Hai luồng tách rời. Luồng huấn luyện chạy ngoại tuyến một lần: dữ liệu thô →
tiền xử lý → huấn luyện và chọn ngưỡng → lưu mô hình ra tệp. Luồng ứng dụng
chạy mỗi lần dùng: người dùng nhập chỉ số → Prediction Layer nạp mô hình và trả
xác suất → Decision Engine phân mức nguy cơ → Recommendation Engine sinh khuyến
nghị → hiển thị. Hai luồng chỉ gặp nhau ở tệp mô hình.

**4.2. Ứng dụng có tự học từ dữ liệu người dùng nhập không?**
Không. Ứng dụng chỉ nạp mô hình có sẵn rồi tính toán, mất vài giây. Muốn mô
hình tốt hơn phải chạy lại luồng huấn luyện và thay tệp mô hình.

**4.3. ⚠️ Nhóm em có gặp lỗi kỹ thuật nào đáng kể không?**
*(Nên chủ động kể — đây là điểm cộng.)*
Có, một lỗi nghiêm trọng gọi là train/serving skew. Mô hình tim mạch và đái
tháo đường được huấn luyện trên dữ liệu đã chuẩn hoá bằng StandardScaler, nhưng
scaler không được lưu lại nên ứng dụng đưa thẳng giá trị thô vào mô hình. Hậu
quả là xác suất gần như đứng yên: một người 25 tuổi khoẻ mạnh vẫn nhận 64,5%
nguy cơ tim mạch.
Cách sửa: dựng lại scaler theo đúng quy trình tiền xử lý gốc, đối chiếu khớp
tới sai số 1e-16 với tập kiểm thử đã lưu, rồi áp dụng trong tầng dự đoán. Sau
khi sửa, người 25 tuổi khoẻ mạnh còn 12,6% và người 70 tuổi nguy cơ cao lên
76,7%. Nhóm em cũng viết một bài kiểm thử tự động để lỗi không tái diễn.

**4.4. Dữ liệu bệnh nhân lưu ở đâu?**
Trong một tệp SQLite cục bộ `data/patient_records.db` với hai bảng `patients`
và `measurements` quan hệ một–nhiều. Mỗi lần bấm lưu là một dòng mới kèm thời
điểm, nhờ đó vẽ được biểu đồ xu hướng. Hạn chế: chưa có xác thực người dùng và
chưa mã hoá, nên chỉ phù hợp bản thử nghiệm.

**4.5. Vì sao chọn Streamlit?**
Vì mục tiêu là chứng minh toàn bộ quy trình chạy được từ đầu tới cuối trong thời
gian có hạn. Streamlit cho phép dựng giao diện bằng Python thuần, không cần tách
frontend/backend. Nếu triển khai thật thì nên tách API riêng để nhiều người dùng
đồng thời và quản lý phiên đăng nhập.

**4.6. Kết quả đánh giá có được lưu tự động không?**
Không, phải bấm nút lưu. Kết quả tạm nằm trong `session_state`, tải lại trang là
mất. Đây là chủ ý: người dùng có thể thử nhiều kịch bản mà không làm bẩn hồ sơ
bệnh nhân.

---

## 5. Trustworthy AI

**5.1. SHAP là gì và dùng để làm gì?**
SHAP đo mức đóng góp của từng đặc trưng vào dự đoán của mô hình, dựa trên lý
thuyết giá trị Shapley trong lý thuyết trò chơi. Nhóm em dùng nó để kiểm tra mô
hình có học đúng thứ nên học không. Kết quả: `sysBP` đứng đầu ở mô hình tăng
huyết áp (0,2009), `ap_hi` đứng đầu ở mô hình tim mạch (0,1458) — phù hợp y văn;
riêng mô hình đái tháo đường thì lệch, và chính SHAP giúp nhóm em phát hiện ra
rò rỉ nhãn ở câu 2.3.

**5.2. SHAP có giải thích cho từng bệnh nhân không?**
Hiện tại thì chưa. Giá trị SHAP được tính ngoại tuyến trên một mẫu 300 dòng của
tập kiểm thử, nên nó mô tả hành vi chung của mô hình. Muốn giải thích cho ca
vừa nhập thì phải tính SHAP trực tiếp lúc dự đoán — đây là hướng phát triển.

**5.3. Đánh giá công bằng thế nào, kết quả ra sao?**
Nhóm em so sánh Accuracy, Precision, Recall, F1 giữa các nhóm giới tính và nhóm
tuổi. Mô hình tăng huyết áp khá đồng đều: chênh lệch Accuracy 0,032 theo giới,
0,042 theo tuổi. Nhưng mô hình đái tháo đường lệch nặng theo nhóm tuổi, chênh
lệch Recall tới 0,304 — nghĩa là có nhóm tuổi bị bỏ sót ca bệnh nhiều hơn hẳn.
Lưu ý về phương pháp: chênh lệch hiệu năng là tín hiệu cần theo dõi, chưa đủ để
kết luận có phân biệt đối xử.

**5.4. Kiểm tra độ bền vững ra sao?**
Thêm nhiễu ngẫu nhiên vào dữ liệu đầu vào ở các mức 1%, 5%, 10% rồi đo mức suy
giảm. Cả hai mô hình đều ổn định: nhiễu 10% chỉ làm F1 giảm dưới 0,02. Điều này
hợp lý vì Random Forest lấy trung bình nhiều cây nên ít nhạy với dao động nhỏ.

**5.5. Vấn đề quyền riêng tư được xử lý thế nào?**
Dữ liệu y tế là thông tin nhạy cảm. Bộ cardio gốc còn chứa cột định danh trực
tiếp `id`, đã được loại khỏi tập đặc trưng. Dữ liệu bệnh nhân trong ứng dụng
lưu cục bộ, không gửi đi đâu. Tuy nhiên chưa có mã hoá và xác thực nên chưa đủ
điều kiện dùng với dữ liệu bệnh nhân thật.

---

## 6. Câu hỏi khó và câu hỏi bẫy

**6.1. "AI" trong đồ án nằm ở đâu?**
Ở ba mô hình Random Forest học từ dữ liệu. Hệ thống không dùng mô hình ngôn ngữ
nào. Trước đây có một mục hỏi đáp nhưng thực chất chỉ dò từ khoá trả về câu trả
lời viết sẵn, nên nhóm em đã gỡ bỏ để tránh gây hiểu nhầm rằng hệ thống có trợ
lý hội thoại.

**6.2. Thầy nhập thử một người trẻ khoẻ mạnh, kết quả có hợp lý không?**
Có. Nữ 25 tuổi, huyết áp 110/70, BMI bình thường cho ra tim mạch 12,6%, đái tháo
đường 3,0%, tăng huyết áp 0,3% — đều mức THẤP. Nam 70 tuổi, huyết áp 185/115,
hút thuốc cho ra 76,7% / 82,0% / 90,3%. Nhóm em có bài kiểm thử tự động chốt
lại đúng hành vi này.

**6.3. Điểm yếu nhất của đồ án là gì?**
Chất lượng dữ liệu. Cụ thể: rò rỉ nhãn ở bộ đái tháo đường khiến AUC báo cáo
cao hơn thực tế, và bộ tim mạch chưa lọc giá trị vô lý nên mô hình chỉ đạt AUC
0,799. Cả hai đều đã được xác định rõ nguyên nhân và có phương án khắc phục cụ
thể trong phần hướng phát triển.

**6.4. Nếu có thêm ba tháng thì em làm gì trước?**
Theo thứ tự: huấn luyện lại mô hình đái tháo đường trên dữ liệu gốc để loại rò
rỉ nhãn; lọc giá trị vô lý của bộ tim mạch rồi huấn luyện lại; hiệu chỉnh xác
suất; rồi mới tới tính SHAP cho từng ca. Ba việc đầu ảnh hưởng trực tiếp tới độ
tin cậy của con số, nên phải làm trước các tính năng mới.

**6.5. Hệ thống này có thể đưa vào dùng thật không?**
Chưa. Cần ít nhất bốn điều kiện: hiệu chỉnh trên dữ liệu Việt Nam, xử lý xong
các vấn đề chất lượng dữ liệu, bổ sung xác thực và mã hoá dữ liệu bệnh nhân, và
có quy trình giám sát sau triển khai. Ở hiện trạng, nó là một prototype nghiên
cứu hoàn chỉnh về mặt quy trình.

**6.6. Làm sao tránh người dùng tự điều trị theo kết quả?**
Sản phẩm không đưa ra chẩn đoán, không kê đơn, không gợi ý thuốc. Khuyến nghị
chỉ dừng ở mức theo dõi chỉ số, ăn uống, vận động và đi khám. Cảnh báo giới hạn
sử dụng hiển thị ở cuối mỗi lần đánh giá và ở thanh bên.

**6.7. Em tự làm phần nào?**
*(Trả lời trung thực theo đúng phần việc của mình. Nếu có dùng công cụ hỗ trợ
thì nói rõ đã dùng vào việc gì và mình kiểm chứng lại thế nào — hội đồng đánh
giá cao sự minh bạch hơn là câu trả lời hoàn hảo.)*

---

## Ba câu nên chủ động nói ra

Nếu có phần tự trình bày, hãy chủ động nêu ba điều này trước khi bị hỏi. Chúng
biến điểm yếu thành bằng chứng về năng lực kỹ thuật:

1. **Phát hiện và sửa lỗi train/serving skew** — mô tả ở câu 4.3, kèm bài kiểm
   thử chống tái diễn.
2. **Phát hiện rò rỉ nhãn nhờ phân tích SHAP** — câu 2.3, kèm con số đo mức ảnh
   hưởng 0,875 → 0,806.
3. **Công bố AUC thật thay vì con số đẹp** — cho thấy nhóm hiểu rằng báo cáo
   trung thực quan trọng hơn kết quả đẹp.
