# Phương thức triển khai
## 1. Chuẩn hoá dữ liệu âm thanh
~~~
normalized_audio = audio_data.astype(float)
normalized_audio = normalized_audio / np.max(np.abs(normalized_audio))
~~~
Chuẩn hóa biên độ tín hiệu về khoảng [-1, 1] bằng cách chia cho giá trị tuyệt đối lớn nhất  
Chú thích: 
- Tránh tràn số (Overflow) và mất dữ liệu (Underflow)	chuyển từ `int` sang `float` để tránh lỗi làm tròn và sai số.
- Đưa tín hiệu về phạm vi [-1, 1]	giúp tín hiệu có biên độ đồng nhất, bất kể dữ liệu gốc có giá trị thế nào.
- Các thuật toán như FFT, HPS làm việc chính xác hơn khi dữ liệu nằm trong khoảng [-1, 1].

## 2. Windowing sử dụng Hanning
```
window = np.hanning(len(normalized_audio))
windowed_signal = normalized_audio * window
```
Giảm rò rì phổ (spectral leakage)  
Chú thích:
- tín hiệu thu vào là hữu hạn và thường không tuần hoàn, 2 đoạn đầu cuối có thể không khớp, khi FFT lặp lại tín hiệu sẽ bị nhảy đột ngột, gây nhiễu
- windowing làm giảm biên độ tín hiệu ở 2 đầu về gần 0, hạn chế nhiễu

## 3. FFT
~~~
freq_spectrum = np.abs(rfft(windowed_signal))
~~~
Chuyển tín hiệu từ miền thời gian sang miền tần số

## 4. Nội suy
~~~
interpolated_magnitude_spectrum = np.interp(
            np.arange(0, len(freq_spectrum), 1 / NUM_HPS),  # x_new
            np.arange(0, len(freq_spectrum)),  # x_old
            freq_spectrum,  # y_old
        )
~~~
Lấy mẫu lại phổ tần số với độ phân giải cao hơn để phục vụ cho Harmonic Product Spectrum (HPS)  

## 5. Chuẩn hoá phổ đã nội suy
~~~
interpolated_magnitude_spectrum = (
            interpolated_magnitude_spectrum
            / np.linalg.norm(interpolated_magnitude_spectrum, ord=2)
        )
~~~
Chuẩn hoá về phạm vi [0,1] để phục vụ cho HPS

## 6. HPS
~~~
hps_spec = copy.deepcopy(interpolated_magnitude_spectrum)
for i in range(1, NUM_HPS):
    # Get the first segment of hps_spec up to length/harmonic
    a = hps_spec[: int(np.ceil(len(interpolated_magnitude_spectrum) / (i + 1)))]

    # Get every (i+1)th sample from interpolated spectrum for harmonic comparison
    b = interpolated_magnitude_spectrum[:: (i + 1)]

    temp_hps_spec = np.multiply(a, b)

    if not any(temp_hps_spec):
        break
    hps_spec = temp_hps_spec
~~~  

HPS (Harmonic Product Spectrum) là một kỹ thuật được sử dụng để tìm tần số cơ bản của một tín hiệu bằng cách nhân phổ tần số với các phiên bản thu nhỏ của chính nó.  

✅ Mục tiêu là tăng cường tần số cơ bản (fundamental frequency) và giảm nhiễu từ các hài âm(harmonics)/bội âm(overtunes) của nó.  
![markdown](http://sv.mazurka.org.uk/MzHarmonicSpectrum/img/hpsdiagram.png)

### Ví dụ minh hoạ: 

#### 📌 Giả sử có một tín hiệu với phổ tần số ban đầu

| Index (Tần số) | 0  | 1  | 2  | 3  | 4  | 5  | 6  | 7  | 8  | 9  |
|---------------|----|----|----|----|----|----|----|----|----|----|
| Biên độ FFT  | 10 | 50 | 80 | 100| 70 | 40 | 30 | 20 | 10 | 5  |

Giả sử **NUM_HPS = 3**, tức là ta sẽ nhân phổ tần số gốc với 2 phiên bản giảm bậc của nó.

---

##### 🔹 Bước 1: HPS Bậc 2 (i = 1)  

- **Lấy phần đầu của `hps_spec` (chỉ lấy 1/2 đầu mảng):**  

| Index | 0  | 1   | 2  | 3   | 4  |
|-------|----|-----|----|-----|----|
| `a`  | 10 | 50  | 80 | 100 | 70 |

- **Lấy mỗi phần tử thứ 2 trong `interpolated_magnitude_spectrum` để nhân:**  

| Index | 0  | 1  | 2  | 3  | 4  |
|-------|----|----|----|----|----|
| `b`  | 10 | 80 | 70 | 30 | 10 |

- **Nhân từng phần tử:**  

| Index | 0   | 1    | 2   | 3    | 4   |
|-------|-----|------|-----|------|-----|
| `hps_spec` | 100 | 4000 | 5600 | 3000 | 700 |

---

##### 🔹 Bước 2: HPS Bậc 3 (i = 2)  

- **Lấy phần đầu của `hps_spec` (chỉ lấy 1/3 đầu mảng):**  

| Index | 0    | 1     | 2    |
|-------|------|-------|------|
| `a`  | 100  | 4000  | 5600 |

- **Lấy mỗi phần tử thứ 3 trong `interpolated_magnitude_spectrum` để nhân:**  

| Index | 0  | 1    | 2  |
|-------|----|------|----|
| `b`  | 10 | 100  | 30 |

- **Nhân từng phần tử:**  

| Index | 0    | 1      | 2     |
|-------|------|--------|-------|
| `hps_spec` | 1000 | 400000 | 168000 |

---

##### 🔹 Kết quả cuối cùng  
Sau khi thực hiện HPS, giá trị lớn nhất trong `hps_spec` xuất hiện tại **index = 1**
