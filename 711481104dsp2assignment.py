# 1. 匯入必要的函式庫
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

# ==============================================================================
# 1. 全域設定與常數
# ==============================================================================
print("--- 1. 載入全域設定 ---")

# --- 繪圖設定 ---
NUM_POINTS_HALF = 4096 # 頻率響應的計算點數 (只算 0 到 pi)

# --- H(z) 濾波器參數 ---
RADIUS = 0.95 # H2(z) 的半徑

# --- Q3 訊號 x[n] 參數 ---
TOTAL_LENGTH = 300     # 訊號總長度
# 長度
N1 = 60
N2 = 60
N3 = 60

# 振幅
A1 = 1.0  # 0.8pi (第一段)
A2 = 1.0  # 0.2pi (第二段)
A3 = 1.0  # 0.4pi (第三段)

L_silence = 0

# 2. 共同資源定義 (H(z) 與 x[n])
# ==============================================================================
# 在此定義的變數，將被後續所有任務共同使用

# --- 2a. 定義 H(z) = H1(z) * (H2(z))^2 濾波器係數 ---
# 這些係數主要供 Q4 的 lfilter (LCCDE) 使用
print("--- 2a. 正在計算 H(z) 濾波器係數... ---")

# H1(z) 係數
b1_filter = [1, -2 * 0.98 * np.cos(0.8 * np.pi), 0.98**2]
a1_filter = [1, -2 * 0.8 * np.cos(0.4 * np.pi), 0.8**2]

# H2(z) 總係數 (由 k=1~4 串聯)
b2_total_filter = np.array([1.0])
a2_total_filter = np.array([1.0])

# (修正了原始程式碼的縮排錯誤)
for k in range(1, 5):
    ck = RADIUS * np.exp(1j * (.15 * np.pi + .02 * np.pi * k))
    a2k = [1, -2 * np.real(ck), np.abs(ck)**2]
    b2k = np.conj(np.flip(a2k))
    b2_total_filter = np.convolve(b2_total_filter, b2k)
    a2_total_filter = np.convolve(a2_total_filter, a2k)

print("--- 濾波器係數計算完畢 ---")

# --- 2b. 定義 Q3 輸入訊號 x[n] ---
print("--- 2b. 正在產生 Q3 輸入訊號 x[n]... ---")
n1_arr = np.arange(N1)
n2_arr = np.arange(N2)
n3_arr = np.arange(N3)

x1_chirp = A1 * np.sin(0.8 * np.pi * n1_arr) * np.hanning(N1)
x2_chirp = A2 * np.sin(0.2 * np.pi * n2_arr) * np.hanning(N2)
x3_chirp = A3 * np.sin(0.4 * np.pi * n3_arr) * np.hanning(N3)

silence = np.zeros(L_silence)

x_n_unpadded = np.concatenate((x1_chirp, silence, x2_chirp, silence, x3_chirp))
x_n_padded = np.pad(x_n_unpadded, (0, TOTAL_LENGTH - len(x_n_unpadded)), 'constant')
print("--- 輸入訊號 x[n] 產生完畢 ---")

# 3. 執行緒序 (Bonus): 分析 H(z) 的頻率響應 (繪製兩張圖)
#    (採用數值穩定的「響應疊加」法)
# ==============================================================================
print("\n--- 執行 Q1 Q2 ：分析 H(z) 頻率響應 (圖 1 & 2)... ---")

# --- 計算 H1(z) 的響應 ---
w, h1_response = signal.freqz(b1_filter, a1_filter, NUM_POINTS_HALF)
w_gd, gd1_vals = signal.group_delay((b1_filter, a1_filter), NUM_POINTS_HALF)

# --- 計算 H2(z) 的總響應 (透過相加) ---
h2_total_response = np.ones(NUM_POINTS_HALF, dtype=np.complex128)
total_h2_gd = np.zeros(NUM_POINTS_HALF)

for k in range(1, 5):
    ck = RADIUS * np.exp(1j * (.15 * np.pi + .02 * np.pi * k))
    a2k = [1, -2 * np.real(ck), np.abs(ck)**2]
    b2k = np.conj(np.flip(a2k))
    
    _, h2k_response = signal.freqz(b2k, a2k, NUM_POINTS_HALF)
    _, gd2k_vals = signal.group_delay((b2k, a2k), NUM_POINTS_HALF)
    
    h2_total_response *= h2k_response
    total_h2_gd += gd2k_vals

# --- 組合得到最終的結果 (全都在 [0, pi] 範圍內) ---
final_h_response = h1_response * (h2_total_response)**2
final_magnitude = np.abs(final_h_response)
principal_phase = np.angle(final_h_response)
unwrapped_phase = np.unwrap(principal_phase)
final_group_delay = gd1_vals + 2 * total_h2_gd 

# --- 繪製結果 (分成兩張圖) ---
plt.style.use('default')
w_plot = w / np.pi # 將頻率軸正規化

# --- (建議) 增加 Pi 刻度標籤的定義 ---
import numpy as np # 確保 numpy 已匯入
tick_locs = [-np.pi, -0.8*np.pi, -0.6*np.pi, -0.4*np.pi, -0.2*np.pi, 0, 
             0.2*np.pi, 0.4*np.pi, 0.6*np.pi, 0.8*np.pi, np.pi]
tick_labels = ['-π', '-0.8π', '-0.6π', '-0.4π', '-0.2π', '0', 
               '0.2π', '0.4π', '0.6π', '0.8π', 'π']

# --- 第一張圖: 相位 (Phase) ---
fig_phase, axs_phase = plt.subplots(2, 1, figsize=(10, 8)) # 2 行, 1 列
fig_phase.suptitle('Bonus Figure 1: Phase Response of H(z)', fontsize=16)

# 圖 1(a): Principal Value Phase (奇對稱)
axs_phase[0].plot(w, principal_phase) # *** 修改：使用 w ***
current_color = axs_phase[0].lines[0].get_color()
axs_phase[0].plot(-w, -principal_phase, color=current_color) # *** 修改：使用 -w ***
axs_phase[0].set_title('Calculated Principal Value of Phase Response', fontsize=12)
axs_phase[0].set_xlabel('Frequency ω (radians)') # *** 修改：X 軸標籤 ***
axs_phase[0].set_ylabel('ARG[H(e$^{jω}$)] (radians)')
axs_phase[0].set_xlim([-np.pi, np.pi]) # *** 修改：X 軸範圍 ***
axs_phase[0].set_xticks(tick_locs) # *** 新增：設定 X 軸刻度 ***
axs_phase[0].set_xticklabels(tick_labels) # *** 新增：設定 X 軸標籤 ***
axs_phase[0].grid(True, linestyle=':')

# 圖 1(b): Continuous (Unwrapped) Phase (奇對稱)
axs_phase[1].plot(w, unwrapped_phase) # *** 修改：使用 w ***
current_color = axs_phase[1].lines[0].get_color()
axs_phase[1].plot(-w, -unwrapped_phase, color=current_color) # *** 修改：使用 -w ***
axs_phase[1].set_title('Calculated Continuous (Unwrapped) Phase Response', fontsize=12)
axs_phase[1].set_xlabel('Frequency ω (radians)') # *** 修改：X 軸標籤 ***
axs_phase[1].set_ylabel('arg[H(e$^{jω}$)] (radians)')
axs_phase[1].set_xlim([-np.pi, np.pi]) # *** 修改：X 軸範圍 ***
axs_phase[1].set_xticks(tick_locs) # *** 新增：設定 X 軸刻度 ***
axs_phase[1].set_xticklabels(tick_labels) # *** 新增：設定 X 軸標籤 ***
axs_phase[1].grid(True, linestyle=':')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show() # *** 顯示第一張圖 ***

print("--- Q1繪圖完成 ---")

# --- (建議) 確保 Pi 刻度標籤已定義 ---
import numpy as np # 確保 numpy 已匯入
tick_locs = [-np.pi, -0.8*np.pi, -0.6*np.pi, -0.4*np.pi, -0.2*np.pi, 0, 
             0.2*np.pi, 0.4*np.pi, 0.6*np.pi, 0.8*np.pi, np.pi]
tick_labels = ['-π', '-0.8π', '-0.6π', '-0.4π', '-0.2π', '0', 
               '0.2π', '0.4π', '0.6π', '0.8π', 'π']

# --- 第二張圖: 群延遲 (Group Delay) & 幅度 (Magnitude) ---
fig_gm, axs_gm = plt.subplots(2, 1, figsize=(10, 8)) # 2 行, 1 列
fig_gm.suptitle('Bonus Figure 2: Group Delay & Magnitude Response of H(z)', fontsize=16)

# 圖 2(a): Group Delay (偶對稱)
axs_gm[0].plot(w, final_group_delay) # *** 修改：使用 w ***
current_color = axs_gm[0].lines[0].get_color()
axs_gm[0].plot(-w, final_group_delay, color=current_color) # *** 修改：使用 -w ***
axs_gm[0].set_title('Calculated Group Delay', fontsize=12)
axs_gm[0].set_xlabel('Frequency ω (radians)') # *** 修改：X 軸標籤 ***
axs_gm[0].set_ylabel('grd[H(e$^{jω}$)] (samples)')
axs_gm[0].set_xlim([-np.pi, np.pi]) # *** 修改：X 軸範圍 ***
axs_gm[0].set_ylim([-50, 200]) # (Y 軸範圍保持不變)
axs_gm[0].set_xticks(tick_locs) # *** 新增：設定 X 軸刻度 ***
axs_gm[0].set_xticklabels(tick_labels) # *** 新增：設定 X 軸標籤 ***
axs_gm[0].grid(True, linestyle=':')

# 圖 2(b): Magnitude Response (偶對稱)
axs_gm[1].plot(w, final_magnitude) # *** 修改：使用 w ***
current_color = axs_gm[1].lines[0].get_color()
axs_gm[1].plot(-w, final_magnitude, color=current_color) # *** 修改：使用 -w ***
axs_gm[1].set_title('Calculated Magnitude of Frequency Response', fontsize=12)
axs_gm[1].set_xlabel('Frequency ω (radians)') # *** 修改：X 軸標籤 ***
axs_gm[1].set_ylabel('|H(e$^{jω}$)|')
axs_gm[1].set_xlim([-np.pi, np.pi]) # *** 修改：X 軸範圍 ***
axs_gm[1].set_xticks(tick_locs) # *** 新增：設定 X 軸刻度 ***
axs_gm[1].set_xticklabels(tick_labels) # *** 新增：設定 X 軸標籤 ***
axs_gm[1].grid(True, linestyle=':')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show() # *** 顯示第二張圖 ***

print("--- Q2繪圖完成 ---")

# 4. 執行緒序 (Q3): 分析 x[n] 的波形與 DTFT
# ==============================================================================
print("\n--- 執行 Q3：分析 x[n] 並繪圖... ---")

# --- 計算 x[n] 的 DTFT ---
# (我們重用在 Q3 區塊定義的 w_plot, 但 DTFT 需重算)
w_q3, h_q3 = signal.freqz(x_n_padded, [1.0], NUM_POINTS_HALF)
dtft_magnitude = np.abs(h_q3)
w_q3_plot = w_q3 / np.pi # Q3 專用的 w_plot

# --- 繪製 Q3 的圖 (2x1) ---
fig_q3, axs_q3 = plt.subplots(2, 1, figsize=(10, 8)) # 2 行, 1 列
fig_q3.suptitle('Q3: Input Signal x[n] and its DTFT', fontsize=16)

# 圖 (a): 時域波形 x[n]
axs_q3[0].plot(x_n_padded)
axs_q3[0].set_title('(a) Waveform of signal x[n]', fontsize=14)
axs_q3[0].set_xlabel('Sample number (n)', fontsize=12)
axs_q3[0].set_ylabel('Amplitude', fontsize=12)
axs_q3[0].set_xlim([0, TOTAL_LENGTH])
axs_q3[0].grid(True, linestyle=':')

# 圖 (b): 頻域幅度 |X(e^jω)|
axs_q3[1].plot(w_q3_plot, dtft_magnitude)
current_color = axs_q3[1].lines[0].get_color()
axs_q3[1].plot(-w_q3_plot, dtft_magnitude, color=current_color)
axs_q3[1].set_title('(b) Magnitude of DTFT of x[n]', fontsize=14)
axs_q3[1].set_xlabel('Normalized Frequency (ω / π)', fontsize=12)
axs_q3[1].set_ylabel('|X(e$^{jω}$)|', fontsize=12)
axs_q3[1].set_xlim([-1, 1])
axs_q3[1].set_xticks(np.arange(-1.0, 1.1, 0.2))
axs_q3[1].grid(True, linestyle=':')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show() # *** 顯示 Q3 的圖窗 ***

print("--- Q3 繪圖完成 ---")

# 5. 執行緒序 (Q4): 將 x[n] 輸入 LCCDE 並繪製 y[n]
# ==============================================================================
print("\n--- 執行 Q4：執行 LCCDE 濾波並繪製 y[n]... ---")

# --- 執行 LCCDE 濾波 (串聯方式) ---
# 我們重用在 2a 區塊定義的濾波器係數
y1_n = signal.lfilter(b1_filter, a1_filter, x_n_padded)
y2_n = signal.lfilter(b2_total_filter, a2_total_filter, y1_n)
y_n_final = signal.lfilter(b2_total_filter, a2_total_filter, y2_n)

# --- 繪製 Q4 的圖 (1x1) ---
fig_q4, ax_q4 = plt.subplots(1, 1, figsize=(12, 4)) # 1 行, 1 列
fig_q4.suptitle('Q4: Output Signal y[n] after LCCDE Filtering', fontsize=16)

ax_q4.plot(y_n_final)
ax_q4.set_title('Waveform of output signal y[n] (Calculated from Formula)', fontsize=14)
ax_q4.set_xlabel('Sample number (n)', fontsize=12)
ax_q4.set_ylabel('Amplitude', fontsize=12)
ax_q4.set_xlim([0, TOTAL_LENGTH])
ax_q4.set_ylim([-6, 6]) # 根據 Q4 實際輸出調整 Y 軸 (之前是 -2 到 2)
ax_q4.grid(True, linestyle=':')

plt.tight_layout()
plt.show() # *** 顯示 Q4 的圖窗 ***

print("\n--- Q4 繪圖完成 ---")
print("\n--- 所有程式執行完畢 ---")