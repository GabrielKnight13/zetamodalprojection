import numpy as np
import matplotlib.pyplot as plt
from mpmath import zetazero
from scipy.signal import butter, filtfilt

# --- Simulation parameters ---
T = 50.0
N = 20000
t = np.linspace(0, T, N)
dt = t[1] - t[0]
f_s = 1 / dt
f_Nyquist = f_s / 2
print(f_Nyquist)
# --- Plant parameters ---
m, c, k = 1.0, 0.2, 10.0
omega_n = np.sqrt(k / m)
f_n = omega_n / (2 * np.pi)

# --- PID gains ---
Kp, Ki, Kd = 30.0, 10.0, 35.0

# --- Derivative filter ---
f_c = 3 * f_n
alpha = np.exp(-2 * np.pi * f_c * dt)

# --- Actuator saturation ---
u_max = 2.0

# --- Step input definition ---
A = 1.0
t_step = 1.0
r_des = np.zeros_like(t)
r_des[t >= t_step] = A

# --- Zeta modal setup ---
start_index = 1
end_index = 75
gammas = np.array([float(zetazero(n).imag) for n in range(start_index, end_index + 1)])
f_modes = gammas / (2 * np.pi)
mask = f_modes <= f_Nyquist
gammas_keep = gammas[mask]

# --- Modal basis construction ---
alpha_decay = 0.1  # Set to 0.0 for no damping
Phi = np.array([np.exp(-alpha_decay * t) * np.sin(w * t) / w for w in gammas_keep]).T
#Phi = np.array([np.sin(w * t) / w for w in gammas_keep]).T
if len(gammas_keep) == 0:
    raise ValueError("No zeta modes below Nyquist. Increase N or reduce mode index range.")

if Phi.ndim != 2:
    raise ValueError("Phi must be a 2D array. Check gammas_keep and modal construction.")

# Ensure r_des is 2D
r_des = r_des.reshape(-1, 1)  # Shape becomes (N, 1)
# --- Least-squares projection ---
a, *_ = np.linalg.lstsq(Phi, r_des, rcond=None)
r_zeta = Phi @ a
#window = np.hanning(len(r_zeta)).reshape(-1, 1)
#r_zeta *= window  # Windowing
r_zeta /= np.max(np.abs(r_zeta))   # Normalize
r_zeta *= A                        # Scale to desired amplitude

# --- Butterworth filtering ---
b, a_filt = butter(4, 0.05)  # 5% of Nyquist
r_zeta = r_zeta.flatten()
r_zeta_smooth = filtfilt(b, a_filt, r_zeta)

# --- PID Simulation ---
def simulate(r):
    y = np.zeros(N)
    v = np.zeros(N)
    u = np.zeros(N)
    e = np.zeros(N)
    int_e = 0.0
    d_e = 0.0

    for i in range(1, N):
        e[i] = r[i].item() - y[i - 1].item()

        int_e += e[i] * dt

        d_raw = (e[i] - e[i - 1]) / dt
        d_e = alpha * d_e + (1 - alpha) * d_raw

        u_unsat = Kp * e[i] + Ki * int_e + Kd * d_e
        u[i] = np.clip(u_unsat, -u_max, u_max)

        if abs(u[i]) >= u_max:
            int_e -= e[i] * dt

        a_val = (u[i] - c * v[i - 1] - k * y[i - 1]) / m
        v[i] = v[i - 1] + a_val * dt
        y[i] = y[i - 1] + v[i] * dt

    rms_error = np.sqrt(np.mean((r - y) ** 2))
    total_effort = np.sum(np.abs(u)) * dt
    return y, u, rms_error, total_effort

# --- Run simulations ---
y_raw, u_raw, _, _ = simulate(r_des)
y_zeta, u_zeta, _, _ = simulate(r_zeta_smooth)

# --- Diagnostics ---
err_raw = r_des - y_raw
rms_pid = np.sqrt(np.mean(err_raw**2))

err_zeta = r_zeta_smooth - y_zeta
rms_error = np.sqrt(np.mean(err_zeta**2))
error_decay = np.all(np.abs(err_zeta[-100:]) < 0.05 * np.max(np.abs(r_zeta_smooth)))
effort_ok = np.max(np.abs(u_zeta)) <= u_max
cont_effort = np.max(np.abs(u_zeta))
cont_effort_pid = np.max(np.abs(u_raw))
stable = error_decay and effort_ok

# --- Print results ---
print(f"RMS Error (Zeta): {rms_error:.4f}")
print(f"RMS Error (PID):  {rms_pid:.4f}")
print(f"Peak Control Effort (Zeta): {cont_effort:.7f}")
print(f"Peak Control Effort (Raw Step): {cont_effort_pid:.2f}")
print(f"Error Decay: {error_decay}")
print(f"System Stable: {stable}")

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))

# --- Subplot 1: Reference Signal ---
plt.subplot(3, 1, 1)
plt.plot(t, err_zeta, label='Reference (Zeta)', color='black', linewidth=2)
plt.ylabel('Amplitude')
plt.title('Zeta-Shaped Reference Signal')
plt.grid(True)
plt.legend()

# --- Subplot 2: Output (Zeta) ---
plt.subplot(3, 1, 2)
plt.plot(t, y_zeta, label='Output (Zeta)', color='blue', linestyle='--')
plt.ylabel('Amplitude')
plt.title('System Output (Zeta Tracking)')
plt.grid(True)
plt.legend()

# --- Subplot 3: Output (Raw PID) ---
plt.subplot(3, 1, 3)
plt.plot(t, y_raw, label='Output (Raw PID)', color='red', linestyle='--')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.title('System Output (Raw Step Input)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

plt.plot(np.abs(a), marker='o')
plt.title('Modal Weights |a|')
plt.xlabel('Mode Index')
plt.ylabel('Weight Magnitude')
plt.grid(True)
plt.show()

def plot_fft(signal, label, dt, ax, color):
    N = len(signal)
    freqs = np.fft.rfftfreq(N, dt)
    spectrum = np.abs(np.fft.rfft(signal)) / N
    ax.plot(freqs, spectrum, color, label=label)

# --- FFT plots ---
fig, axs = plt.subplots(1, 1, figsize=(10,8), sharex=True)

plot_fft(r_zeta_smooth, 'Ref (zeta)', dt, axs, 'g')
plot_fft(y_zeta, 'Output (zeta)', dt, axs, 'b')
plot_fft(u_zeta, 'Control (zeta)', dt, axs, 'r')

axs.axvline(f_n, color='g', linestyle=':', label='f_n')
axs.set_title("Zeta-Projected Step Spectra")
axs.set_xlabel("Frequency [Hz]")
axs.set_xlim(0, 5)
axs.set_ylim(0, 0.005) 
axs.set_ylabel("Amplitude")
axs.legend()
axs.grid(True)

plt.tight_layout()
plt.show()