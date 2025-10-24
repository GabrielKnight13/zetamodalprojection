import numpy as np
import matplotlib.pyplot as plt
from mpmath import zetazero
from scipy.signal import butter, filtfilt
from scipy.optimize import minimize

# --- Time vector setup ---
def setup_time_vector(f_Nyquist, T):
    f_s = 2.5 * f_Nyquist
    dt = 1 / f_s
    N = int(T / dt)
    t = np.linspace(0, T, N, endpoint=False)
    return t, dt, f_s, f_Nyquist, T, N

t, dt, f_s, f_Nyquist, T, N = setup_time_vector(50, 250)

# --- Plant parameters ---
m, c, k = 1.0, 0.5, 7.0
omega_n = np.sqrt(k / m)
f_n = omega_n / (2 * np.pi)
u_max = 10.0

# --- Derivative filter ---
f_c = 3 * f_n
alpha = np.exp(-2 * np.pi * f_c * dt)

# --- Zeta modal reference construction ---
start_index = 1
end_index = 15

gammas = np.array([float(zetazero(n).imag) for n in range(start_index, end_index + 1)])
f_modes = gammas / (2 * np.pi)

mask = f_modes <= f_Nyquist

gammas_keep = gammas[mask]
print(gammas_keep)
alpha_decay = 0.02
Phi = np.array([np.exp(-alpha_decay * t) * np.sin(w * t) / w for w in gammas_keep]).T


def synthetic_sensor_modal(t, freqs=[3, 7, 12], decay=0.02):
    signal = sum(np.exp(-decay * t) * np.sin(2 * np.pi * f * t) for f in freqs)
    return signal / np.max(np.abs(signal))  # normalize



r_des = synthetic_sensor_modal(t).reshape(-1, 1)

#r_des = np.zeros_like(t)
#r_des[t >= 1.0] = 1.0
#r_des = r_des.reshape(-1, 1)
a, *_ = np.linalg.lstsq(Phi, r_des, rcond=None)
r_zeta = Phi @ a
r_zeta = r_zeta.flatten()
r_zeta /= np.max(np.abs(r_zeta))
#r_zeta *= 1.0
r_zeta_smooth = r_zeta
# --- Butterworth filtering ---
#b, a_filt = butter(4, 0.2)
#r_zeta_smooth = filtfilt(b, a_filt, r_zeta)

# --- PID simulation function ---
def simulate(r, Kp, Ki, Kd):
    y = np.zeros(N)
    v = np.zeros(N)
    u = np.zeros(N)
    e = np.zeros(N)
    int_e = 0.0
    d_e = 0.0

    for i in range(1, N):
        e[i] = r[i] - y[i - 1]
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

def objective(params):
    Kp, Ki, Kd = params
    y, u, _, _ = simulate(r_zeta_smooth, Kp, Ki, Kd)
    final_val = r_zeta_smooth[-1]
    amp_error = abs(y[-1] - final_val)
    overshoot = max(0.0, np.max(y) - final_val)
    sat_ratio = np.mean(np.abs(u) >= u_max)
    rms = np.sqrt(np.mean((r_zeta_smooth - y)**2))
    return rms + 10.0 * amp_error + 5.0 * overshoot + 2.0 * sat_ratio

# --- PID optimization ---
initial_guess = [1.0, 1.0, 0.1]
bounds = [(0.5, 5), (0.5, 5), (0.1, 2)]
result = minimize(objective, initial_guess, bounds=bounds)
Kp_opt, Ki_opt, Kd_opt = result.x
print("Optimized PID gains:", Kp_opt, Ki_opt, Kd_opt)

# --- Run simulations ---
y_raw, u_raw, _, total_effort_pid = simulate(r_des.flatten(), Kp_opt, Ki_opt, Kd_opt)
y_zeta, u_zeta, _, total_effort = simulate(r_zeta_smooth, Kp_opt, Ki_opt, Kd_opt)

# --- Diagnostics ---
err_raw = r_des.flatten() - y_raw
rms_pid = np.sqrt(np.mean(err_raw**2))

err_zeta = r_zeta_smooth - y_zeta
rms_error = np.sqrt(np.mean(err_zeta**2))
error_decay = np.all(np.abs(err_zeta[-100:]) < 0.05 * np.max(np.abs(r_zeta_smooth)))
effort_ok = np.max(np.abs(u_zeta)) <= u_max
cont_effort = np.max(np.abs(u_zeta))
cont_effort_pid = np.max(np.abs(u_raw))
stable = error_decay and effort_ok

# --- Print results ---
print(f"ω_n = {omega_n:.2f}, ζ = {c / (2 * omega_n):.2f}, k = {k:.2f}")
print(f"RMS Error (Zeta): {rms_error:.4f}")
print(f"RMS Error (PID):  {rms_pid:.4f}")
print(f"Peak Control Effort (Zeta): {cont_effort:.4f}")
print(f"Peak Control Effort (Raw Step): {cont_effort_pid:.2f}")
print(f"Error Decay: {error_decay}")
print(f"System Stable: {stable}")

# --- Plotting ---
plt.figure(figsize=(12, 8))

# Top: Reference
plt.subplot(3, 1, 1)
plt.plot(t, r_zeta_smooth, label='Reference (Zeta)', color='black', linewidth=2)
plt.ylabel('Amplitude')
plt.title('Zeta-Shaped Reference Signal')
plt.grid(True)
plt.legend()

# Middle: Output from Zeta Input
plt.subplot(3, 1, 2)
plt.plot(t, y_zeta, label='Output (Zeta)', color='blue', linestyle='--')
plt.ylabel('Amplitude')
plt.title('System Output (Zeta Tracking via simulate)')
plt.grid(True)
plt.legend()

# Bottom: Output from Raw Step Input
plt.subplot(3, 1, 3)
plt.plot(t, y_raw, label='Output (Raw Step)', color='red', linestyle='--')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.title('System Output (Raw Step Input via simulate)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# --- Modal weights plot ---
plt.figure(figsize=(8, 4))
plt.plot(np.abs(a), marker='o')
plt.title('Modal Weights |a|')
plt.xlabel('Mode Index')
plt.ylabel('Weight Magnitude')
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(t, r_zeta_smooth - y_zeta, label='Tracking Error', color='purple')
plt.axhline(0, color='gray', linestyle='--')
plt.title('Tracking Error Over Time')
plt.xlabel('Time [s]')
plt.ylabel('Error')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

def plot_fft(signal, label, color):
    freqs = np.fft.rfftfreq(N, dt)
    fft_vals = np.abs(np.fft.rfft(signal))
    plt.plot(freqs, fft_vals, label=label, color=color)

plt.figure(figsize=(10, 5))
plot_fft(r_zeta_smooth, 'Reference (Zeta)', 'black')
plot_fft(y_zeta, 'Output (Zeta)', 'blue')
plot_fft(u_zeta, 'Output (Controller)', 'red')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Magnitude')
plt.xlim(0, 50) 
plt.yscale('log')
plt.title('FFT of Reference and System Outputs')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(t, r_des.flatten(), label='Synthetic Sensor Input', color='gray')

plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.title('Tracking Performance')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
