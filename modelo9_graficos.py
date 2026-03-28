import wfdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, iirnotch, filtfilt, welch, freqz, lfilter

# Configuracion inicial
Fs = 2000 
ruta_base = 'C:/Users/usuario/Desktop/APS/Codigos/Proyecto Final GF APS/DATOS1/' 
resultados_poblacionales = []


#1. Funciones de procesamiento 

def filtrar_emg(senal): #CAdena de procesamiento digital
    b, a = butter(4, [20/(Fs/2), 450/(Fs/2)], btype='bandpass') #Butterworth de 4to orden
    bn, an = iirnotch(50/(Fs/2), 35.0) #Filtro Notch 50 Hz
    limpia = filtfilt(b, a, senal) # Fase cero. 
    return filtfilt(bn, an, limpia)

def calcular_metricas(senal):
    rms = np.sqrt(np.mean(senal**2))
    f, psd = welch(senal, fs=Fs, window='hamming', nperseg=4096) 
    mnf = np.sum(f * psd) / np.sum(psd)
    mdf = f[np.where(np.cumsum(psd) >= np.sum(psd)/2)[0][0]]
    return rms, mnf, mdf, f, psd

# 2. Bloques de graficos

def graficar_bode():
    b_b, a_b = butter(4, [40/(Fs/2), 450/(Fs/2)], btype='bandpass')
    b_n, a_n = iirnotch(50/(Fs/2), 35.0)
    w_b, h_b = freqz(b_b, a_b, worN=8000, fs=Fs)
    w_n, h_n = freqz(b_n, a_n, worN=8000, fs=Fs)
    
    plt.figure(figsize=(10, 5))
    plt.plot(w_b, 20 * np.log10(np.maximum(abs(h_b), 1e-5)), label='Pasa-banda (40-450 Hz)', color='blue', lw=2)
    plt.plot(w_n, 20 * np.log10(np.maximum(abs(h_n), 1e-5)), label='Filtro Notch (50 Hz)', color='red', ls='--')
    plt.title("Grafico 1: Caracterización del Sistema de Filtrado (Bode)")
    plt.xlabel("Frecuencia (Hz)"); plt.ylabel("Ganancia (dB)")
    plt.xlim(0, 500); plt.ylim(-60, 5)
    plt.axhline(-3, color='black', ls=':', alpha=0.5, label='Punto de corte (-3 dB)')
    plt.grid(True, ls='--', alpha=0.5); plt.legend(); plt.tight_layout(); plt.show()

def graficar_analisis_sujeto(emg_total, nombre):
    filtrada = filtrar_emg(emg_total)
    t = np.arange(len(filtrada)) / Fs
    
    # Grafico 2: Envolvente Lineal 
    be, ae = butter(4, 6/(Fs/2), btype='low')
    env = filtfilt(be, ae, np.abs(filtrada))
    
    plt.figure(figsize=(12, 5))
    plt.plot(t, filtrada, color='silver', alpha=0.6, label='EMG Filtrado')
    plt.plot(t, env, color='red', lw=2, label='Envolvente Lineal (6Hz)')
    plt.title(f"Grafico 2: Envolvente de Activación Muscular - {nombre}")
    plt.xlabel("Tiempo (s)"); plt.ylabel("Amplitud (mV)")
    plt.xlim(5, 20); plt.autoscale(enable=True, axis='y', tight=True)
    plt.grid(True, ls='--', alpha=0.5); plt.legend(); plt.tight_layout(); plt.show()

    

def graficar_psd_comparativo(f_i, p_i, f_f, p_f, mdf_i, mdf_f, nombre):
    # Grafico 3: PSD Comparativo con líneas de MDF
    plt.figure(figsize=(10, 5))
    plt.plot(f_i, p_i, label='Inicio (Primeros 30s)', color='forestgreen', lw=1.5)
    plt.plot(f_f, p_f, label='Final (Últimos 30s)', color='crimson', lw=1.5)
    
    # Líneas de Mediana (MDF)
    plt.axvline(mdf_i, color='forestgreen', linestyle='--', lw=2, label=f'MDF Inicio: {round(mdf_i,1)} Hz')
    plt.axvline(mdf_f, color='crimson', linestyle='--', lw=2, label=f'MDF Final: {round(mdf_f,1)} Hz')
    
    plt.fill_between(f_i, p_i, color='forestgreen', alpha=0.1)
    plt.fill_between(f_f, p_f, color='crimson', alpha=0.1)
    
    plt.title(f"Gráfico 3: Densidad Espectral de Potencia (PSD) - {nombre}")
    plt.xlabel("Frecuencia (Hz)"); plt.ylabel("Potencia (V²/Hz)")
    plt.xlim(12, 100); plt.grid(True, ls='--', alpha=0.4); plt.legend(); plt.tight_layout(); plt.show()

# Main principal (Bucle)

graficar_bode()

print("ID    RMS%    MNF%    MDF%")
for i in range(1, 32):
    nombre = f"S{i}"
    try:
        rec = wfdb.rdrecord(ruta_base + nombre) 
        emg = rec.p_signal[:, 0] 
        n = 30 * Fs 
        s_ini = filtrar_emg(emg[:n]) 
        s_fin = filtrar_emg(emg[-n:])
                


        rms_i, mnf_i, mdf_i, f_i, p_i = calcular_metricas(s_ini)
        rms_f, mnf_f, mdf_f, f_f, p_f = calcular_metricas(s_fin)

        #return rms, mnf, mdf, f, psd
        
        #calculo de las variaciones porcentuales
        c_rms = ((rms_f - rms_i) / rms_i) * 100
        c_mnf = ((mnf_f - mnf_i) / mnf_i) * 100
        c_mdf = ((mdf_f - mdf_i) / mdf_i) * 100
        
        if i == 4: # colocar el numero de individuo que se desea analizar 
            graficar_analisis_sujeto(emg, nombre)
            graficar_psd_comparativo(f_i, p_i, f_f, p_f, mdf_i, mdf_f, nombre)
            
        resultados_poblacionales.append([c_rms, c_mnf, c_mdf])
        print(f"{nombre}    {round(c_rms,2)}%    {round(c_mnf,2)}%    {round(c_mdf,2)}%")
            
    except Exception as e:
        print(f"Error procesando {nombre}: {e}")

# Resultados Finales en consola
if resultados_poblacionales:
    promedios = np.mean(resultados_poblacionales, axis=0)
    print("\nPromedios de grupo para 31 individuos\n")
    print(f"RMS: {round(promedios[0], 2)}% | MNF: {round(promedios[1], 2)}% | MDF: {round(promedios[2], 2)}%")