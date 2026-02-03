import wfdb
import numpy as np
from scipy.signal import butter, iirnotch, filtfilt, welch

Fs = 2000
ruta_base = 'DATOS1/' #Colocar direccion de la carpeta con los datos
resultados_poblacionales = []

def filtrar_emg(senal):
    b, a = butter(4, [40/(Fs/2), 450/(Fs/2)], btype='bandpass')
    bn, an = iirnotch(50/(Fs/2), 35.0)
    limpia = filtfilt(b, a, senal)
    limpia = filtfilt(bn, an, limpia)
    return limpia

def calcular_metricas(senal):
    rms = np.sqrt(np.mean(senal**2))
    f, psd = welch(senal, fs=Fs, window='hamming', nperseg=4096)
    mnf = np.sum(f * psd) / np.sum(psd)
    acumulado = np.cumsum(psd)
    mitad = np.sum(psd) / 2
    indice_mdf = np.where(acumulado >= mitad)[0][0]
    mdf = f[indice_mdf]
    return rms, mnf, mdf

print("ID    RMS%    MNF%    MDF%")

for i in range(1, 32):
    nombre = "S" + str(i)
    ruta = ruta_base + nombre
    
    try:
        rec = wfdb.rdrecord(ruta)
        emg = rec.p_signal[:, 0]
        
        n = 30 * Fs
        senal_ini = filtrar_emg(emg[:n])
        senal_fin = filtrar_emg(emg[-n:])
        
        r_ini, n_ini, d_ini = calcular_metricas(senal_ini)
        r_fin, n_fin, d_fin = calcular_metricas(senal_fin)
        
        c_rms = ((r_fin - r_ini) / r_ini) * 100
        c_mnf = ((n_fin - n_ini) / n_ini) * 100
        c_mdf = ((d_fin - d_ini) / d_ini) * 100
        
        resultados_poblacionales.append([c_rms, c_mnf, c_mdf])
        print(nombre, round(c_rms,2), round(c_mnf,2), round(c_mdf,2))
        
    except:
        print("Error procesando", nombre)

if len(resultados_poblacionales) > 0:
    res_matriz = np.array(resultados_poblacionales)
    promedios = np.mean(res_matriz, axis=0)
    print("\n--- RESULTADOS PROMEDIO DEL GRUPO ---")

    print(f"RMS: {round(promedios[0], 2)}% | MNF: {round(promedios[1], 2)}% | MDF: {round(promedios[2], 2)}%")
