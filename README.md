# Pairs Trading with Cointegration & Kalman Filters

##  Descripción General
Este proyecto desarrolla una estrategia de **arbitraje estadístico (Pairs Trading)** aplicando:
- **Cointegración** para detectar relaciones de equilibrio entre activos.
- **Kalman Filters** para ajustar dinámicamente los **hedge ratios** y generar señales.
- **VECM (Vector Error Correction Model)** para modelar el proceso de corrección del spread.

Todo el sistema sigue el marco de **Powell’s Sequential Decision Analysis (SDA)**:
> predict → observe → update → decide → act → learn

---

## Objetivos
1. Identificar pares cointegrados mediante Engle–Granger y Johansen.
2. Implementar dos Kalman Filters (para hedge ratio y para señales).
3. Desarrollar una estrategia *market-neutral* con cobertura dinámica.
4. Evaluar resultados con costos realistas y análisis de desempeño.

---

##  Requisitos Técnicos

### Dependencias
Instala todo con:
```bash
pip install -r requirements.txt
