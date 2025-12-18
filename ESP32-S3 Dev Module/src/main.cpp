#include <Arduino.h>
#include "pulsecounting.h"

#define SENSOR_GPIO     40
#define SAMPLE_TIME_MS  200
#define N_WINDOWS_AVG   10

// Regressão feita em pF: C_est[pF] = m * C_nom[pF] + c
const double CAL_M = 0.9756537206669901;   // m
const double CAL_C = 16.799370297444092;   // c [pF]

// Resistências reais do oscilador (ohm)
const double RA = 500000.0;
const double RB = 1000000.0;

void setup()
{
    Serial.begin(115200);
    delay(1000);

    pcnt_init(SENSOR_GPIO);

    Serial.println(
        "sample_index,elapsed_ms,"
        "f_meas_Hz,f_from_Craw_Hz,f_from_Ccal_Hz,"
        "C_raw_pF,C_cal_pF,"
        "Tgate_s,df_Hz,"
        "dCraw_pF_per_df,dCcal_pF_per_df,"
        "N_counts,bits_eq"
    );
}

void loop()
{
    static uint32_t sampleIndex = 0;
    static uint32_t t0_ms = millis();

    // Reset counter/timer
    if (Serial.available()) {
        char c = Serial.read();
        if (c == 'R' || c == 'r') {
            sampleIndex = 0;
            t0_ms = millis();
            Serial.println("=== RESET ===");
        }
    }

    // -------------------------------------------------
    // 1) Frequência medida (hardware)
    // -------------------------------------------------
    float f_meas = pcnt_measure_frequency_avg(SAMPLE_TIME_MS, N_WINDOWS_AVG);

    uint32_t elapsed_ms = millis() - t0_ms;

    // Tempo efetivo de integração (gate)
    const double Tgate_s = (SAMPLE_TIME_MS * (double)N_WINDOWS_AVG) / 1000.0;

    // Resolução típica em frequência (LSB equivalente)
    const double df_Hz = 1.0 / Tgate_s;

    // Contagens esperadas (pulsos) no gate
    const double N_counts = (f_meas > 0.0f) ? (f_meas * Tgate_s) : 0.0;

    // Bits equivalentes (ENOB-like) da contagem
    const double bits_eq = (N_counts > 0.0) ? (log(N_counts) / log(2.0)) : NAN;

    // Se não houver sinal, evita NaN por divisão por zero no modelo
    if (f_meas < 1.0f) {
        Serial.print(sampleIndex);  Serial.print(',');
        Serial.print(elapsed_ms);   Serial.print(',');
        Serial.print(f_meas, 2);    Serial.print(',');
        Serial.print("nan,nan,nan,nan,");         // f_from_Craw,f_from_Ccal,Craw,Ccal
        Serial.print(Tgate_s, 3);   Serial.print(',');
        Serial.print(df_Hz, 3);     Serial.print(',');
        Serial.print("nan,nan,");                 // dCraw,dCcal
        Serial.print(N_counts, 2);  Serial.print(',');
        Serial.println(bits_eq, 2);

        sampleIndex++;
        delay(50);
        return;
    }

    // -------------------------------------------------
    // 2) Capacitância "raw" pelo modelo teórico do NE555
    // -------------------------------------------------
    double C_raw_F  = pcnt_freq_to_capacitance_F(f_meas, RA, RB);
    double C_raw_pF = C_raw_F * 1e12;

    // -------------------------------------------------
    // 3) Calibração linear (inversa)
    // -------------------------------------------------
    double C_cal_pF = (C_raw_pF - CAL_C) / CAL_M;
    double C_cal_F  = C_cal_pF * 1e-12;

    // -------------------------------------------------
    // 4) Frequência equivalente a cada capacitância
    //     f = 1 / (ln2 * (RA + 2RB) * C)
    // -------------------------------------------------
    double f_from_Craw = NAN;
    double f_from_Ccal = NAN;

    const double K = log(2.0) * (RA + 2.0 * RB);

    if (isfinite(C_raw_F) && C_raw_F > 0.0) {
        f_from_Craw = 1.0 / (K * C_raw_F);
    }
    if (isfinite(C_cal_F) && C_cal_F > 0.0) {
        f_from_Ccal = 1.0 / (K * C_cal_F);
    }

    // -------------------------------------------------
    // 5) Sensibilidade equivalente em pF (por df_Hz)
    //     Como C ~ 1/f  =>  dC ≈ C * (df/f)
    // -------------------------------------------------
    double dCraw_pF_per_df = NAN;
    double dCcal_pF_per_df = NAN;

    if (isfinite(C_raw_pF) && f_meas > 1.0f) {
        dCraw_pF_per_df = fabs(C_raw_pF) * (df_Hz / f_meas);
    }
    if (isfinite(C_cal_pF) && f_meas > 1.0f) {
        dCcal_pF_per_df = fabs(C_cal_pF) * (df_Hz / f_meas);
    }

    // -------------------------------------------------
    // 6) Output CSV
    // -------------------------------------------------
    Serial.print(sampleIndex);        Serial.print(',');
    Serial.print(elapsed_ms);         Serial.print(',');
    Serial.print(f_meas, 2);          Serial.print(',');
    Serial.print(f_from_Craw, 2);     Serial.print(',');
    Serial.print(f_from_Ccal, 2);     Serial.print(',');
    Serial.print(C_raw_pF, 2);        Serial.print(',');
    Serial.print(C_cal_pF, 2);        Serial.print(',');
    Serial.print(Tgate_s, 3);         Serial.print(',');
    Serial.print(df_Hz, 3);           Serial.print(',');
    Serial.print(dCraw_pF_per_df, 4); Serial.print(',');
    Serial.print(dCcal_pF_per_df, 4); Serial.print(',');
    Serial.print(N_counts, 2);        Serial.print(',');
    Serial.println(bits_eq, 2);

    sampleIndex++;
    delay(50);
}
