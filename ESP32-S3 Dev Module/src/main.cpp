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
        "C_raw_pF,C_cal_pF"
    );
}

void loop()
{
    static uint32_t sampleIndex = 0;
    static uint32_t t0_ms = millis();

    if (Serial.available()) {
        char c = Serial.read();
        if (c == 'R' || c == 'r') {
            sampleIndex = 0;
            t0_ms = millis();
            Serial.println("=== RESET ===");
        }
    }

    // 1) Frequência medida (hardware)
    float f_meas = pcnt_measure_frequency_avg(
        SAMPLE_TIME_MS,
        N_WINDOWS_AVG
    );

    uint32_t elapsed_ms = millis() - t0_ms;

    // 2) Capacitância "raw" pelo modelo teórico do NE555
    double C_raw_F  = pcnt_freq_to_capacitance_F(f_meas, RA, RB);
    double C_raw_pF = C_raw_F * 1e12;

    // 3) Calibração linear (inversa)
    double C_cal_pF = (C_raw_pF - CAL_C) / CAL_M;
    double C_cal_F  = C_cal_pF * 1e-12;

    // 4) Frequência equivalente a cada capacitância
    double f_from_Craw =
        1.0 / (log(2.0) * (RA + 2.0 * RB) * C_raw_F);

    double f_from_Ccal =
        1.0 / (log(2.0) * (RA + 2.0 * RB) * C_cal_F);

    // 5) Output CSV
    Serial.print(sampleIndex);      Serial.print(',');
    Serial.print(elapsed_ms);       Serial.print(',');
    Serial.print(f_meas, 2);        Serial.print(',');
    Serial.print(f_from_Craw, 2);   Serial.print(',');
    Serial.print(f_from_Ccal, 2);   Serial.print(',');
    Serial.print(C_raw_pF, 2);      Serial.print(',');
    Serial.println(C_cal_pF, 2);

    sampleIndex++;
    delay(50);
}
