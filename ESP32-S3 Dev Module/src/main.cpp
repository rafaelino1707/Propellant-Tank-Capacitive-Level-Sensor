#include <Arduino.h>
#include "pulsecounting.h"

#define SENSOR_GPIO     38
#define SAMPLE_TIME_MS  200
#define N_WINDOWS_AVG   10

// Regressão feita em pF: C_est[pF] = m * C_nom[pF] + c
const double CAL_M = 0.9532902608039984;   // m
const double CAL_C = 20.628045287878635;   // c  [pF]

void setup()
{
    Serial.begin(115200);
    delay(1000);

    Serial.println("PCNT frequency measurement with averaging + C estimation");
    pcnt_init(SENSOR_GPIO);

    Serial.println("sample_index,elapsed_ms,freq_Hz,C_raw_F,C_cal_F");
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
            Serial.println("=== RESET COUNTER AND TIMER ===");
        }
    }

    // 1) Mede frequência, sem fatores manhosos
    float freq_Hz = pcnt_measure_frequency_avg(SAMPLE_TIME_MS, N_WINDOWS_AVG);

    uint32_t now_ms    = millis();
    uint32_t elapsed_ms = now_ms - t0_ms;

    // 2) Capacitância "crua" pelo modelo teórico do NE555 (em Farad)
    double C_raw_F  = pcnt_freq_to_capacitance_F(freq_Hz, 0.5e6, 1e6);

    // 3) Converter para pF
    double C_raw_pF = C_raw_F * 1e12;

    // 4) Aplicar a inversa da regressão:
    //    C_real_pF = (C_est_pF - c)/m
    double C_cal_pF = (C_raw_pF - CAL_C) / CAL_M;

    // 5) Voltar a Farad para mandar na serial
    double C_cal_F  = C_cal_pF * 1e-12;

    Serial.print(sampleIndex);
    Serial.print(',');
    Serial.print(elapsed_ms);
    Serial.print(',');
    Serial.print(freq_Hz, 3);
    Serial.print(',');
    Serial.print(C_raw_F, 12);
    Serial.print(',');
    Serial.println(C_cal_F, 12);

    sampleIndex++;
    delay(50);
}
