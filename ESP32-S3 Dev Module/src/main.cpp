#include <Arduino.h>
#include "pulsecounting.h"

#define SENSOR_GPIO     38        // NE555 Out Connect
#define SAMPLE_TIME_MS  200      // 0.2 s por janela
#define N_WINDOWS_AVG   10       // 10 janelas -> 2.0 s efetivos

void setup()
{
    Serial.begin(115200);
    delay(1000);

    Serial.println("PCNT frequency measurement with averaging + C estimation");
    pcnt_init(SENSOR_GPIO);

    Serial.println("sample_index,elapsed_ms,freq_Hz,C_est_F");
}

void loop()
{
    static uint32_t sampleIndex = 0;
    static uint32_t t0_ms = millis();

    // Average Frequency with Outliers Rejection
    float freq_Hz = pcnt_measure_frequency_avg(SAMPLE_TIME_MS, N_WINDOWS_AVG);

    uint32_t now_ms = millis();
    uint32_t elapsed_ms = now_ms - t0_ms;

    // Stimating Capacitance
    double C_est_F = pcnt_freq_to_capacitance_F(freq_Hz, 0.5e6, 1e6);

    // --- Reset command via Serial ---
    if (Serial.available()) {
        char c = Serial.read();
        if (c == 'R' || c == 'r') {
            sampleIndex = 0;
            t0_ms = millis();
            Serial.println("=== RESET COUNTER AND TIMER ===");
        }
    }


    // CSV: sample_index,elapsed_ms,freq_Hz,C_est_F
    Serial.print(sampleIndex);
    Serial.print(",");
    Serial.print(elapsed_ms);
    Serial.print(",");
    Serial.print(freq_Hz, 3);
    Serial.print(",");
    Serial.println(C_est_F, 12);

    sampleIndex++;

    // Extra Break
    delay(50);
}
