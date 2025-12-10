#include "pulsecounting.h"

#include <Arduino.h>
#include "driver/pcnt.h"

// =======================================================
//  CONFIGURAÇÃO PCNT
// =======================================================
#define PCNT_UNIT_USED    PCNT_UNIT_0
#define PCNT_CHANNEL_USED PCNT_CHANNEL_0

// =======================================================
//  CALIBRAÇÃO T = a C + b
//  T [s], C [F]
//  -> C = (T - b)/a
//  SUBSTITUI PELOS VALORES DA TUA REGRESSÃO
// =======================================================
const double CAL_A = 1.23e6;   // [s/F]  EXEMPLO
const double CAL_B = 2.5e-4;   // [s]    EXEMPLO

// =======================================================
//  Inicialização do PCNT
// =======================================================
void pcnt_init(int gpio)
{
    pcnt_config_t pcnt_config = {
        .pulse_gpio_num = gpio,
        .ctrl_gpio_num  = PCNT_PIN_NOT_USED,
        .lctrl_mode     = PCNT_MODE_KEEP,
        .hctrl_mode     = PCNT_MODE_KEEP,
        .pos_mode       = PCNT_COUNT_INC,     // conta apenas na subida
        .neg_mode       = PCNT_COUNT_DIS,
        .counter_h_lim  = 32767,
        .counter_l_lim  = 0,
        .unit           = PCNT_UNIT_USED,
        .channel        = PCNT_CHANNEL_USED
    };

    pinMode(gpio, INPUT_PULLUP);

    pcnt_unit_config(&pcnt_config);
    pcnt_counter_pause(PCNT_UNIT_USED);
    pcnt_counter_clear(PCNT_UNIT_USED);
}

// =======================================================
//  Mede frequência numa janela (Hz)
// =======================================================
float pcnt_measure_frequency_window(uint32_t sample_time_ms, int16_t* out_pulses)
{
    pcnt_counter_clear(PCNT_UNIT_USED);
    pcnt_counter_resume(PCNT_UNIT_USED);

    delay(sample_time_ms);

    pcnt_counter_pause(PCNT_UNIT_USED);

    int16_t pulses = 0;
    pcnt_get_counter_value(PCNT_UNIT_USED, &pulses);

    if (out_pulses) {
        *out_pulses = pulses;
    }

    // Proteção contra divisão por zero
    if (sample_time_ms == 0) return 0.0f;

    float freq = pulses * (1000.0f / (float)sample_time_ms);
    return freq;
}

// =======================================================
//  Mede média de N janelas com rejeição de outliers
// =======================================================
//
// Estratégia:
//  1) mede N janelas → guarda em freq[i]
//  2) calcula média provisória
//  3) rejeita valores fora de ±30 % da média provisória
//  4) calcula média final dos restantes
//
float pcnt_measure_frequency_avg(uint32_t sample_time_ms,
                                 uint8_t n_windows)
{
    if (n_windows == 0) return 0.0f;

    const uint8_t NMAX = 16;
    if (n_windows > NMAX) n_windows = NMAX;

    float freq[NMAX];

    // Mede janelas
    for (uint8_t i = 0; i < n_windows; ++i) {
        freq[i] = pcnt_measure_frequency_window(sample_time_ms, nullptr);
    }

    // Média provisória
    float sum = 0.0f;
    for (uint8_t i = 0; i < n_windows; ++i) sum += freq[i];
    float mean = sum / (float)n_windows;

    // Rejeição simples de outliers (30 %)
    const float THRESH = 0.30f;
    float sum_ok = 0.0f;
    uint8_t n_ok = 0;

    for (uint8_t i = 0; i < n_windows; ++i) {
        if (mean == 0.0f) continue;
        float rel_err = fabsf(freq[i] - mean) / mean;
        if (rel_err <= THRESH) {
            sum_ok += freq[i];
            n_ok++;
        }
    }

    if (n_ok == 0) {
        // Se tudo foi rejeitado, volta à média provisória
        return mean;
    }

    return sum_ok / (float)n_ok;
}

// =======================================================
//  Converte f [Hz] -> C [F] via T = a C + b
// =======================================================
double pcnt_freq_to_capacitance_F(float freq_Hz, float R_A, float R_B)
{
    if (freq_Hz <= 0.0f) return NAN;

    double R = R_A + 2*R_B;   // período em s
    double C = 1/(0.693 * R * freq_Hz);    // F
    return C;
}
