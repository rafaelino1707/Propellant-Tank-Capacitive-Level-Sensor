#pragma once

#include <stdint.h>

// Inicializa o PCNT para medir pulsos no GPIO do sensor
void pcnt_init(int gpio);

// Mede a frequência numa única janela de amostragem (ms).
// Devolve Hz, e opcionalmente escreve o número de pulsos em out_pulses.
float pcnt_measure_frequency_window(uint32_t sample_time_ms, int16_t* out_pulses);

// Mede a frequência média usando N janelas consecutivas.
// Faz uma rejeição simples de outliers (valores 30% acima/abaixo da média provisória).
float pcnt_measure_frequency_avg(uint32_t sample_time_ms,
                                 uint8_t n_windows);

// Converte uma frequência medida (Hz) em capacitância (F), usando T = a C + b.
double pcnt_freq_to_capacitance_F(float freq_Hz, float R_A, float R_B);
