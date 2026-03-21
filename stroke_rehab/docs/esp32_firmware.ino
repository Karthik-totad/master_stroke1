/*
 * NeuroRehab ESP32 EMG Firmware
 * 
 * Reads EMG sensor from analog pin and streams over USB Serial.
 * Compatible with the NeuroRehab browser game and Python pipeline.
 * 
 * WIRING:
 *   EMG sensor OUT → GPIO 34 (or any ADC pin)
 *   EMG sensor VCC → 3.3V
 *   EMG sensor GND → GND
 * 
 * OUTPUT FORMAT (115200 baud):
 *   EMG:2048
 *   EMG:1956
 *   EMG:2103
 * 
 * FLASH: Arduino IDE → select your ESP32 board → Upload
 */

#define EMG_PIN     34        // ADC pin connected to EMG sensor
#define BAUD_RATE   115200    // must match browser baud selector
#define SAMPLE_DELAY_MS 1     // 1ms = ~1000 Hz sample rate

void setup() {
  Serial.begin(BAUD_RATE);
  analogReadResolution(12);   // 12-bit ADC → values 0-4095
  analogSetAttenuation(ADC_11db);  // full 0-3.3V range
}

void loop() {
  int val = analogRead(EMG_PIN);
  Serial.println("EMG:" + String(val));
  delay(SAMPLE_DELAY_MS);
}
