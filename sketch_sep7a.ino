#include <WiFi.h>

void setup() {
  Serial.begin(115200);
  delay(1500);

  WiFi.mode(WIFI_STA);
  delay(200);

  Serial.println();
  Serial.println("===== WiSense ESP32 Test =====");

  Serial.print("Chip model: ");
  Serial.println(ESP.getChipModel());

  Serial.print("Chip revision: ");
  Serial.println(ESP.getChipRevision());

  Serial.print("Wi-Fi MAC address: ");
  Serial.println(WiFi.macAddress());

  Serial.println("Board test successful!");
}

void loop() {
}