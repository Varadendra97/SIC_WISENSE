#include <WiFi.h>
#include <esp_now.h>
#include <esp_wifi.h>
#include <freertos/FreeRTOS.h>
#include <freertos/queue.h>

#include <algorithm>
#include <math.h>

// ===================== WiSense radio settings =====================
#define WIFI_CHANNEL 6
#define PRINT_EVERY_N_CSI 5  // About 10 printed/processed CSI samples per second

// ===================== Physical LED connections ==================
// GPIO -> 220/330 ohm resistor -> LED long leg; LED short leg -> GND
#define RED_LED_PIN 25
#define YELLOW_LED_PIN 26
#define GREEN_LED_PIN 27
#define ROOM_LIGHT_PIN 32 // GPIO32 -> 330 ohm resistor -> LED anode; cathode -> GND
constexpr uint32_t LIGHT_WATCHDOG_MS = 5000;
constexpr uint32_t ROOM_IDLE_OFF_MS = 15000;
bool roomIdleTimerActive = false;
uint32_t roomIdleSinceMs = 0;
bool roomLightOn = false;
bool roomAutomationEnabled = false;
void setRoomLight(bool on);
uint32_t lastLightCommandMs = 0;
char lightCommand[32] = {};
size_t lightCommandLength = 0;
bool lightCommandOverflow = false;

// Callback only queues CSI. All printing occurs in loop(), preventing
// LIGHT_ACK messages from interleaving with and corrupting CSI_DATA lines.
constexpr size_t MAX_CSI_BYTES = 1024;
struct QueuedCsi {
  uint32_t count;
  int rssi;
  int channel;
  uint16_t length;
  int8_t raw[MAX_CSI_BYTES];
};
QueueHandle_t csiQueue = nullptr;

// ===================== Local detector settings ====================
// These match the current Python dashboard as closely as possible.
constexpr uint8_t TARGET_SUBCARRIERS = 64;
constexpr uint8_t WINDOW_SIZE = 12;
constexpr uint16_t WARMUP_SCORES = 50;
constexpr uint16_t CALIBRATION_SCORES = 60;
constexpr uint8_t REQUIRED_HIGH_READINGS = 2;

constexpr uint32_t GREEN_HOLD_MS = 1200;
constexpr uint32_t RECENT_ACTIVITY_MS = 30000;

struct CsiPacket {
  uint32_t sequence;
  uint32_t senderTime;
  uint8_t pattern[32];
};

uint8_t senderMac[6] = {0};
volatile bool senderKnown = false;
volatile uint32_t receivedPackets = 0;
volatile uint32_t csiPackets = 0;

// Rolling normalized CSI window: 12 packets x 64 subcarriers.
float csiWindow[WINDOW_SIZE][TARGET_SUBCARRIERS] = {};
uint8_t windowWriteIndex = 0;
uint8_t windowCount = 0;

float calibrationScores[CALIBRATION_SCORES] = {};
uint16_t warmupScoreCount = 0;
uint16_t calibrationScoreCount = 0;

volatile bool localCalibrationDone = false;
volatile float localBaseline = 0.0f;
volatile float localThreshold = 0.0f;
volatile float latestLocalScore = 0.0f;
volatile uint8_t consecutiveHighReadings = 0;
volatile uint32_t lastMovementMs = 0;

enum ActivityState {
  STATE_WAITING,
  STATE_CALIBRATING,
  STATE_NO_MOVEMENT,
  STATE_RECENT_MOVEMENT,
  STATE_MOVEMENT
};

float percentileFromSorted(const float *values, int count, float fraction) {
  if (count <= 0) {
    return 0.0f;
  }

  float position = fraction * (count - 1);
  int lower = (int)floorf(position);
  int upper = (int)ceilf(position);

  if (lower == upper) {
    return values[lower];
  }

  float weight = position - lower;
  return values[lower] * (1.0f - weight) + values[upper] * weight;
}

void finishLocalCalibration() {
  float sortedScores[CALIBRATION_SCORES];

  for (int i = 0; i < CALIBRATION_SCORES; i++) {
    sortedScores[i] = calibrationScores[i];
  }

  std::sort(sortedScores, sortedScores + CALIBRATION_SCORES);

  float baseline = percentileFromSorted(
    sortedScores,
    CALIBRATION_SCORES,
    0.50f
  );

  float quietUpper = percentileFromSorted(
    sortedScores,
    CALIBRATION_SCORES,
    0.90f
  );

  float margin = fmaxf(2.0f, baseline * 0.20f);
  float candidate = quietUpper + margin;
  float safetyCap = baseline + fmaxf(8.0f, baseline * 0.60f);

  float threshold = fminf(candidate, safetyCap);
  threshold = fmaxf(threshold, baseline + 2.0f);

  localBaseline = baseline;
  localThreshold = threshold;
  consecutiveHighReadings = 0;
  lastMovementMs = 0;
  localCalibrationDone = true;

  Serial.print("LOCAL DETECTOR READY | Baseline: ");
  Serial.print(localBaseline, 2);
  Serial.print(" | Threshold: ");
  Serial.println(localThreshold, 2);
}

float calculateLocalActivityScore() {
  float totalVariation = 0.0f;
  int usableSubcarriers = 0;

  for (int carrier = 0; carrier < TARGET_SUBCARRIERS; carrier++) {
    bool usable = true;
    float sum = 0.0f;

    for (int packet = 0; packet < WINDOW_SIZE; packet++) {
      float value = csiWindow[packet][carrier];

      if (value <= 0.0f) {
        usable = false;
        break;
      }

      sum += value;
    }

    if (!usable) {
      continue;
    }

    float mean = sum / WINDOW_SIZE;
    float variance = 0.0f;

    for (int packet = 0; packet < WINDOW_SIZE; packet++) {
      float difference = csiWindow[packet][carrier] - mean;
      variance += difference * difference;
    }

    totalVariation += sqrtf(variance / WINDOW_SIZE);
    usableSubcarriers++;
  }

  if (usableSubcarriers < 10) {
    return -1.0f;
  }

  return (totalVariation / usableSubcarriers) * 100.0f;
}

void processCsiLocally(wifi_csi_info_t *info) {
  const int requiredRawValues = TARGET_SUBCARRIERS * 2;

  if (info->len < requiredRawValues) {
    return;
  }

  float amplitudes[TARGET_SUBCARRIERS];
  float validAmplitudes[TARGET_SUBCARRIERS];
  int validCount = 0;

  // ESP32 CSI order is imaginary, real, imaginary, real, ...
  for (int carrier = 0; carrier < TARGET_SUBCARRIERS; carrier++) {
    int rawIndex = carrier * 2;
    float imaginary = (float)info->buf[rawIndex];
    float real = (float)info->buf[rawIndex + 1];
    float amplitude = sqrtf(real * real + imaginary * imaginary);

    amplitudes[carrier] = amplitude;

    if (amplitude > 0.0f) {
      validAmplitudes[validCount++] = amplitude;
    }
  }

  if (validCount < 10) {
    return;
  }

  std::sort(validAmplitudes, validAmplitudes + validCount);

  float packetMedian;

  if (validCount % 2 == 0) {
    packetMedian =
      (validAmplitudes[validCount / 2 - 1] +
       validAmplitudes[validCount / 2]) /
      2.0f;
  } else {
    packetMedian = validAmplitudes[validCount / 2];
  }

  if (packetMedian <= 0.0f) {
    return;
  }

  for (int carrier = 0; carrier < TARGET_SUBCARRIERS; carrier++) {
    if (amplitudes[carrier] > 0.0f) {
      csiWindow[windowWriteIndex][carrier] =
        amplitudes[carrier] / packetMedian;
    } else {
      csiWindow[windowWriteIndex][carrier] = 0.0f;
    }
  }

  windowWriteIndex = (windowWriteIndex + 1) % WINDOW_SIZE;

  if (windowCount < WINDOW_SIZE) {
    windowCount++;
  }

  if (windowCount < WINDOW_SIZE) {
    return;
  }

  float score = calculateLocalActivityScore();

  if (score < 0.0f) {
    return;
  }

  latestLocalScore = score;

  // Ignore the first scores while the rolling window settles.
  if (warmupScoreCount < WARMUP_SCORES) {
    warmupScoreCount++;
    return;
  }

  // Collect quiet-room scores once, then keep the limits fixed.
  if (!localCalibrationDone) {
    calibrationScores[calibrationScoreCount++] = score;

    if (calibrationScoreCount >= CALIBRATION_SCORES) {
      finishLocalCalibration();
    }

    return;
  }

  if (score > localThreshold) {
    if (consecutiveHighReadings < REQUIRED_HIGH_READINGS) {
      consecutiveHighReadings++;
    }

    if (consecutiveHighReadings >= REQUIRED_HIGH_READINGS) {
      lastMovementMs = millis();
    }
  } else {
    consecutiveHighReadings = 0;
  }
}

ActivityState getActivityState(uint32_t now) {
  if (!senderKnown) {
    return STATE_WAITING;
  }

  if (!localCalibrationDone) {
    return STATE_CALIBRATING;
  }

  uint32_t movementTime = lastMovementMs;

  if (movementTime == 0) {
    return STATE_NO_MOVEMENT;
  }

  uint32_t elapsed = now - movementTime;

  if (elapsed <= GREEN_HOLD_MS) {
    return STATE_MOVEMENT;
  }

  if (elapsed <= RECENT_ACTIVITY_MS) {
    return STATE_RECENT_MOVEMENT;
  }

  return STATE_NO_MOVEMENT;
}

const char *stateName(ActivityState state) {
  switch (state) {
    case STATE_WAITING:
      return "WAITING";
    case STATE_CALIBRATING:
      return "CALIBRATING";
    case STATE_MOVEMENT:
      return "GREEN/MOVEMENT";
    case STATE_RECENT_MOVEMENT:
      return "YELLOW/RECENT";
    default:
      return "RED/NO MOVEMENT";
  }
}

// Start the off timer when green movement state ends, including yellow/recent.
// Repeated AUTO:1 heartbeats do not reset this inactivity timer.
void updateRoomAutomation(ActivityState state, uint32_t now) {
  if (!roomAutomationEnabled || state == STATE_WAITING || state == STATE_CALIBRATING) {
    roomIdleTimerActive = false;
    if (roomLightOn) setRoomLight(false);
    return;
  }
  if (state == STATE_MOVEMENT) {
    roomIdleTimerActive = false;
    if (!roomLightOn) setRoomLight(true);
    return;
  }
  if (!roomLightOn) return;
  if (!roomIdleTimerActive) {
    roomIdleTimerActive = true;
    roomIdleSinceMs = now;
  }
  if ((uint32_t)(now - roomIdleSinceMs) >= ROOM_IDLE_OFF_MS) {
    setRoomLight(false);
  }
}

void updatePhysicalLeds(uint32_t now) {
  ActivityState state = getActivityState(now);
  updateRoomAutomation(state, now);

  // Flash yellow while waiting for the sender or calibrating.
  if (state == STATE_WAITING || state == STATE_CALIBRATING) {
    uint32_t blinkPeriod =
      (state == STATE_WAITING) ? 800UL : 300UL;
    bool yellowOn = ((now / blinkPeriod) % 2) == 0;

    digitalWrite(RED_LED_PIN, LOW);
    digitalWrite(YELLOW_LED_PIN, yellowOn ? HIGH : LOW);
    digitalWrite(GREEN_LED_PIN, LOW);
    return;
  }

  digitalWrite(
    RED_LED_PIN,
    state == STATE_NO_MOVEMENT ? HIGH : LOW
  );
  digitalWrite(
    YELLOW_LED_PIN,
    state == STATE_RECENT_MOVEMENT ? HIGH : LOW
  );
  digitalWrite(
    GREEN_LED_PIN,
    state == STATE_MOVEMENT ? HIGH : LOW
  );

}

// Runs whenever an ESP-NOW packet arrives.
void onDataReceived(
  const esp_now_recv_info_t *receiveInfo,
  const uint8_t *data,
  int dataLength
) {
  if (dataLength != sizeof(CsiPacket)) {
    return;
  }

  if (!senderKnown) {
    memcpy(senderMac, receiveInfo->src_addr, 6);
    senderKnown = true;
  }

  receivedPackets++;
}

// Copy CSI out of the Wi-Fi callback; avoid UART and detector work here.
void onCsiReceived(void *context, wifi_csi_info_t *info) {
  if (info == nullptr || info->buf == nullptr || csiQueue == nullptr) return;
  if (!senderKnown || memcmp(info->mac, senderMac, 6) != 0) return;
  uint32_t count = ++csiPackets;
  if (count % PRINT_EVERY_N_CSI != 0 || info->len > MAX_CSI_BYTES) return;
  QueuedCsi packet = {};
  packet.count = count;
  packet.rssi = info->rx_ctrl.rssi;
  packet.channel = info->rx_ctrl.channel;
  packet.length = info->len;
  memcpy(packet.raw, info->buf, info->len);
  // Never block the Wi-Fi task. A full queue drops this sample.
  xQueueSend(csiQueue, &packet, 0);
}

void consumeQueuedCsi() {
  QueuedCsi packet;
  if (csiQueue == nullptr || xQueueReceive(csiQueue, &packet, 0) != pdTRUE) return;
  wifi_csi_info_t local = {};
  local.len = packet.length;
  local.buf = packet.raw;
  processCsiLocally(&local);
  Serial.print("CSI_DATA,");
  Serial.print(packet.count);
  Serial.print(",");
  Serial.print(packet.rssi);
  Serial.print(",");
  Serial.print(packet.channel);
  Serial.print(",");
  Serial.print(packet.length);
  Serial.print(",[");
  for (int i = 0; i < packet.length; i++) {
    Serial.print((int)packet.raw[i]);
    if (i + 1 < packet.length) Serial.print(",");
  }
  Serial.println("]");
}

void setRoomLight(bool on) {
  roomLightOn = on;
  if (!on) roomIdleTimerActive = false;
  digitalWrite(ROOM_LIGHT_PIN, on ? HIGH : LOW);
  Serial.println(on ? "LIGHT_ACK:1" : "LIGHT_ACK:0");
}

void handleLightCommands(uint32_t now) {
  // Bounded, non-blocking parser. No readStringUntil() or 12-second delay().
  for (int processed = 0; processed < 64 && Serial.available() > 0; processed++) {
    char c = (char)Serial.read();
    if (c == '\r') continue;
    if (c == '\n') {
      lightCommand[lightCommandLength] = '\0';
      if (!lightCommandOverflow) {
        if (strcmp(lightCommand, "AUTO:1") == 0) {
          lastLightCommandMs = now;
          roomAutomationEnabled = true;
          Serial.println(roomLightOn ? "LIGHT_ACK:1" : "LIGHT_ACK:0");
        } else if (strcmp(lightCommand, "AUTO:0") == 0 || strcmp(lightCommand, "LIGHT:0") == 0) {
          lastLightCommandMs = now;
          roomAutomationEnabled = false;
          setRoomLight(false);
        }
      }
      lightCommandLength = 0;
      lightCommandOverflow = false;
    } else if (!lightCommandOverflow) {
      if (lightCommandLength < sizeof(lightCommand) - 1) {
        lightCommand[lightCommandLength++] = c;
      } else {
        lightCommandOverflow = true;
      }
    }
  }
  if (roomAutomationEnabled && (uint32_t)(now - lastLightCommandMs) >= LIGHT_WATCHDOG_MS) {
    roomAutomationEnabled = false;
    setRoomLight(false);
  }
}

void setup() {
  pinMode(ROOM_LIGHT_PIN, OUTPUT);
  digitalWrite(ROOM_LIGHT_PIN, LOW);
  pinMode(RED_LED_PIN, OUTPUT);
  pinMode(YELLOW_LED_PIN, OUTPUT);
  pinMode(GREEN_LED_PIN, OUTPUT);

  digitalWrite(RED_LED_PIN, LOW);
  digitalWrite(YELLOW_LED_PIN, LOW);
  digitalWrite(GREEN_LED_PIN, LOW);

  Serial.begin(115200);
  delay(1500);
  csiQueue = xQueueCreate(8, sizeof(QueuedCsi));
  if (csiQueue == nullptr) {
    Serial.println("ERROR: CSI queue allocation failed");
    return;
  }

  Serial.println();
  Serial.println("===== WiSense CSI Receiver + Offline LEDs + Room Light =====");
  Serial.println("Keep the sensing area EMPTY during startup calibration.");

  WiFi.mode(WIFI_STA);
  WiFi.disconnect();
  delay(200);

  Serial.print("Receiver MAC: ");
  Serial.println(WiFi.macAddress());

  esp_wifi_set_ps(WIFI_PS_NONE);
  esp_wifi_set_bandwidth(WIFI_IF_STA, WIFI_BW_HT20);

  if (esp_wifi_set_channel(
        WIFI_CHANNEL,
        WIFI_SECOND_CHAN_NONE
      ) != ESP_OK) {
    Serial.println("ERROR: Could not set Wi-Fi channel!");
    return;
  }

  Serial.print("Wi-Fi channel: ");
  Serial.println(WIFI_CHANNEL);

  if (esp_now_init() != ESP_OK) {
    Serial.println("ERROR: ESP-NOW initialization failed!");
    return;
  }

  if (esp_now_register_recv_cb(onDataReceived) != ESP_OK) {
    Serial.println("ERROR: ESP-NOW callback failed!");
    return;
  }

  if (esp_wifi_set_promiscuous(true) != ESP_OK) {
    Serial.println("ERROR: Promiscuous mode failed!");
    return;
  }

  wifi_csi_config_t csiConfig = {};
  csiConfig.lltf_en = true;
  csiConfig.htltf_en = true;
  csiConfig.stbc_htltf2_en = true;
  csiConfig.ltf_merge_en = true;
  csiConfig.channel_filter_en = true;
  csiConfig.manu_scale = false;
  csiConfig.shift = 0;

  if (esp_wifi_set_csi_config(&csiConfig) != ESP_OK) {
    Serial.println("ERROR: CSI configuration failed!");
    return;
  }

  if (esp_wifi_set_csi_rx_cb(onCsiReceived, nullptr) != ESP_OK) {
    Serial.println("ERROR: CSI callback failed!");
    return;
  }

  if (esp_wifi_set_csi(true) != ESP_OK) {
    Serial.println("ERROR: CSI could not be enabled!");
    return;
  }

  Serial.println("Receiver ready—waiting for sender...");
}

void loop() {
  static unsigned long previousReport = 0;
  static bool senderAnnounced = false;

  uint32_t now = millis();
  handleLightCommands(now);
  consumeQueuedCsi();
  updatePhysicalLeds(millis());

  if (senderKnown && !senderAnnounced) {
    Serial.print("Sender detected: ");

    for (int i = 0; i < 6; i++) {
      if (senderMac[i] < 16) {
        Serial.print("0");
      }

      Serial.print(senderMac[i], HEX);

      if (i < 5) {
        Serial.print(":");
      }
    }

    Serial.println();
    senderAnnounced = true;
  }

  if (now - previousReport >= 2000) {
    ActivityState state = getActivityState(now);

    Serial.print("STATUS | ESP-NOW packets: ");
    Serial.print(receivedPackets);
    Serial.print(" | CSI packets: ");
    Serial.print(csiPackets);
    Serial.print(" | Offline LED: ");
    Serial.print(stateName(state));

    if (localCalibrationDone) {
      Serial.print(" | Score: ");
      Serial.print(latestLocalScore, 2);
      Serial.print(" | Baseline: ");
      Serial.print(localBaseline, 2);
      Serial.print(" | Threshold: ");
      Serial.print(localThreshold, 2);
    } else if (senderKnown) {
      Serial.print(" | Calibration: ");
      Serial.print(calibrationScoreCount);
      Serial.print("/");
      Serial.print(CALIBRATION_SCORES);
    }

    Serial.println();
    previousReport = now;
  }
}

