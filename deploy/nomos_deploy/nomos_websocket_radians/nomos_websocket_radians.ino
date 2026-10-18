#include <Arduino.h>
#include <WiFi.h>
#include <WiFiUdp.h>
#include <ArduinoJson.h>
#include <SCServo.h>
#include <Wire.h>
#include <Adafruit_BNO055.h>
#include <esp_system.h>
#include <math.h>

// EDIT THESE THREE SETTINGS.
const char* WIFI_SSID = "TP-Link_AD1E";
const char* WIFI_PASSWORD = "96009932";
IPAddress PC_IP(192, 168, 0, 103);
const uint16_t PC_PORT = 5005;
const uint16_t ROBOT_PORT = 5006;

const uint32_t STATE_PERIOD_MS = 20;       // Target 50 Hz, not a guaranteed rate.
const uint32_t COMMAND_TIMEOUT_MS = 300;
const uint32_t MAX_OBSERVATION_AGE_MS = 150;
const float PADDING_RAD = 0;
const float RAD_PER_TICK = 2.0f * PI / 4096.0f;
const uint16_t MOVE_SPEED = 3400;
const uint8_t MOVE_ACCELERATION = 150;
const size_t PACKET_BYTES = 1400;

struct MotorConfig {
  uint8_t id;
  const char* name;
  int minTicks, maxTicks, zeroTicks, direction;
};
// This exact array order is used in every packet. Uncalibrated shoulders omitted.
const MotorConfig motors[] = {
  {2, "hip_roll_l", 2740, 2348, 2365, -1},
  {3, "hip_yaw_l", 3297, 1728, 2588, -1},
  {4, "hip_pitch_l", 2659, 441, 3679, 1},
  {5, "left_knee", 3601, 849, 2419, -1},
  {6, "left_ankle", 3073, 1221, 2034, -1},
  {11, "hip_roll_r", 1140, 750, 1123, -1},
  {12, "hip_yaw_r", 2274, 816, 1550, -1},
  {13, "hip_pitch_r", 1692, 3937, 710, -1},
  {14, "right_knee", 696, 3457, 1889, 1},
  {15, "right_ankle", 1800, 3685, 2834, 1}
};
const int MOTOR_COUNT = sizeof(motors) / sizeof(motors[0]);

struct MotorState {
  bool valid = false;
  int ticks = 0, speedRaw = 0;
  uint32_t sampledMs = 0;
};
struct ImuState {
  bool detected = false, valid = false;
  float quaternion[4] = {1, 0, 0, 0}; // w, x, y, z
  float gyro[3] = {0, 0, 0};         // radians/second
  float acceleration[3] = {0, 0, 0}; // m/s^2, includes gravity
  float accel[3] = {0, 0, 0};        // normalized acceleration, sensor-frame x/y/z
  uint32_t sampledMs = 0;
};
struct RobotState {
  bool armed = false, torqueEnabled = false;
  uint32_t session = 0, stateSeq = 0, lastCommandSeq = 0;
  uint32_t lastCommandMs = 0, armEpoch = 0;
  bool haveCommandSeq = false;
  const char* status = "startup";
};
struct PendingCommand {
  bool ready = false;
  uint32_t seq = 0, stateSeq = 0;
  int16_t targets[MOTOR_COUNT];
};
struct SentState {
  bool valid = false;
  uint32_t seq = 0, sampledMs = 0;
};
MotorState feedback[MOTOR_COUNT];
ImuState imuState;
RobotState robot;
PendingCommand pending;
SentState sentStates[32];
SMS_STS servo;
Adafruit_BNO055 bno(55, 0x28, &Wire);
WiFiUDP udp;
char packetBuffer[PACKET_BYTES + 1];

struct Diagnostics {
  bool verbose = true;
  uint32_t received = 0, accepted = 0, rejected = 0, applied = 0, sent = 0, sendFailed = 0;
  uint32_t lastRxMs = 0, lastSweepMs = 0, lastRejectPrintMs = 0;
  bool printedReject = false;
  const char* lastRejectReason = "none";
  char lastRejectDetail[220] = "none";
};
Diagnostics diagnostics;

// Print at most one rejection per second, but count every rejection and retain detail.
void rejectCommand(const char* reason, const char* format, ...) {
  diagnostics.rejected++;
  diagnostics.lastRejectReason = reason;
  va_list args;
  va_start(args, format);
  vsnprintf(diagnostics.lastRejectDetail, sizeof(diagnostics.lastRejectDetail), format, args);
  va_end(args);
  if (diagnostics.verbose && (!diagnostics.printedReject || millis()-diagnostics.lastRejectPrintMs >= 1000)) {
    Serial.printf("[REJECT %s] %s\n", reason, diagnostics.lastRejectDetail);
    diagnostics.lastRejectPrintMs = millis();
    diagnostics.printedReject = true;
  }
}

void printStatus() {
  int valid = 0;
  for (int i = 0; i < MOTOR_COUNT; i++) if (feedback[i].valid) valid++;
  Serial.printf("[STATUS] WiFi=%s armed=%d torque_requested=%d joints=%d/%d IMU=%d sweep=%lu ms\n",
                WiFi.status()==WL_CONNECTED ? "ONLINE" : "OFFLINE", robot.armed,
                robot.torqueEnabled, valid, MOTOR_COUNT, imuState.valid,
                (unsigned long)diagnostics.lastSweepMs);
  Serial.printf("[UDP TOTAL] RX=%lu accepted=%lu rejected=%lu applied=%lu TX=%lu TXfail=%lu session=%lu epoch=%lu\n",
                (unsigned long)diagnostics.received, (unsigned long)diagnostics.accepted,
                (unsigned long)diagnostics.rejected, (unsigned long)diagnostics.applied,
                (unsigned long)diagnostics.sent, (unsigned long)diagnostics.sendFailed,
                (unsigned long)robot.session, (unsigned long)robot.armEpoch);
  Serial.printf("[CONTROL] status=%s applied_seq=%s%lu RX_age=%lu ms command_age=%lu ms\n",
                robot.status, robot.haveCommandSeq ? "" : "none/",
                (unsigned long)robot.lastCommandSeq,
                (unsigned long)(millis()-diagnostics.lastRxMs),
                (unsigned long)(millis()-robot.lastCommandMs));
  if (diagnostics.rejected)
    Serial.printf("[LAST REJECT %s] %s\n", diagnostics.lastRejectReason, diagnostics.lastRejectDetail);
  if (!diagnostics.received)
    Serial.println("[HINT] NO UDP COMMANDS RECEIVED: check robot IP/5006 and PC socket/IP/5005.");
  else if (!robot.armed)
    Serial.println("[HINT] DISARMED: type arm locally. Streaming alone cannot enable movement.");
}

bool insideRealLimits(int index, int ticks);
float positionRadians(int index, int ticks);

void printFeedback() {
  Serial.println("ID  JOINT                 TICKS     Q_RAD     SPEED_RAW  AGE_MS  STATUS");
  for (int i = 0; i < MOTOR_COUNT; i++) {
    Serial.printf("%2u  %-18s  %5d  ", motors[i].id, motors[i].name, feedback[i].ticks);
    if (feedback[i].valid && insideRealLimits(i, feedback[i].ticks))
      Serial.printf("%9.5f  ", positionRadians(i, feedback[i].ticks));
    else Serial.print("       --  ");
    Serial.printf("%9d  %6lu  %s\n", feedback[i].speedRaw,
                  (unsigned long)(millis()-feedback[i].sampledMs),
                  !feedback[i].valid ? "NO FEEDBACK" :
                  insideRealLimits(i, feedback[i].ticks) ? "OK" : "OUTSIDE REAL LIMITS");
  }
  Serial.printf("[IMU] detected=%d valid=%d normalized_accel=[%.4f,%.4f,%.4f]\n",
                imuState.detected, imuState.valid, imuState.accel[0], imuState.accel[1], imuState.accel[2]);
}

int wrapTicks(int value) {
  int result = value % 4096;
  return result < 0 ? result + 4096 : result;
}
int fromMin(const MotorConfig& motor, int ticks) {
  return wrapTicks(motor.direction * (ticks - motor.minTicks));
}
int intervalSpan(const MotorConfig& motor) { return fromMin(motor, motor.maxTicks); }
int zeroFromMin(const MotorConfig& motor) { return fromMin(motor, motor.zeroTicks); }
bool insideRealLimits(int index, int ticks) {
  return fromMin(motors[index], ticks) <= intervalSpan(motors[index]);
}
float positionRadians(int index, int ticks) {
  return (fromMin(motors[index], ticks) - zeroFromMin(motors[index])) * RAD_PER_TICK;
}
bool targetTicks(int index, float radians, int16_t& ticks) {
  const MotorConfig& motor = motors[index];
  int span = intervalSpan(motor), zero = zeroFromMin(motor);
  int padding = (int)ceilf(PADDING_RAD / RAD_PER_TICK);
  if (span <= 2 * padding || zero > span || !isfinite(radians)) return false;
  float low = (padding - zero) * RAD_PER_TICK;
  float high = (span - padding - zero) * RAD_PER_TICK;
  if (radians < low || radians > high) return false;
  int target = zero + (int)lroundf(radians / RAD_PER_TICK);
  if (target < padding || target > span - padding) return false;
  ticks = wrapTicks(motor.minTicks + motor.direction * target);
  return true;
}

void stopRobot(const char* reason) {
  robot.armed = false;
  robot.torqueEnabled = false;
  robot.status = reason;
  pending.ready = false;
  servo.EnableTorque(254, 0); // Broadcast torque OFF, including shoulders.
  Serial.printf("STOP: %s. Type arm locally to allow commands again.\n", reason);
}
void checkWatchdog() {
  if (robot.torqueEnabled &&
      (WiFi.status() != WL_CONNECTED || millis() - robot.lastCommandMs > COMMAND_TIMEOUT_MS)) {
    Serial.printf("[WATCHDOG] WiFi=%d last applied command=%lu ms ago (limit=%lu ms).\n",
                  WiFi.status(), (unsigned long)(millis()-robot.lastCommandMs),
                  (unsigned long)COMMAND_TIMEOUT_MS);
    stopRobot("network/command timeout");
  }
}
bool observationFresh(uint32_t seq) {
  const SentState& state = sentStates[seq % 32];
  return state.valid && state.seq == seq && millis() - state.sampledMs <= MAX_OBSERVATION_AGE_MS;
}
bool sequenceNewer(uint32_t a, uint32_t b) { return (int32_t)(a - b) > 0; }

// Bounded receive work: keep the highest valid sequence, never operate servos here.
void receiveCommands() {
  for (int count = 0; count < 8; count++) {
    int size = udp.parsePacket();
    if (!size) break;
    diagnostics.received++;
    diagnostics.lastRxMs = millis();
    if (diagnostics.verbose && diagnostics.received == 1)
      Serial.printf("[FIRST RX] from %s:%u, %d bytes. Expected %s:%u.\n",
                    udp.remoteIP().toString().c_str(), udp.remotePort(), size,
                    PC_IP.toString().c_str(), PC_PORT);
    if (udp.remoteIP() != PC_IP || udp.remotePort() != PC_PORT) {
      rejectCommand("SOURCE", "Got %s:%u; require %s:%u. Bind the sending socket to port 5005.",
                    udp.remoteIP().toString().c_str(), udp.remotePort(), PC_IP.toString().c_str(), PC_PORT);
      udp.flush(); continue;
    }
    if (size > (int)PACKET_BYTES) {
      rejectCommand("SIZE", "Packet %d bytes exceeds %u.", size, (unsigned)PACKET_BYTES);
      udp.flush(); continue;
    }
    int received = udp.read((uint8_t*)packetBuffer, size);
    if (received != size) {
      rejectCommand("READ", "Read %d of %d UDP bytes.", received, size); continue;
    }
    packetBuffer[received] = '\0';
    static bool printedPayload = false;
    if (diagnostics.verbose && !printedPayload) {
      Serial.printf("[FIRST EXPECTED-PEER PAYLOAD] %.220s%s\n", packetBuffer,
                    received > 220 ? " ... (truncated)" : "");
      printedPayload = true;
    }
    JsonDocument doc;
    DeserializationError jsonError = deserializeJson(doc, packetBuffer, received);
    if (jsonError) { rejectCommand("JSON", "%s", jsonError.c_str()); continue; }
    if (!doc["session"].is<uint32_t>()) {
      rejectCommand("SESSION", "Missing/incompatible session: require uint32 copied from state."); continue;
    }
    if (doc["session"].as<uint32_t>() != robot.session) {
      rejectCommand("SESSION", "Got %lu; expected boot session %lu.",
                    (unsigned long)doc["session"].as<uint32_t>(), (unsigned long)robot.session); continue;
    }
    const char* command = doc["cmd"] | "";
    if (!strcmp(command, "stop")) { stopRobot("remote stop"); return; }
    if (strcmp(command, "targets")) {
      rejectCommand("CMD", "Got cmd='%s'; require 'targets' (or 'stop').", command); continue;
    }
    if (!robot.armed) {
      rejectCommand("DISARMED", "Type arm in Serial Monitor. UDP cannot arm the robot."); continue;
    }
    if (!doc["epoch"].is<uint32_t>() || doc["epoch"].as<uint32_t>() != robot.armEpoch) {
      rejectCommand("EPOCH", "Missing/wrong epoch; require current epoch %lu from state.",
                    (unsigned long)robot.armEpoch); continue;
    }
    if (!doc["seq"].is<uint32_t>() || !doc["state_seq"].is<uint32_t>()) {
      rejectCommand("SEQUENCE", "seq and state_seq must both be uint32 JSON numbers."); continue;
    }
    uint32_t seq = doc["seq"], stateSeq = doc["state_seq"];
    if ((robot.haveCommandSeq && !sequenceNewer(seq, robot.lastCommandSeq)) ||
        (pending.ready && !sequenceNewer(seq, pending.seq))) {
      rejectCommand("OLD SEQUENCE", "Got seq=%lu; last applied=%lu pending=%s%lu. Increment command seq.",
                    (unsigned long)seq, (unsigned long)robot.lastCommandSeq,
                    pending.ready ? "" : "none/", (unsigned long)pending.seq); continue;
    }
    if (!observationFresh(stateSeq)) {
      const SentState& state = sentStates[stateSeq % 32];
      if (!state.valid || state.seq != stateSeq)
        rejectCommand("STATE UNKNOWN", "state_seq=%lu not in sent-state history; latest state=%lu.",
                      (unsigned long)stateSeq, (unsigned long)(robot.stateSeq-1));
      else rejectCommand("STATE STALE", "state_seq=%lu age=%lu ms; maximum=%lu ms.",
                         (unsigned long)stateSeq, (unsigned long)(millis()-state.sampledMs),
                         (unsigned long)MAX_OBSERVATION_AGE_MS);
      continue;
    }
    JsonArrayConst values = doc["targets_rad"].as<JsonArrayConst>();
    if (values.size() != MOTOR_COUNT) {
      rejectCommand("TARGET ARRAY", "Require targets_rad array of %d values; got %u. targets_deg is NOT accepted.",
                    MOTOR_COUNT, (unsigned)values.size()); continue;
    }
    int16_t targets[MOTOR_COUNT];
    bool valid = true;
    for (int i = 0; i < MOTOR_COUNT; i++) {
      if (!values[i].is<float>()) {
        rejectCommand("TARGET TYPE", "Index %d / ID %u must be a numeric radian target, not null/string.",
                      i, motors[i].id); valid = false; break;
      }
      float requested = values[i].as<float>();
      if (!targetTicks(i, requested, targets[i])) {
        int padding = (int)ceilf(PADDING_RAD / RAD_PER_TICK);
        int zero = zeroFromMin(motors[i]);
        rejectCommand("TARGET LIMIT", "Index %d ID %u (%s): requested %.6f rad; allow %.6f..%.6f rad. WHOLE packet rejected.",
                      i, motors[i].id, motors[i].name, requested,
                      (padding-zero)*RAD_PER_TICK,
                      (intervalSpan(motors[i])-padding-zero)*RAD_PER_TICK);
        valid = false; break;
      }
    }
    if (!valid) { robot.status = "target rejected: limits/format"; continue; }
    memcpy(pending.targets, targets, sizeof(targets));
    pending.seq = seq; pending.stateSeq = stateSeq; pending.ready = true;
    diagnostics.accepted++;
    if (diagnostics.verbose && diagnostics.accepted == 1)
      Serial.printf("[FIRST ACCEPTED] seq=%lu state_seq=%lu: all 10 radian targets validated.\n",
                    (unsigned long)seq, (unsigned long)stateSeq);
  }
}

void applyCommand() {
  if (!pending.ready || !robot.armed) return;
  if (!observationFresh(pending.stateSeq)) {
    rejectCommand("APPLY STALE", "Pending seq=%lu observation expired before application.", (unsigned long)pending.seq);
    pending.ready = false; return;
  }
  uint8_t ids[MOTOR_COUNT], accelerations[MOTOR_COUNT];
  uint16_t speeds[MOTOR_COUNT];
  for (int i = 0; i < MOTOR_COUNT; i++) {
    if (!feedback[i].valid || millis() - feedback[i].sampledMs > MAX_OBSERVATION_AGE_MS ||
        !insideRealLimits(i, feedback[i].ticks)) {
      Serial.printf("[APPLY FAULT] ID %u valid=%d ticks=%d age=%lu ms inside_real_limits=%d\n",
                    motors[i].id, feedback[i].valid, feedback[i].ticks,
                    (unsigned long)(millis()-feedback[i].sampledMs), insideRealLimits(i, feedback[i].ticks));
      stopRobot("invalid/stale joint feedback"); return;
    }
    int delta = fromMin(motors[i], pending.targets[i]) - fromMin(motors[i], feedback[i].ticks);
    if (abs(delta) >= 2048) {
      rejectCommand("MOVE ROUTE", "ID %u current=%d target=%d delta=%d ticks: >= pi radians; use intermediate targets.",
                    motors[i].id, feedback[i].ticks, pending.targets[i], delta);
      robot.status = "target rejected: use intermediate angle";
      pending.ready = false; return;
    }
    ids[i] = motors[i].id;
    speeds[i] = MOVE_SPEED;
    accelerations[i] = MOVE_ACCELERATION;
  }
  // Startup/arm confirms torque OFF; install all targets before enabling torque.
  servo.SyncWritePosEx(ids, MOTOR_COUNT, pending.targets, speeds, accelerations);
  if (!robot.torqueEnabled) {
    Serial.println("[TORQUE] First full target written. Enabling ten calibrated motors...");
    for (int i = 0; i < MOTOR_COUNT; i++) {
      if (servo.EnableTorque(ids[i], 1) != 1 || servo.Error != 0) {
        Serial.printf("[TORQUE FAILED] ID %u status_error=%u.\n", ids[i], servo.Error);
        stopRobot("torque enable failed"); return;
      }
    }
    robot.torqueEnabled = true;
    Serial.println("[TORQUE] All enable commands acknowledged.");
  }
  diagnostics.applied++;
  static uint32_t lastTargetPrintMs = 0;
  if (diagnostics.verbose && (diagnostics.applied == 1 || millis()-lastTargetPrintMs >= 1000)) {
    Serial.printf("[APPLY] seq=%lu state_seq=%lu ticks:", (unsigned long)pending.seq, (unsigned long)pending.stateSeq);
    for (int i = 0; i < MOTOR_COUNT; i++) Serial.printf(" %u:%d", ids[i], pending.targets[i]);
    Serial.println(" (bus write issued; check feedback for actual movement)");
    lastTargetPrintMs = millis();
  }
  robot.lastCommandSeq = pending.seq;
  robot.haveCommandSeq = true;
  robot.lastCommandMs = millis();
  robot.status = "running";
  pending.ready = false;
}

void sampleSensors() {
  for (int i = 0; i < MOTOR_COUNT; i++) {
    bool wasValid = feedback[i].valid;
    feedback[i].valid = servo.FeedBack(motors[i].id) != -1 && servo.Error == 0;
    if (feedback[i].valid) {
      feedback[i].ticks = servo.ReadPos(-1);
      feedback[i].speedRaw = servo.ReadSpeed(-1);
      feedback[i].sampledMs = millis();
      feedback[i].valid = feedback[i].ticks >= 0 && feedback[i].ticks < 4096;
    }
    if (diagnostics.verbose && feedback[i].valid != wasValid)
      Serial.printf("[MOTOR] ID %u %s, ticks=%d status_error=%u\n", motors[i].id,
                    feedback[i].valid ? "FEEDBACK RESTORED" : "FEEDBACK LOST", feedback[i].ticks, servo.Error);
    if (robot.torqueEnabled && (!feedback[i].valid || !insideRealLimits(i, feedback[i].ticks))) {
      Serial.printf("[MOTOR FAULT] ID %u valid=%d ticks=%d inside_real_limits=%d\n",
                    motors[i].id, feedback[i].valid, feedback[i].ticks, insideRealLimits(i, feedback[i].ticks));
      stopRobot("joint feedback fault");
    }
    receiveCommands();
    checkWatchdog();
  }
  bool imuWasValid = imuState.valid;
  imuState.valid = false;
  if (imuState.detected) {
    imuState.sampledMs = millis();
    imu::Quaternion q = bno.getQuat();
    sensors_event_t gyro, accelData;
    bool accelRead = bno.getEvent(&accelData, Adafruit_BNO055::VECTOR_ACCELEROMETER);
    bool gyroRead = bno.getEvent(&gyro, Adafruit_BNO055::VECTOR_GYROSCOPE);
    imuState.quaternion[0] = q.w(); imuState.quaternion[1] = q.x();
    imuState.quaternion[2] = q.y(); imuState.quaternion[3] = q.z();
    imuState.gyro[0] = gyro.gyro.x; imuState.gyro[1] = gyro.gyro.y; imuState.gyro[2] = gyro.gyro.z;
    imuState.acceleration[0] = accelData.acceleration.x;
    imuState.acceleration[1] = accelData.acceleration.y;
    imuState.acceleration[2] = accelData.acceleration.z;
    float accelNorm = sqrtf(imuState.acceleration[0]*imuState.acceleration[0] +
                            imuState.acceleration[1]*imuState.acceleration[1] +
                            imuState.acceleration[2]*imuState.acceleration[2]);
    bool accelValid = accelRead && isfinite(accelNorm) && accelNorm > 1e-6f;
    for (int axis = 0; axis < 3; axis++) {
      // Clear invalid values instead of retaining a stale normalized vector.
      imuState.accel[axis] = accelValid ? imuState.acceleration[axis] / accelNorm : 0;
    }
    float norm = q.w()*q.w() + q.x()*q.x() + q.y()*q.y() + q.z()*q.z();
    imuState.valid = gyroRead && accelValid && isfinite(norm) && norm > 0.5f && norm < 1.5f;
    for (float value : imuState.gyro) imuState.valid &= isfinite(value);
  }
  if (diagnostics.verbose && imuState.valid != imuWasValid)
    Serial.printf("[IMU] %s\n", imuState.valid ? "VALID" : "INVALID");
  if (robot.torqueEnabled && !imuState.valid) stopRobot("IMU invalid");
}

void sendState(uint32_t cycleStartMs) {
  if (WiFi.status() != WL_CONNECTED) return;
  JsonDocument doc;
  uint32_t seq = robot.stateSeq++;
  doc["type"] = "state";
  doc["session"] = robot.session;
  doc["epoch"] = robot.armEpoch;
  doc["seq"] = seq;
  doc["t_ms"] = cycleStartMs;
  doc["sweep_ms"] = millis() - cycleStartMs;
  doc["armed"] = robot.armed;
  doc["torque_enabled"] = robot.torqueEnabled;
  doc["status"] = robot.status;
  if (robot.haveCommandSeq) doc["applied_seq"] = robot.lastCommandSeq;
  else doc["applied_seq"] = nullptr;
  JsonArray ids = doc["ids"].to<JsonArray>();
  JsonArray positions = doc["q_rad"].to<JsonArray>();
  JsonArray speeds = doc["speed_raw"].to<JsonArray>();
  JsonArray ages = doc["joint_age_ms"].to<JsonArray>();
  for (int i = 0; i < MOTOR_COUNT; i++) {
    ids.add(motors[i].id);
    if (feedback[i].valid) {
      ages.add(millis() - feedback[i].sampledMs);
      speeds.add(feedback[i].speedRaw);
      if (insideRealLimits(i, feedback[i].ticks)) positions.add(positionRadians(i, feedback[i].ticks));
      else positions.add(nullptr);
    } else { ages.add(nullptr); speeds.add(nullptr); positions.add(nullptr); }
  }
  doc["imu_valid"] = imuState.valid;
  if (imuState.valid) {
    doc["imu_age_ms"] = millis() - imuState.sampledMs;
    JsonArray quat = doc["quat_wxyz"].to<JsonArray>();
    JsonArray gyro = doc["gyro_rad_s"].to<JsonArray>();
    JsonArray accelRaw = doc["accel_m_s2"].to<JsonArray>();
    JsonArray accelUnit = doc["accel"].to<JsonArray>();
    for (float value : imuState.quaternion) quat.add(value);
    for (float value : imuState.gyro) gyro.add(value);
    for (float value : imuState.acceleration) accelRaw.add(value);
    for (float value : imuState.accel) accelUnit.add(value);
  }
  size_t length = measureJson(doc);
  if (length > PACKET_BYTES) { stopRobot("state packet too large"); return; }
  serializeJson(doc, packetBuffer, sizeof(packetBuffer));
  udp.beginPacket(PC_IP, PC_PORT);
  udp.write((uint8_t*)packetBuffer, length);
  if (udp.endPacket()) {
    sentStates[seq % 32] = {true, seq, cycleStartMs};
    diagnostics.sent++;
    if (diagnostics.verbose && diagnostics.sent == 1)
      Serial.printf("[FIRST TX] state seq=%lu bytes=%u to %s:%u\n", (unsigned long)seq,
                    (unsigned)length, PC_IP.toString().c_str(), PC_PORT);
  } else diagnostics.sendFailed++;
}

void armRobot() {
  Serial.println("[ARM] Checking Wi-Fi, ten servos, torque OFF, positions and IMU...");
  stopRobot("preparing to arm");
  if (WiFi.status() != WL_CONNECTED) { Serial.println("Cannot arm: Wi-Fi offline."); return; }
  for (const MotorConfig& motor : motors) {
    int mode = servo.readByte(motor.id, SMS_STS_MODE);
    int torque = servo.readByte(motor.id, SMS_STS_TORQUE_ENABLE);
    Serial.printf("[ARM CHECK] ID %u (%s) mode=%d (need 0) torque=%d (need 0) status_error=%u\n",
                  motor.id, motor.name, mode, torque, servo.Error);
    if (mode != 0 || torque != 0 || servo.Error != 0) {
      Serial.printf("Cannot arm: ID %u must respond, be in position mode, and have torque OFF.\n", motor.id);
      return;
    }
  }
  sampleSensors();
  if (!imuState.valid) { Serial.println("[ARM FAILED] IMU invalid."); printFeedback(); return; }
  for (int i = 0; i < MOTOR_COUNT; i++) {
    if (!feedback[i].valid || !insideRealLimits(i, feedback[i].ticks)) {
      Serial.printf("[ARM FAILED] ID %u unavailable or outside real limits.\n", motors[i].id); printFeedback(); return;
    }
  }
  robot.armEpoch++;
  robot.haveCommandSeq = false;
  pending.ready = false;
  robot.armed = true;
  robot.status = "armed; awaiting targets";
  Serial.printf("[ARM OK] session=%lu epoch=%lu. Copy BOTH from fresh state packets.\n",
                (unsigned long)robot.session, (unsigned long)robot.armEpoch);
  Serial.println("[ARM OK] Torque remains OFF until valid targets_rad arrives.");
}

void printLimits() {
  int padding = (int)ceilf(PADDING_RAD / RAD_PER_TICK);
  Serial.println("ID  JOINT                 PADDED_MIN_RAD  PADDED_MAX_RAD");
  for (const MotorConfig& motor : motors) {
    int zero = zeroFromMin(motor);
    Serial.printf("%2u  %-18s  %14.5f  %14.5f\n", motor.id, motor.name,
                  (padding-zero)*RAD_PER_TICK, (intervalSpan(motor)-padding-zero)*RAD_PER_TICK);
  }
}
void readSerial() {
  static String line;
  static bool tooLong = false;
  while (Serial.available()) {
    char ch = Serial.read();
    if (ch == '\n') {
      line.trim();
      if (tooLong) Serial.println("Line too long.");
      else if (line == "arm") armRobot();
      else if (line == "stop") stopRobot("serial stop");
      else if (line == "limits") printLimits();
      else if (line == "status") printStatus();
      else if (line == "fb") printFeedback();
      else if (line == "verbose") { diagnostics.verbose = true; Serial.println("[DEBUG] Verbose ON."); }
      else if (line == "quiet") { diagnostics.verbose = false; Serial.println("[DEBUG] Periodic/repeated logs OFF. Stops still printed."); }
      else if (line.length()) Serial.println("Commands: arm, stop, limits, status, fb, verbose, quiet");
      line = ""; tooLong = false;
    } else if (ch != '\r' && !tooLong) {
      if (line.length() < 32) line += ch; else tooLong = true;
    }
  }
}

void setup() {
  Serial.begin(115200);
  Serial2.begin(1000000, SERIAL_8N1, 16, 17);
  servo.pSerial = &Serial2;
  servo.IOTimeOut = 5; // Library inter-byte timeout, milliseconds; verify on your bus.
  delay(200);
  stopRobot("startup");
  robot.session = esp_random();
  Serial.printf("[BOOT] session=%lu protocol=RADIANS targets_rad[10] q_rad[10]\n", (unsigned long)robot.session);
  Serial.printf("[NETWORK CONFIG] PC=%s:%u robot_listen=%u source port MUST be %u\n",
                PC_IP.toString().c_str(), PC_PORT, ROBOT_PORT, PC_PORT);
  Serial.printf("[TIMING] state_period=%lu ms observation_age_max=%lu ms watchdog=%lu ms servo_timeout=%lu ms\n",
                (unsigned long)STATE_PERIOD_MS, (unsigned long)MAX_OBSERVATION_AGE_MS,
                (unsigned long)COMMAND_TIMEOUT_MS, servo.IOTimeOut);
  Serial.println("[JOINT ORDER] 2,3,4,5,6,11,12,13,14,15. Shoulders excluded.");
  Serial.println("[IMPORTANT] ID 11 zero is OUTSIDE padded limits; an all-zero packet is rejected.");
  Wire.begin(21, 19);
  Wire.setTimeOut(10);
  imuState.detected = bno.begin();
  if (imuState.detected) bno.setExtCrystalUse(true);
  Serial.printf("IMU detected: %s\n", imuState.detected ? "yes" : "no");
  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.println("Connecting Wi-Fi. Serial commands: arm, stop, limits, status, fb, verbose, quiet");
  printLimits();
}

void loop() {
  static bool udpStarted = false;
  static uint32_t lastSampleMs = 0;
  static int previousWiFiStatus = -1;
  int currentWiFiStatus = WiFi.status();
  if (currentWiFiStatus != previousWiFiStatus) {
    Serial.printf("[WIFI] status=%d connected=%d\n", currentWiFiStatus, currentWiFiStatus == WL_CONNECTED);
    previousWiFiStatus = currentWiFiStatus;
  }
  if (WiFi.status() == WL_CONNECTED && !udpStarted) {
    udpStarted = udp.begin(ROBOT_PORT);
    if (udpStarted) {
      Serial.print("Robot IP: "); Serial.println(WiFi.localIP());
      Serial.printf("Robot UDP port: %u; PC port: %u\n", ROBOT_PORT, PC_PORT);
    }
  } else if (WiFi.status() != WL_CONNECTED && udpStarted) {
    udp.stop(); udpStarted = false;
  }
  checkWatchdog();
  readSerial();
  if (udpStarted) receiveCommands();
  checkWatchdog();
  applyCommand();
  uint32_t now = millis();
  if (now - lastSampleMs >= STATE_PERIOD_MS) {
    lastSampleMs = now;
    sampleSensors();
    checkWatchdog();
    diagnostics.lastSweepMs = millis()-now;
    sendState(now);
  }
  static uint32_t lastSummaryMs = 0;
  static uint32_t prevRx = 0, prevTx = 0, prevApplied = 0;
  if (diagnostics.verbose && millis()-lastSummaryMs >= 1000) {
    Serial.printf("[RATE / %lu ms] RX=%lu TX=%lu APPLIED=%lu\n",
                  (unsigned long)(millis()-lastSummaryMs),
                  (unsigned long)(diagnostics.received-prevRx),
                  (unsigned long)(diagnostics.sent-prevTx),
                  (unsigned long)(diagnostics.applied-prevApplied));
    printStatus();
    prevRx = diagnostics.received; prevTx = diagnostics.sent; prevApplied = diagnostics.applied;
    lastSummaryMs = millis();
  }
  delay(1);
}
