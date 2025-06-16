#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BME280.h>
#include <MS5611.h>
#include <Adafruit_BNO055.h>
#include <arm_math.h>
#include <TinyGPS++.h>

// Sensor objects
TinyGPSPlus gps;
Adafruit_BME280 bme;
MS5611 ms5611;
Adafruit_BNO055 bno = Adafruit_BNO055(55);
#define SEALEVELPRESSSURE_HPA (1013.25)

float h, alt1, alt2;
double myRealAltitude = 0;
float usedAlt;
float prevPeakEstimate = 0;
float peakSmoothingFactor = 0.9;
unsigned long lastPeakUpdate = 0;
bool peakDetected = false;
unsigned long peakDetectedTime = 0;
float peakAlt = 0;
float sensor_std_dev;

// Unscented KF Configuration
const int n = 2;                            // State dimension (altitude, velocity)
const int sigmaCount = 2 * n + 1;           // Number of sigma points
const float dt = 0.035;                     // Time step
float lambda = 3 - n;                       // Scaling parameter
float alpha = 1e-3, beta = 2, kappa = 0;    // UKF tuning parameters

// State variables
arm_matrix_instance_f32 x;                  // State vector [n x 1]
arm_matrix_instance_f32 P;                  // Covariance [n x n]
arm_matrix_instance_f32 Q;                  // Process noise [n x n]
float R_bme = 1.5;                          // BME measurement noise
float R_ms = 1.5;                          // MS measurement noise

// Sensor fusion
#define VOTING_WINDOW 10
float h_history[VOTING_WINDOW] = {0};
int history_index = 0;

// System state
unsigned long lastStateChange = 0;
const unsigned long stateHoldTime = 100;
unsigned long lastUpdate = 0;
float df = 0;                             // Time difference
float acx = 0, acy = 0, acz = 0;          // Acceleration
float orx = 0, ory = 0, orz = 0;          // Orientation
int state = 1;                            // State machine state
bool parachute1 = false, parachute2 = false;

// ARM Matrix storage
float x_data[2] = {0, 0};
float P_data[4] = {1, 0, 0, 1};
float Q_data[4] = {0.2, 0, 0, 0.2};
float wm[sigmaCount], wc[sigmaCount];     // Weights

float estimatedMaxAltitude = 0;

void setup() {
  Serial.begin(115200);
  while (!Serial);
  Wire.begin();
  Wire.setClock(100000);  // 100 kHz instead of 400 kHz

  Serial.println("I2C Scanner");

  for (byte address = 1; address < 127; address++) {
    Wire.beginTransmission(address);
    if (Wire.endTransmission() == 0) {
      Serial.print("Found I2C device at 0x");
      Serial.println(address, HEX);
    }
    delay(5);
  }
  
  // Initialize sensors
  bno.begin();
  ms5611.begin(); 
  bme.begin(0x76);

  // Initialize ARM matrices
  arm_mat_init_f32(&x, n, 1, x_data);
  arm_mat_init_f32(&P, n, n, P_data);
  arm_mat_init_f32(&Q, n, n, Q_data);

  // Initialize weights
  lambda = alpha * alpha * (n + kappa) - n;
  wm[0] = lambda / (n + lambda);
  wc[0] = wm[0] + (1 - alpha * alpha + beta);
  for (int i = 1; i < sigmaCount; i++) {
    wm[i] = 1 / (2.0 * (n + lambda));
    wc[i] = wm[i];
  }

  // Setup pins
  pinMode(8, OUTPUT);
  pinMode(7, OUTPUT);
  pinMode(5, OUTPUT);
  pinMode(4, OUTPUT);
}

void loop() {
  unsigned long now = millis();
  float elapsed = (now - lastUpdate) / 1000.0;
  if (elapsed < 0.01) return;
  df = elapsed;
  lastUpdate = now;

  // Read sensor data
  readIMUData();
  float alt_bme = getAltBME();
  float alt_ms = getAltMS();
  usedAlt = fuseAltitudes(alt_bme, alt_ms);

  // Unscented KF Prediction
  float Xsig[n][sigmaCount];            // Sigma points
  float Xsig_pred[n][sigmaCount];       // Predicted sigma points
  
  generate_sigma_points(Xsig);
  predict_sigma_points(Xsig, Xsig_pred);
  
  // Convert to ARM matrix format
  arm_matrix_instance_f32 Xsig_mat, Xsig_pred_mat;
  arm_mat_init_f32(&Xsig_mat, n, sigmaCount, (float*)Xsig);
  arm_mat_init_f32(&Xsig_pred_mat, n, sigmaCount, (float*)Xsig_pred);
  
  // Predict mean and covariance
  predict_mean_cov(&Xsig_pred_mat, wm, wc, &Q, &x, &P);
  
  // Update with measurement
  update_with_measurement(usedAlt);

  // State machine
  runStateMachine(x_data[0], alt_bme, alt_ms, x_data[1], acz, orx);

  // Log data
  logSensorData(usedAlt, alt_bme, alt_ms);
}

float safeValue(float v, float fallback) {
  return (isnan(v) || isinf(v)) ? fallback : v;
}

void readIMUData() {
  imu::Vector<3> accel = bno.getVector(Adafruit_BNO055::VECTOR_ACCELEROMETER);
  imu::Vector<3> orient = bno.getVector(Adafruit_BNO055::VECTOR_EULER);
  
  acx = accel.x();
  acy = accel.y();
  acz = accel.z();
  
  orx = orient.x();
  ory = orient.y();
  orz = orient.z();
}

float getAltBME() {
  return safeValue(bme.readAltitude(SEALEVELPRESSSURE_HPA), x_data[0]);
}

float getAltMS() {
  float pressure = ms5611.readPressure();
  if (isnan(pressure) || pressure <= 0) return x_data[0];
  float pressure_hPa = pressure / 50.0f;
  float alt = 44330.0f * (1.0f - pow(pressure_hPa / SEALEVELPRESSSURE_HPA, 0.1903f));
  return safeValue(alt, x_data[0]);
}
/*
float fuseAltitudes(float alt1, float alt2) {
  // Simple fusion with outlier rejection
  if (abs(alt1 - alt2) > 10) {
    return (abs(alt1 - x_data[0]) < abs(alt2 - x_data[0])) ? alt1 : alt2;
  }
  return (alt1 + alt2) / 2.0;
}
*/
// Outlier
float fuseAltitudes(float alt1, float alt2) {
  float diff = abs(alt1 - alt2);
  float threshold = sensor_std_dev * 3.0; // 3σ kuralı

  if (diff > threshold) {
    return (abs(alt1 - x_data[0]) < abs(alt2 - x_data[0])) ? alt1 : alt2;
  }
  return (alt1 + alt2) / 2.0;
}

void generate_sigma_points(float Xsig[n][sigmaCount]) {
  // Calculate matrix square root of P
  float sqrtP[n][n];
  arm_matrix_instance_f32 sqrtP_mat;
  arm_mat_init_f32(&sqrtP_mat, n, n, (float*)sqrtP);
  
  // Compute Cholesky decomposition (P = L*L^T)
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      float val = P_data[i*n + j];
      sqrtP[i][j] = (i == j && val > 0) ? sqrt(val) : 0;
    }
  }
  
  // Scale by (n+lambda)
  float gamma = sqrt(n + lambda);
  arm_mat_scale_f32(&sqrtP_mat, gamma, &sqrtP_mat);
  
  // Generate sigma points
  for (int i = 0; i < n; i++) {
    Xsig[i][0] = x_data[i];  // First point is mean
  }
  
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      Xsig[j][i+1] = x_data[j] + sqrtP[j][i];
      Xsig[j][i+1+n] = x_data[j] - sqrtP[j][i];
    }
  }
}

void predict_sigma_points(float Xsig[n][sigmaCount], float Xsig_pred[n][sigmaCount]) {
  for (int i = 0; i < sigmaCount; i++) {
    float h = Xsig[0][i];
    float v = Xsig[1][i];
    
    // Simple constant velocity model
    Xsig_pred[0][i] = h + v * dt;
    Xsig_pred[1][i] = v;
  }
}

void predict_mean_cov(arm_matrix_instance_f32* Xsig_pred,
                     const float* Wm,
                     const float* Wc,
                     arm_matrix_instance_f32* Q,
                     arm_matrix_instance_f32* x_pred,
                     arm_matrix_instance_f32* P_pred) {
  // 1. Predict mean
  for (int i = 0; i < n; i++) {
    x_pred->pData[i] = 0;
    for (int j = 0; j < sigmaCount; j++) {
      x_pred->pData[i] += Wm[j] * Xsig_pred->pData[i * sigmaCount + j];
    }
  }

  // 2. Predict covariance
  float diff[n];
  float diff_T[n];
  float temp[n * n];
  
  arm_matrix_instance_f32 diff_mat, diff_T_mat, temp_mat;
  arm_mat_init_f32(&diff_mat, n, 1, diff);
  arm_mat_init_f32(&diff_T_mat, 1, n, diff_T);
  arm_mat_init_f32(&temp_mat, n, n, temp);
  
  // Initialize with process noise
  memcpy(P_pred->pData, Q->pData, sizeof(float) * Q->numRows * Q->numCols);
  
  for (int j = 0; j < sigmaCount; j++) {
    // Calculate difference
    for (int i = 0; i < n; i++) {
      diff[i] = Xsig_pred->pData[i * sigmaCount + j] - x_pred->pData[i];
      diff_T[i] = diff[i];
    }
    
    // Outer product
    arm_mat_mult_f32(&diff_mat, &diff_T_mat, &temp_mat);
    
    // Scale and add to covariance
    arm_mat_scale_f32(&temp_mat, Wc[j], &temp_mat);
    arm_mat_add_f32(P_pred, &temp_mat, P_pred);
  }
}

void update_with_measurement(float z_meas) {
  // Measurement prediction
  float z_pred = x_data[0];
  float innov = z_meas - z_pred;
  float R_combined = (R_bme + R_ms) * 0.5f;

  float relInnov = 0;

  if(!isnan(z_pred) && !isnan(innov)){
    relInnov = fabs(innov) / max(fabs(z_pred), 1.0f);
  }

  if (relInnov > 0.05f){
    R_combined *= 10.0f;
  }
  
  // Innovation covariance
  float S = P_data[0] + R_combined;
  
  // Kalman gain
  float K0 = P_data[0] / S;
  float K1 = P_data[2] / S;
  
  // Update state
  float dz = z_meas - z_pred;
  x_data[0] += K0 * dz;
  x_data[1] += K1 * dz;
  
  // Update covariance (Joseph form for stability)
  float IKH[4] = {1 - K0, 0, -K1, 1};
  float P_temp[4];
  
  arm_matrix_instance_f32 IKH_mat, P_mat, P_temp_mat;
  arm_mat_init_f32(&IKH_mat, n, n, IKH);
  arm_mat_init_f32(&P_mat, n, n, P_data);
  arm_mat_init_f32(&P_temp_mat, n, n, P_temp);
  
  // P = (I-KH)*P*(I-KH)' + KRK'
  arm_mat_mult_f32(&IKH_mat, &P_mat, &P_temp_mat);
  arm_mat_trans_f32(&IKH_mat, &P_mat);
  arm_mat_mult_f32(&P_temp_mat, &P_mat, &P_mat);
  
  // Add KRK' term
  P_data[0] += K0 * R_combined * K0;
  P_data[1] += K0 * R_combined * K1;
  P_data[2] += K1 * R_combined * K0;
  P_data[3] += K1 * R_combined * K1;
}

void runStateMachine(float h, float alt1, float alt2, float velocity, float accelZ, float orientationX) {
  
  unsigned long now = millis();


  switch (state) {
    case 1:  // Sensor check

      if (testSensors()) {
        state = 2;
        digitalWrite(8, HIGH);
      }
      break;
      
    case 2:  // Launch detection
      if (isLaunchDetected()) {
        state = 3;
        digitalWrite(7, HIGH);
      }
      break;
      
    case 3:  // Ascent
      if (checkAltitudeCondition(h, alt1, alt2, 4000)) state = 4;
      break;
      
    case 4:  // Apogee detection
      if (peakDetect(h, velocity, accelZ, orientationX)) {
        parachute1 = true;
        state = 5;
        digitalWrite(5, HIGH);
        peakAlt = h;
        peakDetectedTime = millis();
      }
      break;
      
    case 5:  // Descent
      if (h < 1000 && parachute1) {
        parachute2 = true;
        state = 6;
        digitalWrite(4, HIGH);
      }
      break;
      
    case 6:  // Landing
      if (h < 50 && velocity < 5) {
        Serial.println("Landing complete");
      }
      break;
  }
}

bool testSensors() {
  float alt_bme = getAltBME();
  float alt_ms = getAltMS();
  imu::Vector<3> accel = bno.getVector(Adafruit_BNO055::VECTOR_ACCELEROMETER);

  bool bme_ok = !isnan(alt_bme) && alt_bme > -100 && alt_bme < 10000;
  bool ms_ok = !isnan(alt_ms) && alt_ms > -100 && alt_ms < 10000;
  bool bno_ok = !isnan(accel.x()) && !isnan(accel.y()) && !isnan(accel.z());

  return bme_ok && ms_ok && bno_ok;
}

bool isAltitudeRising() {
  if(x.pData[1] > 5 && usedAlt > 900){
    return true;
  }
  return false;
}

bool isLaunchDetected() {
  #define VOTING_WINDOW 10
  float h_history[VOTING_WINDOW] = {0};
  static int counter = 0;
  h_history[history_index] = x.pData[0];
  history_index = (history_index + 1) % VOTING_WINDOW;
  if ((abs(acx) > -1.0 || x.pData[0] - h_history[(history_index + 1) % VOTING_WINDOW] > 2.0) && isAltitudeRising()) {
    counter++;
  } else {
    counter = 0;
  }
  return counter > 10;
}

bool checkAltitudeCondition(double h, double altitude_bme, double altitude_ms, double threshold) {
  int validCount = 0;
  if (h > threshold) validCount++;
  if (altitude_bme > threshold) validCount++;
  if (altitude_ms > threshold) validCount++;
  Serial.print("Altitude Voting: ");
  Serial.println(validCount);
  return (validCount >= 2);
}

void updatePeakPrediction(float currentAltitude, float currentVelocity, float currentAccelZ) {
    const float g = 9.80665f;
    const float minValidAccel = 2.0f;
    
    // Sadece yeterli ivme ve yukarı hareket olduğunda tahmin yap
    if (abs(currentAccelZ) >= minValidAccel && currentVelocity > 0.5f) {
        float netAccel = currentAccelZ - g;  // Gerçek ivme
        float timeToPeak;
        if (fabs(netAccel) > 0.001f){
          timeToPeak = currentVelocity / netAccel;
        }
        
        // Kinematik denklemle zirve tahmini: h = h0 + v*t + 0.5*a*t²
        float newPeakEstimate = currentAltitude + 
                              (currentVelocity * timeToPeak) + 
                              (0.5f * netAccel * timeToPeak * timeToPeak);

        // İlk tahmin
        if (prevPeakEstimate < 1.0f) {
            prevPeakEstimate = newPeakEstimate;
            estimatedMaxAltitude = newPeakEstimate;
        } 
        // Sonraki tahminlerde yumuşak geçiş
        else {
            estimatedMaxAltitude = (1.0f - peakSmoothingFactor) * prevPeakEstimate + 
                                 peakSmoothingFactor * newPeakEstimate;
            prevPeakEstimate = estimatedMaxAltitude;
        }
        
        lastPeakUpdate = millis();
    }
}

// Gelişmiş zirve tespit fonksiyonu (Unscented KF çıktılarını kullanır)
bool peakDetect(float currentAltitude, float currentVelocity, float currentAccelZ, float orientationX) {
    static float prevVelocity = 0;
    static int descentCounter = 0;
    static bool potentialPeak = false;
    const int descentThreshold = 3;
    const float decelerationThreshold = -2.0f;  // m/s²
    
    // 1. Zirve tahminini güncelle
    updatePeakPrediction(currentAltitude, currentVelocity, currentAccelZ);
    
    // 2. Hız değişimini izle
    bool isDescending = (currentVelocity < prevVelocity);
    prevVelocity = currentVelocity;
    
    // 3. Alçalma sayacını güncelle
    if (isDescending) {
        descentCounter = min(descentCounter + 1, descentThreshold + 1);
    } else {
        descentCounter = max(descentCounter - 1, 0);
    }
    
    // 4. İvme kontrolü (yavaşlama)
    float netAccel = currentAccelZ - 9.81f;
    bool isDecelerating = (netAccel < decelerationThreshold);
    
    // 5. Yönelim kontrolü (opsiyonel)
    bool isOrientationChanging = (abs(orientationX) > 15.0f);
    
    // 6. Zirve koşulları
    if (!peakDetected) {
        // Koşul 1: Belirgin alçalma + tahmini zirveye yakınlık
        bool condition1 = (descentCounter >= descentThreshold) && 
                         ((estimatedMaxAltitude - currentAltitude) < 100.0f);
        
        // Koşul 2: Hızın sıfıra yakın ve ivmenin negatif olması
        bool condition2 = (currentVelocity < 5.0f) && isDecelerating;
        
        // Koşul 3: Yönelim değişimi (büyük açı)
        bool condition3 = isOrientationChanging && (currentVelocity < 20.0f);
        
        if (condition1 || condition2 || condition3) {
            if (!potentialPeak) {
                potentialPeak = true;
                return false;
            }
            peakDetected = true;
            peakDetectedTime = millis();
            peakAlt = currentAltitude;
            return true;
        }
    }
    return false;
}

void logSensorData(float usedAlt,float alt_bme,float alt_ms){
    Serial.print("State: "); Serial.println(state);
    Serial.print("usedAlt: "); Serial.println(usedAlt);
    Serial.print("BME alt: "); Serial.println(alt_bme);
    Serial.print("MS alt: "); Serial.println(alt_ms);

    Serial.print("Accel X: "); Serial.print(acx);
    Serial.print(" | Y: "); Serial.print(acy);
    Serial.print(" | Z: "); Serial.println(acz);

    Serial.print("Orientation X: "); Serial.print(orx);
    Serial.print(" | Y: "); Serial.print(ory);
    Serial.print(" | Z: "); Serial.println(orz);

    if (gps.location.isValid()) {
      Serial.print("GPS Lat: "); Serial.print(gps.location.lat(), 6);
      Serial.print(" | Lng: "); Serial.print(gps.location.lng(), 6);
    } else {
      Serial.print("GPS konumu geçersiz");
    }
    Serial.println();
}