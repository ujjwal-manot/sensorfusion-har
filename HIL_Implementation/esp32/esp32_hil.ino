/*
 * SensorFusion-HAR V2 — ESP32 Hardware-in-the-Loop (HIL)
 *
 * 11-Class HAR Model Validation Firmware
 * 
 * This firmware does NOT read from a physical MPU6050.
 * Instead, it waits for a UART binary payload containing 50x6 float values
 * (a 1-second window) from the PC simulator (`hil_server.py`),
 * runs the TFLite Micro inference, and sends back the prediction and confidence.
 */

#include "esp32_v2_useful11_config.h"
#include "model_data.h" // MUST BE GENERATED VIA COLAB TFLITE EXPORT

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// ============================================================================
// Configuration
// ============================================================================
#define TENSOR_ARENA_SIZE  (60 * 1024)  // 60 KB to be safe for 11-class model
#define NUM_CHANNELS SENSORFUSION_V2_INPUT_CHANNELS
#define TIME_STEPS SENSORFUSION_V2_TIME_STEPS
#define NUM_CLASSES SENSORFUSION_V2_NUM_CLASSES
#define EXPECTED_PAYLOAD_SIZE (TIME_STEPS * NUM_CHANNELS * sizeof(float))

// ============================================================================
// Globals
// ============================================================================
static float sensor_buffer[TIME_STEPS][NUM_CHANNELS];

// TFLM
static uint8_t tensor_arena[TENSOR_ARENA_SIZE] __attribute__((aligned(16)));
static tflite::MicroInterpreter* interpreter = nullptr;
static TfLiteTensor* input_tensor = nullptr;
static TfLiteTensor* output_tensor = nullptr;

// ============================================================================
// TFLM Setup
// ============================================================================
bool tflm_init() {
    const tflite::Model* model = tflite::GetModel(model_tflite);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("ERROR: Model schema version mismatch!");
        return false;
    }

    static tflite::AllOpsResolver resolver;

    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, TENSOR_ARENA_SIZE
    );
    interpreter = &static_interpreter;

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("ERROR: AllocateTensors() failed!");
        return false;
    }

    input_tensor = interpreter->input(0);
    output_tensor = interpreter->output(0);
    return true;
}

// ============================================================================
// HIL Communication & Inference
// ============================================================================
void run_inference_and_reply() {
    // We assume the PC already normalized the data, OR we can normalize it here.
    // Let's normalize it here to match the real-world deployment perfectly.
    
    if (input_tensor->type == kTfLiteInt8) {
        float input_scale = input_tensor->params.scale;
        int32_t input_zp = input_tensor->params.zero_point;
        int8_t* input_data = input_tensor->data.int8;
        
        for (int t = 0; t < TIME_STEPS; t++) {
            for (int c = 0; c < NUM_CHANNELS; c++) {
                int idx = t * NUM_CHANNELS + c;
                float normalized = (sensor_buffer[t][c] - SENSORFUSION_V2_MEAN[c]) / SENSORFUSION_V2_STD[c];
                int32_t quantized = (int32_t)roundf(normalized / input_scale) + input_zp;
                if (quantized < -128) quantized = -128;
                if (quantized > 127) quantized = 127;
                input_data[idx] = (int8_t)quantized;
            }
        }
    } else {
        float* input_data = input_tensor->data.f;
        for (int t = 0; t < TIME_STEPS; t++) {
            for (int c = 0; c < NUM_CHANNELS; c++) {
                int idx = t * NUM_CHANNELS + c;
                input_data[idx] = (sensor_buffer[t][c] - SENSORFUSION_V2_MEAN[c]) / SENSORFUSION_V2_STD[c];
            }
        }
    }

    // Run inference
    unsigned long start_us = micros();
    TfLiteStatus status = interpreter->Invoke();
    unsigned long elapsed_us = micros() - start_us;

    if (status != kTfLiteOk) {
        Serial.println("HIL_ERR:InferenceFailed");
        return;
    }

    // Dequantize output
    float output_vals[NUM_CLASSES];
    if (output_tensor->type == kTfLiteInt8) {
        float output_scale = output_tensor->params.scale;
        int32_t output_zp = output_tensor->params.zero_point;
        int8_t* raw_output = output_tensor->data.int8;
        for (int i = 0; i < NUM_CLASSES; i++) {
            output_vals[i] = ((float)raw_output[i] - output_zp) * output_scale;
        }
    } else {
        float* raw_output = output_tensor->data.f;
        for (int i = 0; i < NUM_CLASSES; i++) {
            output_vals[i] = raw_output[i];
        }
    }

    // Softmax and argmax
    int best_class = 0;
    float best_score = output_vals[0];
    for (int i = 1; i < NUM_CLASSES; i++) {
        if (output_vals[i] > best_score) {
            best_score = output_vals[i];
            best_class = i;
        }
    }

    float sum_exp = 0.0f;
    for (int i = 0; i < NUM_CLASSES; i++) {
        sum_exp += expf(output_vals[i] - best_score);
    }
    float confidence = 1.0f / sum_exp;
    float inference_ms = elapsed_us / 1000.0f;

    // Send result back to PC Simulator in formatted string
    // Format: HIL_RES:<predicted_class_index>:<confidence>:<inference_ms>
    Serial.print("HIL_RES:");
    Serial.print(best_class);
    Serial.print(":");
    Serial.print(confidence, 4);
    Serial.print(":");
    Serial.println(inference_ms, 2);
}

// ============================================================================
// Setup & Loop
// ============================================================================
void setup() {
    // High baud rate for fast 1200-byte payload transmission
    Serial.begin(921600); 
    while (!Serial) { delay(10); }

    if (!tflm_init()) {
        Serial.println("HIL_ERR:TFLMInitFailed");
        while (1) { delay(1000); }
    }
    
    // Send ready signal
    Serial.println("HIL_READY");
}

void loop() {
    // Protocol: PC sends a sync string "HIL_SYNC", then 1200 bytes of binary float data.
    if (Serial.available() >= 8) {
        String sync = Serial.readStringUntil('\n');
        sync.trim();
        if (sync == "HIL_SYNC") {
            // Read 1200 bytes
            uint8_t* ptr = (uint8_t*)sensor_buffer;
            size_t read_bytes = 0;
            unsigned long timeout = millis() + 1000;
            
            while (read_bytes < EXPECTED_PAYLOAD_SIZE && millis() < timeout) {
                if (Serial.available()) {
                    ptr[read_bytes++] = Serial.read();
                }
            }
            
            if (read_bytes == EXPECTED_PAYLOAD_SIZE) {
                run_inference_and_reply();
            } else {
                Serial.println("HIL_ERR:TimeoutOrTruncated");
                // Flush the rest
                while(Serial.available()) Serial.read();
            }
        }
    }
}
