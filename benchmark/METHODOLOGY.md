# Benchmark Methodology: JEPA World Model Streaming Performance (JWMSP)

## 1. Overview

The **JWMSP** benchmark measures the end-to-end performance of Joint Embedding Predictive Architecture (JEPA) models in a live streaming context. Unlike static inference benchmarks, JWMSP evaluates the system's ability to maintain temporal synchronization between a continuous input stream and a latent-space world model.

## 2. System Under Test (SUT) Boundary

The SUT is defined as a **Wide Boundary** to capture real-world bottlenecks. It starts from the receipt of an encoded bitstream and ends at the availability of the predicted latent embedding.

### Components of the SUT

1. **Ingress/Demuxer:** Handles the raw transport protocol (e.g., RTSP, Unix Domain Socket).
2. **Hardware Decoder:** Converts compressed video (H.264/H.265) into raw frame buffers.
3. **Pre-processor:** Resizing, normalization, and "Tubelet" formation.
4. **JEPA Encoder ($E$):** Maps pixel-space context to latent embeddings.
5. **JEPA Predictor ($P$):** Projects the future latent state based on context and (optional) actions.

## 3. Metrics Definitions

### A. Temporal Metrics

* **Computational Latency ($L_{comp}$):** Total wall-clock time for the SUT to process a specific inference task (Decode + Prep + Model).
* **Algorithmic Latency ($L_{algo}$):** The physical time duration of the video window required by the model (e.g., $Context\_Length / FPS$).
* **Total System Lag ($L_{sys}$):** The delay between a real-world event and its corresponding prediction.
    * **Formula:** $L_{sys} = L_{algo} + L_{comp}$
* **Time to First Prediction (TTFP):** The "Cold Start" time required to fill the initial context buffer and produce the first embedding.

### B. Sustainability Metrics

* **Real-Time Factor (RTF):** Ratio of processed video duration to processing time.
    * $RTF = \frac{T_{video}}{T_{process}}$. Must stay $\ge 1.0$ for real-time viability.
* **Offered vs. Served Rate:** Comparison of the input stream FPS vs. the output prediction FPS.
* **Jitter:** The variance in $L_{sys}$ over a sustained stream.

### C. Reliability Metrics

* **Drop Rate:** Percentage of frames or "inference units" skipped due to buffer overflows or compute timeouts.

## 4. Measurement & Instrumentation Strategy

### Tracing with Jaeger/OpenTelemetry

Each inference request is tracked using a unique `TraceID` that persists through the following spans. Span names follow a consistent `{subsystem}_{verb}` convention:

| Span Name | Key Attributes | Description |
| :--- | :--- | :--- |
| `input_receive` | `input.type`, `input.size_bytes` | Time to receive request/data from network or file. |
| `input_open` | `input.codec`, `input.source_type` | Container initialization (file header parse) or stream connection (RTSP handshake). |
| `input_decode` | `clip.start_frame`, `clip.end_frame` | Duration of the hardware/software video decoding per clip. |
| `input_preprocess` | `frame_count` | Resizing, normalization, and tensor preparation. |
| `input_action` | `action_dim` | (Optional) Action vector reception for AC models. |
| `jepa_encode` | `patch_count` | Time spent in the ViT/Backbone encoder forward pass. |
| `jepa_predict` | `horizon_step` | Time spent predicting the future latent state. |
| `jepa_pool` | `pool_type` | Temporal aggregation (attentive pooling). |
| `output_postprocess` | `top_k` | Softmax, top-k selection, label mapping. |

**Rendering spans** (outside SUT, optional):

| Span Name | Description |
| :--- | :--- |
| `render_annotate` | Overlay predictions on frames. |
| `render_stream` | Encode and deliver output stream. |

**Lifecycle spans** (server startup, standalone traces):

| Span Name | Key Attributes | Description |
| :--- | :--- | :--- |
| `init_model_load` | `model.path`, `model.device`, `model.load_duration_s` | Model loading at server startup. Standalone trace for cold start analysis. |

### The Load Generator (The Harness)

The harness acts as a **Deterministic Streamer**:

1. **Source:** Streams a high-bitrate `.mp4` at a fixed target FPS.
2. **Action Injection:** For action-conditioned models, the harness sends a synchronized side-channel of control vectors.
3. **Timestamping:** Injects a "World Time" metadata tag into each frame/packet to allow the SUT to calculate $L_{sys}$ accurately.

## 5. Benchmarking Execution Modes

1. **Stress Test (Saturation):** Feed the SUT from a pre-loaded memory buffer to determine the absolute maximum throughput of the Encoder/Predictor pipeline.
2. **Live Simulation:** Feed the SUT via a network socket at the native stream rate (e.g., 30 FPS).
    * **Goal:** Measure the stability of $L_{sys}$ and identify "drift" over a 30-minute window (Soak Test).
3. **Degraded Input Test:** Introduce artificial packet loss or variable framerates to test the robustness of the JEPA context-filling logic.

## 6. Output & Reporting

The benchmark concludes with a **JEPA Efficiency Report** including:

* **P95 and P99 Latencies** for $L_{comp}$ and $L_{sys}$.
* **Throughput vs. Lag Curve:** Visualizing how $L_{sys}$ changes as the offered load increases.
* **Resource Utilization:** GPU/CPU/Memory footprints during steady-state.