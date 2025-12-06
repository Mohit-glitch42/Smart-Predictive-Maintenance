import { useState, useEffect } from "react";
import { predictFailure } from "./api";
import "./App.css";

// 🔹 Global model metrics (from evaluate_cmaps_model.py)
const MODEL_METRICS = {
  rocAuc: 0.695,
  prAuc: 0.507,
  alertThreshold: 0.2, // 20%
};

// 🔹 Global feature importance (from permutation importance)
const FEATURE_IMPORTANCE = [
  { name: "Engine age (time_cycles)", key: "time_cycles", weight: 0.5012 },
  { name: "Sensor 4", key: "sensor_4", weight: 0.1906 },
  { name: "Sensor 12", key: "sensor_12", weight: 0.1388 },
  { name: "Sensor 9", key: "sensor_9", weight: 0.0673 },
  { name: "Sensor 2", key: "sensor_2", weight: 0.0524 },
  { name: "Sensor 8", key: "sensor_8", weight: 0.0497 },
];

const FACTS = [
  "Model: RandomForestClassifier trained on jet engine performance data.",
  "Target label: failed_within_30_cycles (binary 0/1 classification).",
  "Features include temperature, pressure ratio, vibration and maintenance history.",
  "FastAPI serves the ML model; Spring Boot acts as a gateway for the frontend.",
  "Predictions estimate short-term failure risk to support maintenance planning.",
];

function App() {
  const [form, setForm] = useState({
    cyclesSinceMaintenance: "",
    avgTurbineTemp: "",
    compressorPressureRatio: "",
    vibrationLevel: "",
    fuelFlowVariation: "",
    previousFailures: "",
  });

  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);
  const [lastPayload, setLastPayload] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [showTech, setShowTech] = useState(false);
  const [factIndex, setFactIndex] = useState(0);

  useEffect(() => {
    const id = setInterval(() => {
      setFactIndex((prev) => (prev + 1) % FACTS.length);
    }, 6000); // rotate fact every 6 seconds
    return () => clearInterval(id);
  }, []);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setForm((prev) => ({ ...prev, [name]: value }));
  };

  const applyPreset = (type) => {
    if (type === "low") {
      setForm({
        cyclesSinceMaintenance: 15,
        avgTurbineTemp: 720,
        compressorPressureRatio: 31.2,
        vibrationLevel: 0.21,
        fuelFlowVariation: 0.03,
        previousFailures: 0,
      });
    } else if (type === "medium") {
      setForm({
        cyclesSinceMaintenance: 85,
        avgTurbineTemp: 860,
        compressorPressureRatio: 28.4,
        vibrationLevel: 0.41,
        fuelFlowVariation: 0.06,
        previousFailures: 1,
      });
    } else if (type === "high") {
      setForm({
        cyclesSinceMaintenance: 150,
        avgTurbineTemp: 960,
        compressorPressureRatio: 24.1,
        vibrationLevel: 0.82,
        fuelFlowVariation: 0.13,
        previousFailures: 3,
      });
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    setLoading(true);

    const cappedPreviousFailures = Math.min(
      Number(form.previousFailures),
      10 // cap unrealistic values at 10
    );

    const frontendPayload = {
      cyclesSinceMaintenance: Number(form.cyclesSinceMaintenance),
      avgTurbineTemp: Number(form.avgTurbineTemp),
      compressorPressureRatio: Number(form.compressorPressureRatio),
      vibrationLevel: Number(form.vibrationLevel),
      fuelFlowVariation: Number(form.fuelFlowVariation),
      previousFailures: cappedPreviousFailures,
    };

    // For "Last Input Payload" + history display
    setLastPayload(frontendPayload);

    try {
      // Call Spring Boot backend (which then calls FastAPI)
      const res = await predictFailure(frontendPayload);

      // Backend returns: { prediction, probability, riskLevel }
      // Normalize to also expose failureProbability so existing UI keeps working.
      const normalized = {
        ...res,
        failureProbability:
          res.failureProbability != null
            ? res.failureProbability
            : res.probability,
      };

      setResult(normalized);

      const probability =
        normalized.failureProbability != null
          ? normalized.failureProbability
          : null;

      const newEntry = {
        id: Date.now(),
        timestamp: new Date().toLocaleTimeString(),
        riskLevel: normalized.riskLevel,
        prediction: normalized.prediction,
        failureProbability: probability,
        payload: frontendPayload,
      };

      setHistory((prev) => [newEntry, ...prev].slice(0, 5));
    } catch (err) {
      console.error("Prediction error:", err.response?.data || err.message);
      setError("Failed to get prediction. Please check backend / ML service.");
    } finally {
      setLoading(false);
    }
  };

  const getRiskColorClass = (riskLevel) => {
    if (!riskLevel) return "risk-pill";
    const level = riskLevel.toUpperCase();
    if (level === "LOW") return "risk-pill low";
    if (level === "MEDIUM") return "risk-pill medium";
    if (level === "HIGH") return "risk-pill high";
    return "risk-pill";
  };

  const getSafetyMessage = () => {
    if (!result) return "";
    const level = result.riskLevel?.toUpperCase();
    if (level === "LOW") {
      return "Engine is operating in a healthy range. Continue normal operation and follow standard maintenance schedule.";
    }
    if (level === "MEDIUM") {
      return "Engine shows early signs of degradation. Plan maintenance soon and monitor parameters closely.";
    }
    if (level === "HIGH") {
      return "High risk of failure detected. Schedule immediate inspection and avoid high load operation.";
    }
    return "Review engine parameters and maintenance schedule.";
  };

  const probabilityPercent =
    result && result.failureProbability != null
      ? (result.failureProbability * 100).toFixed(1)
      : null;

  return (
    <div className="app-root">
      <header className="app-header">
        <div>
          <h1>Jet Engine Predictive Maintenance</h1>
          <p className="subtitle">
            Real-time failure risk prediction for jet engines
          </p>
        </div>
      </header>

      <main className="app-main">
        <section className="panel panel-left">
          <h2>Engine Parameters</h2>
          <p className="panel-description">
            Enter current sensor readings and maintenance info to estimate
            failure risk within the next cycles.
          </p>

          <div className="preset-row">
            <span>Quick test presets:</span>
            <div className="preset-buttons">
              <button type="button" onClick={() => applyPreset("low")}>
                Low Risk
              </button>
              <button type="button" onClick={() => applyPreset("medium")}>
                Medium Risk
              </button>
              <button type="button" onClick={() => applyPreset("high")}>
                High Risk
              </button>
            </div>
          </div>

          <form onSubmit={handleSubmit} className="form-grid">
            <div className="form-field">
              <label>Cycles Since Maintenance</label>
              <input
                type="number"
                name="cyclesSinceMaintenance"
                value={form.cyclesSinceMaintenance}
                onChange={handleChange}
                required
              />
              <small>Number of cycles since last major service.</small>
            </div>

            <div className="form-field">
              <label>Avg Turbine Temp (°C)</label>
              <input
                type="number"
                name="avgTurbineTemp"
                value={form.avgTurbineTemp}
                onChange={handleChange}
                required
              />
              <small>Higher temperatures increase failure risk.</small>
            </div>

            <div className="form-field">
              <label>Compressor Pressure Ratio</label>
              <input
                type="number"
                step="0.1"
                name="compressorPressureRatio"
                value={form.compressorPressureRatio}
                onChange={handleChange}
                required
              />
              <small>Healthy engines maintain higher pressure ratios.</small>
            </div>

            <div className="form-field">
              <label>Vibration Level (g)</label>
              <input
                type="number"
                step="0.01"
                name="vibrationLevel"
                value={form.vibrationLevel}
                onChange={handleChange}
                required
              />
              <small>Abnormal vibration indicates imbalance or wear.</small>
            </div>

            <div className="form-field">
              <label>Fuel Flow Variation</label>
              <input
                type="number"
                step="0.01"
                name="fuelFlowVariation"
                value={form.fuelFlowVariation}
                onChange={handleChange}
                required
              />
              <small>Instability in fuel flow can signal issues.</small>
            </div>

            <div className="form-field">
              <label>Previous Failures</label>
              <input
                type="number"
                name="previousFailures"
                value={form.previousFailures}
                onChange={handleChange}
                required
              />
              <small>Historical failures increase future risk.</small>
            </div>

            {error && <p className="error-text">{error}</p>}

            <button className="primary-button" type="submit" disabled={loading}>
              {loading ? "Predicting..." : "Predict Failure Risk"}
            </button>
          </form>

          {/* Input help card to use space and guide users */}
          <div className="input-help-card">
            <h3>Input ranges &amp; guidance</h3>
            <ul>
              <li>
                <b>Cycles</b>: 0–300 is healthy, &gt; 300 requires closer
                monitoring.
              </li>
              <li>
                <b>Avg turbine temp</b>: values &gt; 900&nbsp;°C generally push
                risk higher.
              </li>
              <li>
                <b>Vibration</b>: &gt; 0.6 g indicates imbalance or wear.
              </li>
              <li>
                <b>Previous failures</b>: more than one prior failure should be
                treated as serious.
              </li>
            </ul>
            <p className="input-help-note">
              These bands are illustrative for demo purposes and can be tuned to
              match real engine limits.
            </p>
          </div>
        </section>

        <section className="panel panel-right">
          <h2>Prediction Overview</h2>

          {/* Top message or result card */}
          {!result && (
            <p className="empty-state">
              No prediction yet. Enter engine parameters on the left and click{" "}
              <b>"Predict Failure Risk"</b> to see the results.
            </p>
          )}

          {result && (
            <div className="result-card">
              <div className="result-header">
                <span className={getRiskColorClass(result.riskLevel)}>
                  {result.riskLevel || "UNKNOWN"}
                </span>
                <span className="result-label">
                  {result.prediction === 1
                    ? "Likely Failure Soon"
                    : "Stable for Now"}
                </span>
              </div>

              <div className="probability-section">
                <div className="probability-header">
                  <span>Failure Probability</span>
                  <span className="probability-value">
                    {probabilityPercent !== null
                      ? `${probabilityPercent}%`
                      : "N/A"}
                  </span>
                </div>
                <div className="probability-bar">
                  <div
                    className="probability-bar-fill"
                    style={{
                      width:
                        probabilityPercent !== null
                          ? `${Math.min(Math.max(probabilityPercent, 0), 100)}%`
                          : "0%",
                    }}
                  ></div>
                </div>
              </div>

              <div className="recommendation">
                <h3>Recommendation</h3>
                <p>{getSafetyMessage()}</p>
              </div>
            </div>
          )}

          {/* These cards are shown even before first prediction */}

          <div className="metrics-card">
            <h3>Model Metrics</h3>
            <p className="metrics-subtitle">
              Evaluated on the C-MAPSS test set (failure within 30 cycles).
            </p>

            <div className="metrics-grid">
              <div className="metrics-item">
                <span className="metrics-label">ROC-AUC</span>
                <span className="metrics-value">
                  {MODEL_METRICS.rocAuc.toFixed(3)}
                </span>
              </div>
              <div className="metrics-item">
                <span className="metrics-label">PR-AUC</span>
                <span className="metrics-value">
                  {MODEL_METRICS.prAuc.toFixed(3)}
                </span>
              </div>
              <div className="metrics-item">
                <span className="metrics-label">Alert Threshold</span>
                <span className="metrics-value">
                  {(MODEL_METRICS.alertThreshold * 100).toFixed(0)}%
                </span>
              </div>
            </div>

            <p className="metrics-note">
              Alerts are raised when predicted failure probability is{" "}
              <b>≥ {(MODEL_METRICS.alertThreshold * 100).toFixed(0)}%</b>, based
              on ROC / PR curve analysis.
            </p>
          </div>

          <div className="influence-card">
            <h3>Feature Influence (overall)</h3>
            <p className="influence-subtitle">
              Permutation importance on the C-MAPSS test set — shows which
              signals the model relies on most when assessing risk.
            </p>

            <ul className="influence-list">
              {FEATURE_IMPORTANCE.map((f) => (
                <li key={f.key} className="influence-item">
                  <div className="influence-row">
                    <span className="influence-name">{f.name}</span>
                    <span className="influence-percent">
                      {(f.weight * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="influence-bar">
                    <div
                      className="influence-bar-fill"
                      style={{ width: `${Math.max(f.weight * 100, 3)}%` }}
                    />
                  </div>
                </li>
              ))}
            </ul>

            <p className="influence-note">
              For example, if <b>Engine age (time_cycles)</b> has ~50%
              influence, it means engines with many cycles remaining are usually
              low risk, while engines near end-of-life strongly push predictions
              toward higher risk.
            </p>
          </div>

          <div className="tech-card">
            <button
              type="button"
              className="tech-toggle"
              onClick={() => setShowTech((prev) => !prev)}
            >
              <span>Technical details</span>
              <span className="tech-toggle-icon">{showTech ? "▴" : "▾"}</span>
            </button>

            {showTech && (
              <div className="tech-content">
                <div className="tech-section">
                  <div className="tech-section-header">
                    <span>Raw Model Response</span>
                    <span className="tech-badge">Spring Boot → FastAPI</span>
                  </div>
                  <pre className="code-block">
                    {result
                      ? JSON.stringify(result, null, 2)
                      : "// Run a prediction to see raw model response"}
                  </pre>
                </div>

                <div className="tech-section">
                  <div className="tech-section-header">
                    <span>Last Input Payload</span>
                    <span className="tech-badge">Frontend → Backend</span>
                  </div>
                  <pre className="code-block">
                    {lastPayload
                      ? JSON.stringify(lastPayload, null, 2)
                      : "// Run a prediction to see payload"}
                  </pre>
                </div>

                <div className="tech-section tech-notes">
                  <div className="tech-section-header">
                    <span>Model Configuration (documentation)</span>
                  </div>
                  <ul>
                    <li>Algorithm: RandomForestClassifier (scikit-learn)</li>
                    <li>
                      Target: <code>failed_within_30_cycles</code> (0/1)
                    </li>
                    <li>
                      Features: cycles_since_maintenance, avg_turbine_temp,
                      compressor_pressure_ratio, vibration_level,
                      fuel_flow_variation, previous_failures
                    </li>
                    <li>Preprocessing: train/test split + scaling</li>
                    <li>Serving: FastAPI microservice → Spring Boot proxy</li>
                  </ul>
                </div>
              </div>
            )}
          </div>

          <div className="history-card">
            <h3>Recent Predictions</h3>
            {history.length === 0 && (
              <p className="empty-state-small">
                Predictions will appear here as you run them.
              </p>
            )}
            {history.length > 0 && (
              <ul className="history-list">
                {history.map((item) => (
                  <li key={item.id} className="history-item">
                    <div className="history-main">
                      <span className={getRiskColorClass(item.riskLevel)}>
                        {item.riskLevel}
                      </span>
                      <span className="history-time">{item.timestamp}</span>
                    </div>
                    <div className="history-detail">
                      <span>Cycles: {item.payload.cyclesSinceMaintenance}</span>
                      <span>
                        Temp: {item.payload.avgTurbineTemp}°C | Vib:{" "}
                        {item.payload.vibrationLevel}
                      </span>
                      <span>
                        Prob:{" "}
                        {item.failureProbability != null
                          ? `${(item.failureProbability * 100).toFixed(1)}%`
                          : "N/A"}
                      </span>
                    </div>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </section>
      </main>

      {/* Dynamic bottom section with jet engine visual + rotating facts */}
      <section className="bottom-strip">
        <div className="bottom-inner">
          <div className="engine-visual-card">
            <h2 className="bottom-title">Jet Engine Visual</h2>
            <p className="bottom-subtitle">
              Conceptual turbofan view – purely illustrative, not to scale.
            </p>
            <div className="engine-visual">
              <div className="engine-body">
                <div className="engine-intake" />
                <div className="engine-fan">
                  <div className="engine-blade blade-1" />
                  <div className="engine-blade blade-2" />
                  <div className="engine-blade blade-3" />
                  <div className="engine-blade blade-4" />
                </div>
                <div className="engine-core" />
                <div className="engine-exhaust" />
              </div>
            </div>
          </div>

          <div className="engine-info-card">
            <h2 className="bottom-title">Model &amp; Data Insights</h2>
            <p className="bottom-subtitle">
              Live technical context about the predictive maintenance engine.
            </p>

            <div className="fact-pill">
              <span className="fact-label">Now showing</span>
            </div>

            <p className="fact-text">{FACTS[factIndex]}</p>

            <div className="fact-dots">
              {FACTS.map((_, i) => (
                <span
                  key={i}
                  className={`fact-dot ${i === factIndex ? "active" : ""}`}
                />
              ))}
            </div>

            <ul className="bottom-list">
              <li>End-to-end: React → Spring Boot → FastAPI → scikit-learn.</li>
              <li>
                Supports scenario testing via low / medium / high presets.
              </li>
              <li>
                Designed to be explainable with technical details visible.
              </li>
            </ul>
          </div>
        </div>
      </section>
    </div>
  );
}

export default App;
