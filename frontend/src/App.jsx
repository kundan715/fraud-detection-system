import { useEffect, useState } from "react";
import { getMetrics, getSample, predict } from "./api.js";

const pct = (x, d = 1) => `${(x * 100).toFixed(d)}%`;

function RiskMeter({ result }) {
  const { fraud_probability: p, threshold, is_fraud: flagged, risk_level: level } = result;
  const label = flagged ? "Likely fraud" : level === "medium" ? "Needs a look" : "Looks legitimate";
  return (
    <section className={`meter-block ${flagged ? "is-flagged" : ""}`} aria-live="polite">
      <div className="meter-head">
        <p className="score">{pct(p)}</p>
        <div>
          <p className="verdict">{label}</p>
          <p className="muted">chance this transaction is fraud</p>
        </div>
      </div>
      <div className="strip" role="img" aria-label={`Score ${pct(p)}, alert threshold ${pct(threshold)}`}>
        <div className="strip-safe" style={{ width: `${threshold * 100}%` }} />
        <div className="strip-alert" style={{ left: `${threshold * 100}%` }} />
        <div className="notch" style={{ left: `${threshold * 100}%` }}>
          <span>alert at {pct(threshold, 0)}</span>
        </div>
        <div className="marker" style={{ left: `${Math.min(p, 0.995) * 100}%` }} />
      </div>
      <div className="strip-scale"><span>0%</span><span>100%</span></div>
    </section>
  );
}

function Explanation({ items }) {
  const max = Math.max(...items.map((i) => Math.abs(i.contribution)), 0.01);
  return (
    <section>
      <h2>Why this score</h2>
      <p className="muted small">
        Each bar shows how much a field pushed the score toward fraud (right) or toward legitimate (left).
      </p>
      <ul className="bars">
        {items.map((i) => (
          <li key={i.feature}>
            <span className="bar-name">{i.feature}</span>
            <span className="bar-track">
              <span
                className={`bar ${i.direction}`}
                style={{ width: `${(Math.abs(i.contribution) / max) * 50}%` }}
              />
            </span>
            <span className="bar-val">{i.contribution > 0 ? "+" : ""}{i.contribution.toFixed(2)}</span>
          </li>
        ))}
      </ul>
    </section>
  );
}

function Benchmarks({ meta }) {
  const rows = Object.entries(meta.models);
  const d = meta.dataset;
  return (
    <section className="benchmarks">
      <h2>How the model was chosen</h2>
      <p className="muted small">
        Trained on {d.rows.toLocaleString()} transactions, {d.fraud.toLocaleString()} of them fraud ({pct(d.fraud_rate, 2)}).
        Scores below come from a held-out test set of {d.split.test.toLocaleString()} transactions ({d.test_fraud} fraud).
      </p>
      <div className="table-wrap">
        <table>
          <thead>
            <tr><th>Model</th><th>PR-AUC</th><th>Precision</th><th>Recall</th><th>Missed fraud</th></tr>
          </thead>
          <tbody>
            {rows.map(([name, m]) => (
              <tr key={name} className={name === meta.serving_model ? "serving" : ""}>
                <td>{name}{name === meta.serving_model && <em> in use</em>}</td>
                <td>{m.pr_auc.toFixed(3)}</td>
                <td>{pct(m.precision)}</td>
                <td>{pct(m.recall)}</td>
                <td>{m.confusion.fn} of {m.confusion.tp + m.confusion.fn}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

export default function App() {
  const [meta, setMeta] = useState(null);
  const [features, setFeatures] = useState(null);
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [sampleKind, setSampleKind] = useState("");

  useEffect(() => {
    getMetrics().then(setMeta).catch((e) => setError(e.message));
    loadSample("legit");
  }, []);

  async function loadSample(kind) {
    setError("");
    try {
      const s = await getSample(kind);
      setFeatures(s.features);
      setSampleKind(kind);
      setResult(null);
    } catch (e) {
      setError(e.message);
    }
  }

  async function analyze() {
    setBusy(true);
    setError("");
    try {
      setResult(await predict(features));
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }

  const setValue = (name, raw) => {
    setFeatures({ ...features, [name]: raw === "" ? "" : Number(raw) });
    setResult(null);
    setSampleKind("");
  };

  const pcaFields = features ? Object.keys(features).filter((k) => k !== "Amount") : [];
  const hasEmpty = features && Object.values(features).some((v) => v === "" || Number.isNaN(v));

  return (
    <div className="page">
      <header>
        <h1>Fraud Check</h1>
        <p className="lede">Score a card transaction and see which fields drove the result.</p>
      </header>

      {error && <p className="error" role="alert">{error}</p>}

      <main className="layout">
        <section className="panel">
          <h2>Transaction</h2>
          <div className="sample-row">
            <button className={`ghost ${sampleKind === "legit" ? "on" : ""}`} onClick={() => loadSample("legit")}>
              Load a normal transaction
            </button>
            <button className={`ghost ${sampleKind === "fraud" ? "on" : ""}`} onClick={() => loadSample("fraud")}>
              Load a known fraud case
            </button>
          </div>

          {features && (
            <>
              <label className="field">
                <span>Amount</span>
                <input
                  type="number" min="0" step="0.01" value={features.Amount}
                  onChange={(e) => setValue("Amount", e.target.value)}
                />
              </label>

              <details>
                <summary>Anonymised card features (V1 to V28)</summary>
                <p className="muted small">
                  These come from a PCA transform of the original card data, so they have no readable meaning on their own.
                </p>
                <div className="v-grid">
                  {pcaFields.map((k) => (
                    <label key={k} className="v-field">
                      <span>{k}</span>
                      <input
                        type="number" step="any" value={features[k]}
                        onChange={(e) => setValue(k, e.target.value)}
                      />
                    </label>
                  ))}
                </div>
              </details>

              <button className="primary" onClick={analyze} disabled={busy || hasEmpty}>
                {busy ? "Scoring…" : "Score transaction"}
              </button>
              {hasEmpty && <p className="muted small">Fill in every field to score.</p>}
            </>
          )}
        </section>

        <div className="results">
          {result ? (
            <>
              <RiskMeter result={result} />
              <Explanation items={result.explanation} />
            </>
          ) : (
            <section className="empty">
              <h2>No score yet</h2>
              <p className="muted">Pick a sample or edit the fields, then score the transaction.</p>
            </section>
          )}
        </div>
      </main>

      {meta && <Benchmarks meta={meta} />}
    </div>
  );
}
