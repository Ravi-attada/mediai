const COLORS = [
  'linear-gradient(90deg,#6c63ff,#4ecdc4)',
  'linear-gradient(90deg,#ff6584,#ffb400)',
  'linear-gradient(90deg,#00d68f,#4ecdc4)',
]

export default function ExplanationChart({ pneumoniaShap, diabetesShap }) {
  const allKeys = new Set([
    ...Object.keys(pneumoniaShap || {}),
    ...Object.keys(diabetesShap || {})
  ])

  if (!allKeys.size) return null

  return (
    <div className="shap-card fade-in">
      <h3>🔍 Cross-Modal Explainability (SHAP Contributions)</h3>
      <p style={{ color: 'var(--muted)', fontSize: '.88rem', marginBottom: '1.2rem' }}>
        Shows how much each data source contributed to the final prediction — our novel research contribution.
      </p>

      <div style={{ marginBottom: '1.5rem' }}>
        <div style={{ fontSize: '.85rem', fontWeight: 600, marginBottom: '.8rem', color: 'var(--primary)' }}>
          🫁 Pneumonia — Modality Contributions
        </div>
        {Object.entries(pneumoniaShap || {}).map(([key, val], i) => (
          <div key={key} className="shap-bar-row">
            <div className="shap-bar-label">{key}</div>
            <div className="shap-bar-outer">
              <div className="shap-bar-inner"
                style={{ width: `${Math.min(100, val * 300)}%`, background: COLORS[i % COLORS.length] }} />
            </div>
            <div className="shap-bar-val">{(val * 100).toFixed(1)}%</div>
          </div>
        ))}
      </div>

      <div>
        <div style={{ fontSize: '.85rem', fontWeight: 600, marginBottom: '.8rem', color: '#ff6584' }}>
          🩸 Diabetes — Modality Contributions
        </div>
        {Object.entries(diabetesShap || {}).map(([key, val], i) => (
          <div key={key} className="shap-bar-row">
            <div className="shap-bar-label">{key}</div>
            <div className="shap-bar-outer">
              <div className="shap-bar-inner"
                style={{ width: `${Math.min(100, val * 300)}%`, background: COLORS[i % COLORS.length] }} />
            </div>
            <div className="shap-bar-val">{(val * 100).toFixed(1)}%</div>
          </div>
        ))}
      </div>
    </div>
  )
}
