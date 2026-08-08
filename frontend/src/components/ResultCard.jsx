export default function ResultCard({ result, type }) {
  if (!result) return null
  const { risk_percentage, risk_level, recommendation, modalities_used, missing_modalities } = result

  return (
    <div className={`result-card ${type} fade-in`}>
      <div className="result-disease">
        {type === 'pneumonia' ? '🫁 Pneumonia Risk' : '🩸 Diabetes Risk'}
      </div>
      <div className="result-score-row">
        <div className="result-score" style={{
          color: risk_level === 'High' ? 'var(--red)'
               : risk_level === 'Moderate' ? 'var(--yellow)'
               : risk_level === 'Low' ? 'var(--primary2)'
               : 'var(--green)'
        }}>
          {risk_percentage}%
        </div>
        <span className={`risk-badge ${risk_level}`}>{risk_level}</span>
      </div>

      <div className="progress-bar-wrap">
        <div className="progress-label"><span>Risk Score</span><span>{risk_percentage}%</span></div>
        <div className="progress-bar">
          <div className={`progress-fill fill-${risk_level}`} style={{ width: `${risk_percentage}%` }} />
        </div>
      </div>

      <div className="result-recommendation">{recommendation}</div>

      <div style={{ marginTop: '1rem' }}>
        <div style={{ fontSize: '.78rem', color: 'var(--muted)', marginBottom: '.4rem' }}>
          Data sources used:
        </div>
        <div className="modality-tags">
          {modalities_used?.map(m => <span key={m} className="modality-tag">{m}</span>)}
          {missing_modalities?.map(m => <span key={m} className="modality-tag missing">⚠️ {m} missing</span>)}
        </div>
      </div>
    </div>
  )
}
