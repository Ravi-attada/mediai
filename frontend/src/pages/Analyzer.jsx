import { useState, useCallback } from 'react'
import axios from 'axios'
import toast from 'react-hot-toast'
import { useDropzone } from 'react-dropzone'

// In production (Vercel), point to HuggingFace backend.
// In development (localhost), use Vite proxy.
const API_BASE = import.meta.env.VITE_BACKEND_URL || ''

/* ── Lab field definitions ─────────────────────────── */
const LAB_FIELDS = {
  pneumonia: [
    { key: 'wbc',         label: 'WBC Count',      unit: '×10³/µL', placeholder: '4–11'     },
    { key: 'crp',         label: 'CRP',             unit: 'mg/L',    placeholder: '0–10'     },
    { key: 'temperature', label: 'Temperature',     unit: '°C',      placeholder: '36.1–37.2'},
    { key: 'spo2',        label: 'SpO₂',            unit: '%',       placeholder: '95–100'   },
  ],
  diabetes: [
    { key: 'blood_glucose', label: 'Blood Glucose (F)', unit: 'mg/dL',   placeholder: '70–100'  },
    { key: 'hba1c',         label: 'HbA1c',             unit: '%',       placeholder: '4.0–5.7' },
    { key: 'bmi',           label: 'BMI',               unit: 'kg/m²',   placeholder: '18.5–25' },
    { key: 'cholesterol',   label: 'Cholesterol',       unit: 'mg/dL',   placeholder: '<200'    },
  ]
}

const SHAP_COLORS = [
  'linear-gradient(90deg,#6c63ff,#a855f7)',
  'linear-gradient(90deg,#06b6d4,#10b981)',
  'linear-gradient(90deg,#f43f5e,#f59e0b)',
]

function riskColor(level) {
  return level === 'High' ? 'var(--red)' : level === 'Moderate' ? 'var(--yellow)' : level === 'Low' ? 'var(--teal)' : 'var(--green)'
}

function statusChipClass(status) {
  if (!status) return 'normal'
  const s = status.toUpperCase()
  if (s === 'NORMAL') return 'normal'
  if (s === 'HIGH' || s === 'LOW' || s === 'DIABETIC' || s === 'OBESE') return 'abnormal'
  return 'warning'
}

/* ── Dropzone sub-component ── */
function XrayDropzone({ onFile }) {
  const [name, setName] = useState('')
  const onDrop = useCallback(files => {
    if (files[0]) { setName(files[0].name); onFile(files[0]) }
  }, [onFile])
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop, accept: {'image/*':['.png','.jpg','.jpeg','.dcm']}, maxFiles: 1
  })
  return (
    <div {...getRootProps()} className={`dropzone${isDragActive?' active':''}`}>
      <input {...getInputProps()} />
      <span className="dropzone-icon">🫁</span>
      {isDragActive
        ? <p>Drop it here...</p>
        : <><p>Drag &amp; drop chest X-ray<br /><small>PNG · JPG · DICOM supported</small></p></>
      }
      {name && <div className="file-chip">✅ {name}</div>}
    </div>
  )
}

/* ── Disease Result Card ── */
function DiseaseCard({ result, type }) {
  if (!result) return null
  const { risk_percentage, risk_level, recommendation, modalities_used, missing_modalities } = result
  return (
    <div className={`disease-card ${type}`}>
      <div className="dc-tag">{type === 'pneumonia' ? '🫁 Pneumonia' : '🩸 Diabetes'} &nbsp;·&nbsp; Risk Assessment</div>
      <div className="dc-score-row">
        <div className="dc-score" style={{color: riskColor(risk_level)}}>{risk_percentage}%</div>
        <span className={`risk-pill ${risk_level}`}>{risk_level} Risk</span>
      </div>
      <div className="prog-wrap">
        <div className="prog-labels"><span>Risk Score</span><span>{risk_percentage}%</span></div>
        <div className="prog-track">
          <div className={`prog-fill ${risk_level}`} style={{width:`${risk_percentage}%`}} />
        </div>
      </div>
      <div className={`dc-rec ${risk_level}`}>{recommendation}</div>
      <div style={{fontSize:'.75rem',color:'var(--text-2)',marginBottom:'.5rem',fontWeight:600}}>DATA SOURCES</div>
      <div className="mod-chips">
        {modalities_used?.map(m => <span key={m} className="mod-chip">✓ {m}</span>)}
        {missing_modalities?.map(m => <span key={m} className="mod-chip missing">⚠ {m} absent</span>)}
      </div>
    </div>
  )
}

/* ── SHAP Chart ── */
function ShapChart({ pneumoniaShap, diabetesShap }) {
  const pEntries = Object.entries(pneumoniaShap || {})
  const dEntries = Object.entries(diabetesShap  || {})
  if (!pEntries.length && !dEntries.length) return null
  return (
    <div className="shap-card">
      <div className="shap-header">
        <h3>Cross-Modal Explainability</h3>
        <span className="shap-badge">Novel Contribution</span>
      </div>
      <p className="shap-desc">
        Shows how much each data modality (X-ray image, clinical text, blood tests) contributed
        to the final disease risk prediction — a key research novelty of this MTech project.
      </p>
      {pEntries.length > 0 && (
        <div className="shap-section">
          <div className="shap-section-title"><span>🫁</span> Pneumonia — Modality Attribution</div>
          {pEntries.map(([key, val], i) => (
            <div key={key} className="shap-row">
              <div className="shap-label">{key}</div>
              <div className="shap-track">
                <div className="shap-fill" style={{width:`${Math.min(100,val*350)}%`, background: SHAP_COLORS[i%3]}} />
              </div>
              <div className="shap-pct" style={{color:'var(--teal)'}}>{(val*100).toFixed(1)}%</div>
            </div>
          ))}
        </div>
      )}
      {dEntries.length > 0 && (
        <div className="shap-section">
          <div className="shap-section-title"><span>🩸</span> Diabetes — Modality Attribution</div>
          {dEntries.map(([key, val], i) => (
            <div key={key} className="shap-row">
              <div className="shap-label">{key}</div>
              <div className="shap-track">
                <div className="shap-fill" style={{width:`${Math.min(100,val*350)}%`, background: SHAP_COLORS[i%3]}} />
              </div>
              <div className="shap-pct" style={{color:'#f59e0b'}}>{(val*100).toFixed(1)}%</div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

/* ══════════════════════════════════════════
   MAIN ANALYZER PAGE
══════════════════════════════════════════ */
export default function Analyzer() {
  const [clinicalText, setClinicalText] = useState('')
  const [xrayFile, setXrayFile]         = useState(null)
  const [labs, setLabs]                 = useState({})
  const [loading, setLoading]           = useState(false)
  const [results, setResults]           = useState(null)

  const setLab = (key, val) => setLabs(prev => ({...prev, [key]: val}))

  const handleAnalyze = async () => {
    const hasText  = clinicalText.trim().length > 0
    const hasImage = xrayFile !== null
    const hasLabs  = Object.values(labs).some(v => v !== '')
    if (!hasText && !hasImage && !hasLabs) {
      toast.error('Please provide at least one input.')
      return
    }
    setLoading(true); setResults(null)
    try {
      const fd = new FormData()
      if (hasText)  fd.append('clinical_text', clinicalText)
      if (hasImage) fd.append('image', xrayFile)
      if (hasLabs) {
        const obj = {}
        Object.entries(labs).forEach(([k,v]) => { if (v !== '') obj[k] = parseFloat(v) })
        fd.append('blood_data', JSON.stringify(obj))
      }
      const { data } = await axios.post(`${API_BASE}/api/analyze`, fd, {headers:{'Content-Type':'multipart/form-data'}})
      setResults(data)
      toast.success('Analysis complete!')
    } catch(err) {
      toast.error(err.response?.data?.detail || 'Analysis failed. Is the backend running?')
    } finally { setLoading(false) }
  }

  const summary = results?.results?.summary

  return (
    <div className="page analyzer-page">

      <div className="page-header">
        <h1>🏥 Medical Report <span className="grad-text">Analyzer</span></h1>
        <p>Provide any combination of inputs — our AI handles missing modalities automatically (novel contribution)</p>
      </div>

      {/* ── Input Row ── */}
      <div className="input-grid">

        {/* X-ray */}
        <div className="glass-card">
          <div className="card-header">
            <div className="card-icon">🫁</div>
            <span className="card-title">Chest X-ray</span>
            <span className="card-badge">Vision Module</span>
          </div>
          <XrayDropzone onFile={setXrayFile} />
          <p style={{fontSize:'.76rem',color:'var(--text-3)',marginTop:'.6rem'}}>
            For Pneumonia detection · Leave empty if unavailable
          </p>
        </div>

        {/* Clinical Text */}
        <div className="glass-card">
          <div className="card-header">
            <div className="card-icon">📝</div>
            <span className="card-title">Clinical Notes</span>
            <span className="card-badge">NLP Module</span>
          </div>
          <textarea
            placeholder={`Enter doctor notes, symptoms or discharge summary...\n\nExample: Patient presents with fever 38.5°C, productive cough, difficulty breathing. Crackles heard. SpO₂ 93%. HbA1c 7.2%, excessive thirst reported.`}
            value={clinicalText}
            onChange={e => setClinicalText(e.target.value)}
          />
        </div>
      </div>

      {/* ── Blood Tests ── */}
      <div className="glass-card" style={{marginBottom:'1rem'}}>
        <div className="card-header">
          <div className="card-icon">🩸</div>
          <span className="card-title">Blood Test Results</span>
          <span className="card-badge">Tabular Module</span>
        </div>
        <p style={{fontSize:'.84rem',color:'var(--text-2)',marginBottom:'1.2rem'}}>
          Enter available values — blank fields are handled by missing modality robustness (our research novelty)
        </p>

        <div className="labs-section-label">Pneumonia Indicators</div>
        <div className="labs-grid" style={{marginBottom:'1rem'}}>
          {LAB_FIELDS.pneumonia.map(f => (
            <div key={f.key} className="lab-field">
              <label>{f.label}</label>
              <div className="lab-input-wrap">
                <input type="number" step="any" placeholder={`Normal: ${f.placeholder}`}
                  value={labs[f.key]||''} onChange={e => setLab(f.key, e.target.value)} />
                <span className="lab-unit">{f.unit}</span>
              </div>
            </div>
          ))}
        </div>

        <div className="labs-section-label">Diabetes Indicators</div>
        <div className="labs-grid">
          {LAB_FIELDS.diabetes.map(f => (
            <div key={f.key} className="lab-field">
              <label>{f.label}</label>
              <div className="lab-input-wrap">
                <input type="number" step="any" placeholder={`Normal: ${f.placeholder}`}
                  value={labs[f.key]||''} onChange={e => setLab(f.key, e.target.value)} />
                <span className="lab-unit">{f.unit}</span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* ── Analyze Button ── */}
      <div className="analyze-wrap">
        <button className="analyze-btn" onClick={handleAnalyze} disabled={loading}>
          {loading ? <><div className="spinner"/>Analyzing all modalities...</> : <>🔬 Run Multimodal Analysis</>}
        </button>
      </div>

      {/* ── Results ── */}
      {results && (
        <div className="results-wrap">

          {/* Summary bar */}
          {summary && (
            <div className="summary-bar">
              <span>📊</span>
              <div><span className="sum-label">Primary Concern</span><br /><span className="sum-val">{summary.primary_concern}</span></div>
              <div className="sum-divider" />
              <div><span className="sum-label">Overall Risk</span><br /><span className="sum-val">{(summary.overall_risk_score*100).toFixed(1)}%</span></div>
              <div className="sum-divider" />
              <div><span className="sum-label">Modalities Used</span><br /><span className="sum-val">{results.input_summary?.text_provided && results.input_summary?.image_provided && results.input_summary?.labs_provided ? 'All 3' : 'Partial — handled ✓'}</span></div>
            </div>
          )}

          {/* Disease Cards */}
          <div className="disease-grid">
            <DiseaseCard result={results.results?.pneumonia} type="pneumonia" />
            <DiseaseCard result={results.results?.diabetes}  type="diabetes"  />
          </div>

          {/* SHAP */}
          <ShapChart
            pneumoniaShap={results.results?.pneumonia?.shap_contributions}
            diabetesShap={results.results?.diabetes?.shap_contributions}
          />

          {/* Blood Test Breakdown Table */}
          {results.module_outputs?.tabular && (() => {
            const rows = results.module_outputs.tabular.abnormal_flags || []
            return rows.length > 0 && (
              <div className="blood-table-card">
                <h3>🧪 Abnormal Blood Markers</h3>
                <div style={{overflowX:'auto'}}>
                  <table>
                    <thead>
                      <tr>
                        {['Marker','Your Value','Normal Range','Status'].map(h =>
                          <th key={h}>{h}</th>)}
                      </tr>
                    </thead>
                    <tbody>
                      {rows.map((r, i) => (
                        <tr key={i}>
                          <td style={{fontWeight:600}}>{r.marker}</td>
                          <td style={{color: 'var(--red)', fontWeight:700}}>{r.value}</td>
                          <td style={{color:'var(--text-2)'}}>{r.normal}</td>
                          <td>
                            <span className={`status-chip ${r.status === 'HIGH' ? 'High' : 'Low'}`}>{r.status}</span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )
          })()}

          {/* Individual module scores */}
          {(results.module_outputs?.nlp || results.module_outputs?.vision) && (
            <div className="blood-table-card" style={{marginTop:'1.2rem'}}>
              <h3>🔬 Module-Level Scores</h3>
              <div style={{display:'grid', gridTemplateColumns:'repeat(auto-fit,minmax(200px,1fr))', gap:'1rem', marginTop:'1rem'}}>
                {results.module_outputs?.nlp && ['pneumonia','diabetes'].map(d => (
                  <div key={d} style={{padding:'.9rem 1.1rem',background:'rgba(255,255,255,0.03)',borderRadius:'10px',border:'1px solid var(--border)'}}>
                    <div style={{fontSize:'.72rem',color:'var(--text-3)',fontWeight:700,textTransform:'uppercase',letterSpacing:'.06em',marginBottom:'.3rem'}}>
                      📝 NLP · {d}
                    </div>
                    <div style={{fontSize:'1.6rem',fontWeight:800,fontFamily:"'Space Grotesk',sans-serif",color:'var(--teal)'}}>
                      {results.module_outputs.nlp[`${d}_risk`]?.toFixed(0)}%
                    </div>
                    <div style={{fontSize:'.78rem',color:'var(--text-2)',marginTop:'.2rem'}}>
                      {results.module_outputs.nlp.key_findings?.filter(f => f.category.toLowerCase().includes(d)).map(f => f.term).slice(0,3).join(', ') || '—'}
                    </div>
                  </div>
                ))}
                {results.module_outputs?.vision && (
                  <div style={{padding:'.9rem 1.1rem',background:'rgba(255,255,255,0.03)',borderRadius:'10px',border:'1px solid var(--border)'}}>
                    <div style={{fontSize:'.72rem',color:'var(--text-3)',fontWeight:700,textTransform:'uppercase',letterSpacing:'.06em',marginBottom:'.3rem'}}>
                      🫁 Vision · Pneumonia
                    </div>
                    <div style={{fontSize:'1.6rem',fontWeight:800,fontFamily:"'Space Grotesk',sans-serif",color:'var(--grad-1)'}}>
                      {results.module_outputs.vision.pneumonia_risk?.toFixed(0)}%
                    </div>
                    <div style={{fontSize:'.78rem',color:'var(--text-2)',marginTop:'.2rem'}}>
                      {results.module_outputs.vision.radiological_findings?.slice(0,2).join(' • ') || '—'}
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}

        </div>
      )}
    </div>
  )
}
