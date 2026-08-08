import { Link } from 'react-router-dom'

const features = [
  { icon: '📝', title: 'NLP Module', desc: 'Analyzes clinical notes using Bio_ClinicalBERT — a 110M-parameter BERT model pre-trained on clinical text, with dual-head output for Pneumonia & Diabetes.', tag: 'Month 3–4' },
  { icon: '🫁', title: 'Vision Module', desc: 'DenseNet121 deep learning architecture detects consolidation, opacity and infiltrates from chest X-ray images. Trained on 336K X-rays.', tag: 'Month 5–6' },
  { icon: '🩸', title: 'Tabular Module', desc: 'Clinical DNN (4-layer MLP) processes blood tests (HbA1c, WBC, SpO₂, CRP). Novel missing-value masking handles absent lab values.', tag: 'Month 3–4' },
  { icon: '⚡', title: 'Attention Fusion', desc: 'Cross-Modal Attention Network fuses Vision + NLP + Tabular embeddings. Learns which modality to trust most — works even when inputs are missing.', tag: 'Month 7' },
  { icon: '🔍', title: 'SHAP Explainability', desc: 'Shows exactly which modality (image, text, labs) drove the prediction — our novel research contribution.', tag: 'Month 8' },
  { icon: '🚀', title: 'Live Deployment', desc: 'Full-stack FastAPI + React app — deployable to Render / HuggingFace Spaces for free.', tag: 'Month 9' },
]

export default function Home() {
  return (
    <div className="page">

      {/* ── Hero ── */}
      <section className="hero">
        <div className="hero-badge">
          <span className="badge-dot" />
          MTech Research Project &nbsp;·&nbsp; AI + Healthcare
        </div>

        <h1 className="hero-title">
          AI-Powered Medical<br />
          <span className="grad-text">Report Analyzer</span><br />
          & Disease Risk Predictor
        </h1>

        <p className="hero-subtitle">
          Multi-Modal Deep Learning combining <strong>NLP</strong>, <strong>Computer Vision</strong>,
          and <strong>Tabular ML</strong> to predict Pneumonia &amp; Diabetes risk
          from clinical reports — with explainable AI.
        </p>

        <div className="hero-pills">
          {['Pneumonia Detection','Diabetes Prediction','SHAP Explainability','Missing Modality Robust','Multi-Disease Framework'].map(p =>
            <span key={p} className="hero-pill">{p}</span>)}
        </div>

        <div className="hero-actions">
          <Link to="/analyzer" className="btn btn-primary">🔬 Try the Analyzer</Link>
          <Link to="/about"    className="btn btn-glass">📄 Research Details</Link>
        </div>

        {/* Browser Preview Card */}
        <div className="hero-preview">
          <div className="preview-bar">
            <span className="preview-dot"/><span className="preview-dot"/><span className="preview-dot"/>
            <div className="preview-url">localhost:3000/analyzer</div>
          </div>
          <div className="preview-body">
            <div className="preview-metric">
              <div className="preview-metric-label">🫁 Pneumonia Risk</div>
              <div className="preview-metric-val" style={{color:'#f43f5e'}}>72%</div>
              <div className="preview-metric-bar">
                <div className="preview-metric-fill" style={{width:'72%',background:'linear-gradient(90deg,#f43f5e,#fb7185)'}}/>
              </div>
            </div>
            <div className="preview-metric">
              <div className="preview-metric-label">🩸 Diabetes Risk</div>
              <div className="preview-metric-val" style={{color:'#f59e0b'}}>48%</div>
              <div className="preview-metric-bar">
                <div className="preview-metric-fill" style={{width:'48%',background:'linear-gradient(90deg,#f59e0b,#fbbf24)'}}/>
              </div>
            </div>
            <div className="preview-metric">
              <div className="preview-metric-label">Primary Concern</div>
              <div className="preview-metric-val" style={{fontSize:'1.2rem',paddingTop:'.3rem'}}>🫁 Pneumonia</div>
            </div>
            <div className="preview-metric">
              <div className="preview-metric-label">Modalities Used</div>
              <div className="preview-metric-val" style={{fontSize:'1.2rem',paddingTop:'.3rem'}}>NLP + Vision + Labs</div>
            </div>
          </div>
        </div>
      </section>

      {/* ── Stats ── */}
      <div style={{padding:'0 1.5rem 3rem'}}>
        <div className="stats-row">
          {[
            {val:'3',     label:'Input Modalities'},
            {val:'2',     label:'Diseases Detected'},
            {val:'SHAP',  label:'Explainability Method'},
            {val:'✓',     label:'Missing Modality Robust'},
            {val:'Free',  label:'Open Datasets Used'},
          ].map(s => (
            <div key={s.label} className="stat-item">
              <span className="stat-val">{s.val}</span>
              <span className="stat-label">{s.label}</span>
            </div>
          ))}
        </div>
      </div>

      {/* ── Features ── */}
      <section className="section">
        <div className="section-inner">
          <div className="section-tag">Research Contributions</div>
          <h2 className="section-title">What Makes This <span className="grad-text">Novel?</span></h2>
          <p className="section-desc">Addressing real gaps found in 25+ published papers (PubMed 2023–2025)</p>
          <div className="features-grid">
            {features.map(f => (
              <div key={f.title} className="feature-card">
                <div className="feat-icon-wrap">{f.icon}</div>
                <h3>{f.title}</h3>
                <p>{f.desc}</p>
                <span className="feat-tag">{f.tag}</span>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── Pipeline ── */}
      <section className="section" style={{paddingTop:0}}>
        <div className="section-inner">
          <div className="section-tag">Architecture</div>
          <h2 className="section-title">AI <span className="grad-text">Pipeline</span></h2>
          <div className="pipeline">
            {[
              {icon:'📝', name:'Clinical Text', sub:'NLP Module'},
              null,
              {icon:'🫁', name:'Chest X-ray',   sub:'Vision Module'},
              null,
              {icon:'🩸', name:'Blood Tests',   sub:'Tabular Module'},
              null,
              {icon:'⚡', name:'Fusion Layer',  sub:'Attention Weights'},
              null,
              {icon:'📊', name:'Risk Score',    sub:'+ SHAP Explanation'},
            ].map((s, i) => s
              ? <div key={i} className="pipe-step">
                  <span className="pipe-icon">{s.icon}</span>
                  <div className="pipe-name">{s.name}</div>
                  <div className="pipe-sub">{s.sub}</div>
                </div>
              : <span key={i} className="pipe-arrow">→</span>
            )}
          </div>
        </div>
      </section>

    </div>
  )
}
