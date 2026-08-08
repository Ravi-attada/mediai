const tech = [
  {name:'ClinicalBERT',  role:'NLP Module'}, {name:'DenseNet121', role:'Vision Module'},
  {name:'Clinical DNN', role:'Tabular ML'},  {name:'Cross-Modal Attention', role:'Fusion Layer'},
  {name:'SHAP',          role:'Explainability'},{name:'FastAPI',  role:'Backend API'},
  {name:'React.js',      role:'Frontend'},   {name:'PyTorch',    role:'DL Framework'},
]

const gaps = [
  {title:'No Unified Multi-Disease Framework', desc:'Existing papers target ONE disease only. This project uses ONE model for Pneumonia + Diabetes simultaneously.'},
  {title:'Poor Cross-Modal Explainability',    desc:'No paper shows which modality (image/text/labs) contributed most. Our attention attribution solves this.'},
  {title:'Missing Modality Not Handled',       desc:'All existing systems require ALL inputs. Ours works with any subset — critical for real clinical settings.'},
  {title:'No Real-Time Deployment',            desc:'Research models are offline only. This project includes a live web app with REST API deployment.'},
]

const datasets = [
  {name:'CheXpert',  desc:'224,316 chest X-ray images',               source:'Stanford ML Group'},
  {name:'NIH X-rays', desc:'112,120 chest X-rays (combined with CheXpert)', source:'NIH / Kaggle'},
  {name:'MTSamples', desc:'4,999 clinical notes and transcriptions',       source:'Kaggle'},
  {name:'PIMA',      desc:'Blood test data for diabetes prediction',    source:'Kaggle / UCI'},
]

const timeline = [
  {m:'Month 1–2',  t:'Literature Review + Dataset Preparation',                              done:true},
  {m:'Month 3–4',  t:'NLP Module — Bio_ClinicalBERT on MTSamples clinical notes',            done:true},
  {m:'Month 5–6',  t:'Vision Module — DenseNet121 on CheXpert + NIH (336K X-rays)',         done:true},
  {m:'Month 7',    t:'Multimodal Fusion — Cross-Modal Attention Network (Novel)',         done:true},
  {m:'Month 8',    t:'Explainability — Cross-Modal Attention Attribution + SHAP',        done:true},
  {m:'Month 9',    t:'Web App + Deployment — FastAPI + React full-stack',                done:true},
  {m:'Month 10',   t:'Thesis Writing + Research Paper Submission',                        done:false},
]

export default function About() {
  return (
    <div className="page about-page">

      <div className="about-hero">
        <span className="section-tag">MTech Research Project</span>
        <h1><span className="grad-text">AI-Powered Medical</span><br />Report Analyzer & Disease Risk Predictor</h1>
        <p style={{marginTop:'.8rem'}}>
          Multi-Modal AI System for Automated Clinical Report Analysis and Early Disease Risk Prediction
          using NLP and Deep Learning — targeting <strong>Pneumonia</strong> and <strong>Diabetes</strong>.
        </p>
      </div>

      {/* Problem Statement */}
      <div className="about-section">
        <h2>📋 Problem Statement</h2>
        <p>
          Clinical diagnosis requires analyzing multiple heterogeneous data sources — medical images,
          textual reports, and structured laboratory values. Existing AI systems handle these modalities
          in isolation and target only one disease. This project builds a <strong>unified multi-modal
          framework</strong> that fuses all three data types with attention-based fusion.
        </p>
        <p>
          The system predicts risk for Pneumonia (from chest X-ray + symptoms + blood markers) and
          Diabetes (from blood glucose, HbA1c, BMI + clinical notes) — simultaneously, in one pipeline,
          with cross-modal explainability.
        </p>
      </div>

      {/* Research Gaps */}
      <div className="about-section">
        <h2>🔬 Research Gaps Addressed</h2>
        <p style={{marginBottom:'1rem'}}>Identified from 25+ papers searched on PubMed (2023–2025):</p>
        <div className="gap-list">
          {gaps.map((g, i) => (
            <div key={i} className="gap-item">
              <div className="gap-num">{i+1}</div>
              <div>
                <div className="gap-title">{g.title}</div>
                <div className="gap-desc">{g.desc}</div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Novel Contributions */}
      <div className="about-section">
        <h2>✨ Novel Contributions</h2>
        {[
          ['Cross-modal explainability',   'SHAP attribution showing which modality (image/text/labs) drove each prediction'],
          ['Missing modality robustness',  'Attention reweighting — model works even when some inputs are absent'],
          ['Multi-disease unified model',  'One framework for Pneumonia + Diabetes (no existing paper does this)'],
          ['Clinical deployment',          'Full-stack web app with REST API — not just an offline research notebook'],
        ].map(([title, desc]) => (
          <p key={title} style={{marginBottom:'.5rem'}}>
            ✅ <strong>{title}</strong> — {desc}
          </p>
        ))}
      </div>

      {/* Tech Stack */}
      <div className="about-section">
        <h2>⚙️ Technology Stack</h2>
        <div className="tech-grid">
          {tech.map(t => (
            <div key={t.name} className="tech-chip">
              <div className="tech-chip-name">{t.name}</div>
              <div className="tech-chip-role">{t.role}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Datasets */}
      <div className="about-section">
        <h2>📁 Datasets Used</h2>
        <div style={{display:'flex',flexDirection:'column',gap:'.7rem',marginTop:'.5rem'}}>
          {datasets.map(d => (
            <div key={d.name} style={{display:'flex',gap:'1rem',alignItems:'center',padding:'.9rem 1.1rem',background:'var(--bg-card)',border:'1px solid var(--border)',borderRadius:'var(--radius-sm)'}}>
              <div style={{width:90,fontWeight:700,color:'var(--grad-1)',fontSize:'.9rem',flexShrink:0}}>{d.name}</div>
              <div style={{flex:1,fontSize:'.88rem',color:'var(--text-2)'}}>{d.desc}</div>
              <div style={{fontSize:'.78rem',padding:'.2rem .7rem',borderRadius:'50px',background:'rgba(16,185,129,0.1)',color:'var(--green)',fontWeight:600,whiteSpace:'nowrap'}}>{d.source}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Timeline */}
      <div className="about-section">
        <h2>🗓️ Project Timeline</h2>
        <div className="timeline">
          {timeline.map((r,i) => (
            <div key={i} className="tl-item">
              <div className={`tl-dot ${r.done?'done':'pending'}`} />
              <div className="tl-month">{r.m}</div>
              <div className="tl-task">{r.t}</div>
              <div className="tl-status">{r.done ? '✅ Done' : '⏳ Pending'}</div>
            </div>
          ))}
        </div>
      </div>

    </div>
  )
}
