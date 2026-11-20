"""Build a self-contained, interactive HTML dashboard from pipeline outputs."""
import base64
import html
import json

import pandas as pd

from config import FIG_DIR, OUT_DIR


def image_data(filename):
    path = FIG_DIR / filename
    if not path.exists():
        return ""
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def safe_json(value):
    """Keep embedded JSON from accidentally closing its script element."""
    return json.dumps(value, ensure_ascii=False).replace("<", "\\u003c")


def figure(filename, title, description):
    src = image_data(filename)
    if not src:
        return f'<article class="panel"><h3>{html.escape(title)}</h3><p class="muted">Run the pipeline to generate this chart.</p></article>'
    return (f'<article class="panel chart"><div class="panel-heading"><div><h3>{html.escape(title)}</h3>'
            f'<p class="muted">{html.escape(description)}</p></div></div>'
            f'<img src="{src}" alt="{html.escape(title)}"></article>')


def table(headers, rows, class_name=""):
    th = "".join(f"<th>{html.escape(str(x))}</th>" for x in headers)
    body = "".join("<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in row) + "</tr>" for row in rows)
    return f'<div class="table-wrap"><table class="{class_name}"><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>'


def run():
    eda = json.loads((OUT_DIR / "eda_stats.json").read_text(encoding="utf-8"))
    metrics = json.loads((OUT_DIR / "metrics.json").read_text(encoding="utf-8"))
    scores_path = OUT_DIR / "churn_scores.csv"
    scores = pd.read_csv(scores_path) if scores_path.exists() else pd.DataFrame(
        columns=["customer_id", "churn_probability", "risk_band"])
    best_name = metrics["best_model"]
    best = metrics["models"][best_name]
    risk_counts = scores["risk_band"].value_counts().to_dict() if not scores.empty else {}

    metric_rows = []
    for name, values in metrics["models"].items():
        metric_rows.append([name, f'{values["roc_auc"]:.3f}', f'{values["cv_auc"]:.3f}',
                            f'{values["accuracy"]:.1%}', f'{values["precision"]:.1%}',
                            f'{values["recall"]:.1%}', f'{values["f1"]:.3f}',
                            "Selected" if name == best_name else "—"])
    segment_rows = [[f'Segment {s["segment"]}', f'{s["customers"]:,}', f'{s["tenure"]:.1f} mo',
                     f'${s["monthly"]:.2f}', f'{s["tickets"]:.1f}', f'{s["churn_pct"]:.1f}%']
                    for s in metrics.get("segments", [])]
    cm = metrics.get("confusion_matrix", [[0, 0], [0, 0]])
    top_features = list(metrics.get("top_features", {}).items())
    risk_data = scores.fillna("").to_dict(orient="records")

    cards = [
        ("Customers analyzed", f'{eda["customers"]:,}', "Clean customer records"),
        ("Overall churn", f'{eda["churn_rate_pct"]:.1f}%', "Observed churn rate"),
        ("Average tenure", f'{eda["avg_tenure"]:.1f} mo', "Across all customers"),
        ("Average monthly bill", f'${eda["avg_monthly"]:.2f}', "Per customer"),
        ("Best model", best_name, f'ROC-AUC {best["roc_auc"]:.3f}'),
    ]
    card_html = "".join(f'<article class="kpi"><span>{html.escape(k)}</span><strong>{html.escape(str(v))}</strong><small>{html.escape(str(note))}</small></article>' for k, v, note in cards)
    feature_html = "".join(f'<div class="feature-row"><div><span>{html.escape(name.replace("_", " "))}</span><b>{value:.1%}</b></div><div class="track"><i style="width:{min(value / max((v for _, v in top_features), default=1) * 100, 100):.1f}%"></i></div></div>' for name, value in top_features)

    chart_specs = [
        ("fig_churn_distribution.png", "Churn mix", "Share of customers who stayed or left."),
        ("fig_churn_contract.png", "Churn by contract", "Month-to-month plans have the highest churn."),
        ("fig_churn_tenure.png", "Churn by tenure", "Churn rate across customer tenure groups."),
        ("fig_churn_internet.png", "Churn by internet service", "Compare churn across service types."),
        ("fig_churn_payment.png", "Churn by payment method", "Churn rates across payment preferences."),
        ("fig_monthly_charges.png", "Monthly charges", "Distribution of bills for retained and churned customers."),
        ("fig_correlation.png", "Feature correlation", "Pairwise numeric correlations, including churn."),
        ("fig_roc.png", "Model ROC curves", "Hold-out true positive rate versus false positive rate."),
        ("fig_confusion.png", "Best model confusion matrix", "Hold-out classification counts at a 0.5 threshold."),
        ("fig_feature_importance.png", "Top churn drivers", "Features with the strongest contribution in the selected model."),
        ("fig_segments.png", "Customer segment map", "Customers grouped by tenure and monthly charges."),
    ]
    charts_html = "".join(figure(*spec) for spec in chart_specs)
    risk_counts_html = "".join(f'<div class="risk-stat {html.escape(str(level).lower())}"><span>{html.escape(str(level))} risk</span><b>{int(count):,}</b></div>' for level, count in [("High", risk_counts.get("High", 0)), ("Medium", risk_counts.get("Medium", 0)), ("Low", risk_counts.get("Low", 0))])

    html_doc = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Customer Churn | Analytics</title>
<style>
:root{{--navy:#142b4a;--blue:#2563eb;--teal:#0f9f91;--red:#d94a5b;--amber:#d89922;--ink:#17243a;--muted:#718096;--line:#e7ecf2;--bg:#f4f7fb;--white:#fff}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 "Segoe UI",Arial,sans-serif}}button,input{{font:inherit}}
.layout{{min-height:100vh}}aside{{position:fixed;inset:0 auto 0 0;width:236px;background:var(--navy);color:white;padding:24px 16px;display:flex;flex-direction:column;z-index:4}}
.brand{{display:flex;align-items:center;gap:11px;padding:4px 10px 28px;font-size:15px;font-weight:700}}.brand-mark{{width:34px;height:34px;border-radius:10px;background:#2c70d9;display:grid;place-items:center;font-size:18px}}
.nav-label{{color:#91a7c1;font-size:10px;letter-spacing:.12em;text-transform:uppercase;padding:10px 12px}}nav a{{display:block;padding:10px 12px;margin:2px 0;color:#d5deea;text-decoration:none;border-radius:8px;font-size:13px}}nav a:hover,nav a.active{{background:#25466d;color:white}}.side-note{{margin-top:auto;padding:14px 12px;color:#9db0c5;font-size:11px;border-top:1px solid #2b4666}}
main{{margin-left:236px;max-width:1600px;padding:26px 34px 52px}}.topbar{{display:flex;justify-content:space-between;align-items:center;margin-bottom:25px}}.eyebrow{{color:var(--blue);font-size:11px;font-weight:700;letter-spacing:.12em;text-transform:uppercase}}h1{{font-size:27px;line-height:1.2;margin:4px 0 6px;letter-spacing:-.025em}}.subtitle,.muted{{color:var(--muted)}}.subtitle{{margin:0}}.pill{{background:#e5f5f2;color:#08796f;padding:7px 11px;border-radius:99px;font-size:11px;font-weight:700}}
.section{{margin:34px 0 0;scroll-margin-top:20px}}.section-head{{display:flex;justify-content:space-between;align-items:end;margin-bottom:14px}}h2{{font-size:18px;margin:0}}.section-head p{{margin:3px 0 0;color:var(--muted);font-size:12px}}.kpis{{display:grid;grid-template-columns:repeat(5,minmax(130px,1fr));gap:13px}}.kpi,.panel{{background:var(--white);border:1px solid var(--line);border-radius:12px;box-shadow:0 2px 7px #203b5a08}}.kpi{{padding:17px 18px;min-height:112px}}.kpi span,.kpi small{{display:block;color:var(--muted);font-size:11px}}.kpi strong{{display:block;font-size:22px;margin:8px 0 2px;color:var(--navy);letter-spacing:-.02em}}.kpi:nth-child(2) strong{{color:var(--red)}}
.grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}}.panel{{padding:17px;min-width:0}}.panel-heading{{display:flex;justify-content:space-between;gap:12px}}h3{{font-size:14px;margin:0 0 2px}}.panel .muted{{font-size:11px;margin:0}}.chart img{{display:block;width:100%;height:auto;margin-top:10px;border-radius:6px}}.insight-grid{{display:grid;grid-template-columns:1.1fr .9fr;gap:14px}}.feature-row{{margin-top:15px}}.feature-row>div:first-child{{display:flex;justify-content:space-between;gap:10px;font-size:11px;text-transform:capitalize}}.feature-row b{{color:var(--navy)}}.track{{height:6px;background:#edf1f6;border-radius:9px;margin-top:6px;overflow:hidden}}.track i{{height:100%;display:block;background:linear-gradient(90deg,#2d7ce4,#53b8b0);border-radius:9px}}
.summary-list{{list-style:none;padding:0;margin:8px 0 0}}.summary-list li{{padding:12px 0;border-bottom:1px solid var(--line);display:flex;justify-content:space-between;gap:10px}}.summary-list li:last-child{{border-bottom:0}}.summary-list span{{color:var(--muted)}}.summary-list b{{text-align:right}}.table-wrap{{overflow:auto}}table{{border-collapse:collapse;width:100%;font-size:12px;white-space:nowrap}}th{{text-align:left;color:var(--muted);background:#f7f9fc;font-size:10px;text-transform:uppercase;letter-spacing:.06em}}td,th{{padding:11px 12px;border-bottom:1px solid var(--line)}}tbody tr:last-child td{{border-bottom:0}}tbody tr:hover{{background:#f8fafc}}.best{{font-weight:700;color:var(--teal)}}.cm-grid{{display:grid;grid-template-columns:repeat(2,1fr);gap:8px;margin-top:14px}}.cm-cell{{background:#f3f7fc;padding:14px;border-radius:8px}}.cm-cell b{{display:block;font-size:20px;color:var(--navy)}}.cm-cell span{{font-size:11px;color:var(--muted)}}
.risk-stats{{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:14px}}.risk-stat{{padding:12px 14px;border-radius:9px;background:#f7f8fb}}.risk-stat span,.risk-stat b{{display:block}}.risk-stat span{{font-size:11px;color:var(--muted)}}.risk-stat b{{font-size:20px;margin-top:3px}}.risk-stat.high b{{color:var(--red)}}.risk-stat.medium b{{color:var(--amber)}}.risk-stat.low b{{color:var(--teal)}}.toolbar{{display:flex;gap:8px;margin:12px 0}}.toolbar input,.toolbar select{{border:1px solid var(--line);background:white;padding:9px 11px;border-radius:8px;color:var(--ink)}}.toolbar input{{flex:1;min-width:150px}}.risk-badge{{padding:4px 8px;border-radius:99px;font-size:10px;font-weight:700}}.risk-badge.high{{background:#ffeaed;color:#b52d40}}.risk-badge.medium{{background:#fff4d8;color:#996500}}.risk-badge.low{{background:#e1f6f0;color:#08796f}}.pagination{{display:flex;justify-content:space-between;align-items:center;margin-top:12px;color:var(--muted);font-size:11px}}.pagination button{{border:1px solid var(--line);background:white;border-radius:6px;padding:6px 10px;cursor:pointer}}.pagination button:disabled{{opacity:.45;cursor:default}}footer{{margin-top:32px;padding-top:15px;border-top:1px solid var(--line);font-size:11px;color:var(--muted)}}
@media(max-width:1050px){{aside{{width:190px}}main{{margin-left:190px;padding:22px}}.kpis{{grid-template-columns:repeat(3,1fr)}}}}@media(max-width:720px){{aside{{position:static;width:auto;padding:12px 14px}}.brand{{padding:0 4px 10px}}nav{{display:flex;overflow:auto;gap:4px}}.nav-label,.side-note{{display:none}}nav a{{white-space:nowrap;margin:0;padding:8px}}main{{margin:0;padding:20px 14px 40px}}.topbar{{align-items:flex-start;gap:10px}}h1{{font-size:22px}}.pill{{display:none}}.kpis{{grid-template-columns:repeat(2,1fr);gap:9px}}.kpi{{padding:13px;min-height:98px}}.kpi strong{{font-size:19px}}.grid,.insight-grid{{grid-template-columns:1fr}}.risk-stats{{gap:6px}}.risk-stat{{padding:9px}}}}
</style></head><body><div class="layout">
<aside><div class="brand"><div class="brand-mark">↗</div><span>CHURN<br>INTELLIGENCE</span></div><div class="nav-label">Workspace</div><nav>
<a class="active" href="#overview">Overview</a><a href="#drivers">Churn drivers</a><a href="#models">Model performance</a><a href="#segments">Customer segments</a><a href="#risk">Risk worklist</a></nav>
<div class="side-note">Analytics dashboard<br>Source: synthetic customer dataset</div></aside>
<main><header class="topbar"><div><div class="eyebrow">Customer analytics</div><h1>Churn overview</h1><p class="subtitle">Explore customer behavior, model quality, and accounts that may need attention.</p></div><span class="pill">PIPELINE OUTPUT</span></header>
<section class="section" id="overview"><div class="section-head"><div><h2>Business snapshot</h2><p>Current dataset and scoring summary.</p></div></div><div class="kpis">{card_html}</div>
<div class="grid" style="margin-top:14px">{figure(*chart_specs[0])}{figure(*chart_specs[1])}</div></section>
<section class="section" id="drivers"><div class="section-head"><div><h2>What is associated with churn?</h2><p>Observed group rates and model signals help prioritize investigation.</p></div></div>
<div class="grid">{''.join(figure(*x) for x in chart_specs[2:7])}</div>
<div class="insight-grid" style="margin-top:14px"><article class="panel"><h3>Leading model signals</h3><p class="muted">Relative importance values from {html.escape(best_name)}.</p>{feature_html}</article>
<article class="panel"><h3>Churn rates by customer group</h3><p class="muted">Rates are descriptive and do not establish cause.</p><ul class="summary-list">
<li><span>Contract type</span><b>{max(eda['by_contract'].items(), key=lambda x:x[1])[0]} · {max(eda['by_contract'].values()):.1f}%</b></li>
<li><span>Tenure band</span><b>{max(eda['by_tenure'].items(), key=lambda x:x[1])[0]} months · {max(eda['by_tenure'].values()):.1f}%</b></li>
<li><span>Internet service</span><b>{max(eda['by_internet'].items(), key=lambda x:x[1])[0]} · {max(eda['by_internet'].values()):.1f}%</b></li>
<li><span>Payment method</span><b>{max(eda['by_payment'].items(), key=lambda x:x[1])[0]} · {max(eda['by_payment'].values()):.1f}%</b></li>
<li><span>Correlation with tenure</span><b>{eda['corr_with_churn'].get('tenure_months', 0):.3f}</b></li></ul></article></div></section>
<section class="section" id="models"><div class="section-head"><div><h2>Model performance</h2><p>Hold-out metrics compare three classifiers. ROC-AUC is used to select the saved model.</p></div></div>
<div class="panel">{table(['Model','ROC-AUC','CV AUC','Accuracy','Precision','Recall','F1','Status'], metric_rows)}<div class="cm-grid">
<div class="cm-cell"><b>{cm[0][0]:,}</b><span>Correctly identified retained</span></div><div class="cm-cell"><b>{cm[0][1]:,}</b><span>Retained flagged as churn</span></div>
<div class="cm-cell"><b>{cm[1][0]:,}</b><span>Churn missed by model</span></div><div class="cm-cell"><b>{cm[1][1]:,}</b><span>Correctly identified churn</span></div></div></div>
<div class="grid" style="margin-top:14px">{figure(*chart_specs[7])}{figure(*chart_specs[8])}</div></section>
<section class="section" id="segments"><div class="section-head"><div><h2>Customer segments</h2><p>K-means groups based on tenure, monthly charges, and support tickets.</p></div></div>
<div class="panel">{table(['Segment','Customers','Avg. tenure','Avg. monthly bill','Support tickets','Churn rate'], segment_rows)}</div><div class="grid" style="margin-top:14px">{figure(*chart_specs[10])}{figure(*chart_specs[9])}</div></section>
<section class="section" id="risk"><div class="section-head"><div><h2>Customer risk worklist</h2><p>Scores are model estimates; use them to guide review, not as a guarantee of churn.</p></div></div>
<div class="panel"><div class="risk-stats">{risk_counts_html}</div><div class="toolbar"><input id="riskSearch" type="search" placeholder="Search customer ID"><select id="riskFilter"><option value="">All risk levels</option><option>High</option><option>Medium</option><option>Low</option></select></div>
<div class="table-wrap"><table><thead><tr><th>Customer ID</th><th>Churn probability</th><th>Risk band</th></tr></thead><tbody id="riskRows"></tbody></table></div>
<div class="pagination"><span id="pageInfo"></span><div><button id="prevPage">Previous</button> <button id="nextPage">Next</button></div></div></div></section>
<footer>Customer Churn Analytics · Synthetic data for demonstration · Model scores are estimates based on the generated dataset.</footer></main></div>
<script>
const riskData={safe_json(risk_data)};let page=0;const pageSize=25;
function renderRisk(){{const q=document.getElementById('riskSearch').value.trim().toLowerCase();const band=document.getElementById('riskFilter').value;const filtered=riskData.filter(x=>String(x.customer_id).toLowerCase().includes(q)&&(!band||x.risk_band===band));const pages=Math.max(1,Math.ceil(filtered.length/pageSize));page=Math.min(page,pages-1);const visible=filtered.slice(page*pageSize,(page+1)*pageSize);document.getElementById('riskRows').innerHTML=visible.map(x=>`<tr><td>${{x.customer_id}}</td><td>${{(Number(x.churn_probability)*100).toFixed(1)}}%</td><td><span class="risk-badge ${{String(x.risk_band).toLowerCase()}}">${{x.risk_band}}</span></td></tr>`).join('')||'<tr><td colspan="3">No matching customers.</td></tr>';document.getElementById('pageInfo').textContent=`${{filtered.length.toLocaleString()}} customers · page ${{page+1}} of ${{pages}}`;document.getElementById('prevPage').disabled=page===0;document.getElementById('nextPage').disabled=page>=pages-1;}}
document.getElementById('riskSearch').addEventListener('input',()=>{{page=0;renderRisk()}});document.getElementById('riskFilter').addEventListener('change',()=>{{page=0;renderRisk()}});document.getElementById('prevPage').addEventListener('click',()=>{{page--;renderRisk()}});document.getElementById('nextPage').addEventListener('click',()=>{{page++;renderRisk()}});renderRisk();
const links=[...document.querySelectorAll('nav a')];const sections=links.map(a=>document.querySelector(a.getAttribute('href')));const observer=new IntersectionObserver(entries=>entries.forEach(e=>{{if(e.isIntersecting)links.forEach(a=>a.classList.toggle('active',a.getAttribute('href')==='#'+e.target.id))}}),{{rootMargin:'-20% 0px -70% 0px'}});sections.forEach(s=>s&&observer.observe(s));
</script></body></html>'''
    (OUT_DIR / "dashboard.html").write_text(html_doc, encoding="utf-8")
    print(f"Saved {OUT_DIR / 'dashboard.html'}")


if __name__ == "__main__":
    run()
