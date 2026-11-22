"""Builds a simple static HTML report from the metrics dict (no extra libraries)."""

from html import escape

STYLE = """
body { font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif;
       max-width: 880px; margin: 2rem auto; padding: 0 1rem; color: #222; background: #fff; }
h1 { margin-bottom: 0.2rem; }
h2 { margin-top: 2rem; font-size: 1.15rem; }
.sub { color: #666; margin-top: 0; }
.cards { display: flex; flex-wrap: wrap; gap: 0.8rem; margin: 1.5rem 0; }
.card { flex: 1 1 170px; border: 1px solid #ddd; border-radius: 8px; padding: 0.7rem 1rem; }
.card .label { font-size: 0.75rem; color: #666; text-transform: uppercase; letter-spacing: 0.04em; }
.card .value { font-size: 1.35rem; font-weight: 600; margin-top: 0.2rem; }
table { border-collapse: collapse; width: 100%; }
th, td { border: 1px solid #ddd; padding: 6px 10px; text-align: left; font-size: 0.92rem; }
th { background: #f5f5f5; }
tr.best td { font-weight: 600; background: #fbf3f5; }
.bar { height: 10px; background: #7b2d43; border-radius: 3px; }
table.cm { width: auto; }
.cm td { text-align: center; width: 90px; }
.cm th { text-align: center; }
.footer { color: #888; font-size: 0.8rem; margin-top: 2.5rem; }
"""


def _cards(m):
    test = m["test"]
    items = [
        ("Selected model", m["best_model"].replace("_", " ")),
        ("Test accuracy", f"{test['accuracy']:.1%}"),
        ("Test macro F1", f"{test['macro_f1']:.3f}"),
        ("Train / test rows", f"{m['dataset']['n_train']} / {m['dataset']['n_test']}"),
    ]
    return "".join(
        f'<div class="card"><div class="label">{label}</div>'
        f'<div class="value">{escape(value)}</div></div>'
        for label, value in items
    )


def _cv_table(m):
    ranked = sorted(m["cv_results"].items(), key=lambda kv: kv[1]["mean_f1"], reverse=True)
    rows = []
    for name, res in ranked:
        css = ' class="best"' if name == m["best_model"] else ""
        rows.append(
            f"<tr{css}><td>{escape(name)}</td><td>{res['mean_f1']:.4f}</td>"
            f"<td>{res['std_f1']:.4f}</td>"
            f'<td style="width:40%"><div class="bar" style="width:{res["mean_f1"] * 100:.1f}%"></div></td></tr>'
        )
    return (
        "<table><tr><th>Model</th><th>Mean F1</th><th>Std</th><th></th></tr>"
        + "".join(rows)
        + "</table>"
    )


def _confusion_matrix(m):
    cm = m["test"]["confusion_matrix"]
    names = m["class_names"]
    header = "".join(f"<th>pred {escape(n)}</th>" for n in names)
    rows = []
    for name, row in zip(names, cm):
        total = sum(row) or 1
        cells = []
        for value in row:
            share = value / total
            color = "#fff" if share > 0.5 else "#222"
            cells.append(
                f'<td style="background: rgba(123, 45, 67, {share:.2f}); color: {color}">{value}</td>'
            )
        rows.append(f"<tr><th>true {escape(name)}</th>{''.join(cells)}</tr>")
    return f'<table class="cm"><tr><th></th>{header}</tr>{"".join(rows)}</table>'


def _per_class(m):
    report = m["test"]["per_class"]
    rows = []
    for name in m["class_names"]:
        r = report[name]
        rows.append(
            f"<tr><td>{escape(name)}</td><td>{r['precision']:.3f}</td>"
            f"<td>{r['recall']:.3f}</td><td>{r['f1-score']:.3f}</td><td>{int(r['support'])}</td></tr>"
        )
    return (
        "<table><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr>"
        + "".join(rows)
        + "</table>"
    )


def write_report(metrics, path):
    s = metrics["settings"]
    trained_at = metrics["trained_at"].replace("T", " ").replace("+00:00", " UTC")
    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Wine Model Report</title>
<style>{STYLE}</style>
</head>
<body>
<h1>Wine cultivar classifier</h1>
<p class="sub">Trained {escape(trained_at)} &middot; scikit-learn {escape(metrics["sklearn_version"])}
&middot; {s["cv_folds"]}-fold CV &middot; test size {s["test_size"]} &middot; seed {s["random_state"]}</p>

<div class="cards">{_cards(metrics)}</div>

<h2>Model comparison (cross validation, macro F1)</h2>
{_cv_table(metrics)}

<h2>Confusion matrix (test split)</h2>
{_confusion_matrix(metrics)}

<h2>Per-class scores (test split)</h2>
{_per_class(metrics)}

<p class="footer">Created by train.py inside the wine-lab1 container.</p>
</body>
</html>
"""
    with open(path, "w") as f:
        f.write(page)
