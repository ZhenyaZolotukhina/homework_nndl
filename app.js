// app.js

let data = [];
let importance = [];

document.getElementById("loadBtn").addEventListener("click", loadData);

async function loadData() {
  const train = await fetchCSV("train.csv", "train");
  const test = await fetchCSV("test.csv", "test");

  test.forEach(r => r.Survived = null);

  data = [...train, ...test];

  preprocess();
  computeImportance();
  populateFeatureSelect();
  renderImportance();
  renderConclusion();
}

function fetchCSV(path, source) {
  return new Promise(resolve => {
    Papa.parse(path, {
      download: true,
      header: true,
      dynamicTyping: true,
      complete: res => {
        res.data.forEach(r => r.source = source);
        resolve(res.data);
      }
    });
  });
}

function preprocess() {
  data.forEach(r => {
    r.Sex = r.Sex || "Missing";
    r.Embarked = r.Embarked || "Missing";
    r.FamilySize = (r.SibSp || 0) + (r.Parch || 0) + 1;
  });
}

function computeImportance() {
  const train = data.filter(r => r.source === "train");

  importance = [];

  // Sex (categorical)
  importance.push({
    feature: "Sex",
    score: cramersV(train.map(r => r.Sex), train.map(r => r.Survived))
  });

  // Pclass
  importance.push({
    feature: "Pclass",
    score: cramersV(train.map(r => r.Pclass), train.map(r => r.Survived))
  });

  // Age
  importance.push({
    feature: "Age",
    score: Math.abs(pointBiserial(
      train.map(r => r.Age),
      train.map(r => r.Survived)
    ))
  });

  importance.sort((a, b) => b.score - a.score);
}

function populateFeatureSelect() {
  const select = document.getElementById("featureSelect");
  select.innerHTML = "";
  importance.forEach(i => {
    const opt = document.createElement("option");
    opt.value = i.feature;
    opt.textContent = i.feature;
    select.appendChild(opt);
  });
  select.addEventListener("change", () => renderFeature(select.value));
  renderFeature(select.value);
}

function renderFeature(feature) {
  const train = data.filter(r => r.source === "train");

  const groups = {};
  train.forEach(r => {
    const k = r[feature];
    if (k == null) return;
    if (!groups[k]) groups[k] = { n: 0, s: 0 };
    groups[k].n++;
    groups[k].s += r.Survived === 1 ? 1 : 0;
  });

  const cats = Object.keys(groups);
  const rates = cats.map(c => groups[c].s / groups[c].n);

  Plotly.newPlot("barSurvival", [{
    type: "bar",
    x: cats,
    y: rates
  }], {
    yaxis: { tickformat: ".0%" },
    margin: { t: 30 }
  });

  const died = train.filter(r => r.Survived === 0).map(r => r[feature]).filter(v => v != null);
  const surv = train.filter(r => r.Survived === 1).map(r => r[feature]).filter(v => v != null);

  Plotly.newPlot("distPlot", [
    { type: "histogram", x: died, name: "Died", opacity: 0.6 },
    { type: "histogram", x: surv, name: "Survived", opacity: 0.6 }
  ], {
    barmode: "overlay",
    margin: { t: 30 }
  });
}

function renderImportance() {
  Plotly.newPlot("importancePlot", [{
    type: "bar",
    x: importance.map(i => i.score).reverse(),
    y: importance.map(i => i.feature).reverse(),
    orientation: "h"
  }], {
    margin: { t: 30 }
  });
}

function renderConclusion() {
  const top = importance[0];
  document.getElementById("finalText").innerHTML = `
    <p><b>Most important feature:</b> <code>${top.feature}</code></p>
    <p>
      Based on EDA, <code>${top.feature}</code> has the strongest relationship
      with survival. It shows the largest difference in survival rates between
      its categories and ranks highest by correlation / association metrics.
    </p>
    <p>
      This conclusion is supported by both visual analysis (bar plots,
      distributions) and numerical importance ranking.
    </p>
  `;
}

/* ---------- stats ---------- */

function pointBiserial(x, y) {
  let xs = [], ys = [];
  for (let i = 0; i < x.length; i++) {
    if (x[i] != null && (y[i] === 0 || y[i] === 1)) {
      xs.push(x[i]);
      ys.push(y[i]);
    }
  }
  const mean = arr => arr.reduce((a, b) => a + b, 0) / arr.length;
  const mx = mean(xs);
  const sx = Math.sqrt(xs.reduce((a, v) => a + (v - mx) ** 2, 0) / xs.length);

  const x1 = xs.filter((_, i) => ys[i] === 1);
  const x0 = xs.filter((_, i) => ys[i] === 0);

  const p = x1.length / xs.length;
  const q = 1 - p;

  return ((mean(x1) - mean(x0)) / sx) * Math.sqrt(p * q);
}

function cramersV(x, y) {
  const catsX = [...new Set(x)];
  const catsY = [...new Set(y)];
  const table = catsX.map(() => catsY.map(() => 0));

  for (let i = 0; i < x.length; i++) {
    const xi = catsX.indexOf(x[i]);
    const yi = catsY.indexOf(y[i]);
    if (xi >= 0 && yi >= 0) table[xi][yi]++;
  }

  const n = x.length;
  const rowSums = table.map(r => r.reduce((a, b) => a + b, 0));
  const colSums = catsY.map((_, j) => table.reduce((a, r) => a + r[j], 0));

  let chi2 = 0;
  for (let i = 0; i < table.length; i++) {
    for (let j = 0; j < table[0].length; j++) {
      const e = (rowSums[i] * colSums[j]) / n;
      chi2 += (table[i][j] - e) ** 2 / e;
    }
  }

  return Math.sqrt(chi2 / (n * Math.min(catsX.length - 1, catsY.length - 1)));
}
