// app.js
/* Titanic EDA Dashboard (browser-only, GitHub Pages friendly)
   - Loads train.csv and test.csv (auto fetch ./train.csv ./test.csv OR file inputs)
   - Merges into one dataset with `source` column
   - Preprocess + feature engineering
   - Interactive EDA: missingness, univariate, multivariate, importance, conclusion
*/

const App = (() => {
  const state = {
    rawTrain: null,
    rawTest: null,
    data: null,            // merged + processed
    imputeReport: null,
    featureTypes: null,    // { numeric:Set, categorical:Set }
    importance: null,      // array of {feature,type,score,metric,details}
    ready: false
  };

  // ---------- DOM ----------
  const el = {
    statusPill: document.getElementById("statusPill"),
    warnBox: document.getElementById("warnBox"),
    dataTag: document.getElementById("dataTag"),

    trainFile: document.getElementById("trainFile"),
    testFile: document.getElementById("testFile"),
    btnLoad: document.getElementById("btnLoad"),
    btnUseSample: document.getElementById("btnUseSample"),
    btnDownloadProcessed: document.getElementById("btnDownloadProcessed"),

    sourceFilter: document.getElementById("sourceFilter"),
    missingAsCategory: document.getElementById("missingAsCategory"),
    rareThreshold: document.getElementById("rareThreshold"),

    kpiRows: document.getElementById("kpiRows"),
    kpiCols: document.getElementById("kpiCols"),
    kpiTrain: document.getElementById("kpiTrain"),
    kpiSurv: document.getElementById("kpiSurv"),

    tabs: document.getElementById("tabs"),
    tabOverview: document.getElementById("tab-overview"),
    tabPreproc: document.getElementById("tab-preproc"),
    tabMissing: document.getElementById("tab-missing"),
    tabUnivariate: document.getElementById("tab-univariate"),
    tabMultivariate: document.getElementById("tab-multivariate"),
    tabImportance: document.getElementById("tab-importance"),
    tabConclusion: document.getElementById("tab-conclusion"),

    // Overview plots
    plotSource: document.getElementById("plotSource"),
    plotKeyRates: document.getElementById("plotKeyRates"),

    // Preproc
    plotImpute: document.getElementById("plotImpute"),
    featureList: document.getElementById("featureList"),

    // Missingness
    missingSourceView: document.getElementById("missingSourceView"),
    plotMissing: document.getElementById("plotMissing"),

    // Univariate
    uniFeature: document.getElementById("uniFeature"),
    uniMode: document.getElementById("uniMode"),
    uniTrainOnly: document.getElementById("uniTrainOnly"),
    plotUni: document.getElementById("plotUni"),
    uniTitle: document.getElementById("uniTitle"),
    uniInsights: document.getElementById("uniInsights"),

    // Multivariate
    mvX: document.getElementById("mvX"),
    mvY: document.getElementById("mvY"),
    mvTrainOnly: document.getElementById("mvTrainOnly"),
    plotMV: document.getElementById("plotMV"),
    mvTitle: document.getElementById("mvTitle"),

    // Importance
    impTopK: document.getElementById("impTopK"),
    plotImp: document.getElementById("plotImp"),
    impTableBody: document.getElementById("impTableBody"),
    impNote: document.getElementById("impNote"),

    // Conclusion
    finalConclusion: document.getElementById("finalConclusion"),
    plotTopProof: document.getElementById("plotTopProof"),
    talkTrack: document.getElementById("talkTrack"),
  };

  // ---------- Utils ----------
  const setStatus = (text) => { el.statusPill.textContent = `status: ${text}`; };
  const warn = (html) => {
    if (!html) { el.warnBox.style.display = "none"; el.warnBox.innerHTML = ""; return; }
    el.warnBox.style.display = "block";
    el.warnBox.innerHTML = html;
  };

  const isNil = (v) => v === null || v === undefined || (typeof v === "number" && Number.isNaN(v));
  const toNumber = (v) => {
    if (v === null || v === undefined) return null;
    const s = String(v).trim();
    if (s === "" || s.toLowerCase() === "nan") return null;
    const n = Number(s);
    return Number.isFinite(n) ? n : null;
  };
  const toStr = (v) => {
    if (v === null || v === undefined) return null;
    const s = String(v).trim();
    return s === "" ? null : s;
  };
  const clamp = (x, a, b) => Math.max(a, Math.min(b, x));

  const quantile = (arr, q) => {
    const xs = arr.filter(v => Number.isFinite(v)).slice().sort((a,b)=>a-b);
    if (!xs.length) return null;
    const pos = (xs.length - 1) * q;
    const base = Math.floor(pos);
    const rest = pos - base;
    if (xs[base + 1] === undefined) return xs[base];
    return xs[base] + rest * (xs[base + 1] - xs[base]);
  };

  const mode = (arr) => {
    const m = new Map();
    for (const v of arr) {
      if (v === null || v === undefined) continue;
      m.set(v, (m.get(v) || 0) + 1);
    }
    let best = null, bestC = -1;
    for (const [k,c] of m.entries()) {
      if (c > bestC) { bestC = c; best = k; }
    }
    return best;
  };

  const unique = (arr) => Array.from(new Set(arr));

  const downloadText = (filename, text) => {
    const blob = new Blob([text], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  };

  const toCSV = (rows) => {
    const cols = unique(rows.flatMap(r => Object.keys(r)));
    const esc = (v) => {
      if (v === null || v === undefined) return "";
      const s = String(v);
      if (/[",\n\r]/.test(s)) return `"${s.replace(/"/g,'""')}"`;
      return s;
    };
    const head = cols.join(",");
    const lines = rows.map(r => cols.map(c => esc(r[c])).join(","));
    return [head, ...lines].join("\n");
  };

  // ---------- CSV loading ----------
  async function fetchText(url) {
    const res = await fetch(url, { cache: "no-store" });
    if (!res.ok) throw new Error(`fetch failed: ${url} (${res.status})`);
    return await res.text();
  }

  function parseCSVText(text) {
    return new Promise((resolve, reject) => {
      Papa.parse(text, {
        header: true,
        skipEmptyLines: true,
        dynamicTyping: false,
        complete: (results) => {
          if (results.errors?.length) {
            // still resolve but warn
            resolve({ data: results.data, errors: results.errors });
          } else resolve({ data: results.data, errors: [] });
        },
        error: reject
      });
    });
  }

  async function readFileAsText(file) {
    return new Promise((resolve, reject) => {
      const r = new FileReader();
      r.onload = () => resolve(r.result);
      r.onerror = reject;
      r.readAsText(file);
    });
  }

  async function loadFromRepoOrFiles({ preferRepo = false } = {}) {
    setStatus("loading");
    warn("");

    try {
      let trainText = null;
      let testText = null;

      const trainFile = el.trainFile.files?.[0];
      const testFile = el.testFile.files?.[0];

      if (preferRepo) {
        trainText = await fetchText("./train.csv");
        testText  = await fetchText("./test.csv");
      } else if (trainFile && testFile) {
        trainText = await readFileAsText(trainFile);
        testText = await readFileAsText(testFile);
      } else {
        // fallback: try repo first, then error
        try {
          trainText = await fetchText("./train.csv");
          testText  = await fetchText("./test.csv");
        } catch {
          throw new Error("Не вижу train.csv/test.csv рядом со страницей и/или не выбраны файлы в форме.");
        }
      }

      const [trainParsed, testParsed] = await Promise.all([parseCSVText(trainText), parseCSVText(testText)]);
      if (trainParsed.errors?.length || testParsed.errors?.length) {
        warn(
          `<b>CSV parser warnings:</b><br/>` +
          [...trainParsed.errors, ...testParsed.errors].slice(0,3).map(e => `• ${e.message}`).join("<br/>")
        );
      }

      state.rawTrain = trainParsed.data;
      state.rawTest = testParsed.data;

      setStatus("processing");
      const { data, imputeReport, featureTypes } = preprocess(state.rawTrain, state.rawTest);
      state.data = data;
      state.imputeReport = imputeReport;
      state.featureTypes = featureTypes;

      setStatus("scoring importance");
      state.importance = computeImportance(data, {
        missingAsCategory: el.missingAsCategory.value === "yes",
        rareThreshold: Number(el.rareThreshold.value || 0.02)
      });

      state.ready = true;
      setStatus("ready");
      el.dataTag.textContent = "loaded";
      el.dataTag.classList.add("ok");
      el.btnDownloadProcessed.disabled = false;

      fillSelectors();
      renderAll();
    } catch (e) {
      console.error(e);
      setStatus("error");
      state.ready = false;
      el.dataTag.textContent = "error";
      el.dataTag.classList.remove("ok");
      warn(`<b>Ошибка загрузки:</b> ${escapeHtml(String(e.message || e))}`);
    }
  }

  // ---------- Preprocessing + Feature Engineering ----------
  function preprocess(trainRows, testRows) {
    // Clone + add source + unify columns
    const tTrain = trainRows.map(r => ({ ...r, source: "train" }));
    const tTest  = testRows.map(r => ({ ...r, source: "test" }));

    // Ensure Survived exists in all rows
    for (const r of tTest) if (!("Survived" in r)) r.Survived = null;

    // Union of keys
    const cols = unique([...tTrain, ...tTest].flatMap(r => Object.keys(r)));
    const normalize = (r) => {
      const out = {};
      for (const c of cols) out[c] = (c in r) ? r[c] : null;
      return out;
    };
    let merged = [...tTrain.map(normalize), ...tTest.map(normalize)];

    // Type coercion for known Kaggle Titanic schema
    const numericCols = new Set([
      "PassengerId", "Survived", "Pclass", "Age", "SibSp", "Parch", "Fare"
    ]);

    merged = merged.map(r => {
      const o = { ...r };
      // normalize blanks to null
      for (const k of Object.keys(o)) {
        const v = o[k];
        if (typeof v === "string") {
          const s = v.trim();
          o[k] = s === "" ? null : v;
        }
      }
      // numeric coercion
      for (const k of numericCols) {
        if (k in o) o[k] = toNumber(o[k]);
      }
      // categorical normalization
      if ("Sex" in o) o.Sex = normalizeCategory(o.Sex);
      if ("Embarked" in o) o.Embarked = normalizeCategory(o.Embarked);
      if ("Cabin" in o) o.Cabin = toStr(o.Cabin);
      if ("Ticket" in o) o.Ticket = toStr(o.Ticket);
      if ("Name" in o) o.Name = toStr(o.Name);
      return o;
    });

    // Feature engineering (before imputation where helpful)
    // Title from Name
    merged.forEach(r => {
      r.Title = extractTitle(r.Name);
    });

    // Deck from Cabin
    merged.forEach(r => {
      r.Deck = extractDeck(r.Cabin);
    });

    // FamilySize, IsAlone
    merged.forEach(r => {
      const sibsp = toNumber(r.SibSp) ?? 0;
      const parch = toNumber(r.Parch) ?? 0;
      const fs = sibsp + parch + 1;
      r.FamilySize = fs;
      r.IsAlone = fs === 1 ? 1 : 0;
    });

    // Ticket group size (count passengers with same ticket)
    const ticketCounts = new Map();
    for (const r of merged) {
      const t = r.Ticket;
      if (!t) continue;
      ticketCounts.set(t, (ticketCounts.get(t) || 0) + 1);
    }
    merged.forEach(r => {
      const c = r.Ticket ? (ticketCounts.get(r.Ticket) || 1) : 1;
      r.TicketGroupSize = c;
    });

    // FarePerPerson
    merged.forEach(r => {
      const fare = toNumber(r.Fare);
      const g = toNumber(r.TicketGroupSize) ?? 1;
      r.FarePerPerson = (fare === null) ? null : (g > 0 ? fare / g : fare);
    });

    // Imputation
    const report = {
      Age: { before: 0, after: 0, filled: 0, strategy: "median by (Sex×Pclass) fallback global median" },
      Fare:{ before: 0, after: 0, filled: 0, strategy: "median by Pclass fallback global median" },
      Embarked:{ before: 0, after: 0, filled: 0, strategy: "mode (train)" },
      Title:{ before: 0, after: 0, filled: 0, strategy: "Unknown for missing/rare" },
      Deck:{ before: 0, after: 0, filled: 0, strategy: "Missing for empty cabin" }
    };

    // Count before
    report.Age.before = merged.filter(r => isNil(r.Age)).length;
    report.Fare.before = merged.filter(r => isNil(r.Fare)).length;
    report.Embarked.before = merged.filter(r => isNil(r.Embarked)).length;
    report.Title.before = merged.filter(r => isNil(r.Title)).length;
    report.Deck.before = merged.filter(r => isNil(r.Deck)).length;

    // Age median by Sex×Pclass (train+test combined for stable median)
    const ageGroups = new Map(); // key -> array ages
    const allAges = [];
    for (const r of merged) {
      const a = toNumber(r.Age);
      if (a !== null) {
        allAges.push(a);
        const key = `${r.Sex || "Missing"}|${r.Pclass ?? "Missing"}`;
        if (!ageGroups.has(key)) ageGroups.set(key, []);
        ageGroups.get(key).push(a);
      }
    }
    const globalAgeMed = quantile(allAges, 0.5);
    const ageMedByKey = new Map();
    for (const [k, arr] of ageGroups.entries()) ageMedByKey.set(k, quantile(arr, 0.5));

    merged.forEach(r => {
      if (isNil(r.Age)) {
        const key = `${r.Sex || "Missing"}|${r.Pclass ?? "Missing"}`;
        r.Age = ageMedByKey.get(key) ?? globalAgeMed ?? null;
      }
    });

    // Fare median by Pclass
    const fareByClass = new Map();
    const allFares = [];
    for (const r of merged) {
      const f = toNumber(r.Fare);
      if (f !== null) {
        allFares.push(f);
        const k = String(r.Pclass ?? "Missing");
        if (!fareByClass.has(k)) fareByClass.set(k, []);
        fareByClass.get(k).push(f);
      }
    }
    const globalFareMed = quantile(allFares, 0.5);
    const fareMedByClass = new Map();
    for (const [k, arr] of fareByClass.entries()) fareMedByClass.set(k, quantile(arr, 0.5));
    merged.forEach(r => {
      if (isNil(r.Fare)) {
        const k = String(r.Pclass ?? "Missing");
        r.Fare = fareMedByClass.get(k) ?? globalFareMed ?? null;
      }
      // update FarePerPerson after Fare imputation
      const g = toNumber(r.TicketGroupSize) ?? 1;
      r.FarePerPerson = (r.Fare === null) ? null : (g > 0 ? r.Fare / g : r.Fare);
    });

    // Embarked mode from train only (common approach)
    const embarkedTrain = merged.filter(r => r.source === "train").map(r => r.Embarked).filter(v => v != null);
    const embarkedMode = mode(embarkedTrain) ?? mode(merged.map(r => r.Embarked).filter(v => v != null));
    merged.forEach(r => {
      if (isNil(r.Embarked)) r.Embarked = embarkedMode ?? "Missing";
    });

    // Deck: if missing -> Missing
    merged.forEach(r => { if (isNil(r.Deck)) r.Deck = "Missing"; });

    // Title: normalize rare titles into buckets (keeps EDA clean)
    merged.forEach(r => { if (isNil(r.Title)) r.Title = "Unknown"; });
    merged.forEach(r => { r.Title = normalizeTitle(r.Title); });

    // Update report after
    report.Age.after = merged.filter(r => isNil(r.Age)).length;
    report.Fare.after = merged.filter(r => isNil(r.Fare)).length;
    report.Embarked.after = merged.filter(r => isNil(r.Embarked)).length;
    report.Title.after = merged.filter(r => isNil(r.Title)).length;
    report.Deck.after = merged.filter(r => isNil(r.Deck)).length;

    report.Age.filled = report.Age.before - report.Age.after;
    report.Fare.filled = report.Fare.before - report.Fare.after;
    report.Embarked.filled = report.Embarked.before - report.Embarked.after;
    report.Title.filled = report.Title.before - report.Title.after;
    report.Deck.filled = report.Deck.before - report.Deck.after;

    // Identify feature types (numeric vs categorical) for dashboard
    const featureTypes = inferFeatureTypes(merged);

    return { data: merged, imputeReport: report, featureTypes };
  }

  function normalizeCategory(v) {
    const s = toStr(v);
    if (!s) return null;
    return s;
  }

  function extractTitle(name) {
    const s = toStr(name);
    if (!s) return null;
    // Typical format: "Last, Title. First ..."
    const m = s.match(/,\s*([^.]*)\./);
    if (!m) return null;
    return m[1].trim();
  }

  function normalizeTitle(title) {
    if (!title) return "Unknown";
    const t = title.trim();
    const common = new Set(["Mr", "Mrs", "Miss", "Master"]);
    if (common.has(t)) return t;
    // group known variants
    const royalty = new Set(["Lady","Countess","Dona","Sir","Jonkheer"]);
    const officer = new Set(["Dr","Rev","Col","Major","Capt"]);
    const nobility = new Set(["Don"]);
    if (royalty.has(t)) return "Royalty";
    if (officer.has(t)) return "Officer";
    if (nobility.has(t)) return "Nobility";
    // sometimes: Mme, Mlle, Ms
    if (t === "Mme") return "Mrs";
    if (t === "Mlle" || t === "Ms") return "Miss";
    return "Other";
  }

  function extractDeck(cabin) {
    const s = toStr(cabin);
    if (!s) return null;
    // cabin can have multiple like "C23 C25 C27"
    return s.trim()[0].toUpperCase();
  }

  function inferFeatureTypes(rows) {
    const cols = unique(rows.flatMap(r => Object.keys(r)));
    const numeric = new Set();
    const categorical = new Set();

    // Heuristic: if > 90% parsable as number => numeric (excluding ids with too many unique)
    for (const c of cols) {
      if (c === "Name" || c === "Ticket" || c === "Cabin") {
        categorical.add(c);
        continue;
      }
      const vals = rows.map(r => r[c]).filter(v => v !== null && v !== undefined);
      if (!vals.length) { categorical.add(c); continue; }
      let numCount = 0;
      for (const v of vals) {
        if (typeof v === "number" && Number.isFinite(v)) numCount++;
        else {
          const n = toNumber(v);
          if (n !== null) numCount++;
        }
      }
      const ratio = numCount / vals.length;

      // treat PassengerId as numeric but not for importance; still numeric for plots
      if (ratio >= 0.90) numeric.add(c);
      else categorical.add(c);
    }

    // Force some known categorical
    ["Sex","Embarked","Title","Deck","source"].forEach(c => categorical.add(c));
    // Force some known numeric
    ["Pclass","Age","SibSp","Parch","Fare","FamilySize","IsAlone","TicketGroupSize","FarePerPerson","Survived"].forEach(c => numeric.add(c));

    // Remove overlaps
    for (const c of numeric) categorical.delete(c);

    return { numeric, categorical };
  }

  // ---------- Filtering ----------
  function getRowsBySource(rows, source) {
    if (source === "all") return rows;
    return rows.filter(r => r.source === source);
  }

  function getTrainRows(rows) {
    return rows.filter(r => r.source === "train" && (r.Survived === 0 || r.Survived === 1));
  }

  // ---------- Missingness ----------
  function computeMissingness(rows) {
    const cols = unique(rows.flatMap(r => Object.keys(r)));
    const out = cols.map(c => {
      const total = rows.length;
      const miss = rows.reduce((acc, r) => acc + (isMissingValue(r[c]) ? 1 : 0), 0);
      return { col: c, missing: miss, total, ratio: total ? miss / total : 0 };
    });
    out.sort((a,b)=>b.ratio - a.ratio);
    return out;
  }

  function isMissingValue(v) {
    if (v === null || v === undefined) return true;
    if (typeof v === "number") return !Number.isFinite(v);
    if (typeof v === "string") return v.trim() === "";
    return false;
  }

  // ---------- Importance Metrics ----------
  // Numeric: point-biserial correlation between x and y (binary)
  function pointBiserial(xArr, yArr) {
    // yArr is 0/1; return correlation r
    const xs = [];
    const ys = [];
    for (let i=0;i<xArr.length;i++){
      const x = xArr[i], y = yArr[i];
      if (!Number.isFinite(x) || !(y === 0 || y === 1)) continue;
      xs.push(x); ys.push(y);
    }
    const n = xs.length;
    if (n < 10) return null;

    let meanX = 0;
    for (const v of xs) meanX += v;
    meanX /= n;

    let sdx = 0;
    for (const v of xs) sdx += (v - meanX) ** 2;
    sdx = Math.sqrt(sdx / (n - 1));
    if (!Number.isFinite(sdx) || sdx === 0) return null;

    // group means
    let n1=0,n0=0, m1=0,m0=0;
    for (let i=0;i<n;i++){
      if (ys[i] === 1) { n1++; m1 += xs[i]; }
      else { n0++; m0 += xs[i]; }
    }
    if (n1 === 0 || n0 === 0) return null;
    m1 /= n1; m0 /= n0;

    // r_pb = (m1 - m0) / sdx * sqrt(p*q)
    const p = n1 / n;
    const q = n0 / n;
    const r = ((m1 - m0) / sdx) * Math.sqrt(p*q);
    return Number.isFinite(r) ? r : null;
  }

  // Cramer's V for categorical X and binary Y
  function cramersV(xArr, yArr) {
    // Build contingency table
    const xVals = [];
    const yVals = [];
    for (let i=0;i<xArr.length;i++){
      const x = xArr[i], y = yArr[i];
      if (x === null || x === undefined) continue;
      if (!(y === 0 || y === 1)) continue;
      xVals.push(String(x));
      yVals.push(String(y));
    }
    const n = xVals.length;
    if (n < 10) return null;

    const xCats = unique(xVals);
    const yCats = unique(yVals); // should be 2
    if (xCats.length < 2 || yCats.length < 2) return null;

    const table = Array.from({ length: xCats.length }, () => Array(yCats.length).fill(0));
    const xi = new Map(xCats.map((v,i)=>[v,i]));
    const yi = new Map(yCats.map((v,i)=>[v,i]));

    for (let i=0;i<n;i++){
      table[xi.get(xVals[i])][yi.get(yVals[i])] += 1;
    }

    // chi-square
    const rowSums = table.map(r => r.reduce((a,b)=>a+b,0));
    const colSums = Array(yCats.length).fill(0);
    for (let j=0;j<yCats.length;j++){
      for (let i=0;i<xCats.length;i++) colSums[j] += table[i][j];
    }
    let chi2 = 0;
    for (let i=0;i<xCats.length;i++){
      for (let j=0;j<yCats.length;j++){
        const expected = (rowSums[i]*colSums[j]) / n;
        if (expected > 0) chi2 += ((table[i][j]-expected)**2)/expected;
      }
    }

    const r = xCats.length;
    const k = yCats.length;
    const phi2 = chi2 / n;
    // Cramer's V
    const v = Math.sqrt(phi2 / Math.min(k-1, r-1));
    return Number.isFinite(v) ? v : null;
  }

  function collapseRareCategories(values, thresholdRatio = 0.02) {
    // values: array of strings (can include null)
    const clean = values.map(v => (v === null || v === undefined) ? null : String(v));
    const total = clean.filter(v => v != null).length;
    if (!total || thresholdRatio <= 0) return clean;

    const counts = new Map();
    for (const v of clean) {
      if (v == null) continue;
      counts.set(v, (counts.get(v) || 0) + 1);
    }
    const rare = new Set();
    for (const [k,c] of counts.entries()) {
      if (c / total < thresholdRatio) rare.add(k);
    }
    return clean.map(v => (v != null && rare.has(v)) ? "Other" : v);
  }

  function computeImportance(allRows, opts) {
    const rows = getTrainRows(allRows); // importance on train only
    const y = rows.map(r => r.Survived);

    const { numeric, categorical } = state.featureTypes;

    // Candidate features (exclude leakage/ids/high-cardinality text)
    const exclude = new Set(["Survived","PassengerId","Name","Ticket","Cabin"]);
    const candidates = [];

    for (const f of numeric) if (!exclude.has(f)) candidates.push({ f, type: "numeric" });
    for (const f of categorical) if (!exclude.has(f)) candidates.push({ f, type: "categorical" });

    const missingAsCategory = !!opts.missingAsCategory;
    const rareThreshold = Number.isFinite(opts.rareThreshold) ? opts.rareThreshold : 0.02;

    const scored = [];

    for (const { f, type } of candidates) {
      if (type === "numeric") {
        const x = rows.map(r => toNumber(r[f]));
        const rpb = pointBiserial(x, y);
        if (rpb === null) continue;
        scored.push({
          feature: f,
          type,
          score: Math.abs(rpb),
          metric: "abs(point-biserial r)",
          details: { r: rpb }
        });
      } else {
        let x = rows.map(r => r[f]);
        x = x.map(v => {
          if (v === null || v === undefined || String(v).trim() === "") return missingAsCategory ? "Missing" : null;
          return String(v);
        });
        x = collapseRareCategories(x, rareThreshold);

        // if too many categories, skip (keeps EDA meaningful)
        const k = unique(x.filter(v => v != null)).length;
        if (k > 30) continue;

        const v = cramersV(x, y);
        if (v === null) continue;
        scored.push({
          feature: f,
          type,
          score: v,
          metric: "Cramér’s V",
          details: { categories: k }
        });
      }
    }

    scored.sort((a,b)=>b.score - a.score);
    return scored;
  }

  // ---------- Rendering ----------
  function fillSelectors() {
    if (!state.ready) return;

    const { numeric, categorical } = state.featureTypes;

    // Univariate feature list: prioritize common + engineered
    const priority = [
      "Sex","Pclass","Age","Fare","Embarked",
      "Title","Deck","FamilySize","IsAlone","TicketGroupSize","FarePerPerson",
      "SibSp","Parch"
    ];
    const all = unique([...numeric, ...categorical]).filter(f => f !== "Survived");
    const ordered = unique([...priority.filter(f => all.includes(f)), ...all.filter(f => !priority.includes(f))]);

    el.uniFeature.innerHTML = ordered.map(f => `<option value="${escapeAttr(f)}">${escapeHtml(f)}</option>`).join("");

    // Multivariate selectors: allow all except Survived/source
    const mvAll = ordered.filter(f => f !== "source");
    const toOptions = (list) => list.map(f => `<option value="${escapeAttr(f)}">${escapeHtml(f)}</option>`).join("");
    el.mvX.innerHTML = toOptions(mvAll);
    el.mvY.innerHTML = toOptions(mvAll);

    // sensible defaults
    if (mvAll.includes("Age")) el.mvX.value = "Age";
    if (mvAll.includes("Fare")) el.mvY.value = "Fare";
  }

  function renderAll() {
    if (!state.ready) return;
    renderKPIs();
    renderOverview();
    renderPreproc();
    renderMissingness();
    renderUnivariate();
    renderMultivariate();
    renderImportance();
    renderConclusion();
  }

  function renderKPIs() {
    const data = state.data;
    const cols = unique(data.flatMap(r => Object.keys(r)));
    const train = getTrainRows(data);

    el.kpiRows.textContent = String(data.length);
    el.kpiCols.textContent = String(cols.length);
    el.kpiTrain.textContent = String(train.length);

    const surv = train.length ? (train.reduce((a,r)=>a + (r.Survived === 1 ? 1 : 0),0) / train.length) : 0;
    el.kpiSurv.textContent = train.length ? `${(surv*100).toFixed(1)}%` : "—";
  }

  function renderOverview() {
    const data = state.data;

    // Source pie
    const sources = ["train","test"];
    const counts = sources.map(s => data.filter(r => r.source === s).length);
    Plotly.newPlot(el.plotSource, [{
      type: "pie",
      labels: sources,
      values: counts,
      hole: 0.55,
      textinfo: "label+percent",
      hovertemplate: "%{label}: %{value}<extra></extra>"
    }], baseLayout(""), { displayModeBar: false, responsive: true });

    // Key survival rates (train only): Sex, Pclass, Embarked (if exist)
    const train = getTrainRows(data);
    const keys = ["Sex","Pclass","Embarked","Title","Deck"];
    const available = keys.filter(k => train.some(r => r[k] != null));
    const traces = [];

    for (const k of available.slice(0, 3)) {
      const rates = computeSurvivalRateByCategory(train, k, {
        missingAsCategory: true,
        rareThreshold: 0.02,
        topN: 12
      });
      traces.push({
        type: "bar",
        name: k,
        x: rates.map(d => d.category),
        y: rates.map(d => d.rate),
        hovertemplate: `${k}=%{x}<br>Survival rate=%{y:.3f}<br>n=%{customdata}<extra></extra>`,
        customdata: rates.map(d => d.n)
      });
    }

    Plotly.newPlot(el.plotKeyRates, traces, baseLayout("", {
      barmode: "group",
      yaxis: { title: "Survival rate", tickformat: ".0%", range: [0,1] },
      xaxis: { title: "Category" }
    }), { displayModeBar: false, responsive: true });
  }

  function renderPreproc() {
    const rep = state.imputeReport;
    const keys = Object.keys(rep);
    const filled = keys.map(k => rep[k].filled);
    const before = keys.map(k => rep[k].before);
    const after = keys.map(k => rep[k].after);

    const trace1 = { type:"bar", name:"missing before", x: keys, y: before, hovertemplate:"%{x}<br>before=%{y}<extra></extra>" };
    const trace2 = { type:"bar", name:"filled", x: keys, y: filled, hovertemplate:"%{x}<br>filled=%{y}<extra></extra>" };
    const trace3 = { type:"bar", name:"missing after", x: keys, y: after, hovertemplate:"%{x}<br>after=%{y}<extra></extra>" };

    Plotly.newPlot(el.plotImpute, [trace1, trace2, trace3], baseLayout("", {
      barmode: "group",
      yaxis: { title: "Count" },
      xaxis: { title: "Field" }
    }), { displayModeBar: false, responsive: true });

    // Feature list
    const { numeric, categorical } = state.featureTypes;
    const list = [
      `<div><b>Numeric:</b> ${[...numeric].sort().map(x => `<span class="mono">${escapeHtml(x)}</span>`).join(", ")}</div>`,
      `<div style="margin-top:8px;"><b>Categorical:</b> ${[...categorical].sort().map(x => `<span class="mono">${escapeHtml(x)}</span>`).join(", ")}</div>`
    ].join("");
    el.featureList.innerHTML = list;
  }

  function renderMissingness() {
    const source = el.missingSourceView.value;
    const rows = getRowsBySource(state.data, source);
    const miss = computeMissingness(rows);

    Plotly.newPlot(el.plotMissing, [{
      type: "bar",
      x: miss.map(d => d.col),
      y: miss.map(d => d.ratio),
      customdata: miss.map(d => d.missing),
      hovertemplate: "%{x}<br>missing=%{customdata} (%{y:.1%})<extra></extra>"
    }], baseLayout("", {
      yaxis: { title: "Missing ratio", tickformat: ".0%" },
      xaxis: { title: "Column", tickangle: -35 }
    }), { displayModeBar: false, responsive: true });
  }

  function renderUnivariate() {
    const feature = el.uniFeature.value;
    const mode = el.uniMode.value;
    const trainOnly = el.uniTrainOnly.value === "yes";
    const missingAsCategory = el.missingAsCategory.value === "yes";
    const rareThreshold = Number(el.rareThreshold.value || 0.02);

    const baseRows = getRowsBySource(state.data, el.sourceFilter.value);
    const rows = trainOnly ? getTrainRows(baseRows) : baseRows;

    const { numeric, categorical } = state.featureTypes;
    const isNum = numeric.has(feature) && feature !== "Survived";
    const isCat = categorical.has(feature);

    el.uniTitle.textContent = `Univariate: ${feature}`;

    // If no Survived, only distribution mode makes sense
    const canUseSurv = rows.some(r => r.Survived === 0 || r.Survived === 1);

    const chosen = (mode === "auto")
      ? (canUseSurv ? (isCat ? "survival_rate" : "distribution") : "distribution")
      : mode;

    if (chosen === "distribution") {
      if (isNum) {
        renderNumericDistribution(rows, feature, canUseSurv);
      } else {
        renderCategoricalDistribution(rows, feature, { missingAsCategory, rareThreshold });
      }
    } else if (chosen === "survival_rate") {
      if (!canUseSurv) {
        Plotly.purge(el.plotUni);
        el.uniInsights.innerHTML = `<div class="muted">Нужно train-only или наличие Survived для расчёта survival rate.</div>`;
        return;
      }
      renderSurvivalRate(rows, feature, { isNum, isCat, missingAsCategory, rareThreshold });
    }

    el.uniInsights.innerHTML = buildUnivariateInsights(rows, feature, { isNum, isCat, missingAsCategory, rareThreshold });
  }

  function renderNumericDistribution(rows, feature, canUseSurv) {
    const xAll = rows.map(r => toNumber(r[feature])).filter(v => v !== null);
    if (!xAll.length) {
      Plotly.purge(el.plotUni);
      return;
    }

    const traces = [];
    if (canUseSurv) {
      const x0 = rows.filter(r => r.Survived === 0).map(r => toNumber(r[feature])).filter(v => v !== null);
      const x1 = rows.filter(r => r.Survived === 1).map(r => toNumber(r[feature])).filter(v => v !== null);

      traces.push({
        type: "histogram",
        name: "Died (0)",
        x: x0,
        opacity: 0.65,
        nbinsx: 30,
        hovertemplate: "Died<br>value=%{x}<extra></extra>"
      });
      traces.push({
        type: "histogram",
        name: "Survived (1)",
        x: x1,
        opacity: 0.65,
        nbinsx: 30,
        hovertemplate: "Survived<br>value=%{x}<extra></extra>"
      });

      Plotly.newPlot(el.plotUni, traces, baseLayout("", {
        barmode: "overlay",
        xaxis: { title: feature },
        yaxis: { title: "Count" }
      }), { displayModeBar: false, responsive: true });
    } else {
      traces.push({
        type: "histogram",
        name: feature,
        x: xAll,
        nbinsx: 30,
        hovertemplate: "value=%{x}<extra></extra>"
      });
      Plotly.newPlot(el.plotUni, traces, baseLayout("", {
        xaxis: { title: feature },
        yaxis: { title: "Count" }
      }), { displayModeBar: false, responsive: true });
    }
  }

  function renderCategoricalDistribution(rows, feature, { missingAsCategory, rareThreshold }) {
    let x = rows.map(r => r[feature]);
    x = x.map(v => {
      if (v === null || v === undefined || String(v).trim() === "") return missingAsCategory ? "Missing" : null;
      return String(v);
    }).filter(v => v != null);

    if (!x.length) { Plotly.purge(el.plotUni); return; }

    x = collapseRareCategories(x, rareThreshold);

    const counts = new Map();
    for (const v of x) counts.set(v, (counts.get(v) || 0) + 1);
    const items = [...counts.entries()].map(([k,c]) => ({ k, c })).sort((a,b)=>b.c-a.c).slice(0, 20);

    Plotly.newPlot(el.plotUni, [{
      type: "bar",
      x: items.map(d => d.k),
      y: items.map(d => d.c),
      hovertemplate: `${feature}=%{x}<br>n=%{y}<extra></extra>`
    }], baseLayout("", {
      xaxis: { title: feature, tickangle: -25 },
      yaxis: { title: "Count" }
    }), { displayModeBar: false, responsive: true });
  }

  function renderSurvivalRate(rows, feature, { isNum, isCat, missingAsCategory, rareThreshold }) {
    if (isCat) {
      const rates = computeSurvivalRateByCategory(rows, feature, { missingAsCategory, rareThreshold, topN: 16 });
      Plotly.newPlot(el.plotUni, [{
        type: "bar",
        x: rates.map(d => d.category),
        y: rates.map(d => d.rate),
        customdata: rates.map(d => d.n),
        hovertemplate: `${feature}=%{x}<br>rate=%{y:.2%}<br>n=%{customdata}<extra></extra>`
      }], baseLayout("", {
        xaxis: { title: feature, tickangle: -25 },
        yaxis: { title: "Survival rate", tickformat: ".0%", range: [0,1] }
      }), { displayModeBar: false, responsive: true });
    } else if (isNum) {
      // Bin numeric into quantiles
      const xs = rows.map(r => toNumber(r[feature])).filter(v => v !== null);
      if (xs.length < 20) { Plotly.purge(el.plotUni); return; }

      const q = [0, .2, .4, .6, .8, 1].map(p => quantile(xs, p));
      const bins = [];
      for (let i=0;i<q.length-1;i++){
        const a = q[i], b = q[i+1];
        bins.push({ a, b, label: `${fmtNum(a)}–${fmtNum(b)}` });
      }

      const stats = bins.map(bin => {
        const inBin = rows.filter(r => {
          const v = toNumber(r[feature]);
          if (v === null) return false;
          // include right edge in last bin
          const last = (bin === bins[bins.length - 1]);
          return last ? (v >= bin.a && v <= bin.b) : (v >= bin.a && v < bin.b);
        });
        const n = inBin.length;
        const s = n ? inBin.reduce((acc,r)=>acc + (r.Survived === 1 ? 1 : 0),0) : 0;
        return { label: bin.label, rate: n ? s/n : 0, n };
      });

      Plotly.newPlot(el.plotUni, [{
        type: "bar",
        x: stats.map(d => d.label),
        y: stats.map(d => d.rate),
        customdata: stats.map(d => d.n),
        hovertemplate: `${feature} bin=%{x}<br>rate=%{y:.2%}<br>n=%{customdata}<extra></extra>`
      }], baseLayout("", {
        xaxis: { title: `${feature} (quantile bins)`, tickangle: -25 },
        yaxis: { title: "Survival rate", tickformat: ".0%", range: [0,1] }
      }), { displayModeBar: false, responsive: true });
    } else {
      Plotly.purge(el.plotUni);
    }
  }

  function computeSurvivalRateByCategory(rows, feature, { missingAsCategory = true, rareThreshold = 0.02, topN = 12 } = {}) {
    let x = rows.map(r => r[feature]);
    x = x.map(v => {
      if (v === null || v === undefined || String(v).trim() === "") return missingAsCategory ? "Missing" : null;
      return String(v);
    });

    x = collapseRareCategories(x, rareThreshold);

    const stats = new Map(); // cat -> {n, s}
    for (let i=0;i<rows.length;i++){
      const cat = x[i];
      const y = rows[i].Survived;
      if (cat == null) continue;
      if (!(y === 0 || y === 1)) continue;
      if (!stats.has(cat)) stats.set(cat, { n: 0, s: 0 });
      const obj = stats.get(cat);
      obj.n += 1;
      obj.s += (y === 1 ? 1 : 0);
    }

    let arr = [...stats.entries()].map(([category, v]) => ({
      category, n: v.n, rate: v.n ? v.s / v.n : 0
    }));

    // sort by n desc for stability, then show topN
    arr.sort((a,b)=>b.n-a.n);
    arr = arr.slice(0, topN);

    // resort by rate desc for interpretation
    arr.sort((a,b)=>b.rate-a.rate);

    return arr;
  }

  function buildUnivariateInsights(rows, feature, { isNum, isCat, missingAsCategory, rareThreshold }) {
    const train = rows.filter(r => r.Survived === 0 || r.Survived === 1);
    if (!train.length) return `<div class="muted">Нет Survived в выбранной выборке — доступны только распределения.</div>`;

    if (isCat) {
      const rates = computeSurvivalRateByCategory(train, feature, { missingAsCategory, rareThreshold, topN: 12 });
      if (!rates.length) return `<div class="muted">Недостаточно данных для ${escapeHtml(feature)}.</div>`;
      const best = rates[0];
      const worst = rates[rates.length - 1];
      return `
        <div>• Лучшая категория по выживанию: <b>${escapeHtml(best.category)}</b> — <b>${(best.rate*100).toFixed(1)}%</b> (n=${best.n}).</div>
        <div style="margin-top:6px;">• Худшая категория: <b>${escapeHtml(worst.category)}</b> — <b>${(worst.rate*100).toFixed(1)}%</b> (n=${worst.n}).</div>
        <div style="margin-top:8px;" class="muted">Идея для вывода: если разница большая и стабильная (достаточные n), признак сильно связан с Survived.</div>
      `;
    }

    if (isNum) {
      const x0 = train.filter(r => r.Survived === 0).map(r => toNumber(r[feature])).filter(v => v !== null);
      const x1 = train.filter(r => r.Survived === 1).map(r => toNumber(r[feature])).filter(v => v !== null);
      if (x0.length < 10 || x1.length < 10) return `<div class="muted">Недостаточно чисел для сравнения групп.</div>`;
      const m0 = mean(x0), m1 = mean(x1);
      const med0 = quantile(x0, 0.5), med1 = quantile(x1, 0.5);
      const rpb = pointBiserial(train.map(r => toNumber(r[feature])), train.map(r => r.Survived));
      return `
        <div>• Среднее: died=${fmtNum(m0)}, survived=${fmtNum(m1)}.</div>
        <div style="margin-top:6px;">• Медиана: died=${fmtNum(med0)}, survived=${fmtNum(med1)}.</div>
        <div style="margin-top:6px;">• Связь с Survived (point-biserial r): <b>${rpb === null ? "—" : rpb.toFixed(3)}</b>.</div>
        <div style="margin-top:8px;" class="muted">Идея: если распределения сильно различаются + |r| заметный → важный признак.</div>
      `;
    }

    return `<div class="muted">Тип признака не распознан.</div>`;
  }

  function renderMultivariate() {
    const xF = el.mvX.value;
    const yF = el.mvY.value;
    const trainOnly = el.mvTrainOnly.value === "yes";
    const baseRows = getRowsBySource(state.data, el.sourceFilter.value);
    const rows = trainOnly ? getTrainRows(baseRows) : baseRows;

    const { numeric, categorical } = state.featureTypes;
    const xIsNum = numeric.has(xF);
    const yIsNum = numeric.has(yF);
    const xIsCat = categorical.has(xF);
    const yIsCat = categorical.has(yF);

    el.mvTitle.textContent = `Multivariate: ${xF} vs ${yF}`;

    const hasSurv = rows.some(r => r.Survived === 0 || r.Survived === 1);

    // numeric-numeric => scatter (colored by Survived if available)
    if (xIsNum && yIsNum) {
      const pts = rows.map(r => ({
        x: toNumber(r[xF]),
        y: toNumber(r[yF]),
        s: (r.Survived === 0 || r.Survived === 1) ? r.Survived : null
      })).filter(p => p.x !== null && p.y !== null);

      if (!pts.length) { Plotly.purge(el.plotMV); return; }

      if (hasSurv) {
        const p0 = pts.filter(p => p.s === 0);
        const p1 = pts.filter(p => p.s === 1);
        Plotly.newPlot(el.plotMV, [
          { type:"scatter", mode:"markers", name:"Died (0)", x: p0.map(p=>p.x), y: p0.map(p=>p.y),
            marker:{ size:7, opacity:0.65 }, hovertemplate:`${xF}=%{x}<br>${yF}=%{y}<extra></extra>` },
          { type:"scatter", mode:"markers", name:"Survived (1)", x: p1.map(p=>p.x), y: p1.map(p=>p.y),
            marker:{ size:7, opacity:0.65 }, hovertemplate:`${xF}=%{x}<br>${yF}=%{y}<extra></extra>` }
        ], baseLayout("", {
          xaxis: { title: xF },
          yaxis: { title: yF }
        }), { displayModeBar: false, responsive: true });
      } else {
        Plotly.newPlot(el.plotMV, [{
          type:"scatter", mode:"markers", name:"points",
          x: pts.map(p=>p.x), y: pts.map(p=>p.y),
          marker:{ size:7, opacity:0.65 },
          hovertemplate:`${xF}=%{x}<br>${yF}=%{y}<extra></extra>`
        }], baseLayout("", {
          xaxis: { title: xF },
          yaxis: { title: yF }
        }), { displayModeBar: false, responsive: true });
      }
      return;
    }

    // cat-numeric => boxplot grouped by category, colored by Survived if available
    if (xIsCat && yIsNum) {
      renderCatNum(rows, xF, yF, hasSurv);
      return;
    }
    if (xIsNum && yIsCat) {
      renderCatNum(rows, yF, xF, hasSurv); // swap
      return;
    }

    // cat-cat => heatmap of survival rate if Survived else count heatmap
    if (xIsCat && yIsCat) {
      renderCatCat(rows, xF, yF, hasSurv);
      return;
    }

    Plotly.purge(el.plotMV);
  }

  function renderCatNum(rows, catF, numF, hasSurv) {
    const missingAsCategory = el.missingAsCategory.value === "yes";
    const rareThreshold = Number(el.rareThreshold.value || 0.02);

    let cats = rows.map(r => r[catF]);
    cats = cats.map(v => {
      if (v === null || v === undefined || String(v).trim() === "") return missingAsCategory ? "Missing" : null;
      return String(v);
    });
    cats = collapseRareCategories(cats, rareThreshold);

    const ys = rows.map(r => toNumber(r[numF]));
    const s = rows.map(r => (r.Survived === 0 || r.Survived === 1) ? r.Survived : null);

    // Keep top categories by count (max 12)
    const counts = new Map();
    for (const c of cats) if (c != null) counts.set(c, (counts.get(c) || 0) + 1);
    const topCats = [...counts.entries()].sort((a,b)=>b[1]-a[1]).slice(0,12).map(([k])=>k);

    const traces = [];
    if (hasSurv) {
      // two sets of boxplots per category
      const cats0 = topCats.map(() => []);
      const cats1 = topCats.map(() => []);
      for (let i=0;i<rows.length;i++){
        const c = cats[i];
        const y = ys[i];
        const sv = s[i];
        if (c == null || y == null || !topCats.includes(c) || sv == null) continue;
        const idx = topCats.indexOf(c);
        if (sv === 0) cats0[idx].push(y);
        else cats1[idx].push(y);
      }
      traces.push({
        type:"box",
        name:"Died (0)",
        x: topCats.flatMap((c, idx) => Array(cats0[idx].length).fill(c)),
        y: cats0.flat(),
        boxpoints: false
      });
      traces.push({
        type:"box",
        name:"Survived (1)",
        x: topCats.flatMap((c, idx) => Array(cats1[idx].length).fill(c)),
        y: cats1.flat(),
        boxpoints: false
      });
      Plotly.newPlot(el.plotMV, traces, baseLayout("", {
        boxmode: "group",
        xaxis: { title: catF, tickangle: -20 },
        yaxis: { title: numF }
      }), { displayModeBar: false, responsive: true });
    } else {
      const byCat = new Map(topCats.map(c => [c, []]));
      for (let i=0;i<rows.length;i++){
        const c = cats[i], y = ys[i];
        if (c == null || y == null || !byCat.has(c)) continue;
        byCat.get(c).push(y);
      }
      const x = [], y = [];
      for (const c of topCats) {
        for (const v of byCat.get(c)) { x.push(c); y.push(v); }
      }
      Plotly.newPlot(el.plotMV, [{
        type:"box",
        name: numF,
        x, y,
        boxpoints: false
      }], baseLayout("", {
        xaxis: { title: catF, tickangle: -20 },
        yaxis: { title: numF }
      }), { displayModeBar: false, responsive: true });
    }
  }

  function renderCatCat(rows, xF, yF, hasSurv) {
    const missingAsCategory = el.missingAsCategory.value === "yes";
    const rareThreshold = Number(el.rareThreshold.value || 0.02);

    let xs = rows.map(r => r[xF]);
    let ys = rows.map(r => r[yF]);

    xs = xs.map(v => {
      if (v === null || v === undefined || String(v).trim() === "") return missingAsCategory ? "Missing" : null;
      return String(v);
    });
    ys = ys.map(v => {
      if (v === null || v === undefined || String(v).trim() === "") return missingAsCategory ? "Missing" : null;
      return String(v);
    });

    xs = collapseRareCategories(xs, rareThreshold);
    ys = collapseRareCategories(ys, rareThreshold);

    // take top categories
    const top = (arr) => {
      const c = new Map();
      for (const v of arr) if (v != null) c.set(v, (c.get(v)||0)+1);
      return [...c.entries()].sort((a,b)=>b[1]-a[1]).slice(0,10).map(([k])=>k);
    };
    const xCats = top(xs);
    const yCats = top(ys);

    const grid = Array.from({ length: yCats.length }, () => Array(xCats.length).fill(null));
    const gridN = Array.from({ length: yCats.length }, () => Array(xCats.length).fill(0));
    const gridS = Array.from({ length: yCats.length }, () => Array(xCats.length).fill(0));

    for (let i=0;i<rows.length;i++){
      const a = xs[i], b = ys[i];
      if (a == null || b == null) continue;
      const xi = xCats.indexOf(a);
      const yi = yCats.indexOf(b);
      if (xi < 0 || yi < 0) continue;
      gridN[yi][xi] += 1;
      if (hasSurv && rows[i].Survived === 1) gridS[yi][xi] += 1;
    }

    if (hasSurv) {
      for (let j=0;j<yCats.length;j++){
        for (let i=0;i<xCats.length;i++){
          grid[j][i] = gridN[j][i] ? (gridS[j][i] / gridN[j][i]) : null;
        }
      }
      Plotly.newPlot(el.plotMV, [{
        type:"heatmap",
        x: xCats,
        y: yCats,
        z: grid,
        hovertemplate: `${yF}=%{y}<br>${xF}=%{x}<br>survival rate=%{z:.2%}<extra></extra>`
      }], baseLayout("", {
        xaxis: { title: xF },
        yaxis: { title: yF }
      }), { displayModeBar: false, responsive: true });
    } else {
      for (let j=0;j<yCats.length;j++){
        for (let i=0;i<xCats.length;i++){
          grid[j][i] = gridN[j][i] || 0;
        }
      }
      Plotly.newPlot(el.plotMV, [{
        type:"heatmap",
        x: xCats,
        y: yCats,
        z: grid,
        hovertemplate: `${yF}=%{y}<br>${xF}=%{x}<br>count=%{z}<extra></extra>`
      }], baseLayout("", {
        xaxis: { title: xF },
        yaxis: { title: yF }
      }), { displayModeBar: false, responsive: true });
    }
  }

  function renderImportance() {
    const topK = Number(el.impTopK.value || 12);
    const imp = (state.importance || []).slice(0, topK);

    // bar chart
    Plotly.newPlot(el.plotImp, [{
      type:"bar",
      x: imp.map(d => d.score).reverse(),
      y: imp.map(d => d.feature).reverse(),
      orientation: "h",
      hovertemplate: "feature=%{y}<br>score=%{x:.4f}<extra></extra>"
    }], baseLayout("", {
      xaxis: { title: "Importance score" },
      yaxis: { title: "Feature" }
    }), { displayModeBar: false, responsive: true });

    // table
    el.impTableBody.innerHTML = imp.map((d, i) => `
      <tr>
        <td>${i+1}</td>
        <td class="mono">${escapeHtml(d.feature)}</td>
        <td>${escapeHtml(d.type)}</td>
        <td>${d.score.toFixed(4)}</td>
        <td class="mono">${escapeHtml(d.metric)}</td>
      </tr>
    `).join("");

    const top = imp[0];
    el.impNote.innerHTML = top
      ? `Top-1 сейчас: <b class="mono">${escapeHtml(top.feature)}</b> (score=${top.score.toFixed(4)}, metric=${escapeHtml(top.metric)}).`
      : `<span class="muted">Нет результата importance (проверь данные).</span>`;
  }

  function renderConclusion() {
    const imp = state.importance || [];
    if (!imp.length) {
      el.finalConclusion.innerHTML = `<div class="muted">Нет importance → нечего выводить.</div>`;
      Plotly.purge(el.plotTopProof);
      el.talkTrack.innerHTML = "";
      return;
    }

    const top = imp[0];
    const second = imp[1];

    const train = getTrainRows(state.data);

    // Build narrative
    const why = buildWhyText(top, second);
    const guidance = buildTalkTrack(top);

    el.finalConclusion.innerHTML = `
      <div style="font-size:14px; line-height:1.55;">
        <p><b>Самый важный признак по EDA:</b> <span class="mono">${escapeHtml(top.feature)}</span>.</p>
        <p>${why}</p>
        <p class="muted">Важно: это вывод по <b>train</b> (где известен Survived). Для test мы можем смотреть распределения, но не survival rate.</p>
      </div>
    `;

    el.talkTrack.innerHTML = guidance;

    // Proof plot for top feature
    const { numeric, categorical } = state.featureTypes;
    const isNum = numeric.has(top.feature);
    const isCat = categorical.has(top.feature);

    if (isCat) {
      const rates = computeSurvivalRateByCategory(train, top.feature, {
        missingAsCategory: true,
        rareThreshold: Number(el.rareThreshold.value || 0.02),
        topN: 16
      });
      Plotly.newPlot(el.plotTopProof, [{
        type:"bar",
        x: rates.map(d => d.category),
        y: rates.map(d => d.rate),
        customdata: rates.map(d => d.n),
        hovertemplate: `${top.feature}=%{x}<br>rate=%{y:.2%}<br>n=%{customdata}<extra></extra>`
      }], baseLayout("", {
        xaxis: { title: top.feature, tickangle: -25 },
        yaxis: { title: "Survival rate", tickformat: ".0%", range: [0,1] }
      }), { displayModeBar: false, responsive: true });
    } else if (isNum) {
      // show distributions overlay
      renderNumericDistribution(train, top.feature, true);
      // copy plot into conclusion container by re-plotting
      const x0 = train.filter(r => r.Survived === 0).map(r => toNumber(r[top.feature])).filter(v => v !== null);
      const x1 = train.filter(r => r.Survived === 1).map(r => toNumber(r[top.feature])).filter(v => v !== null);

      Plotly.newPlot(el.plotTopProof, [
        { type:"histogram", name:"Died (0)", x:x0, opacity:0.65, nbinsx:30, hovertemplate:"Died<br>%{x}<extra></extra>" },
        { type:"histogram", name:"Survived (1)", x:x1, opacity:0.65, nbinsx:30, hovertemplate:"Survived<br>%{x}<extra></extra>" }
      ], baseLayout("", {
        barmode: "overlay",
        xaxis: { title: top.feature },
        yaxis: { title: "Count" }
      }), { displayModeBar: false, responsive: true });
    } else {
      Plotly.purge(el.plotTopProof);
    }
  }

  function buildWhyText(top, second) {
    const f = top.feature;
    const metric = top.metric;
    const score = top.score;

    // We try to provide a concrete EDA-friendly explanation based on feature
    const lower = f.toLowerCase();

    if (lower === "sex") {
      // typical Titanic: females survive much more
      const train = getTrainRows(state.data);
      const rates = computeSurvivalRateByCategory(train, "Sex", { missingAsCategory:true, rareThreshold:0.0, topN: 10 });
      const by = new Map(rates.map(d => [d.category.toLowerCase(), d]));
      const female = by.get("female");
      const male = by.get("male");
      if (female && male) {
        const diff = (female.rate - male.rate);
        return `Признак <span class="mono">Sex</span> показывает самую сильную и стабильную разницу в выживаемости: у категории <b>${escapeHtml(female.category)}</b> survival rate ≈ <b>${(female.rate*100).toFixed(1)}%</b>, а у <b>${escapeHtml(male.category)}</b> ≈ <b>${(male.rate*100).toFixed(1)}%</b>. Разрыв ≈ <b>${(diff*100).toFixed(1)} п.п.</b>, что визуально видно на графике и подтверждается метрикой importance (${escapeHtml(metric)}=${score.toFixed(4)}).` +
               (second ? ` Следующий по силе признак — <span class="mono">${escapeHtml(second.feature)}</span>, но его эффект слабее.` : "");
      }
      return `Признак <span class="mono">Sex</span> даёт максимальную разницу в выживаемости между категориями (видно на графике survival rate) и поэтому занимает top-1 по метрике importance (${escapeHtml(metric)}=${score.toFixed(4)}).` +
             (second ? ` Второй по силе: <span class="mono">${escapeHtml(second.feature)}</span>.` : "");
    }

    if (lower === "pclass") {
      return `Признак <span class="mono">Pclass</span> (класс билета) отражает социально-экономический статус и доступ к шлюпкам: survival rate у 1-го класса обычно заметно выше, чем у 3-го. Это даёт сильную связь с <span class="mono">Survived</span> по метрике importance (${escapeHtml(metric)}=${score.toFixed(4)}).` +
             (second ? ` Второй по силе признак: <span class="mono">${escapeHtml(second.feature)}</span>.` : "");
    }

    if (lower === "age") {
      return `Признак <span class="mono">Age</span> показывает различия в распределениях между выжившими и погибшими (например, эффект “women and children first”). В EDA это видно по гистограммам/квантильным бинам survival rate и подтверждается importance (${escapeHtml(metric)}=${score.toFixed(4)}).` +
             (second ? ` Следующий по силе — <span class="mono">${escapeHtml(second.feature)}</span>.` : "");
    }

    // generic
    return `По ранжированию важности на train, самый сильный сигнал даёт <span class="mono">${escapeHtml(f)}</span> (importance: ${escapeHtml(metric)}=${score.toFixed(4)}). Это означает, что распределение/категории ${escapeHtml(f)} максимально отличаются между <span class="mono">Survived=1</span> и <span class="mono">Survived=0</span> среди всех рассмотренных признаков.` +
           (second ? ` Следующий по силе признак — <span class="mono">${escapeHtml(second.feature)}</span>.` : "");
  }

  function buildTalkTrack(top) {
    const f = top.feature;
    return `
      <ol style="margin:0; padding-left:18px; line-height:1.55;">
        <li><b>Формулировка:</b> “На основе EDA самый важный признак — <span class="mono">${escapeHtml(f)}</span>.”</li>
        <li><b>Визуальное доказательство:</b> покажи график “Survival rate vs ${escapeHtml(f)}” (вкладка Conclusion / Univariate).</li>
        <li><b>Численное доказательство:</b> “В ранжировании importance ${escapeHtml(f)} имеет максимальный score.”</li>
        <li><b>Интерпретация:</b> объясни причинно-следственную гипотезу (соц. роль, доступ к шлюпкам, приоритет эвакуации и т.д.).</li>
        <li><b>Оговорка:</b> “Это EDA-связь на train, не причинность и не финальная ML-модель.”</li>
      </ol>
      <div class="muted" style="margin-top:8px;">
        Совет: для усиления аргумента сравни также <span class="mono">Pclass</span>, <span class="mono">Title</span>, <span class="mono">Deck</span> — они часто идут сразу после top-1.
      </div>
    `;
  }

  // ---------- Common layout ----------
  function baseLayout(title, extra = {}) {
    return Object.assign({
      title: { text: title, font: { size: 12 } },
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      margin: { l: 50, r: 20, t: 30, b: 65 },
      font: { color: "rgba(232,236,255,.92)" },
      xaxis: { gridcolor: "rgba(255,255,255,.08)", zerolinecolor: "rgba(255,255,255,.08)" },
      yaxis: { gridcolor: "rgba(255,255,255,.08)", zerolinecolor: "rgba(255,255,255,.08)" },
      legend: { orientation: "h", y: 1.08 }
    }, extra);
  }

  // ---------- Helpers ----------
  function mean(arr) {
    const xs = arr.filter(v => Number.isFinite(v));
    if (!xs.length) return null;
    return xs.reduce((a,b)=>a+b,0) / xs.length;
  }
  function fmtNum(x) {
    if (x === null || x === undefined || !Number.isFinite(x)) return "—";
    const ax = Math.abs(x);
    if (ax >= 100) return x.toFixed(0);
    if (ax >= 10) return x.toFixed(1);
    return x.toFixed(2);
  }
  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, (c) => ({
      "&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"
    }[c]));
  }
  function escapeAttr(s) {
    return escapeHtml(s).replace(/"/g, "&quot;");
  }

  // ---------- Tabs ----------
  function setActiveTab(tab) {
    // buttons
    [...el.tabs.querySelectorAll(".tab")].forEach(t => {
      t.classList.toggle("active", t.dataset.tab === tab);
    });
    // sections
    const map = {
      overview: el.tabOverview,
      preproc: el.tabPreproc,
      missing: el.tabMissing,
      univariate: el.tabUnivariate,
      multivariate: el.tabMultivariate,
      importance: el.tabImportance,
      conclusion: el.tabConclusion
    };
    for (const [k, node] of Object.entries(map)) {
      node.classList.toggle("hidden", k !== tab);
    }

    // rerender plots for correct sizing
    setTimeout(() => {
      tryResizePlots();
    }, 50);
  }

  function tryResizePlots() {
    const ids = [
      el.plotSource, el.plotKeyRates, el.plotImpute, el.plotMissing,
      el.plotUni, el.plotMV, el.plotImp, el.plotTopProof
    ];
    for (const node of ids) {
      if (!node) continue;
      try { Plotly.Plots.resize(node); } catch {}
    }
  }

  // ---------- Events ----------
  function bindEvents() {
    el.btnLoad.addEventListener("click", () => loadFromRepoOrFiles({ preferRepo: false }));
    el.btnUseSample.addEventListener("click", () => loadFromRepoOrFiles({ preferRepo: true }));

    el.sourceFilter.addEventListener("change", () => {
      if (!state.ready) return;
      renderOverview();
      renderUnivariate();
      renderMultivariate();
      renderConclusion();
    });

    el.missingAsCategory.addEventListener("change", () => {
      if (!state.ready) return;
      state.importance = computeImportance(state.data, {
        missingAsCategory: el.missingAsCategory.value === "yes",
        rareThreshold: Number(el.rareThreshold.value || 0.02)
      });
      renderImportance();
      renderUnivariate();
      renderMultivariate();
      renderConclusion();
    });

    el.rareThreshold.addEventListener("change", () => {
      if (!state.ready) return;
      state.importance = computeImportance(state.data, {
        missingAsCategory: el.missingAsCategory.value === "yes",
        rareThreshold: Number(el.rareThreshold.value || 0.02)
      });
      renderImportance();
      renderUnivariate();
      renderMultivariate();
      renderConclusion();
    });

    el.missingSourceView.addEventListener("change", () => {
      if (!state.ready) return;
      renderMissingness();
    });

    el.uniFeature.addEventListener("change", () => state.ready && renderUnivariate());
    el.uniMode.addEventListener("change", () => state.ready && renderUnivariate());
    el.uniTrainOnly.addEventListener("change", () => state.ready && renderUnivariate());

    el.mvX.addEventListener("change", () => state.ready && renderMultivariate());
    el.mvY.addEventListener("change", () => state.ready && renderMultivariate());
    el.mvTrainOnly.addEventListener("change", () => state.ready && renderMultivariate());

    el.impTopK.addEventListener("change", () => state.ready && renderImportance());

    el.btnDownloadProcessed.addEventListener("click", () => {
      if (!state.ready) return;
      const csv = toCSV(state.data);
      downloadText("processed.csv", csv);
    });

    el.tabs.addEventListener("click", (e) => {
      const t = e.target.closest(".tab");
      if (!t) return;
      setActiveTab(t.dataset.tab);
    });

    window.addEventListener("resize", () => tryResizePlots());
  }

  // ---------- Init ----------
  function init() {
    bindEvents();
    setActiveTab("overview");
    setStatus("idle");
  }

  return { init };
})();

document.addEventListener("DOMContentLoaded", () => App.init());
