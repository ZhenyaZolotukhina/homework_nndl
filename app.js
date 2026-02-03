// app.js
// Titanic HW2: Shallow Neural Network factor discovery (TensorFlow.js, browser-only)
//
// Reuse note (schema swap points):
// - Update SCHEMA below (TARGET, FEATURES, ID)
// - Update categorical levels if needed (or infer dynamically)
// - Keep the same preprocessing/template pipeline

(() => {
  // -----------------------------
  // Schema (assignment-required)
  // -----------------------------
  const SCHEMA = {
    TARGET: "Survived",
    ID: "PassengerId",
    FEATURES: ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"],
    // Categorical fields among FEATURES:
    CATEGORICAL: ["Pclass", "Sex", "Embarked"],
    NUMERIC: ["Age", "SibSp", "Parch", "Fare"],
  };

  // -----------------------------
  // App state
  // -----------------------------
  const state = {
    trainRows: null,
    testRows: null,
    // preprocessing artifacts
    stats: null, // { ageMedian, fareMedian, embarkedMode, meanAge, stdAge, meanFare, stdFare, ... }
    featureNames: null,
    XTrain: null,
    yTrain: null,
    XVal: null,
    yVal: null,
    valProbs: null, // Float32Array
    // model
    model: null,
    // metric cache
    roc: null, // {fpr:[], tpr:[], auc:number}
    ready: {
      data: false,
      preproc: false,
      model: false,
      trained: false,
      evaluated: false,
    }
  };

  // -----------------------------
  // DOM
  // -----------------------------
  const el = {
    statusPill: document.getElementById("statusPill"),
    dataPill: document.getElementById("dataPill"),
    errBox: document.getElementById("errBox"),

    trainFile: document.getElementById("trainFile"),
    testFile: document.getElementById("testFile"),
    btnFetch: document.getElementById("btnFetch"),
    btnLoad: document.getElementById("btnLoad"),
    btnPreprocess: document.getElementById("btnPreprocess"),
    btnBuildModel: document.getElementById("btnBuildModel"),
    btnTrain: document.getElementById("btnTrain"),
    btnEvaluate: document.getElementById("btnEvaluate"),
    btnPredict: document.getElementById("btnPredict"),
    btnExportModel: document.getElementById("btnExportModel"),

    optFamilySize: document.getElementById("optFamilySize"),
    optIsAlone: document.getElementById("optIsAlone"),

    epochs: document.getElementById("epochs"),
    batchSize: document.getElementById("batchSize"),

    thrSlider: document.getElementById("thrSlider"),
    thrValue: document.getElementById("thrValue"),

    kpiRows: document.getElementById("kpiRows"),
    kpiPosRate: document.getElementById("kpiPosRate"),
    kpiFeat: document.getElementById("kpiFeat"),
    kpiAuc: document.getElementById("kpiAuc"),

    preprocOut: document.getElementById("preprocOut"),
    modelOut: document.getElementById("modelOut"),
    metricsOut: document.getElementById("metricsOut"),

    visPreview: document.getElementById("visPreview"),
    visMissing: document.getElementById("visMissing"),
    visBars: document.getElementById("visBars"),
    visFit: document.getElementById("visFit"),
    visRoc: document.getElementById("visRoc"),
    visConf: document.getElementById("visConf"),
  };

  // -----------------------------
  // Small helpers
  // -----------------------------
  const setStatus = (s) => { el.statusPill.textContent = `status: ${s}`; };
  const showError = (msg) => {
    if (!msg) { el.errBox.style.display = "none"; el.errBox.innerHTML = ""; return; }
    el.errBox.style.display = "block";
    el.errBox.innerHTML = `<b>Error:</b> ${escapeHtml(msg)}`;
  };
  const setDataPill = (ok, text) => {
    el.dataPill.textContent = text;
    el.dataPill.style.borderColor = ok ? "rgba(125,255,178,.35)" : "rgba(255,255,255,.12)";
    el.dataPill.style.color = ok ? "rgba(125,255,178,.95)" : "rgba(170,185,255,.85)";
  };

  const isNil = (v) => v === null || v === undefined || (typeof v === "number" && Number.isNaN(v));
  const toStr = (v) => {
    if (v === null || v === undefined) return null;
    const s = String(v).trim();
    return s === "" ? null : s;
  };
  const toNumber = (v) => {
    if (v === null || v === undefined) return null;
    const s = String(v).trim();
    if (s === "" || s.toLowerCase() === "nan") return null;
    const n = Number(s);
    return Number.isFinite(n) ? n : null;
  };
  const mean = (arr) => arr.reduce((a,b)=>a+b,0) / (arr.length || 1);
  const std = (arr, m) => {
    const v = arr.reduce((a,x)=>a + (x-m)*(x-m), 0) / Math.max(1, arr.length - 1);
    return Math.sqrt(v);
  };
  const quantile = (arr, q) => {
    const xs = arr.filter(Number.isFinite).slice().sort((a,b)=>a-b);
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
    for (const [k,c] of m.entries()) if (c > bestC) { bestC = c; best = k; }
    return best;
  };
  const shuffle = (arr) => {
    for (let i = arr.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [arr[i], arr[j]] = [arr[j], arr[i]];
    }
  };
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
  const toCSV = (rows, cols) => {
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
  const escapeHtml = (s) => String(s).replace(/[&<>"']/g, (c) => ({
    "&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"
  }[c]));

  // -----------------------------
  // CSV loading (fetch or file input)
  // -----------------------------
  async function fetchText(url) {
    const res = await fetch(url, { cache: "no-store" });
    if (!res.ok) throw new Error(`Fetch failed: ${url} (${res.status})`);
    return await res.text();
  }
  function parseCSVText(text) {
    return new Promise((resolve, reject) => {
      Papa.parse(text, {
        header: true,
        skipEmptyLines: true,
        dynamicTyping: false,
        complete: (results) => resolve({ rows: results.data, errors: results.errors || [] }),
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

  async function loadData({ preferRepo }) {
    setStatus("loading data");
    showError("");

    try {
      let trainText, testText;

      if (preferRepo) {
        trainText = await fetchText("./train.csv");
        testText  = await fetchText("./test.csv");
      } else {
        const fTrain = el.trainFile.files?.[0];
        const fTest  = el.testFile.files?.[0];
        if (!fTrain || !fTest) throw new Error("Please select both train.csv and test.csv in the file inputs, or use Fetch from repo.");
        trainText = await readFileAsText(fTrain);
        testText = await readFileAsText(fTest);
      }

      const [trainParsed, testParsed] = await Promise.all([parseCSVText(trainText), parseCSVText(testText)]);
      if (trainParsed.errors.length || testParsed.errors.length) {
        // not fatal; still continue
        console.warn("CSV parse warnings:", [...trainParsed.errors, ...testParsed.errors].slice(0,5));
      }

      // Normalize rows: keep only non-empty objects
      const trainRows = trainParsed.rows.filter(r => r && Object.keys(r).length);
      const testRows  = testParsed.rows.filter(r => r && Object.keys(r).length);

      // Basic schema check
      for (const col of [SCHEMA.ID, ...SCHEMA.FEATURES, SCHEMA.TARGET]) {
        if (!(col in trainRows[0])) throw new Error(`train.csv is missing expected column: ${col}`);
      }
      for (const col of [SCHEMA.ID, ...SCHEMA.FEATURES]) {
        if (!(col in testRows[0])) throw new Error(`test.csv is missing expected column: ${col}`);
      }

      state.trainRows = trainRows;
      state.testRows = testRows;
      state.ready.data = true;

      setDataPill(true, "data: loaded");
      setStatus("data loaded");

      el.btnPreprocess.disabled = false;

      renderDataInspection();
    } catch (e) {
      console.error(e);
      setDataPill(false, "data: error");
      setStatus("error");
      showError(e.message || String(e));
    }
  }

  // -----------------------------
  // Inspection (preview + missing + simple EDA bars)
  // -----------------------------
  function renderDataInspection() {
    const rows = state.trainRows;
    if (!rows?.length) return;

    // KPIs
    const y = rows.map(r => toNumber(r[SCHEMA.TARGET])).filter(v => v === 0 || v === 1);
    const posRate = y.reduce((a,b)=>a + (b === 1 ? 1 : 0), 0) / (y.length || 1);
    el.kpiRows.textContent = String(rows.length);
    el.kpiPosRate.textContent = `${(posRate*100).toFixed(1)}%`;

    // Preview table (first 10 rows)
    const preview = rows.slice(0, 10).map(r => ({
      PassengerId: r.PassengerId,
      Survived: r.Survived,
      Pclass: r.Pclass,
      Sex: r.Sex,
      Age: r.Age,
      SibSp: r.SibSp,
      Parch: r.Parch,
      Fare: r.Fare,
      Embarked: r.Embarked
    }));
    tfvis.render.table(
      { name: "Train preview (first 10 rows)", tab: "Inspection", dom: el.visPreview },
      { headers: Object.keys(preview[0]), values: preview.map(o => Object.values(o)) },
      { height: 280 }
    );

    // Missing % per selected columns
    const cols = [SCHEMA.ID, SCHEMA.TARGET, ...SCHEMA.FEATURES];
    const miss = cols.map(c => {
      const m = rows.reduce((acc, r) => acc + (toStr(r[c]) === null ? 1 : 0), 0);
      return { col: c, missingPct: (m / rows.length) * 100 };
    }).sort((a,b)=>b.missingPct - a.missingPct);

    tfvis.render.barchart(
      { name: "Missing % (train, selected columns)", tab: "Inspection", dom: el.visMissing },
      miss.map(d => ({ x: d.col, y: d.missingPct })),
      { xLabel: "Column", yLabel: "Missing %", height: 320 }
    );

    // Survival by Sex and Pclass (simple EDA bars)
    const bySex = survivalRateByCategory(rows, "Sex");
    const byPclass = survivalRateByCategory(rows, "Pclass");

    const series = [
      { name: "Survival by Sex", values: bySex.map(d => ({ x: d.category, y: d.rate })) },
      { name: "Survival by Pclass", values: byPclass.map(d => ({ x: d.category, y: d.rate })) },
    ];

    tfvis.render.barchart(
      { name: "Quick EDA: Survival rate", tab: "Inspection", dom: el.visBars },
      series,
      { xLabel: "Category", yLabel: "Survival rate", height: 320 }
    );
  }

  function survivalRateByCategory(rows, feature) {
    const stats = new Map();
    for (const r of rows) {
      const y = toNumber(r[SCHEMA.TARGET]);
      if (!(y === 0 || y === 1)) continue;
      const cat = toStr(r[feature]) ?? "Missing";
      if (!stats.has(cat)) stats.set(cat, { n: 0, s: 0 });
      const o = stats.get(cat);
      o.n += 1;
      o.s += (y === 1 ? 1 : 0);
    }
    const arr = [...stats.entries()].map(([category, v]) => ({ category, rate: v.s / v.n, n: v.n }));
    // sort by n desc for stability
    arr.sort((a,b)=>b.n - a.n);
    return arr;
  }

  // -----------------------------
  // Preprocessing (impute + standardize + one-hot)
  // -----------------------------
  function preprocess() {
    setStatus("preprocessing");
    showError("");

    if (!state.ready.data) { showError("Load data first."); return; }

    const train = state.trainRows;
    const test = state.testRows;

    // Extract arrays from train for statistics (imputation + scaling)
    const ageTrain = train.map(r => toNumber(r.Age)).filter(v => v !== null);
    const fareTrain = train.map(r => toNumber(r.Fare)).filter(v => v !== null);
    const embarkedTrain = train.map(r => toStr(r.Embarked)).filter(v => v !== null);

    const ageMedian = quantile(ageTrain, 0.5);
    const fareMedian = quantile(fareTrain, 0.5);
    const embarkedMode = mode(embarkedTrain) ?? "S";

    // We standardize Age and Fare using train stats after imputation
    const ageImputedForStats = train.map(r => toNumber(r.Age) ?? ageMedian).filter(v => v !== null);
    const fareImputedForStats = train.map(r => toNumber(r.Fare) ?? fareMedian).filter(v => v !== null);

    const meanAge = mean(ageImputedForStats);
    const stdAge = std(ageImputedForStats, meanAge) || 1;
    const meanFare = mean(fareImputedForStats);
    const stdFare = std(fareImputedForStats, meanFare) || 1;

    state.stats = { ageMedian, fareMedian, embarkedMode, meanAge, stdAge, meanFare, stdFare };

    // Categorical levels:
    // For Titanic we know typical levels, but we infer from train+test to be safe.
    const sexLevels = uniqueNonNull([...train, ...test].map(r => (toStr(r.Sex) ?? "Missing").toLowerCase()));
    const pclassLevels = uniqueNonNull([...train, ...test].map(r => String(toNumber(r.Pclass) ?? "Missing")));
    const embarkedLevels = uniqueNonNull([...train, ...test].map(r => toStr(r.Embarked) ?? "Missing"));

    // For stability, sort levels
    sexLevels.sort();
    pclassLevels.sort((a,b)=>Number(a)-Number(b));
    embarkedLevels.sort();

    // Feature list (final)
    const useFamily = el.optFamilySize.checked;
    const useAlone = el.optIsAlone.checked;

    const featureNames = [];
    // numeric (standardized where applicable)
    featureNames.push("Age_z", "Fare_z", "SibSp", "Parch");
    if (useFamily) featureNames.push("FamilySize");
    if (useAlone) featureNames.push("IsAlone");
    // one-hot
    for (const lv of pclassLevels) featureNames.push(`Pclass=${lv}`);
    for (const lv of sexLevels) featureNames.push(`Sex=${lv}`);
    for (const lv of embarkedLevels) featureNames.push(`Embarked=${lv}`);

    state.featureNames = featureNames;
    el.kpiFeat.textContent = String(featureNames.length);

    // Build X/y tensors
    const { X, y, ids } = buildXY(train, {
      sexLevels, pclassLevels, embarkedLevels,
      useFamily, useAlone
    });

    // Stratified 80/20 split
    const split = stratifiedSplit(y, 0.8);
    const XTrain = X.gather(split.trainIdx);
    const yTrain = y.gather(split.trainIdx);
    const XVal = X.gather(split.valIdx);
    const yVal = y.gather(split.valIdx);

    // Clean previous tensors/model references (avoid memory growth)
    disposeIfTensor(state.XTrain); disposeIfTensor(state.yTrain);
    disposeIfTensor(state.XVal); disposeIfTensor(state.yVal);

    state.XTrain = XTrain; state.yTrain = yTrain;
    state.XVal = XVal; state.yVal = yVal;

    // Free base tensors
    X.dispose(); y.dispose();
    split.trainIdx.dispose(); split.valIdx.dispose();

    state.ready.preproc = true;
    state.ready.model = false;
    state.ready.trained = false;
    state.ready.evaluated = false;

    el.btnBuildModel.disabled = false;
    el.btnTrain.disabled = true;
    el.btnEvaluate.disabled = true;
    el.btnPredict.disabled = true;
    el.btnExportModel.disabled = true;
    el.thrSlider.disabled = true;
    el.kpiAuc.textContent = "—";

    // Print preprocessing summary
    el.preprocOut.innerHTML = `
      <b>Preprocessing summary</b><br/>
      • Imputation: Age → median (${fmt(ageMedian)}), Fare → median (${fmt(fareMedian)}), Embarked → mode ("${escapeHtml(embarkedMode)}")<br/>
      • Standardization: Age_z=(Age-mean)/std (mean=${fmt(meanAge)}, std=${fmt(stdAge)}), Fare_z=(Fare-mean)/std (mean=${fmt(meanFare)}, std=${fmt(stdFare)})<br/>
      • One-hot levels: Pclass(${pclassLevels.join(", ")}), Sex(${sexLevels.join(", ")}), Embarked(${embarkedLevels.join(", ")})<br/>
      • Final feature dimension: <span class="mono">${featureNames.length}</span><br/>
      • Tensors: XTrain=${state.XTrain.shape.join("×")}, yTrain=${state.yTrain.shape.join("×")} | XVal=${state.XVal.shape.join("×")}, yVal=${state.yVal.shape.join("×")}
    `;

    setStatus("preprocessed");
  }

  function buildXY(rows, cfg) {
    const { sexLevels, pclassLevels, embarkedLevels, useFamily, useAlone } = cfg;
    const { ageMedian, fareMedian, embarkedMode, meanAge, stdAge, meanFare, stdFare } = state.stats;

    const X = [];
    const y = [];
    const ids = [];

    for (const r of rows) {
      const id = toNumber(r[SCHEMA.ID]);
      const target = toNumber(r[SCHEMA.TARGET]);

      // keep only labeled rows for training (some CSVs may have blanks)
      if (!(target === 0 || target === 1)) continue;

      const age = toNumber(r.Age) ?? ageMedian;
      const fare = toNumber(r.Fare) ?? fareMedian;
      const sibsp = toNumber(r.SibSp) ?? 0;
      const parch = toNumber(r.Parch) ?? 0;

      const embarkedRaw = toStr(r.Embarked);
      const embarked = embarkedRaw ?? embarkedMode;

      const sex = (toStr(r.Sex) ?? "Missing").toLowerCase();
      const pclass = String(toNumber(r.Pclass) ?? "Missing");

      // Numeric part (standardize Age, Fare)
      const rowX = [];
      rowX.push((age - meanAge) / stdAge);
      rowX.push((fare - meanFare) / stdFare);
      rowX.push(sibsp);
      rowX.push(parch);

      // Engineered optional
      if (useFamily || useAlone) {
        const familySize = sibsp + parch + 1;
        if (useFamily) rowX.push(familySize);
        if (useAlone) rowX.push(familySize === 1 ? 1 : 0);
      }

      // One-hot: Pclass
      for (const lv of pclassLevels) rowX.push(pclass === lv ? 1 : 0);
      // One-hot: Sex
      for (const lv of sexLevels) rowX.push(sex === lv ? 1 : 0);
      // One-hot: Embarked
      for (const lv of embarkedLevels) rowX.push(embarked === lv ? 1 : 0);

      X.push(rowX);
      y.push([target]);
      ids.push(id);
    }

    const xTensor = tf.tensor2d(X, [X.length, X[0].length], "float32");
    const yTensor = tf.tensor2d(y, [y.length, 1], "float32");
    return { X: xTensor, y: yTensor, ids };
  }

  function stratifiedSplit(yTensor, trainFrac = 0.8) {
    // yTensor shape [N,1] values 0/1
    const yArr = Array.from(yTensor.dataSync()).map(v => (v >= 0.5 ? 1 : 0));
    const idx0 = [];
    const idx1 = [];
    for (let i=0;i<yArr.length;i++) (yArr[i] === 1 ? idx1 : idx0).push(i);

    shuffle(idx0); shuffle(idx1);

    const n0Train = Math.floor(idx0.length * trainFrac);
    const n1Train = Math.floor(idx1.length * trainFrac);

    const trainIdx = idx0.slice(0, n0Train).concat(idx1.slice(0, n1Train));
    const valIdx = idx0.slice(n0Train).concat(idx1.slice(n1Train));

    shuffle(trainIdx); shuffle(valIdx);

    return {
      trainIdx: tf.tensor1d(trainIdx, "int32"),
      valIdx: tf.tensor1d(valIdx, "int32")
    };
  }

  function uniqueNonNull(arr) {
    const s = new Set();
    for (const v of arr) {
      if (v === null || v === undefined) continue;
      const t = String(v).trim();
      if (t === "") continue;
      s.add(t);
    }
    return Array.from(s);
  }

  function disposeIfTensor(t) { try { if (t && typeof t.dispose === "function") t.dispose(); } catch {} }
  function fmt(x) { return Number.isFinite(x) ? x.toFixed(3) : "—"; }

  // -----------------------------
  // Model (shallow NN)
  // -----------------------------
  function buildModel() {
    setStatus("building model");
    showError("");

    if (!state.ready.preproc) { showError("Preprocess first."); return; }

    // Dispose old model if exists
    if (state.model) { try { state.model.dispose(); } catch {} state.model = null; }

    const inputDim = state.XTrain.shape[1];

    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 16, activation: "relu", inputShape: [inputDim] }));
    model.add(tf.layers.dense({ units: 1, activation: "sigmoid" }));

    model.compile({
      optimizer: tf.train.adam(),
      loss: "binaryCrossentropy",
      metrics: ["accuracy"]
    });

    state.model = model;
    state.ready.model = true;
    state.ready.trained = false;
    state.ready.evaluated = false;

    el.btnTrain.disabled = false;
    el.btnEvaluate.disabled = true;
    el.btnPredict.disabled = true;
    el.btnExportModel.disabled = true;
    el.thrSlider.disabled = true;
    el.kpiAuc.textContent = "—";

    // Print summary (manual)
    el.modelOut.innerHTML = `
      <b>Model summary</b><br/>
      • Architecture: Dense(16, relu) → Dense(1, sigmoid)<br/>
      • Input dim: <span class="mono">${inputDim}</span><br/>
      • Optimizer: Adam<br/>
      • Loss: binaryCrossentropy<br/>
      • Metrics: accuracy
    `;

    // Also show tfjs-vis model summary panel
    tfvis.show.modelSummary({ name: "Model Summary", tab: "Model" }, model);

    setStatus("model ready");
  }

  // -----------------------------
  // Training (tfjs-vis + early stopping)
  // -----------------------------
  async function train() {
    setStatus("training");
    showError("");

    if (!state.ready.model) { showError("Build a model first."); return; }

    const epochs = clampInt(Number(el.epochs.value || 50), 5, 200);
    const batchSize = clampInt(Number(el.batchSize.value || 32), 8, 512);

    // Early stopping callback on val_loss (patience=5)
    const patience = 5;
    let best = Infinity;
    let badCount = 0;

    const earlyStop = {
      onEpochEnd: async (epoch, logs) => {
        const v = logs?.val_loss;
        if (!Number.isFinite(v)) return;
        if (v < best - 1e-6) {
          best = v;
          badCount = 0;
        } else {
          badCount += 1;
          if (badCount >= patience) {
            state.model.stopTraining = true;
          }
        }
      }
    };

    // Clear previous training vis
    el.visFit.innerHTML = "";
    tfvis.visor().open();

    const fitCallbacks = tfvis.show.fitCallbacks(
      { name: "Training (Loss & Accuracy)", tab: "Training", dom: el.visFit },
      ["loss", "acc", "val_loss", "val_acc"],
      { callbacks: ["onEpochEnd"] }
    );

    await state.model.fit(state.XTrain, state.yTrain, {
      epochs,
      batchSize,
      validationData: [state.XVal, state.yVal],
      shuffle: true,
      callbacks: [fitCallbacks, earlyStop]
    });

    state.ready.trained = true;
    state.ready.evaluated = false;

    el.btnEvaluate.disabled = false;
    el.btnPredict.disabled = true;
    el.btnExportModel.disabled = false;

    setStatus("trained");
  }

  // -----------------------------
  // Metrics: ROC/AUC + Confusion matrix with slider
  // -----------------------------
  async function evaluate() {
    setStatus("evaluating");
    showError("");

    if (!state.ready.trained) { showError("Train the model first."); return; }

    // Predict validation probabilities
    const probsT = state.model.predict(state.XVal);
    const probs = Array.from(await probsT.data());
    probsT.dispose();

    const yTrue = Array.from(state.yVal.dataSync()).map(v => (v >= 0.5 ? 1 : 0));

    state.valProbs = probs;

    // ROC + AUC
    state.roc = computeRocAuc(yTrue, probs);
    el.kpiAuc.textContent = Number.isFinite(state.roc.auc) ? state.roc.auc.toFixed(3) : "—";

    // Plot ROC
    const rocSeries = [{
      x: state.roc.fpr,
      y: state.roc.tpr
    }];

    tfvis.render.linechart(
      { name: `ROC Curve (AUC=${state.roc.auc.toFixed(3)})`, tab: "Metrics", dom: el.visRoc },
      { values: rocSeries, series: ["ROC"] },
      { xLabel: "False Positive Rate", yLabel: "True Positive Rate", height: 320 }
    );

    // Enable threshold slider and render at current threshold
    el.thrSlider.disabled = false;
    el.thrSlider.value = el.thrSlider.value || "0.50";
    updateThresholdUI(Number(el.thrSlider.value));

    state.ready.evaluated = true;
    el.btnPredict.disabled = false;

    setStatus("evaluated");
  }

  function computeRocAuc(yTrue, probs) {
    // Sort by probability desc; compute ROC by sweeping threshold at each unique prob
    const pairs = probs.map((p, i) => ({ p, y: yTrue[i] }));
    pairs.sort((a,b)=>b.p - a.p);

    const P = yTrue.reduce((a,b)=>a + (b === 1 ? 1 : 0), 0);
    const N = yTrue.length - P;

    let tp = 0, fp = 0;
    let prevP = Infinity;

    const tpr = [0];
    const fpr = [0];

    for (const item of pairs) {
      if (item.p !== prevP) {
        // record point
        tpr.push(P ? tp / P : 0);
        fpr.push(N ? fp / N : 0);
        prevP = item.p;
      }
      if (item.y === 1) tp++;
      else fp++;
    }
    // final point
    tpr.push(1);
    fpr.push(1);

    // AUC trapezoid on FPR axis
    let auc = 0;
    for (let i = 1; i < fpr.length; i++) {
      const dx = fpr[i] - fpr[i - 1];
      const avgY = (tpr[i] + tpr[i - 1]) / 2;
      auc += dx * avgY;
    }
    return { fpr, tpr, auc };
  }

  function updateThresholdUI(thr) {
    const t = clamp(thr, 0, 1);
    el.thrValue.textContent = `threshold: ${t.toFixed(2)}`;

    if (!state.valProbs || !state.ready.trained) return;

    const yTrue = Array.from(state.yVal.dataSync()).map(v => (v >= 0.5 ? 1 : 0));
    const yPred = state.valProbs.map(p => (p >= t ? 1 : 0));

    const { tp, fp, tn, fn } = confusion(yTrue, yPred);
    const precision = tp + fp ? tp / (tp + fp) : 0;
    const recall = tp + fn ? tp / (tp + fn) : 0;
    const f1 = (precision + recall) ? (2 * precision * recall / (precision + recall)) : 0;
    const acc = (tp + tn) / Math.max(1, (tp + tn + fp + fn));

    // Confusion matrix for tfjs-vis expects [[tn, fp],[fn,tp]]
    tfvis.render.confusionMatrix(
      { name: "Confusion Matrix (val)", tab: "Metrics", dom: el.visConf },
      { values: [[tn, fp], [fn, tp]], tickLabels: ["0", "1"] }
    );

    el.metricsOut.innerHTML = `
      <b>Threshold metrics (validation)</b><br/>
      • Accuracy: <span class="mono">${acc.toFixed(3)}</span><br/>
      • Precision: <span class="mono">${precision.toFixed(3)}</span><br/>
      • Recall: <span class="mono">${recall.toFixed(3)}</span><br/>
      • F1: <span class="mono">${f1.toFixed(3)}</span><br/>
      <span class="muted small">tp=${tp}, fp=${fp}, tn=${tn}, fn=${fn}</span>
    `;
  }

  function confusion(yTrue, yPred) {
    let tp=0, fp=0, tn=0, fn=0;
    for (let i=0;i<yTrue.length;i++){
      const yt = yTrue[i], yp = yPred[i];
      if (yt === 1 && yp === 1) tp++;
      else if (yt === 0 && yp === 1) fp++;
      else if (yt === 0 && yp === 0) tn++;
      else if (yt === 1 && yp === 0) fn++;
    }
    return { tp, fp, tn, fn };
  }

  // -----------------------------
  // Inference on test + export CSVs
  // -----------------------------
  async function predictAndDownload() {
    setStatus("predicting test");
    showError("");

    if (!state.ready.evaluated) { showError("Evaluate first (to enable threshold and metrics)."); return; }

    const thr = Number(el.thrSlider.value || 0.5);

    // Build XTest in the same way (impute + scale + one-hot)
    const XTest = buildXTest(state.testRows, {
      useFamily: el.optFamilySize.checked,
      useAlone: el.optIsAlone.checked
    });

    const probsT = state.model.predict(XTest);
    const probs = Array.from(await probsT.data());
    probsT.dispose();
    XTest.dispose();

    // Export: submission.csv + probabilities.csv
    const sub = [];
    const probOut = [];

    for (let i=0;i<state.testRows.length;i++){
      const id = toNumber(state.testRows[i][SCHEMA.ID]);
      const p = probs[i];
      const pred = (p >= thr ? 1 : 0);
      sub.push({ PassengerId: id, Survived: pred });
      probOut.push({ PassengerId: id, probability: p });
    }

    downloadText("submission.csv", toCSV(sub, ["PassengerId","Survived"]));
    downloadText("probabilities.csv", toCSV(probOut, ["PassengerId","probability"]));

    setStatus("done (downloads started)");
  }

  function buildXTest(rows, { useFamily, useAlone }) {
    // We must rebuild the same one-hot level sets that were used during preprocessing.
    // Since we inferred levels from train+test during preprocessing, we re-infer them again here from current data.
    // (In a more complex app you'd store them in state; here we keep it simple and deterministic.)

    const train = state.trainRows;
    const test = state.testRows;

    const sexLevels = uniqueNonNull([...train, ...test].map(r => (toStr(r.Sex) ?? "Missing").toLowerCase()));
    const pclassLevels = uniqueNonNull([...train, ...test].map(r => String(toNumber(r.Pclass) ?? "Missing")));
    const embarkedLevels = uniqueNonNull([...train, ...test].map(r => toStr(r.Embarked) ?? "Missing"));

    sexLevels.sort();
    pclassLevels.sort((a,b)=>Number(a)-Number(b));
    embarkedLevels.sort();

    const { ageMedian, fareMedian, embarkedMode, meanAge, stdAge, meanFare, stdFare } = state.stats;

    const X = [];

    for (const r of rows) {
      const age = toNumber(r.Age) ?? ageMedian;
      const fare = toNumber(r.Fare) ?? fareMedian;
      const sibsp = toNumber(r.SibSp) ?? 0;
      const parch = toNumber(r.Parch) ?? 0;

      const embarkedRaw = toStr(r.Embarked);
      const embarked = embarkedRaw ?? embarkedMode;

      const sex = (toStr(r.Sex) ?? "Missing").toLowerCase();
      const pclass = String(toNumber(r.Pclass) ?? "Missing");

      const rowX = [];
      rowX.push((age - meanAge) / stdAge);
      rowX.push((fare - meanFare) / stdFare);
      rowX.push(sibsp);
      rowX.push(parch);

      if (useFamily || useAlone) {
        const familySize = sibsp + parch + 1;
        if (useFamily) rowX.push(familySize);
        if (useAlone) rowX.push(familySize === 1 ? 1 : 0);
      }

      for (const lv of pclassLevels) rowX.push(pclass === lv ? 1 : 0);
      for (const lv of sexLevels) rowX.push(sex === lv ? 1 : 0);
      for (const lv of embarkedLevels) rowX.push(embarked === lv ? 1 : 0);

      X.push(rowX);
    }

    return tf.tensor2d(X, [X.length, X[0].length], "float32");
  }

  // -----------------------------
  // Export model
  // -----------------------------
  async function exportModel() {
    showError("");
    if (!state.model) { showError("No model to export."); return; }
    setStatus("exporting model");
    await state.model.save("downloads://titanic-tfjs");
    setStatus("model exported");
  }

  // -----------------------------
  // Events
  // -----------------------------
  function bindEvents() {
    el.btnFetch.addEventListener("click", () => loadData({ preferRepo: true }));
    el.btnLoad.addEventListener("click", () => loadData({ preferRepo: false }));

    el.btnPreprocess.addEventListener("click", () => {
      preprocess();
      // enable build model button after preprocessing
      el.btnBuildModel.disabled = !state.ready.preproc;
    });

    el.btnBuildModel.addEventListener("click", () => buildModel());

    el.btnTrain.addEventListener("click", async () => {
      el.btnTrain.disabled = true;
      try { await train(); }
      finally { el.btnTrain.disabled = false; }
    });

    el.btnEvaluate.addEventListener("click", async () => {
      el.btnEvaluate.disabled = true;
      try { await evaluate(); }
      finally { el.btnEvaluate.disabled = false; }
    });

    el.thrSlider.addEventListener("input", () => updateThresholdUI(Number(el.thrSlider.value)));

    el.btnPredict.addEventListener("click", async () => {
      el.btnPredict.disabled = true;
      try { await predictAndDownload(); }
      finally { el.btnPredict.disabled = false; }
    });

    el.btnExportModel.addEventListener("click", async () => {
      el.btnExportModel.disabled = true;
      try { await exportModel(); }
      finally { el.btnExportModel.disabled = false; }
    });

    // If toggles change after preprocessing, user should re-preprocess and rebuild model
    const toggleHandler = () => {
      if (!state.ready.data) return;
      state.ready.preproc = false;
      state.ready.model = false;
      state.ready.trained = false;
      state.ready.evaluated = false;
      el.btnPreprocess.disabled = false;
      el.btnBuildModel.disabled = true;
      el.btnTrain.disabled = true;
      el.btnEvaluate.disabled = true;
      el.btnPredict.disabled = true;
      el.btnExportModel.disabled = true;
      el.thrSlider.disabled = true;
      el.kpiAuc.textContent = "—";
      el.preprocOut.innerHTML = `<b>Note:</b> Feature toggles changed — run <b>Preprocess</b> again.`;
      el.modelOut.innerHTML = "";
      el.metricsOut.innerHTML = "";
    };
    el.optFamilySize.addEventListener("change", toggleHandler);
    el.optIsAlone.addEventListener("change", toggleHandler);
  }

  function init() {
    setStatus("idle");
    setDataPill(false, "data: not loaded");
    showError("");

    // Initial button states
    el.btnPreprocess.disabled = true;
    el.btnBuildModel.disabled = true;
    el.btnTrain.disabled = true;
    el.btnEvaluate.disabled = true;
    el.btnPredict.disabled = true;
    el.btnExportModel.disabled = true;
    el.thrSlider.disabled = true;

    bindEvents();
  }

  function clamp(x, a, b) { return Math.max(a, Math.min(b, x)); }
  function clampInt(x, a, b) {
    if (!Number.isFinite(x)) return a;
    return Math.max(a, Math.min(b, Math.floor(x)));
  }

  document.addEventListener("DOMContentLoaded", init);
})();
