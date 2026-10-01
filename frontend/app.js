(() => {
  const API = ""; // same-origin: FastAPI serves this file, so relative paths work.

  const el = (id) => document.getElementById(id);

  const apiStatus = el("apiStatus");
  const sourceButtons = document.querySelectorAll(".source-toggle[role='tablist'] .source-btn");
  const uploadSource = el("uploadSource");
  const sampleSource = el("sampleSource");
  const dropzone = el("dropzone");
  const fileInput = el("fileInput");
  const dzTitle = el("dzTitle");
  const dropzoneSecondary = el("dropzoneSecondary");
  const secondFileInput = el("secondFileInput");
  const dzSecondaryTitle = el("dzSecondaryTitle");
  const sampleSelect = el("sampleSelect");
  const samplePairSelect = el("samplePairSelect");
  const municipalitySelect = el("municipalitySelect");
  const backendSelect = el("backendSelect");
  const backendRow = el("backendRow");
  const modeToggle = el("modeToggle");
  const modeButtons = document.querySelectorAll("#modeToggle .source-btn");
  let extractionMode = "cv";
  const topKInput = el("topKInput");
  const allFieldsInput = el("allFieldsInput");
  const runBtn = el("runBtn");
  const runHint = el("runHint");

  const progressPanel = el("progressPanel");
  const jobStatusChip = el("jobStatusChip");
  const stageList = el("stageList");
  const logConsole = el("logConsole");

  const resultsPanel = el("resultsPanel");
  const errorPanel = el("errorPanel");
  const errorMessage = el("errorMessage");
  const downloadBtn = el("downloadBtn");

  let activeSource = "upload";
  let selectedFile = null;
  let selectedPairFile = null;
  let samplePlanList = [];
  let lastReport = null;
  let pollTimer = null;

  // ---------- regulation library elements ----------
  const regMunicipalityInput = el("regMunicipalityInput");
  const regMunicipalityList = el("regMunicipalityList");
  const regForceInput = el("regForceInput");
  const regDropzone = el("regDropzone");
  const regFileInput = el("regFileInput");
  const regDzTitle = el("regDzTitle");
  const regIngestBtn = el("regIngestBtn");
  const regRunHint = el("regRunHint");
  const regLogConsole = el("regLogConsole");
  const regDocTable = el("regDocTable");
  const regLibHint = el("regLibHint");
  let regSelectedFiles = [];
  let regPollTimer = null;

  // ---------- boot ----------
  async function boot() {
    try {
      const res = await fetch(`${API}/health`);
      if (res.ok) {
        apiStatus.dataset.state = "ok";
        apiStatus.querySelector(".label").textContent = "backend online";
      } else {
        throw new Error("bad status");
      }
    } catch {
      apiStatus.dataset.state = "down";
      apiStatus.querySelector(".label").textContent = "backend unreachable";
    }

    await refreshMunicipalities();

    try {
      const plans = await fetchJSON(`${API}/api/sample-plans`);
      samplePlanList = plans.sample_plans || [];
      sampleSelect.innerHTML = "";
      for (const p of samplePlanList) {
        const opt = document.createElement("option");
        opt.value = p;
        opt.textContent = p;
        sampleSelect.appendChild(opt);
      }
      if (!samplePlanList.length) {
        sampleSelect.innerHTML = `<option value="">no sample plans found</option>`;
      }
      refreshSamplePairOptions();
    } catch {
      sampleSelect.innerHTML = `<option value="">could not load sample plans</option>`;
    }

    updateRunState();
    updateRegRunState();
  }

  async function refreshMunicipalities() {
    let list = ["BBMP"];
    try {
      const munis = await fetchJSON(`${API}/api/municipalities`);
      if (munis.municipalities && munis.municipalities.length) list = munis.municipalities;
    } catch { /* keep default */ }

    municipalitySelect.innerHTML = "";
    for (const m of list) {
      const opt = document.createElement("option");
      opt.value = m;
      opt.textContent = m;
      municipalitySelect.appendChild(opt);
    }

    regMunicipalityList.innerHTML = "";
    for (const m of list) {
      const opt = document.createElement("option");
      opt.value = m;
      regMunicipalityList.appendChild(opt);
    }
    if (!regMunicipalityInput.value && list.length) regMunicipalityInput.value = list[0];
    loadRegLibrary();
  }

  async function fetchJSON(url, opts) {
    const res = await fetch(url, opts);
    if (!res.ok) {
      const text = await res.text().catch(() => "");
      throw new Error(`${res.status} ${res.statusText} ${text}`);
    }
    return res.json();
  }

  // ---------- source toggle ----------
  sourceButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      sourceButtons.forEach((b) => b.classList.remove("is-active"));
      btn.classList.add("is-active");
      activeSource = btn.dataset.source;
      uploadSource.hidden = activeSource !== "upload";
      sampleSource.hidden = activeSource !== "sample";
      updateRunState();
    });
  });

  // ---------- extraction mode toggle (Section 7: CV / Vision / Fusion) ----------
  modeButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      modeButtons.forEach((b) => { b.classList.remove("is-active"); b.setAttribute("aria-checked", "false"); });
      btn.classList.add("is-active");
      btn.setAttribute("aria-checked", "true");
      extractionMode = btn.dataset.mode;
      // CV mode never calls Vision (Section 6); the backend select is only
      // meaningful for Vision/Fusion modes, so hide it otherwise.
      backendRow.hidden = extractionMode === "cv";
    });
  });

  // ---------- upload ----------
  dropzone.addEventListener("click", () => fileInput.click());
  dropzone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); }
  });
  ["dragover", "dragenter"].forEach((evt) =>
    dropzone.addEventListener(evt, (e) => { e.preventDefault(); dropzone.classList.add("is-drag"); })
  );
  ["dragleave", "drop"].forEach((evt) =>
    dropzone.addEventListener(evt, (e) => { e.preventDefault(); dropzone.classList.remove("is-drag"); })
  );
  dropzone.addEventListener("drop", (e) => {
    const f = e.dataTransfer.files && e.dataTransfer.files[0];
    if (f) setFile(f);
  });
  fileInput.addEventListener("change", () => {
    if (fileInput.files[0]) setFile(fileInput.files[0]);
  });

  function setFile(f) {
    const name = f.name.toLowerCase();
    if (!name.endsWith(".pdf") && !name.endsWith(".dxf")) {
      dzTitle.textContent = "Only PDF or DXF files are supported — try again";
      selectedFile = null;
    } else {
      selectedFile = f;
      dzTitle.textContent = f.name;
    }
    // The secondary (cross-check) dropzone is only meaningful once a
    // primary file is chosen -- it needs to know the primary's extension
    // to reject a same-format pair, and showing it before that is just
    // clutter for the (default) single-file user.
    dropzoneSecondary.hidden = !selectedFile;
    if (!selectedFile) {
      selectedPairFile = null;
      dzSecondaryTitle.textContent = "Optional: add the matching PDF/DXF to cross-check";
    }
    updateRunState();
  }

  dropzoneSecondary.addEventListener("click", () => secondFileInput.click());
  dropzoneSecondary.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); secondFileInput.click(); }
  });
  ["dragover", "dragenter"].forEach((evt) =>
    dropzoneSecondary.addEventListener(evt, (e) => { e.preventDefault(); dropzoneSecondary.classList.add("is-drag"); })
  );
  ["dragleave", "drop"].forEach((evt) =>
    dropzoneSecondary.addEventListener(evt, (e) => { e.preventDefault(); dropzoneSecondary.classList.remove("is-drag"); })
  );
  dropzoneSecondary.addEventListener("drop", (e) => {
    const f = e.dataTransfer.files && e.dataTransfer.files[0];
    if (f) setPairFile(f);
  });
  secondFileInput.addEventListener("change", () => {
    if (secondFileInput.files[0]) setPairFile(secondFileInput.files[0]);
  });

  function setPairFile(f) {
    const name = f.name.toLowerCase();
    const primaryExt = selectedFile ? selectedFile.name.toLowerCase().split(".").pop() : null;
    const ext = name.split(".").pop();
    if (ext !== "pdf" && ext !== "dxf") {
      dzSecondaryTitle.textContent = "Only PDF or DXF files are supported — try again";
      selectedPairFile = null;
    } else if (primaryExt && ext === primaryExt) {
      dzSecondaryTitle.textContent = `Must be the opposite format of the primary file (.${primaryExt}) — try again`;
      selectedPairFile = null;
    } else {
      selectedPairFile = f;
      dzSecondaryTitle.textContent = f.name;
    }
    updateRunState();
  }

  sampleSelect.addEventListener("change", () => { refreshSamplePairOptions(); updateRunState(); });
  samplePairSelect.addEventListener("change", updateRunState);

  // A sample can only be cross-checked against another bundled sample that
  // shares its filename stem and has the opposite extension -- mirrors the
  // backend's own `_resolve_pdf_dxf_pair` pairing rule (one PDF, one DXF).
  function refreshSamplePairOptions() {
    const primary = sampleSelect.value;
    samplePairSelect.innerHTML = `<option value="">None</option>`;
    if (!primary) return;
    const dot = primary.lastIndexOf(".");
    const stem = dot === -1 ? primary : primary.slice(0, dot);
    const primaryExt = dot === -1 ? "" : primary.slice(dot + 1).toLowerCase();
    for (const p of samplePlanList) {
      if (p === primary) continue;
      const pDot = p.lastIndexOf(".");
      const pStem = pDot === -1 ? p : p.slice(0, pDot);
      const pExt = pDot === -1 ? "" : p.slice(pDot + 1).toLowerCase();
      if (pStem === stem && pExt !== primaryExt) {
        const opt = document.createElement("option");
        opt.value = p;
        opt.textContent = p;
        samplePairSelect.appendChild(opt);
      }
    }
  }

  function updateRunState() {
    const ready =
      (activeSource === "upload" && selectedFile) ||
      (activeSource === "sample" && sampleSelect.value);
    runBtn.disabled = !ready;
    runHint.textContent = ready
      ? "Ready. This can take a minute or two, depending on vision/RAG settings."
      : activeSource === "upload"
      ? "Select a plan to begin."
      : "Choose a sample plan to begin.";
  }

  // ---------- regulation library: upload + ingest ----------
  regDropzone.addEventListener("click", () => regFileInput.click());
  regDropzone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); regFileInput.click(); }
  });
  ["dragover", "dragenter"].forEach((evt) =>
    regDropzone.addEventListener(evt, (e) => { e.preventDefault(); regDropzone.classList.add("is-drag"); })
  );
  ["dragleave", "drop"].forEach((evt) =>
    regDropzone.addEventListener(evt, (e) => { e.preventDefault(); regDropzone.classList.remove("is-drag"); })
  );
  regDropzone.addEventListener("drop", (e) => {
    const files = Array.from(e.dataTransfer.files || []).filter((f) => /\.(pdf|txt)$/i.test(f.name));
    if (files.length) setRegFiles(files);
  });
  regFileInput.addEventListener("change", () => {
    if (regFileInput.files.length) setRegFiles(Array.from(regFileInput.files));
  });
  regMunicipalityInput.addEventListener("input", () => { updateRegRunState(); loadRegLibrary(); });

  function setRegFiles(files) {
    regSelectedFiles = files.filter((f) => /\.(pdf|txt)$/i.test(f.name));
    regDzTitle.textContent = regSelectedFiles.length
      ? `${regSelectedFiles.length} file(s): ${regSelectedFiles.map((f) => f.name).join(", ")}`
      : "Only .pdf / .txt files are supported — try again";
    updateRegRunState();
  }

  function updateRegRunState() {
    const ready = regMunicipalityInput.value.trim() && regSelectedFiles.length;
    regIngestBtn.disabled = !ready;
    regRunHint.textContent = ready
      ? `Ready to ingest ${regSelectedFiles.length} file(s) into ${regMunicipalityInput.value.trim().toUpperCase()}.`
      : "Choose a municipality and at least one file to begin.";
  }

  regIngestBtn.addEventListener("click", startIngest);

  async function startIngest() {
    regIngestBtn.disabled = true;
    regLogConsole.hidden = false;
    regLogConsole.innerHTML = "";
    const municipality = regMunicipalityInput.value.trim().toUpperCase();

    try {
      const fd = new FormData();
      fd.append("municipality", municipality);
      for (const f of regSelectedFiles) fd.append("files", f);
      fd.append("force", regForceInput.checked ? "true" : "false");
      const res = await fetchJSON(`${API}/api/regulations/upload`, { method: "POST", body: fd });
      pollIngestJob(res.job_id, municipality);
    } catch (err) {
      appendRegLog({ t: new Date().toISOString(), message: `ERROR: ${err.message || err}` });
      regIngestBtn.disabled = false;
    }
  }

  let regSeenLogCount = 0;

  function pollIngestJob(jobId, municipality) {
    regSeenLogCount = 0;
    if (regPollTimer) clearInterval(regPollTimer);
    regPollTimer = setInterval(async () => {
      try {
        const job = await fetchJSON(`${API}/api/regulations/jobs/${jobId}`);
        for (const entry of job.log.slice(regSeenLogCount)) appendRegLog(entry);
        regSeenLogCount = job.log.length;
        if (job.status === "done" || job.status === "error") {
          clearInterval(regPollTimer);
          regIngestBtn.disabled = false;
          if (job.status === "error") appendRegLog({ t: new Date().toISOString(), message: `ERROR: ${job.error || "unknown error"}` });
          if (job.status === "done") {
            regSelectedFiles = [];
            regDzTitle.textContent = "Drop rule PDFs here, or click to browse";
            updateRegRunState();
            await refreshMunicipalities();
            regMunicipalityInput.value = municipality;
            loadRegLibrary();
          }
        }
      } catch (err) {
        clearInterval(regPollTimer);
        appendRegLog({ t: new Date().toISOString(), message: `ERROR: ${err.message || err}` });
        regIngestBtn.disabled = false;
      }
    }, 1000);
  }

  function appendRegLog(entry) {
    const line = document.createElement("div");
    line.className = "log-line";
    if (/ERROR|FATAL/.test(entry.message)) line.classList.add("err");
    else if (/WARNING/.test(entry.message)) line.classList.add("warn");
    const stamp = new Date(entry.t).toLocaleTimeString();
    line.innerHTML = `<span class="t">${stamp}</span>${escapeHTML(entry.message)}`;
    regLogConsole.appendChild(line);
    regLogConsole.scrollTop = regLogConsole.scrollHeight;
  }

  async function loadRegLibrary() {
    const municipality = regMunicipalityInput.value.trim().toUpperCase();
    if (!municipality) {
      regDocTable.innerHTML = "";
      regLibHint.textContent = "select a municipality above to view";
      return;
    }
    regLibHint.textContent = "loading…";
    try {
      const data = await fetchJSON(`${API}/api/regulations/${encodeURIComponent(municipality)}`);
      regLibHint.textContent = `${data.total_chunks} chunk(s) indexed across ${data.documents.length} document(s)`;
      regDocTable.innerHTML = "";
      if (!data.documents.length) {
        regDocTable.innerHTML = `<p style="color:var(--paper-faint);font-family:var(--mono);font-size:13px;">Nothing indexed yet for ${escapeHTML(municipality)}. Upload PDFs above.</p>`;
        return;
      }
      for (const doc of data.documents) {
        const row = document.createElement("div");
        row.className = "doc-row";
        row.innerHTML = `<span class="doc-name">${escapeHTML(doc.source_file)}</span><span class="doc-tag">${escapeHTML(doc.doc_type)}</span><span class="doc-tag">${escapeHTML(doc.city)}</span><span class="doc-count">${doc.pages} page(s) · ${doc.chunks} chunk(s)</span>`;
        regDocTable.appendChild(row);
      }
    } catch (err) {
      regLibHint.textContent = "";
      regDocTable.innerHTML = `<p style="color:var(--fail);font-family:var(--mono);font-size:13px;">Could not load library: ${escapeHTML(String(err.message || err))}</p>`;
    }
  }

  // ---------- run ----------
  runBtn.addEventListener("click", startRun);

  async function startRun() {
    resetPanels();
    progressPanel.hidden = false;
    progressPanel.scrollIntoView({ behavior: "smooth", block: "start" });
    runBtn.disabled = true;

    const municipality = municipalitySelect.value || "BBMP";
    const mode = extractionMode; // "cv" | "vision" | "fusion" -- Section 7
    const vision = mode !== "cv";
    const backend = vision ? (backendSelect.value || "api") : null;
    const topK = topKInput.value ? parseInt(topKInput.value, 10) : null;
    const allFields = allFieldsInput.checked;

    try {
      let jobId;
      if (activeSource === "upload" && selectedFile) {
        const fd = new FormData();
        fd.append("file", selectedFile);
        if (selectedPairFile) fd.append("second_file", selectedPairFile);
        fd.append("municipality", municipality);
        fd.append("vision", vision ? "true" : "false");
        fd.append("mode", mode);
        if (backend) fd.append("backend", backend);
        if (topK) fd.append("top_k", String(topK));
        fd.append("all_fields", allFields ? "true" : "false");
        const res = await fetchJSON(`${API}/api/analyze/upload`, { method: "POST", body: fd });
        jobId = res.job_id;
      } else {
        const res = await fetchJSON(`${API}/api/analyze/sample`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            sample_plan: sampleSelect.value,
            second_sample_plan: samplePairSelect.value || null,
            municipality,
            vision,
            mode,
            backend,
            top_k: topK,
            all_fields: allFields,
          }),
        });
        jobId = res.job_id;
      }
      pollJob(jobId);
    } catch (err) {
      showError(String(err.message || err));
      runBtn.disabled = false;
    }
  }

  function resetPanels() {
    resultsPanel.hidden = true;
    errorPanel.hidden = true;
    logConsole.innerHTML = "";
    stageList.querySelectorAll("li").forEach((li) => li.classList.remove("is-active", "is-done"));
    jobStatusChip.textContent = "pending";
    jobStatusChip.dataset.s = "pending";
    if (pollTimer) clearInterval(pollTimer);
  }

  let seenLogCount = 0;

  function pollJob(jobId) {
    seenLogCount = 0;
    pollTimer = setInterval(async () => {
      try {
        const job = await fetchJSON(`${API}/api/jobs/${jobId}`);
        renderJob(job);
        if (job.status === "done" || job.status === "error") {
          clearInterval(pollTimer);
          runBtn.disabled = false;
          if (job.status === "done") { job.result.job_id = jobId; renderResult(job.result); }
          if (job.status === "error") showError(job.error || "Unknown error");
        }
      } catch (err) {
        clearInterval(pollTimer);
        showError(String(err.message || err));
        runBtn.disabled = false;
      }
    }, 1200);
  }

  function renderJob(job) {
    jobStatusChip.textContent = job.status;
    jobStatusChip.dataset.s = job.status;

    for (const entry of job.log.slice(seenLogCount)) {
      appendLog(entry);
    }
    seenLogCount = job.log.length;

    const allText = job.log.map((l) => l.message).join(" ");
    setStage(1, /\[1\/4\]/.test(allText), /\[2\/4\]/.test(allText));
    setStage(2, /\[2\/4\]/.test(allText), /\[3\/4\]/.test(allText));
    setStage(3, /\[3\/4\]/.test(allText), /\[4\/4\]/.test(allText));
    setStage(4, /\[4\/4\]/.test(allText), job.status === "done");
  }

  function setStage(n, started, done) {
    const li = stageList.querySelector(`li[data-stage="${n}"]`);
    if (!li) return;
    li.classList.toggle("is-active", started && !done);
    li.classList.toggle("is-done", done);
  }

  function appendLog(entry) {
    const line = document.createElement("div");
    line.className = "log-line";
    if (/ERROR|FATAL/.test(entry.message)) line.classList.add("err");
    else if (/WARNING/.test(entry.message)) line.classList.add("warn");
    const t = new Date(entry.t);
    const stamp = t.toLocaleTimeString();
    line.innerHTML = `<span class="t">${stamp}</span>${escapeHTML(entry.message)}`;
    logConsole.appendChild(line);
    logConsole.scrollTop = logConsole.scrollHeight;
  }

  function escapeHTML(s) {
    return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }

  function showError(msg) {
    errorPanel.hidden = false;
    errorMessage.textContent = msg;
  }

  // ---------- results ----------
  function renderResult(report) {
    lastReport = report;
    resultsPanel.hidden = false;
    const overall = report.compliance.overall_status;
    el("verdictStamp").dataset.verdict = overall;
    el("verdictText").textContent = overall.replace(/_/g, " ");
    el("metaPlan").textContent = report.pipeline.source_dxf
      ? `${report.pipeline.source_pdf} + ${report.pipeline.source_dxf} (cross-checked)`
      : report.pipeline.source_pdf;
    el("metaMunicipality").textContent = report.pipeline.municipality;
    el("metaChecks").textContent = report.compliance.rule_results.length;
    el("metaDrafts").textContent = report.draft_rules.length;

    const metadataGrid = el("metadataGrid");
    metadataGrid.innerHTML = "";
    const meta = report.plan?.metadata || {};
    const metaPairs = [
      ["Applicant", meta.applicant_name], ["Building type", meta.building_type],
      ["Address", meta.address], ["Plot / site no.", meta.plot_number],
      ["Floor count", meta.floor_count], ["Total built-up area", meta.total_built_up_area],
      ["Plinth area", meta.plinth_area]
    ];
    for (const [k,v] of metaPairs) {
      if (v === null || v === undefined || v === "") continue;
      const card=document.createElement("div"); card.className="meta-card";
      card.innerHTML=`<div class="k">${escapeHTML(k)}</div><div class="v">${escapeHTML(String(v))}</div>`;
      metadataGrid.appendChild(card);
    }
    if (!metadataGrid.children.length) metadataGrid.innerHTML='<div class="meta-card"><div class="v">No explicit metadata was recovered.</div></div>';

    const dimGrid = el("dimensionGrid"); dimGrid.innerHTML = "";
    const s = report.plan_summary;
    const dims = [
      ["building_use","Building use",s.building.use,"context"], ["development_area","Development area",report.plan?.development_area,"context"],
      ["plot.width","Plot width",s.plot.width,"number"], ["plot.depth","Plot depth",s.plot.depth,"number"], ["plot.area","Plot area",s.plot.area,"number"],
      ["building.width","Building width",s.building.width,"number"], ["building.depth","Building depth",s.building.depth,"number"], ["building.footprint_area","Building area",s.building.footprint_area,"number"], ["building.floor_count","Floor count",s.building.floor_count,"integer"],
      ["road.width","Road width",s.road.width,"number"], ["setbacks.front","Setback — front",s.setbacks.front,"number"], ["setbacks.rear","Setback — rear",s.setbacks.rear,"number"],
      ["setbacks.left","Setback — left",s.setbacks.left,"number"], ["setbacks.right","Setback — right",s.setbacks.right,"number"], ["coverage","Plot coverage",s.coverage,"number"], ["far","FAR",s.far,"number"],
      ["building.height_estimated","Estimated building height",s.building_height_estimated,"number"],
      ["building.height_excluding_stilt","Height excluding stilt (draft Table 8)",report.plan?.building_height_excluding_stilt,"number"]
    ];
    for (const [field,label,v,kind] of dims) {
      const card=document.createElement("div"); card.className="dim-card";
      const value=v?.value ?? ""; const unit=v?.unit || "";
      // confidence is serialized as a structured object by Pydantic
      // ({level, score, reason}), not always as a string. Normalize both
      // shapes so editing/re-rendering never crashes the frontend.
      const rawConf=v?.confidence;
      const confValue = typeof rawConf === "string" ? rawConf : (rawConf?.level ?? v?.status ?? "missing");
      const conf=String(confValue || "missing").toLowerCase();
      let editor = null;
      if (kind === "context") {
        if (field === "building_use") {
          const options = [["","Select use"],["residential","Residential"],["commercial","Commercial"],["public_semi_public","Public & Semi-Public"],["traffic_transportation","Traffic & Transportation"],["public_utility","Public Utility"],["hospital","Hospital"],["health_centre_nursing_home","Health centre / nursing home"],["nursery_primary_school","Nursery / primary school"],["secondary_school","Secondary school"],["college","College"]];
          editor = `<select class="edit-value" data-field="${field}" aria-label="${escapeHTML(label)}">${options.map(([val,text]) => `<option value="${val}" ${val===value?'selected':''}>${escapeHTML(text)}</option>`).join("")}</select>`;
        } else {
          editor = `<select class="edit-value" data-field="${field}" aria-label="${escapeHTML(label)}"><option value="">Select area</option><option value="A" ${value==='A'?'selected':''}>A — Intensely developed</option><option value="B" ${value==='B'?'selected':''}>B — Moderately developed</option><option value="C" ${value==='C'?'selected':''}>C — Sparsely developed</option></select>`;
        }
      } else {
        // Every numeric field remains editable even when extraction returned
        // no value. Previously a missing value rendered only as "—", which
        // made it impossible to supply required applicability inputs such as
        // floor_count for high-rise rules.
        const inputType = kind === "integer" ? "number" : "number";
        const step = kind === "integer" ? "1" : "any";
        const unitHint = unit || ({
          "plot.area": "m²",
          "building.footprint_area": "m²",
          "coverage": "%",
          "far": "",
          "building.floor_count": "floors"
        }[field] || "m");
        const minAttr = kind === "integer" ? ` min="0"` : "";
        editor = `<input class="edit-value" data-field="${field}" type="${inputType}" step="${step}"${minAttr} value="${value === "" ? "" : escapeHTML(String(value))}" placeholder="Enter value" data-unit="${escapeHTML(unitHint)}" aria-label="${escapeHTML(label)}">`;
      }
      card.dataset.field=field;
      const editedBadge = v?.edited ? `<span class="conf-chip" data-c="edited" title="Manually edited">edited</span>` : "";
      // A PDF<->DXF conflict deliberately does NOT change this field's own
      // confidence level (see backend/spatial_reasoning/
      // pdf_dxf_reconciliation.py) -- it rides on `.conflict` alone, so it
      // needs its own badge regardless of the confidence chip shown above.
      // `.conflict` is also used for non-DXF findings (e.g. a setback that
      // cannot physically fit the resolved plot/building extent), so only
      // call it "DXF disagreed" when the DXF extractor is actually one of the
      // conflicting sources -- otherwise a PDF-only run would claim a DXF
      // disagreement that never happened.
      const conflictFromDxf = (v?.conflict?.conflicting_sources || []).includes("dxf_extractor");
      const dxfConflictBadge = v?.conflict
        ? (conflictFromDxf
            ? `<span class="conf-chip" data-c="pdf_dxf_conflict" title="${escapeHTML(v.conflict.description)}">DXF disagreed</span>`
            : `<span class="conf-chip" data-c="conflicting" title="${escapeHTML(v.conflict.description)}">inconsistent</span>`)
        : "";
      card.innerHTML=`<div class="dim-label">${escapeHTML(label)}</div>${editor}<span class="conf-chip" data-c="${conf}">${conf}</span>${editedBadge}${dxfConflictBadge}`;
      dimGrid.appendChild(card);
    }

    const suggestions = el("suggestionsList"); suggestions.innerHTML="";
    const list=report.suggestions || [];
    if (!list.length) suggestions.innerHTML='<div class="suggestion"><strong>NO CURRENT FAILURES</strong><p>No deterministic FAIL result was returned. Review INSUFFICIENT_DATA or REQUIRES_REVIEW items if present.</p></div>';
    for (const item of list) {
      const div=document.createElement("div"); div.className="suggestion";
      div.innerHTML=`<strong>${escapeHTML(item.field)}</strong><p>${escapeHTML(item.suggestion)}</p>`;
      suggestions.appendChild(div);
    }

    // Runtime compliance is driven by rules.json. RAG is not used to generate
    // or alter rules during this check.
    const chunkMap = {};
    for (const fieldChunks of Object.values(report.retrieval || {})) {
      for (const c of fieldChunks) if (c.chunk_id) chunkMap[c.chunk_id] = c;
    }

    const table = el("resultsTable"); table.innerHTML="";
    const allResults = report.compliance.rule_results;
    const showAll = el("showNotApplicableToggle")?.checked;
    // By default, hide NOT_APPLICABLE rows: on a plan with an unresolved
    // building_use/development_area, the vast majority of the 200+ rule
    // bank legitimately doesn't apply and isn't useful noise -- what's
    // actionable is PASS / FAIL / INSUFFICIENT_DATA / REQUIRES_REVIEW /
    // CONFLICTING_EVIDENCE. The checkbox above still lets anyone inspect
    // the full NOT_APPLICABLE set (e.g. to sanity-check applicability
    // logic itself) without deleting that data.
    const visibleResults = showAll ? allResults : allResults.filter((r) => r.status !== "NOT_APPLICABLE");
    const hiddenCount = allResults.length - visibleResults.length;
    const filterNote = el("resultsFilterNote");
    if (filterNote) {
      filterNote.textContent = hiddenCount > 0 && !showAll
        ? `showing ${visibleResults.length} of ${allResults.length} — ${hiddenCount} NOT_APPLICABLE hidden`
        : `${allResults.length} rule(s)`;
    }
    if (!visibleResults.length) table.innerHTML='<p style="color:var(--paper-faint);font-family:var(--mono);font-size:13px;">No fields had enough grounded evidence to evaluate.</p>';
    for (const r of visibleResults) {
      const row=document.createElement("details"); row.className="result-row";
      const usedChunks = [];

      const evidenceHTML = usedChunks.length
        ? `<div class="evidence-list">${usedChunks.map((c) => `
            <div class="evidence-chunk">
              <div class="evidence-chunk-head">
                <span class="evidence-clause">${escapeHTML(c.clause_ref || "unlabelled clause")}</span>
                <span class="evidence-source">${escapeHTML(c.source_file || "")}${c.page_num ? ` · p.${c.page_num}` : ""}</span>
                ${(c.rrf_score ?? c.score) != null ? `<span class="evidence-score">score ${(c.rrf_score ?? c.score).toFixed(3)}</span>` : ""}
              </div>
              <pre class="evidence-text">${escapeHTML(c.text || "")}</pre>
            </div>`).join("")}</div>`
        : `<p class="evidence-empty">This result was evaluated directly from the authoritative municipality ruleset; RAG is not part of the decision.</p>`;

      row.innerHTML=`<summary><span class="status-chip" data-s="${escapeHTML(r.status)}">${escapeHTML(r.status.replace(/_/g," "))}</span><span class="result-field">${escapeHTML(r.source_field || "")}</span><span class="result-desc">${escapeHTML(r.required_value_description || "no requirement text")}</span></summary><div class="result-detail"><div>${escapeHTML(r.explanation || "")}</div>${r.citation ? `<span class="citation">${escapeHTML(r.citation)}${r.source_page ? ` · source p.${r.source_page}` : ""}</span>` : ""}<div class="evidence-heading">Deterministic rule source</div>${evidenceHTML}</div>`;
      table.appendChild(row);
    }
    el("fusionAudit").textContent=JSON.stringify(report.fusion || {}, null, 2);
    const pdfDxfWrap = el("pdfDxfPanelWrap");
    if (report.pdf_dxf_reconciliation) {
      pdfDxfWrap.hidden = false;
      el("pdfDxfPanel").textContent = JSON.stringify(report.pdf_dxf_reconciliation, null, 2);
    } else {
      pdfDxfWrap.hidden = true;
      el("pdfDxfPanel").textContent = "";
    }
    el("editStatus").textContent="";
  }

  // Re-render the compliance table (without re-running anything) when the
  // "show NOT_APPLICABLE" checkbox is toggled.
  el("showNotApplicableToggle")?.addEventListener("change", () => {
    if (lastReport) renderResult(lastReport);
  });

  el("saveEditsBtn").addEventListener("click", async () => {
    if (!lastReport) return;
    const updates={};
    document.querySelectorAll(".edit-value").forEach(input => {
      if (input.tagName === "SELECT") {
        if (input.value) updates[input.dataset.field]={value:input.value};
        return;
      }
      const raw = input.value.trim();
      if (!raw) return;
      const value = input.dataset.field === "building.floor_count" ? parseInt(raw, 10) : parseFloat(raw);
      if (Number.isFinite(value)) updates[input.dataset.field]={value, unit:input.dataset.unit};
    });
    if (!lastReport.job_id) {
      el("editStatus").textContent="This result predates the editable job state.";
      return;
    }
    el("editStatus").textContent="Rechecking…";
    try {
      const updated=await fetchJSON(`${API}/api/jobs/${lastReport.job_id}/edit`, {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({updates})});
      updated.job_id=lastReport.job_id;
      renderResult(updated);
      el("editStatus").textContent="Updated and re-evaluated.";
    } catch(err){ el("editStatus").textContent=`Edit failed: ${err.message || err}`; }
  });

  el("downloadPdfBtn").addEventListener("click", () => {
    if (!lastReport?.job_id) return;
    const a=document.createElement("a"); a.href=`${API}/api/jobs/${lastReport.job_id}/report.pdf`; a.download=`buildcheck_${lastReport.job_id}.pdf`; a.click();
  });

  downloadBtn.addEventListener("click", () => {
    if (!lastReport) return;
    const blob = new Blob([JSON.stringify(lastReport, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `buildcheck_report_${(lastReport.pipeline.source_pdf || "plan").replace(/\.pdf$/i, "")}.json`;
    a.click();
    URL.revokeObjectURL(url);
  });

  boot();
})();
