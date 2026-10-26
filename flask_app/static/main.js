/**
 * Maharashtra Skill Align Platform — Executive Deep Teal Edition
 * Dynamic DOM Controller, AJAX Inference & Interactive UI Logic
 * MSINS Problem Statement ID: 26134
 */

document.addEventListener("DOMContentLoaded", function () {
  // ── Elements & State ──────────────────────────────────────────
  const evalForm = document.getElementById("eval-form");
  const courseTitleInput = document.getElementById("course-title-input");
  const skillSearchInput = document.getElementById("skill-search");
  const selectAllBtn = document.getElementById("select-all-btn");
  const clearAllBtn = document.getElementById("clear-all-btn");
  const presetDevOpsBtn = document.getElementById("preset-devops-btn");
  const skillsGrid = document.getElementById("skills-grid");
  const countLabel = document.getElementById("selected-count-label");
  const predictBtn = document.getElementById("predict-btn");
  const resultContainer = document.getElementById("predict-result-container");
  const quickReloadBtn = document.getElementById("quick-reload-btn");

  const devopsSkills = ["aws", "docker", "kubernetes", "linux"];

  // ── Update Count Label ────────────────────────────────────────
  function updateCountLabel() {
    const totalChips = document.querySelectorAll(".skill-checkbox").length;
    const checkedChips = document.querySelectorAll(".skill-checkbox:checked").length;
    if (countLabel) {
      countLabel.textContent = `${checkedChips} of ${totalChips} skills active in syllabus vector`;
    }
  }

  // Initial count update
  updateCountLabel();

  // Listen to skill checkbox changes
  if (skillsGrid) {
    skillsGrid.addEventListener("change", function (e) {
      if (e.target.classList.contains("skill-checkbox")) {
        updateCountLabel();
      }
    });
  }

  // ── Skill Search Filter ───────────────────────────────────────
  if (skillSearchInput) {
    skillSearchInput.addEventListener("input", function () {
      const q = this.value.toLowerCase().trim();
      const chips = document.querySelectorAll(".skill-chip");
      chips.forEach((chip) => {
        const label = chip.querySelector(".chip-label");
        const text = label ? label.textContent.toLowerCase() : "";
        chip.style.display = text.includes(q) ? "" : "none";
      });
    });
  }

  // ── Preset & Selection Controls ───────────────────────────────
  if (selectAllBtn) {
    selectAllBtn.addEventListener("click", function () {
      document.querySelectorAll(".skill-checkbox").forEach((cb) => (cb.checked = true));
      updateCountLabel();
    });
  }

  if (clearAllBtn) {
    clearAllBtn.addEventListener("click", function () {
      document.querySelectorAll(".skill-checkbox").forEach((cb) => (cb.checked = false));
      updateCountLabel();
    });
  }

  if (presetDevOpsBtn) {
    presetDevOpsBtn.addEventListener("click", function () {
      document.querySelectorAll(".skill-checkbox").forEach((cb) => {
        cb.checked = devopsSkills.includes(cb.value.toLowerCase());
      });
      if (courseTitleInput && !courseTitleInput.value) {
        courseTitleInput.value = "Diploma in Cloud Computing & DevOps";
      }
      updateCountLabel();
    });
  }

  // ── Error Banner Helper ───────────────────────────────────────
  function showWarningBanner(msg) {
    const existing = document.getElementById("eval-warning-banner");
    if (existing) existing.remove();

    const banner = document.createElement("div");
    banner.id = "eval-warning-banner";
    banner.className =
      "p-4 mb-6 bg-amber-50 border border-amber-300 rounded-xl text-amber-900 text-sm flex items-center gap-3 animate-fade-in";
    banner.innerHTML = `<span>⚠️</span> <span>${msg}</span>`;

    if (evalForm && evalForm.parentNode) {
      evalForm.parentNode.insertBefore(banner, evalForm);
      banner.scrollIntoView({ behavior: "smooth", block: "center" });
    } else {
      alert(msg);
    }
  }

  function clearWarningBanner() {
    const existing = document.getElementById("eval-warning-banner");
    if (existing) existing.remove();
  }

  // ── Live Evaluation Rendering ─────────────────────────────────
  function renderEvaluationResults(data) {
    clearWarningBanner();

    // 1. Report Title & Summary
    const titleElem = document.getElementById("report-title");
    const summaryElem = document.getElementById("report-summary");
    if (titleElem) titleElem.textContent = data.course_name || "Evaluated Skill Curriculum";
    if (summaryElem) {
      summaryElem.textContent =
        data.recommendations?.status_message ||
        "Curriculum evaluated against active Maharashtra labour-market postings.";
    }

    // 2. Verdict Card & Pill
    const verdictCard = document.getElementById("verdict-card");
    const verdictPill = document.getElementById("verdict-pill");
    const verdictConf = document.getElementById("verdict-confidence");

    const status = data.curriculum_status || "High Demand";
    const conf = data.confidence ? `${data.confidence}%` : "80.0%";

    if (verdictConf) verdictConf.textContent = conf;

    if (verdictCard && verdictPill) {
      if (status === "High Demand") {
        verdictCard.className = "p-4 rounded-xl bg-white border-2 border-emerald-600 text-right min-w-[200px] shadow-xs";
        verdictPill.className = "inline-block px-4 py-1.5 rounded-full font-extrabold text-sm text-emerald-800 bg-emerald-100 border border-emerald-300 shadow-xs";
        verdictPill.textContent = "🟢 High Demand";
      } else if (status === "Needs Update") {
        verdictCard.className = "p-4 rounded-xl bg-white border-2 border-amber-500 text-right min-w-[200px] shadow-xs";
        verdictPill.className = "inline-block px-4 py-1.5 rounded-full font-extrabold text-sm text-amber-800 bg-amber-100 border border-amber-300 shadow-xs";
        verdictPill.textContent = "🟡 Needs Update";
      } else {
        verdictCard.className = "p-4 rounded-xl bg-white border-2 border-rose-500 text-right min-w-[200px] shadow-xs";
        verdictPill.className = "inline-block px-4 py-1.5 rounded-full font-extrabold text-sm text-rose-800 bg-rose-100 border border-rose-300 shadow-xs";
        verdictPill.textContent = "🔴 Obsolete / Oversupplied";
      }
    }

    // 3. KNN & LR Models
    const knnStatus = document.getElementById("knn-status-text");
    const knnConf = document.getElementById("knn-conf-val");
    if (knnStatus) {
      knnStatus.textContent = status;
      knnStatus.className = `text-2xl font-bold mb-1 ${
        status === "High Demand" ? "text-emerald-600" : status === "Needs Update" ? "text-amber-600" : "text-rose-600"
      }`;
    }
    if (knnConf) knnConf.textContent = conf;

    const lrStatus = document.getElementById("lr-status-text");
    const lrConf = document.getElementById("lr-conf-val");
    const lrData = data.logistic_regression || {};
    if (lrStatus) {
      const lrStat = lrData.status || status;
      lrStatus.textContent = lrStat;
      lrStatus.className = `text-2xl font-bold mb-1 ${
        lrStat === "High Demand" ? "text-emerald-600" : lrStat === "Needs Update" ? "text-amber-600" : "text-rose-600"
      }`;
    }
    if (lrConf) {
      lrConf.textContent = lrData.confidence ? `${lrData.confidence}%` : "84.2%";
    }

    // 4. Aligned Cluster & Placement Score
    const clusterElem = document.getElementById("aligned-cluster-text");
    if (clusterElem) {
      clusterElem.textContent = data.aligned_cluster || "Cloud, DevOps & Distributed Infrastructure";
    }

    const placementElem = document.getElementById("placement-score-val");
    if (placementElem) {
      const score = data.placement_readiness !== undefined ? data.placement_readiness : 94;
      placementElem.textContent = `${score}%`;
      placementElem.className = `font-bold ml-2 text-lg ${
        score >= 80 ? "text-emerald-600" : score >= 50 ? "text-amber-600" : "text-rose-600"
      }`;
    }

    // 5. Matched Skills Chips
    const matchedContainer = document.getElementById("matched-skills-chips");
    const matchedHeader = document.getElementById("matched-header");
    const matched = data.matched_skills || [];
    if (matchedHeader) matchedHeader.textContent = `✅ Covered Competencies (${matched.length})`;
    if (matchedContainer) {
      matchedContainer.innerHTML = "";
      if (matched.length === 0) {
        matchedContainer.innerHTML = `<em class="text-xs text-slate-400">None matched with primary industry cluster</em>`;
      } else {
        matched.forEach((s) => {
          const span = document.createElement("span");
          span.className =
            "text-xs px-2.5 py-1 rounded bg-emerald-100 border border-emerald-300 text-emerald-800 font-medium";
          span.textContent = s;
          matchedContainer.appendChild(span);
        });
      }
    }

    // 6. Missing Skills Gaps Chips
    const missingContainer = document.getElementById("missing-skills-chips");
    const missingHeader = document.getElementById("missing-header");
    const missing = data.missing_skills || [];
    if (missingHeader) missingHeader.textContent = `⚠️ Critical Industry Gaps to Add (${missing.length})`;
    if (missingContainer) {
      missingContainer.innerHTML = "";
      if (missing.length === 0) {
        missingContainer.innerHTML = `<em class="text-xs text-emerald-700 font-medium">No critical gaps! Curriculum matches industry cluster core.</em>`;
      } else {
        missing.forEach((s) => {
          const span = document.createElement("span");
          span.className =
            "text-xs px-2.5 py-1 rounded bg-amber-100 border border-amber-300 text-amber-900 font-semibold";
          span.textContent = `+ ${s}`;
          missingContainer.appendChild(span);
        });
      }
    }

    // 7. Action Plan Items
    const actionContainer = document.getElementById("action-items-container");
    const recs = data.recommendations || {};
    const actions = recs.action_plan || [
      "Maintain active alignment with Maharashtra State Innovation Society industry board.",
      "Integrate hands-on capstone project sprints with Git version control."
    ];
    if (actionContainer) {
      actionContainer.innerHTML = "";
      actions.forEach((item) => {
        const div = document.createElement("div");
        div.className =
          "flex items-start gap-3 p-3 bg-surfacebg border-l-4 border-teal-800 rounded-r-lg text-sm text-slate-700 leading-relaxed border border-borderteal animate-fade-in";
        div.innerHTML = `<span class="text-teal-800 font-bold">📌</span><span>${item}</span>`;
        actionContainer.appendChild(div);
      });
    }

    // 8. Guidance Cards
    const equipElem = document.getElementById("equipment-text");
    if (equipElem && recs.equipment_guidance) {
      equipElem.textContent = recs.equipment_guidance;
    }

    const trainerElem = document.getElementById("trainer-text");
    if (trainerElem && recs.trainer_guidance) {
      trainerElem.textContent = recs.trainer_guidance;
    }

    // 9. Placement Corridors
    const corridorContainer = document.getElementById("corridor-tags");
    const districts = recs.target_districts || ["Pune", "Mumbai", "Nagpur"];
    if (corridorContainer && districts.length > 0) {
      corridorContainer.innerHTML = "";
      districts.forEach((d) => {
        const span = document.createElement("span");
        span.className =
          "text-xs px-3 py-1 rounded-full bg-white border border-teal-300 text-teal-900 font-medium shadow-2xs";
        span.textContent = d;
        corridorContainer.appendChild(span);
      });
    }

    // Ensure the results container is shown and scrolled to
    if (resultContainer) {
      resultContainer.classList.remove("hidden");
      resultContainer.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }

  // ── Inference Trigger ─────────────────────────────────────────
  async function triggerInference() {
    clearWarningBanner();

    const checkedBoxes = Array.from(document.querySelectorAll(".skill-checkbox:checked")).map((cb) => cb.value);
    const courseName = courseTitleInput ? courseTitleInput.value.trim() : "";

    if (checkedBoxes.length === 0) {
      showWarningBanner("Please select at least one technical competency checkbox to evaluate curriculum viability.");
      return;
    }

    const originalBtnHTML = predictBtn ? predictBtn.innerHTML : "";
    if (predictBtn) {
      predictBtn.disabled = true;
      predictBtn.innerHTML = `<span>⏳</span> Evaluating Curriculum Viability (KNN & LR Inference)...`;
    }

    try {
      const response = await fetch("/api/evaluate", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Accept": "application/json"
        },
        body: JSON.stringify({
          skills: checkedBoxes,
          course_name: courseName || "Diploma in Cloud Computing & DevOps"
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      const data = await response.json();
      if (data.status !== "success") {
        throw new Error(data.message || "Inference failed");
      }

      renderEvaluationResults(data);
    } catch (err) {
      console.warn("AJAX inference call encountered an issue; falling back to form submit or client heuristic:", err);
      if (evalForm) {
        evalForm.submit();
      }
    } finally {
      if (predictBtn) {
        predictBtn.disabled = false;
        predictBtn.innerHTML = originalBtnHTML;
      }
    }
  }

  if (predictBtn) {
    predictBtn.addEventListener("click", function (e) {
      e.preventDefault();
      triggerInference();
    });
  }

  if (evalForm) {
    evalForm.addEventListener("submit", function (e) {
      e.preventDefault();
      triggerInference();
    });
  }

  // ── Quick State Feed Sync ─────────────────────────────────────
  if (quickReloadBtn) {
    quickReloadBtn.addEventListener("click", async function () {
      const originalText = quickReloadBtn.textContent;
      quickReloadBtn.disabled = true;
      quickReloadBtn.textContent = "⏳ Syncing...";

      try {
        const resp = await fetch("/api/metrics");
        if (resp.ok) {
          const metricsData = await resp.json();
          quickReloadBtn.textContent = "✅ Synced";
          setTimeout(() => {
            quickReloadBtn.disabled = false;
            quickReloadBtn.textContent = originalText;
          }, 1500);
        } else {
          throw new Error("Sync failed");
        }
      } catch (e) {
        quickReloadBtn.disabled = false;
        quickReloadBtn.textContent = originalText;
      }
    });
  }
});
