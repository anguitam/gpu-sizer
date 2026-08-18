import React, { useState, useMemo } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";

/* ============================================================
   data/gpus.js
   Nebius self-serve catalog. Rates are per GPU-hour, USD.
   VERIFY specs against NVIDIA datasheets before publishing.
   ============================================================ */

const GPUS = [
  {
    id: "h100",
    label: "HGX H100",
    vramGiB: 80,
    bwTBs: 3.35,
    tflopsBF16: 990,
    rtCores: false,
    nvlink: true,
    nodeSize: 8,
    onDemand: 3.85,
    committed: 2.15,
  },
  {
    id: "h200",
    label: "HGX H200",
    vramGiB: 141,
    bwTBs: 4.8,
    tflopsBF16: 990,
    rtCores: false,
    nvlink: true,
    nodeSize: 8,
    onDemand: 4.5,
    committed: 2.45,
  },
  {
    id: "b200",
    label: "HGX B200",
    vramGiB: 180,
    bwTBs: 8.0,
    tflopsBF16: 2250,
    rtCores: false,
    nvlink: true,
    nodeSize: 8,
    onDemand: 7.15,
    committed: 3.95,
  },
  {
    id: "l40s",
    label: "L40S",
    vramGiB: 48,
    bwTBs: 0.86,
    tflopsBF16: 181,
    rtCores: true,
    nvlink: false,
    nodeSize: 1,
    onDemand: 1.82,
    committed: 0.9,
  },
  {
    id: "rtxpro6000",
    label: "RTX PRO 6000",
    vramGiB: 96,
    bwTBs: 1.8,
    tflopsBF16: 250,
    rtCores: true,
    nvlink: false,
    nodeSize: 1,
    onDemand: 1.8,
    committed: 0.95,
  },
];

const STORAGE = {
  sharedFsGiBMonth: 0.08,
  objectGiBMonth: 0.0147,
};

/* ============================================================
   data/assumptions.js  — every magic number lives here
   ============================================================ */

const DEFAULTS = {
  vramUsableFrac: 0.85,
  activationMult: 1.3,
  mfuSingleNode: 0.4,
  mfuMultiNode: 0.32,
  tokensPerFrame: 256, // 224px image / 14px patch = 16x16
};

const MODES = {
  full: { label: "Full fine-tune (AdamW)", bytesPerParam: 16 },
  lora: { label: "LoRA / adapter", bytesPerParam: 3 },
  inference: { label: "Inference serving", bytesPerParam: 2 },
};

const PRESETS = {
  vla: {
    label: "VLA policy training",
    kind: "training",
    paramsB: 7,
    mode: "full",
    cameras: 3,
    historyFrames: 2,
    datasetFramesM: 20,
    epochs: 3,
    datasetTB: 40,
  },
  worldmodel: {
    label: "World-model pretraining",
    kind: "training",
    paramsB: 30,
    mode: "full",
    cameras: 1,
    historyFrames: 16,
    datasetFramesM: 60,
    epochs: 2,
    datasetTB: 300,
  },
  finetune: {
    label: "Policy fine-tune (LoRA)",
    kind: "training",
    paramsB: 7,
    mode: "lora",
    cameras: 2,
    historyFrames: 2,
    datasetFramesM: 2,
    epochs: 5,
    datasetTB: 4,
  },
  sim: {
    label: "Large-scale sim rollout",
    kind: "sim",
    parallelEnvs: 4096,
    vramPerEnvGiB: 0.6,
    stepsPerSecPerEnv: 60,
    totalStepsB: 2,
    datasetTB: 20,
  },
};

/* ============================================================
   lib/  — pure model. No React below this line until §UI.
   ============================================================ */

const pow2Ceil = (n) => Math.pow(2, Math.ceil(Math.log2(Math.max(1, n))));
const fmt = (n, d = 0) =>
  n.toLocaleString("en-US", { maximumFractionDigits: d, minimumFractionDigits: d });
const usd = (n) =>
  n >= 1000 ? `$${fmt(n)}` : `$${fmt(n, 2)}`;

function seqLength(cfg, a) {
  return cfg.cameras * cfg.historyFrames * a.tokensPerFrame;
}

function memoryFootprint(cfg, a) {
  const bpp = MODES[cfg.mode].bytesPerParam;
  const stateGiB = (cfg.paramsB * 1e9 * bpp) / 1024 ** 3;
  const totalGiB = stateGiB * a.activationMult;
  return { bpp, stateGiB, totalGiB };
}

function sizeCluster(totalGiB, gpu, a) {
  const usable = gpu.vramGiB * a.vramUsableFrac;
  const minGpus = Math.ceil(totalGiB / usable);
  const gpus = gpu.nodeSize > 1 ? Math.max(pow2Ceil(minGpus), 1) : minGpus;
  const nodes = Math.ceil(gpus / gpu.nodeSize);
  return { usable, minGpus, gpus, nodes, needsIB: nodes > 1 };
}

function trainingTime(cfg, gpu, gpus, nodes, a) {
  const seq = seqLength(cfg, a);
  const tokens = cfg.datasetFramesM * 1e6 * seq * cfg.epochs;
  const flops = 6 * cfg.paramsB * 1e9 * tokens;
  const mfu = nodes > 1 ? a.mfuMultiNode : a.mfuSingleNode;
  const effective = gpu.tflopsBF16 * 1e12 * mfu * gpus;
  const hours = flops / effective / 3600;
  return { seq, tokens, flops, mfu, hours };
}

function simSizing(cfg, gpu, a) {
  const usable = gpu.vramGiB * a.vramUsableFrac;
  const envsPerGpu = Math.max(1, Math.floor(usable / cfg.vramPerEnvGiB));
  const gpus = Math.ceil(cfg.parallelEnvs / envsPerGpu);
  const nodes = Math.ceil(gpus / Math.max(gpu.nodeSize, 1));
  const stepsPerSec = cfg.parallelEnvs * cfg.stepsPerSecPerEnv;
  const hours = (cfg.totalStepsB * 1e9) / stepsPerSec / 3600;
  return { envsPerGpu, gpus, nodes, stepsPerSec, hours, needsIB: false };
}

function costOf(gpus, hours, rate, datasetTB, months = 1) {
  const gpuHours = gpus * hours;
  const compute = gpuHours * rate;
  const storage = datasetTB * 1024 * STORAGE.sharedFsGiBMonth * months;
  return { gpuHours, compute, storage, total: compute + storage };
}

/* ---- orchestrator: returns spec + a human-readable derivation ---- */

function evaluate(cfg, gpu, a) {
  const trace = [];
  const notes = [];

  if (cfg.kind === "sim") {
    if (!gpu.rtCores) {
      return {
        gpu,
        feasible: false,
        reason: "No RT cores — cannot run rasterized/ray-traced simulation",
        trace: [
          `${gpu.label} has no RT cores; Isaac Sim / Omniverse rendering requires them`,
        ],
      };
    }
    const s = simSizing(cfg, gpu, a);
    trace.push(
      `${fmt(gpu.vramGiB)} GiB × ${a.vramUsableFrac} usable = ${fmt(
        gpu.vramGiB * a.vramUsableFrac,
        1
      )} GiB per GPU`
    );
    trace.push(
      `÷ ${cfg.vramPerEnvGiB} GiB per env = ${s.envsPerGpu} parallel envs per GPU`
    );
    trace.push(
      `${fmt(cfg.parallelEnvs)} envs ÷ ${s.envsPerGpu} = ${s.gpus} GPUs (${
        s.nodes
      } node${s.nodes > 1 ? "s" : ""})`
    );
    trace.push(
      `${fmt(cfg.totalStepsB, 1)}B steps ÷ ${fmt(
        s.stepsPerSec
      )} steps/s = ${fmt(s.hours, 1)} h wall clock`
    );
    if (!gpu.nvlink)
      notes.push("PCIe only — fine here, rollouts are embarrassingly parallel");

    const od = costOf(s.gpus, s.hours, gpu.onDemand, cfg.datasetTB);
    const cm = costOf(s.gpus, s.hours, gpu.committed, cfg.datasetTB);
    return { gpu, feasible: true, ...s, trace, notes, od, cm };
  }

  const m = memoryFootprint(cfg, a);
  trace.push(
    `${cfg.paramsB}B params × ${m.bpp} bytes/param (${MODES[cfg.mode].label}) = ${fmt(
      m.stateGiB
    )} GiB`
  );
  trace.push(
    `× ${a.activationMult} activation overhead = ${fmt(m.totalGiB)} GiB resident`
  );

  const c = sizeCluster(m.totalGiB, gpu, a);
  trace.push(
    `${gpu.label}: ${gpu.vramGiB} GiB × ${a.vramUsableFrac} = ${fmt(
      c.usable,
      1
    )} GiB usable → ${c.minGpus} GPU minimum`
  );
  if (c.gpus !== c.minGpus)
    trace.push(`rounded to ${c.gpus} (power of two for tensor/data parallel)`);
  if (c.nodes > 1) {
    trace.push(
      `${c.gpus} GPUs > ${gpu.nodeSize}/node → ${c.nodes} nodes → InfiniBand fabric required`
    );
    notes.push("Multi-node: MFU drops, interconnect becomes a cost driver");
  } else {
    trace.push(`fits in one ${gpu.nodeSize}-GPU node → NVLink sufficient, no IB`);
  }
  if (!gpu.nvlink && c.gpus > 1)
    notes.push("PCIe only — no NVLink; sharded training will be bandwidth-bound");

  const t = trainingTime(cfg, gpu, c.gpus, c.nodes, a);
  trace.push(
    `seq len = ${cfg.cameras} cam × ${cfg.historyFrames} frames × ${a.tokensPerFrame} tok = ${fmt(
      t.seq
    )} tokens/sample`
  );
  trace.push(
    `${fmt(cfg.datasetFramesM)}M samples × ${fmt(t.seq)} × ${
      cfg.epochs
    } epochs = ${t.tokens.toExponential(2)} tokens`
  );
  trace.push(
    `6 × ${cfg.paramsB}B × tokens = ${t.flops.toExponential(
      2
    )} FLOPs ÷ (${gpu.tflopsBF16} TF × ${t.mfu} MFU × ${c.gpus}) = ${fmt(
      t.hours,
      1
    )} h`
  );

  const od = costOf(c.gpus, t.hours, gpu.onDemand, cfg.datasetTB);
  const cm = costOf(c.gpus, t.hours, gpu.committed, cfg.datasetTB);
  return { gpu, feasible: true, ...c, ...t, trace, notes, od, cm };
}

/* ============================================================
   §UI
   ============================================================ */

const Field = ({ label, children, hint }) => (
  <label className="block mb-3">
    <span className="block text-[10px] uppercase tracking-widest text-neutral-500 mb-1">
      {label}
    </span>
    {children}
    {hint && <span className="block text-[10px] text-neutral-600 mt-1">{hint}</span>}
  </label>
);

const inputCls =
  "w-full bg-neutral-900 border border-neutral-700 text-emerald-300 px-2 py-1.5 text-sm font-mono focus:outline-none focus:border-emerald-500";

export default function GpuSizer() {
  const [presetKey, setPresetKey] = useState("vla");
  const [cfg, setCfg] = useState(PRESETS.vla);
  const [a, setA] = useState(DEFAULTS);
  const [committed, setCommitted] = useState(false);
  const [utilization, setUtilization] = useState(0.9);
  const [showAssumptions, setShowAssumptions] = useState(false);

  const set = (k) => (e) => {
    const v = e.target.type === "number" ? parseFloat(e.target.value) || 0 : e.target.value;
    setCfg((c) => ({ ...c, [k]: v }));
  };

  const loadPreset = (k) => {
    setPresetKey(k);
    setCfg(PRESETS[k]);
  };

  const results = useMemo(
    () => GPUS.map((g) => evaluate(cfg, g, a)),
    [cfg, a]
  );

  const feasible = results.filter((r) => r.feasible);
  const best = useMemo(() => {
    if (!feasible.length) return null;
    return [...feasible].sort(
      (x, y) =>
        (committed ? x.cm.total : x.od.total) -
        (committed ? y.cm.total : y.od.total)
    )[0];
  }, [feasible, committed]);

  const rate = (r) => (committed ? r.gpu.committed : r.gpu.onDemand);
  const cost = (r) => (committed ? r.cm : r.od);

  const chartData = feasible.map((r) => ({
    name: r.gpu.label.replace("HGX ", ""),
    cost: Math.round(cost(r).total),
    hours: Math.round(r.hours),
    best: best && r.gpu.id === best.gpu.id,
  }));

  return (
    <div className="min-h-screen bg-neutral-950 text-neutral-300 font-mono p-4 sm:p-6">
      <div className="max-w-6xl mx-auto">
        {/* header */}
        <div className="border-b border-neutral-800 pb-3 mb-5">
          <div className="text-emerald-500 text-sm">
            martin@webspace:~$ <span className="text-neutral-300">gpu-sizer</span>
          </div>
          <h1 className="text-xl sm:text-2xl text-neutral-100 mt-2 tracking-tight">
            Physical AI cluster sizing
          </h1>
          <p className="text-xs text-neutral-500 mt-1 max-w-2xl">
            Describe a robotics workload. Get a cluster spec, the derivation behind
            it, and what it costs on Nebius.
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-[300px_1fr] gap-6">
          {/* ---------- INPUTS ---------- */}
          <div>
            <div className="text-[10px] uppercase tracking-widest text-neutral-600 mb-2">
              Workload
            </div>
            <div className="grid grid-cols-2 gap-1 mb-4">
              {Object.entries(PRESETS).map(([k, p]) => (
                <button
                  key={k}
                  onClick={() => loadPreset(k)}
                  className={`text-[11px] px-2 py-1.5 border text-left leading-tight ${
                    presetKey === k
                      ? "border-emerald-500 text-emerald-400 bg-emerald-950"
                      : "border-neutral-800 text-neutral-500 hover:border-neutral-600"
                  }`}
                >
                  {p.label}
                </button>
              ))}
            </div>

            {cfg.kind === "training" ? (
              <>
                <Field label="Model size (B params)">
                  <input type="number" className={inputCls} value={cfg.paramsB} onChange={set("paramsB")} />
                </Field>
                <Field label="Training mode">
                  <select className={inputCls} value={cfg.mode} onChange={set("mode")}>
                    {Object.entries(MODES).map(([k, m]) => (
                      <option key={k} value={k}>{m.label}</option>
                    ))}
                  </select>
                </Field>
                <div className="grid grid-cols-2 gap-2">
                  <Field label="Cameras">
                    <input type="number" className={inputCls} value={cfg.cameras} onChange={set("cameras")} />
                  </Field>
                  <Field label="History frames">
                    <input type="number" className={inputCls} value={cfg.historyFrames} onChange={set("historyFrames")} />
                  </Field>
                </div>
                <div className="grid grid-cols-2 gap-2">
                  <Field label="Samples (M)">
                    <input type="number" className={inputCls} value={cfg.datasetFramesM} onChange={set("datasetFramesM")} />
                  </Field>
                  <Field label="Epochs">
                    <input type="number" className={inputCls} value={cfg.epochs} onChange={set("epochs")} />
                  </Field>
                </div>
              </>
            ) : (
              <>
                <Field label="Parallel environments">
                  <input type="number" className={inputCls} value={cfg.parallelEnvs} onChange={set("parallelEnvs")} />
                </Field>
                <div className="grid grid-cols-2 gap-2">
                  <Field label="VRAM / env (GiB)">
                    <input type="number" step="0.1" className={inputCls} value={cfg.vramPerEnvGiB} onChange={set("vramPerEnvGiB")} />
                  </Field>
                  <Field label="Steps/s per env">
                    <input type="number" className={inputCls} value={cfg.stepsPerSecPerEnv} onChange={set("stepsPerSecPerEnv")} />
                  </Field>
                </div>
                <Field label="Total rollout steps (B)">
                  <input type="number" step="0.1" className={inputCls} value={cfg.totalStepsB} onChange={set("totalStepsB")} />
                </Field>
              </>
            )}

            <Field label="Dataset on shared FS (TB)">
              <input type="number" className={inputCls} value={cfg.datasetTB} onChange={set("datasetTB")} />
            </Field>

            <div className="border-t border-neutral-800 pt-3 mt-4">
              <button
                onClick={() => setCommitted(!committed)}
                className={`w-full text-[11px] px-2 py-2 border mb-3 ${
                  committed
                    ? "border-emerald-500 text-emerald-400 bg-emerald-950"
                    : "border-neutral-700 text-neutral-400"
                }`}
              >
                {committed ? "◉ Committed rate" : "○ On-demand rate"}
              </button>

              <Field label={`Cluster utilization — ${Math.round(utilization * 100)}%`}>
                <input
                  type="range" min="0.3" max="1" step="0.05"
                  value={utilization}
                  onChange={(e) => setUtilization(parseFloat(e.target.value))}
                  className="w-full accent-emerald-500"
                />
              </Field>
            </div>

            <button
              onClick={() => setShowAssumptions(!showAssumptions)}
              className="text-[10px] uppercase tracking-widest text-neutral-600 hover:text-neutral-400 mt-2"
            >
              {showAssumptions ? "− " : "+ "}Assumptions
            </button>
            {showAssumptions && (
              <div className="mt-2 space-y-2 border-l border-neutral-800 pl-3">
                {[
                  ["vramUsableFrac", "Usable VRAM fraction"],
                  ["activationMult", "Activation multiplier"],
                  ["mfuSingleNode", "MFU, single node"],
                  ["mfuMultiNode", "MFU, multi-node"],
                ].map(([k, label]) => (
                  <Field key={k} label={label}>
                    <input
                      type="number" step="0.01" className={inputCls} value={a[k]}
                      onChange={(e) => setA({ ...a, [k]: parseFloat(e.target.value) || 0 })}
                    />
                  </Field>
                ))}
              </div>
            )}
          </div>

          {/* ---------- RESULTS ---------- */}
          <div>
            {best && (
              <div className="border border-emerald-800 bg-emerald-950 bg-opacity-20 p-4 mb-4">
                <div className="text-[10px] uppercase tracking-widest text-emerald-600 mb-2">
                  Recommended spec
                </div>
                <div className="text-2xl text-emerald-300 mb-1">
                  {best.gpus}× {best.gpu.label}
                </div>
                <div className="text-xs text-neutral-400">
                  {best.nodes} node{best.nodes > 1 ? "s" : ""} ·{" "}
                  {best.needsIB ? "InfiniBand fabric required" : best.gpu.nvlink ? "NVLink, single node" : "PCIe"} ·{" "}
                  {fmt(best.hours, 1)} h to completion
                </div>

                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-4 pt-3 border-t border-emerald-900">
                  {[
                    ["GPU-hours", fmt(cost(best).gpuHours)],
                    ["Compute", usd(cost(best).compute)],
                    ["Storage/mo", usd(cost(best).storage)],
                    ["Total", usd(cost(best).total)],
                  ].map(([l, v]) => (
                    <div key={l}>
                      <div className="text-[10px] text-neutral-500 uppercase">{l}</div>
                      <div className="text-sm text-neutral-100">{v}</div>
                    </div>
                  ))}
                </div>

                <div className="mt-3 pt-3 border-t border-emerald-900 text-xs">
                  <span className="text-neutral-500">Effective cost per useful GPU-hour at {Math.round(utilization * 100)}% utilization: </span>
                  <span className="text-amber-400">
                    {usd(rate(best) / utilization)}
                  </span>
                  <span className="text-neutral-600"> (list {usd(rate(best))})</span>
                </div>
                <div className="text-xs mt-1">
                  <span className="text-neutral-500">Committed vs on-demand on this job: </span>
                  <span className="text-emerald-400">
                    {usd(best.od.total - best.cm.total)} saved (
                    {Math.round((1 - best.cm.total / best.od.total) * 100)}%)
                  </span>
                </div>
              </div>
            )}

            {/* derivation — the signature element */}
            {best && (
              <div className="border border-neutral-800 p-4 mb-4">
                <div className="text-[10px] uppercase tracking-widest text-neutral-600 mb-3">
                  Derivation
                </div>
                <div className="space-y-1.5">
                  {best.trace.map((line, i) => (
                    <div key={i} className="text-xs flex gap-2">
                      <span className="text-neutral-700 select-none">
                        {String(i + 1).padStart(2, "0")}
                      </span>
                      <span className="text-emerald-600 select-none">→</span>
                      <span className="text-neutral-400">{line}</span>
                    </div>
                  ))}
                </div>
                {best.notes?.length > 0 && (
                  <div className="mt-3 pt-3 border-t border-neutral-800 space-y-1">
                    {best.notes.map((n, i) => (
                      <div key={i} className="text-xs text-amber-500">! {n}</div>
                    ))}
                  </div>
                )}
              </div>
            )}

            {/* SKU comparison */}
            <div className="border border-neutral-800 p-4 mb-4">
              <div className="text-[10px] uppercase tracking-widest text-neutral-600 mb-3">
                Cost to completion by SKU
              </div>
              <div style={{ height: 200 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={chartData} margin={{ top: 5, right: 5, bottom: 5, left: 5 }}>
                    <CartesianGrid strokeDasharray="2 4" stroke="#262626" vertical={false} />
                    <XAxis dataKey="name" stroke="#525252" tick={{ fontSize: 10, fontFamily: "monospace" }} />
                    <YAxis stroke="#525252" tick={{ fontSize: 10, fontFamily: "monospace" }} tickFormatter={(v) => `$${v >= 1000 ? Math.round(v / 1000) + "k" : v}`} />
                    <Tooltip
                      contentStyle={{ background: "#0a0a0a", border: "1px solid #404040", fontFamily: "monospace", fontSize: 11 }}
                      formatter={(v, n) => (n === "cost" ? [usd(v), "total"] : v)}
                    />
                    <Bar dataKey="cost">
                      {chartData.map((d, i) => (
                        <Cell key={i} fill={d.best ? "#10b981" : "#404040"} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* full table */}
            <div className="border border-neutral-800 p-4">
              <div className="text-[10px] uppercase tracking-widest text-neutral-600 mb-3">
                All SKUs
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="text-neutral-600 text-left">
                      {["SKU", "VRAM", "GPUs", "Nodes", "Hours", "Total"].map((h) => (
                        <th key={h} className="font-normal pb-2 pr-3">{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {results.map((r) => (
                      <tr key={r.gpu.id} className="border-t border-neutral-900">
                        <td className="py-2 pr-3 text-neutral-300">{r.gpu.label}</td>
                        <td className="py-2 pr-3 text-neutral-500">{r.gpu.vramGiB} GiB</td>
                        {r.feasible ? (
                          <>
                            <td className="py-2 pr-3 text-neutral-300">{r.gpus}</td>
                            <td className="py-2 pr-3 text-neutral-500">{r.nodes}</td>
                            <td className="py-2 pr-3 text-neutral-500">{fmt(r.hours, 1)}</td>
                            <td className={`py-2 pr-3 ${best && r.gpu.id === best.gpu.id ? "text-emerald-400" : "text-neutral-300"}`}>
                              {usd(cost(r).total)}
                            </td>
                          </>
                        ) : (
                          <td colSpan={4} className="py-2 text-neutral-600 italic">
                            {r.reason}
                          </td>
                        )}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            <p className="text-[10px] text-neutral-600 mt-4 leading-relaxed">
              Sizing heuristic, not a benchmark. MFU is a constant, not measured. Activation
              overhead is a multiplier, not a formula. No MoE, pipeline-bubble, or
              gradient-accumulation modeling. Rates are a snapshot of published Nebius
              list pricing and exclude egress. Every assumption above is editable —
              if a number looks wrong, change it and see what moves.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
