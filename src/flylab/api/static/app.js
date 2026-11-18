"use strict";
const el = (id) => document.getElementById(id);
const colors = ["#002fa7", "#b63c20", "#008275", "#60656f"];
let configs = [], run = null, socket = null, points = [], spikes = [], sample = [], latest = [];
const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;
let paused = reducedMotion, dropped = 0, omitted = 0;
const terminal = (s) => ["succeeded", "cancelled", "failed"].includes(s);
const pretty = (s) => ({starting:"Starting", running:"Running", cancelling:"Stopping", succeeded:"Finished", cancelled:"Stopped", failed:"Failed"})[s] || s;
const number = (n, digits = 1) => Number(n).toLocaleString("en", {maximumFractionDigits:digits});
async function api(path, payload) {
  const options = payload === undefined ? {} : {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload)};
  const response = await fetch("/api/v1/" + path, options);
  const body = await response.json();
  if (!response.ok) throw new Error(body.error || "The service could not complete this request.");
  return body;
}
function error(message) { el("error").textContent = message || ""; el("error").hidden = !message; }
function controls() {
  const active = run && !terminal(run.state);
  for (const id of ["config", "backend", "seed", "steps"]) el(id).disabled = !!active;
  el("start").disabled = !!active || !configs.length;
  el("stop").disabled = !active || run.state === "cancelling";
  el("state").textContent = run ? pretty(run.state) : "Idle";
}
function configChanged() {
  const cfg = configs.find((c) => c.name === el("config").value);
  if (!cfg) return;
  el("backend").value = cfg.backend; el("seed").value = cfg.seed; el("steps").value = cfg.steps;
  el("config-description").textContent = cfg.description || cfg.experiment_id;
  durationChanged();
}
function durationChanged() {
  const cfg = configs.find((c) => c.name === el("config").value);
  if (cfg) el("duration").textContent = number(el("steps").valueAsNumber * cfg.dt_ms) + " ms simulated · dt " + cfg.dt_ms + " ms";
}
function setSummary(summary) {
  if (!summary || !summary.dataset) return;
  sample = summary.raster_sample || [];
  el("dataset").textContent = summary.dataset.dataset_id + " · " + number(summary.dataset.n_neurons, 0) + " neurons";
  el("model").textContent = summary.model.parameter_set + " · " + number(summary.dataset.n_edges, 0) + " directed edges";
  el("labels").replaceChildren();
  for (const [kind,label] of Object.entries(summary.evidence)) {
    const badge = document.createElement("span"); badge.className = "badge"; badge.textContent = kind.replaceAll("_"," ") + " · " + label; el("labels").append(badge);
  }
  el("manifest").textContent = JSON.stringify(summary,null,2);
  if (summary.preview_pace_ms_per_bin) {
    el("pace").hidden = false; el("pace").textContent = "Visual preview: " + summary.preview_pace_ms_per_bin + " ms wall-time delay per bin. Model time and inputs are unchanged.";
  }
}
function interventionList(items) {
  el("interventions").replaceChildren();
  if (!items.length) {
    const li = document.createElement("li"); li.textContent = run && terminal(run.state) ? "No active run." : "No intervention active in this bin."; el("interventions").append(li);
  }
  for (const item of items) {
    const li = document.createElement("li"), small = document.createElement("small");
    li.textContent = (item.population || item.pre_population || "Selected edges") + " · " + item.kind.replaceAll("_"," ");
    small.textContent = item.mode || (item.rate_hz ? item.rate_hz + " Hz requested" : item.amplitude_mV ? item.amplitude_mV + " mV drive" : "Configured lesion");
    li.append(small); el("interventions").append(li);
  }
}
function updateTable() {
  el("latest").replaceChildren(); el("legend").replaceChildren();
  latest.forEach((pop, i) => {
    const row = document.createElement("tr");
    for (const value of [pop.name, number(pop.n_neurons,0), number(pop.rate_hz,2)]) {
      const cell = document.createElement("td"); cell.textContent = value; row.append(cell);
    }
    el("latest").append(row);
    const entry = document.createElement("span"); entry.textContent = pop.name; entry.className = "c" + (i % colors.length); el("legend").append(entry);
  });
}
function axes(canvas, yLabel, maxY) {
  const box = canvas.getBoundingClientRect(), scale = devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.round(box.width * scale)); canvas.height = Math.max(1, Math.round(box.height * scale));
  const ctx = canvas.getContext("2d"); ctx.scale(scale,scale);
  const area = {x:48,y:16,w:Math.max(1,box.width-62),h:Math.max(1,box.height-45)};
  ctx.font = "10px Helvetica, Arial, sans-serif"; ctx.fillStyle = "#60656f"; ctx.fillText(yLabel,3,10);
  const end = points.length ? points[points.length-1].simulation_time_ms : 100;
  const start = points.length ? Math.max(0, points[0].simulation_time_ms - points[0].bin_ms) : 0;
  const x = (t) => area.x + (t-start) / Math.max(1,end-start) * area.w;
  const y = (v) => area.y + area.h - v / Math.max(1,maxY) * area.h;
  for (let i=0;i<=4;i++) {
    const py = area.y + i*area.h/4;
    ctx.strokeStyle = "#e8e9ed"; ctx.beginPath(); ctx.moveTo(area.x,py); ctx.lineTo(area.x+area.w,py); ctx.stroke();
    ctx.fillText(number(maxY*(1-i/4),1),3,py+3);
    const t = start+(end-start)*i/4; const px = x(t);
    ctx.fillText(number(t,0),Math.min(box.width-35,px-7),box.height-10);
  }
  ctx.fillText("ms",box.width-18,box.height-10);
  return {ctx,area,x,y,start,end};
}
function draw(force = false) {
  if ((paused && !force) || document.hidden) return;
  const max = Math.max(1,...points.flatMap((p) => p.populations.map((v) => v.rate_hz))) * 1.15;
  const chart = axes(el("rates"),"Hz / neuron",max);
  latest.forEach((pop,i) => {
    chart.ctx.strokeStyle = colors[i%colors.length]; chart.ctx.lineWidth = 1.8;
    chart.ctx.setLineDash(pop.name === "__all__" ? [4,3] : []);
    chart.ctx.beginPath();
    points.forEach((p,j) => {
      const v = p.populations.find((v) => v.name === pop.name);
      if (v) { const x=chart.x(p.simulation_time_ms),y=chart.y(v.rate_hz); if(j)chart.ctx.lineTo(x,y);else chart.ctx.moveTo(x,y); }
    });
    chart.ctx.stroke(); chart.ctx.setLineDash([]);
  });
  const raster = axes(el("raster"),"sample row",Math.max(1,sample.length));
  const rows = new Map(sample.map((r,i)=>[r.idx,i]));
  spikes.forEach((s) => {
    if (!rows.has(s.idx) || s.t_ms < raster.start || s.t_ms > raster.end) return;
    const row = rows.get(s.idx), i = latest.findIndex((v)=>v.name===s.population);
    raster.ctx.fillStyle = colors[Math.max(0,i)%colors.length];
    raster.ctx.fillRect(raster.x(s.t_ms),raster.y(row+.5)-2,1.4,4);
  });
}
async function refreshRun() {
  if (!run) return;
  const rid = run.id;
  try {
    const detail = await api("experiments/"+rid);
    if (!run || run.id !== rid) return;
    run = detail; controls(); setSummary(run.manifest_summary);
    el("wall-time").textContent = number(run.wall_seconds,1) + " s";
    if (terminal(run.state)) { interventionList([]); if(run.error)error(run.error); }
  } catch (e) { error(e.message); }
}
function follow(info) {
  if (socket) socket.close();
  run = info; points=[];spikes=[];sample=[];latest=[];dropped=0;omitted=0;
  const sessionId = info.id;
  el("sim-time").textContent="0";el("progress").value=0;el("rate-empty").hidden=false;el("raster-empty").hidden=false;
  el("experiment-title").textContent=info.config;el("run-id").textContent=info.id.slice(0,12);
  controls();setSummary(info.manifest_summary);
  history.replaceState(null,"","?run="+encodeURIComponent(info.id));
  socket = new WebSocket((location.protocol==="https:"?"wss://":"ws://")+location.host+"/api/v1/experiments/"+info.id+"/stream");
  socket.onopen=()=>{if(run&&run.id===sessionId)el("stream-state").textContent="Connected";};
  socket.onmessage=async(event)=>{
    if(!run||run.id!==sessionId)return;
    const data=JSON.parse(event.data);
    dropped += data.dropped_before || 0;
    if(data.type==="manifest")setSummary(data.summary);
    if(data.type==="population_rates"){
      points.push(data);if(points.length>200)points.shift();
      spikes.push(...data.spikes);if(spikes.length>4000)spikes.splice(0,spikes.length-4000);
      latest=data.populations;omitted+=data.spikes_omitted||0;
      if(run.state!=="cancelling")run.state="running";
      el("wall-time").textContent=number(data.wall_seconds,1)+" s";
      el("sim-time").textContent=number(data.simulation_time_ms);
      el("progress").value=Math.min(100,100*data.simulation_time_ms/run.duration_ms);
      el("bin-info").textContent=number(data.bin_ms)+" ms bins · Hz per neuron";
      el("rate-empty").hidden=true;el("raster-empty").hidden=spikes.length>0;
      el("sample-info").textContent=sample.length+" sampled neurons · "+omitted+" sampled events omitted · "+dropped+" old messages dropped";
      updateTable();interventionList(data.active_interventions);controls();draw();
    }
    if(data.type==="status"){run.state=data.state;controls();await refreshRun();draw(reducedMotion);}
  };
  socket.onerror=()=>{if(run&&run.id===sessionId)el("stream-state").textContent="Connection error";};
  socket.onclose=async()=>{
    if(!run||run.id!==sessionId)return;
    await refreshRun();el("stream-state").textContent=run&&terminal(run.state)?"Finished · sampled preview retained":"Disconnected · refresh to reconnect";
  };
}
el("run-form").addEventListener("submit",async(event)=>{
  event.preventDefault();error("");el("start").disabled=true;
  try { const info=await api("experiments",{config:el("config").value,overrides:{seed:el("seed").valueAsNumber,backend:el("backend").value,steps:el("steps").valueAsNumber}});follow(info); }
  catch(e){error(e.message);controls();}
});
el("stop").addEventListener("click",async()=>{if(!run)return;try{run=await api("experiments/"+run.id+"/cancel",{});controls();}catch(e){error(e.message);}});
el("config").addEventListener("change",configChanged);el("steps").addEventListener("input",durationChanged);
function pauseText(){el("pause").textContent=paused?"Resume charts":"Pause charts";el("pause").setAttribute("aria-pressed",String(paused));}
el("pause").addEventListener("click",()=>{paused=!paused;pauseText();draw();});
window.addEventListener("resize",()=>draw());
document.addEventListener("visibilitychange",()=>draw(reducedMotion&&run&&terminal(run.state)));
async function init(){
  try{
    const [health,available]=await Promise.all([api("health"),api("configs")]);configs=available.configs;
    el("service").textContent="Local service connected";el("config").replaceChildren();
    for(const c of configs){const o=document.createElement("option");o.value=c.name;o.textContent=c.name;el("config").append(o);}
    if(configs.some((c)=>c.name==="synthetic.yaml"))el("config").value="synthetic.yaml";
    el("steps").max=health.limits.max_steps;configChanged();controls();pauseText();draw(true);
    const rid=new URLSearchParams(location.search).get("run")||health.active_run;
    if(rid)follow(await api("experiments/"+encodeURIComponent(rid)));
  }catch(e){error(e.message);el("service").textContent="Local service unavailable";}
}
setInterval(()=>{if(run&&!terminal(run.state)&&!document.hidden)refreshRun();},1500);
init();
