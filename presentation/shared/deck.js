/* Shared deck runtime: scaling, navigation, ball animation, video slots, results table. */
(function(){
const stage=document.getElementById('stage');
const slides=[...document.querySelectorAll('.slide')];
const pn=document.getElementById('pn');
const qs=new URLSearchParams(location.search);
let cur=0;

function fit(){const s=Math.min(innerWidth/1920,innerHeight/1080);stage.style.transform=`translate(-50%,-50%) scale(${s})`}
addEventListener('resize',fit);fit();

/* ---- results table: window.RESULTS = {in_dist:{base:null,rl:null},heldout:{base:null,rl:null}} ---- */
document.querySelectorAll('[data-r]').forEach(td=>{
  const [a,b]=td.dataset.r.split('.');
  const v=(window.RESULTS&&window.RESULTS[a])?window.RESULTS[a][b]:null;
  if(v===null||v===undefined){td.className='tbd';td.innerHTML='<span>TBD</span>'}
  else{td.className='val';td.textContent=(typeof v==='number'?v.toFixed(0)+'%':v)}
});

/* ---- KPI: <div data-kpi="heldout"> shows "base% -> after%" ---- */
document.querySelectorAll('[data-kpi]').forEach(el=>{
  const r=(window.RESULTS||{})[el.dataset.kpi];
  if(r&&typeof r.base==='number'&&typeof r.rl==='number'){
    el.innerHTML=`${r.base.toFixed(0)}%<span class="ar">→</span><span class="after">${r.rl.toFixed(0)}%</span>`;
  }else{el.innerHTML='<span class="tbdk">TBD</span>'}
});

/* ---- improvement curve: window.CURVE = {steps:[..], in_dist:[..], heldout:[..], xlabel:"Training step"} ---- */
document.querySelectorAll('.curve').forEach(box=>{
  const C=window.CURVE;
  if(!C||!C.steps){box.innerHTML='<div class="curve-ph">CHART<br><small>set window.CURVE in index.html</small></div>';return}
  const W=960,H=560,L=96,R=210,T=30,B=100,pw=W-L-R,ph=H-T-B;
  const xs=C.steps,xmin=xs[0],xmax=xs[xs.length-1];
  const X=v=>L+(v-xmin)/(xmax-xmin)*pw, Y=v=>T+(1-v/100)*ph;
  let g='';
  [0,25,50,75,100].forEach(v=>{g+=`<line x1="${L}" x2="${L+pw}" y1="${Y(v)}" y2="${Y(v)}" stroke="#2a303c" stroke-width="2"/><text x="${L-16}" y="${Y(v)+8}" text-anchor="end" font-size="24" fill="#9aa3b2">${v}</text>`});
  xs.forEach(v=>{g+=`<text x="${X(v)}" y="${T+ph+36}" text-anchor="middle" font-size="24" fill="#9aa3b2">${v}</text>`});
  g+=`<text x="${L+pw/2}" y="${H-24}" text-anchor="middle" font-size="26" fill="#9aa3b2">${C.xlabel||'Training step'}</text>`;
  g+=`<text transform="translate(28 ${T+ph/2}) rotate(-90)" text-anchor="middle" font-size="26" fill="#9aa3b2">Success rate (%)</text>`;
  const series=[['heldout','Held-out','#ffa13d'],['in_dist','In-distribution','#5b9bff']];
  series.forEach(([k,name,col])=>{
    const ys=C[k]; if(!ys)return;
    const pts=xs.map((x,i)=>`${X(x)},${Y(ys[i])}`).join(' ');
    g+=`<polyline points="${pts}" fill="none" stroke="${col}" stroke-width="6" stroke-linejoin="round" stroke-linecap="round"/>`;
    xs.forEach((x,i)=>{g+=`<circle cx="${X(x)}" cy="${Y(ys[i])}" r="8" fill="${col}"/>`});
    const last=ys[ys.length-1];
    g+=`<text x="${X(xmax)+18}" y="${Y(last)+9}" font-size="30" font-weight="700" fill="${col}">${last.toFixed(0)}%</text>`;
    g+=`<text x="${X(xmax)+18}" y="${Y(last)+40}" font-size="22" fill="${col}">${name}</text>`;
  });
  box.innerHTML=`<svg viewBox="0 0 ${W} ${H}" style="width:100%;height:auto;display:block">${g}</svg>`;
});

/* ---- badge on slides that show placeholder numbers: set window.FAKE=true ---- */
if(window.FAKE){
  document.querySelectorAll('.slide').forEach(sl=>{
    if(sl.querySelector('[data-r],[data-kpi],.curve')){
      const b=document.createElement('div');b.className='fakebadge';b.textContent='FAKE DATA · PREVIEW ONLY';sl.appendChild(b);
    }
  });
}

/* ---- video slots: drop files into ../media/ and they play automatically ---- */
const vids=[];
document.querySelectorAll('.vslot[data-src]').forEach(slot=>{
  const src=slot.dataset.src;
  if(/^data:image\//.test(src)||/\.(png|jpe?g|gif|webp|svg)$/i.test(src)){
    const im=document.createElement('img');
    im.addEventListener('load',()=>slot.classList.add('has'));
    im.addEventListener('error',()=>slot.classList.remove('has'));
    im.src=src;slot.appendChild(im);return;
  }
  const v=document.createElement('video');
  v.muted=true;v.loop=true;v.playsInline=true;v.preload='auto';
  v.addEventListener('loadeddata',()=>{slot.classList.add('has');if(qs.has('clean'))try{v.currentTime=2.5}catch(e){}});
  v.addEventListener('error',()=>slot.classList.remove('has'));
  v.src=(qs.has('clean')&&!src.startsWith('data:'))?src+'#t=2.5':src;
  slot.appendChild(v);vids.push({v,slide:slot.closest('.slide')});
});
function syncVideos(){vids.forEach(({v,slide})=>{if(slide===slides[cur]){v.play().catch(()=>{})}else{v.pause()}})}

/* ---- Physics animation: closed-form answer for a sample scene ---- */
const P={x0:100,v:400,wall:800,r:20,e:0.6,dt:0.2,n:16};
P.xc=P.wall-P.r;P.tc=(P.xc-P.x0)/P.v;P.T=(P.n-1)*P.dt;
const xAt=t=>t<=P.tc?P.x0+P.v*t:P.xc-P.e*P.v*(t-P.tc);
const LOOP=P.T+1.0;   // keep in sync with LOOP in build_video.py (4.0 s)
const anim=document.getElementById('anim');
let ball,tl,raf=null,t0=0;
if(anim){
  const g=document.getElementById('ghosts');
  for(let k=0;k<P.n;k++){
    const c=document.createElementNS('http://www.w3.org/2000/svg','circle');
    c.setAttribute('cx',xAt(k*P.dt));c.setAttribute('cy',130);c.setAttribute('r',6);
    c.setAttribute('fill','#ffa13d');g.appendChild(c);
  }
  ball=document.getElementById('ball');tl=document.getElementById('tlabel');
}
function setT(t){const tt=Math.min(t,P.T);ball.setAttribute('cx',xAt(tt));tl.textContent='t = '+tt.toFixed(2)+' s'}
function startAnim(){
  cancelAnimationFrame(raf);
  if(!anim||!slides[cur].contains(anim))return;
  if(qs.has('t')){setT(parseFloat(qs.get('t')));return}
  t0=performance.now();
  (function step(now){setT(((now-t0)/1000)%LOOP);raf=requestAnimationFrame(step)})(t0);
}

function show(i){
  cur=Math.max(0,Math.min(slides.length-1,i));
  slides.forEach((s,k)=>s.classList.toggle('active',k===cur));
  if(pn)pn.textContent=(cur+1)+' / '+slides.length;
  history.replaceState(null,'','#'+(cur+1)+(location.search||''));
  startAnim();syncVideos();
}
addEventListener('keydown',e=>{
  if(['ArrowRight','PageDown',' ','Enter'].includes(e.key)){e.preventDefault();show(cur+1)}
  else if(['ArrowLeft','PageUp','Backspace'].includes(e.key)){e.preventDefault();show(cur-1)}
  else if(e.key==='Home')show(0);else if(e.key==='End')show(slides.length-1);
  else if(e.key==='f'||e.key==='F'){document.fullscreenElement?document.exitFullscreen():document.documentElement.requestFullscreen()}
});
addEventListener('click',e=>{if(e.target.tagName==='VIDEO')return;show(e.clientX>innerWidth/2?cur+1:cur-1)});
let tx=null;addEventListener('touchstart',e=>tx=e.touches[0].clientX);
addEventListener('touchend',e=>{if(tx===null)return;const d=e.changedTouches[0].clientX-tx;if(Math.abs(d)>40)show(cur+(d<0?1:-1));tx=null});

const h=parseInt(location.hash.slice(1));show(Number.isFinite(h)?h-1:0);
if(qs.has('clean')){const el=document.querySelector('.hint');if(el)el.style.display='none'}
})();
