import { projectCameraPoint, nearestLabelCandidates } from './nearest-labels.js';
// Based on Mnemos_Concept_Forge.html; node cores add a bounded soft halo.
// Original drawing helpers and radial gradient are retained.
// The wrapper supplies live graph data and lets React own animation scheduling.
export function createReferenceRenderer(cx) {
let W=1,H=1,state,traces=[],episodes=[],concepts=[],hypergraphs=[],relations=[],byId={};
const TAU=Math.PI*2;
const clamp=(x,a,b)=>Math.min(b,Math.max(a,x));
const GROUP_COLOR={};
const requestAnimationFrame=()=>0;
function conceptNode(id){return byId[id]||state.metas.find(m=>m.id===id)}
function currentMeta(){return state.metas.find(m=>m.id===state.selected?.id)||state.metas.at(-1)}
function nearestHyper(){return hypergraphs.find(h=>h.id===state.selectedHyperId)}
function rand(seed){let t=seed>>>0;return()=>{t+=0x6D2B79F5;let z=t;z=Math.imul(z^(z>>>15),z|1);z^=z+Math.imul(z^(z>>>7),z|61);return ((z^(z>>>14))>>>0)/4294967296}}
function dot(a,b){let v=0;for(let k=0;k<a.length;k++)v+=a[k]*b[k];return v}
const r=rand(517),stars=Array.from({length:280},()=>[r(),r(),.3+r()*.7]);
function rotate(p){const ca=Math.cos(state.az),sa=Math.sin(state.az),ce=Math.cos(state.el),se=Math.sin(state.el);const [x,y,z]=p;const xr=x*ca+z*sa,zr=-x*sa+z*ca;return[xr,y*ce-zr*se,y*se+zr*ce]}
function proj(point){return projectCameraPoint(rotate(point),W,H,state.zoom)}
function line(a,b,col,opacity=.3,width=1,dash=false){cx.save();cx.globalAlpha=opacity;cx.strokeStyle=col;cx.lineWidth=width;if(dash)cx.setLineDash([3,5]);cx.beginPath();cx.moveTo(a.x,a.y);cx.lineTo(b.x,b.y);cx.stroke();cx.restore()}
function dot2(x,y,r,col,opacity=1){cx.save();cx.globalAlpha=opacity;cx.beginPath();cx.arc(x,y,r,0,TAU);cx.fillStyle=col;cx.fill();cx.restore()}
// Bloom only the solid semantic-node core. Canvas save/restore keeps the
// shadow out of edges, labels, background stars, and subsequent draw calls.
function nodeCore(x,y,r,col,opacity=1){
 cx.save();
 cx.shadowColor=col;
 cx.shadowBlur=Math.min(7,Math.max(4,r*1.6));
 dot2(x,y,r,col,opacity);
 cx.restore();
}
function glow(x,y,r,col,a=.2){cx.save();cx.globalAlpha=a;const g=cx.createRadialGradient(x,y,0,x,y,r);g.addColorStop(0,col);g.addColorStop(.20,col+'77');g.addColorStop(1,col+'00');cx.fillStyle=g;cx.fillRect(x-r,y-r,2*r,2*r);cx.restore()}
function objectDot(n,p,r,color,opacity){
 if(!n.displayColor){if(n.kind==='trace')dot2(p.x,p.y,r,color,opacity);else nodeCore(p.x,p.y,r,color,opacity);return}
 cx.save();cx.globalAlpha=opacity;cx.fillStyle=color;cx.shadowColor=color;cx.shadowBlur=5;
 const sides={entity:4,fact:4,state:3,concept:6,rule:4}[n.objectType];
 cx.beginPath();
 if(!sides)cx.arc(p.x,p.y,r,0,TAU);
 else for(let i=0;i<sides;i++){const a=i*TAU/sides+(n.objectType==='entity'?Math.PI/4:-Math.PI/2),x=p.x+Math.cos(a)*r*1.35,y=p.y+Math.sin(a)*r*1.35;i?cx.lineTo(x,y):cx.moveTo(x,y)}
 cx.closePath();cx.fill();cx.shadowBlur=0;
 if(!['accepted','resolved','recorded','promoted'].includes(n.status)){cx.setLineDash([2,3]);cx.strokeStyle=color;cx.beginPath();cx.arc(p.x,p.y,r+4,0,TAU);cx.stroke()}
 if(n.sessionIndex){const x=p.x+r+3,y=p.y+r+3;cx.fillStyle=['#88baff','#ffca8d','#f4abd7','#dbe4ec'][Math.min(3,n.sessionIndex-1)];cx.beginPath();if(n.sessionIndex===1)cx.arc(x,y,1.8,0,TAU);else if(n.sessionIndex===2)cx.rect(x-1.8,y-1.8,3.6,3.6);else{cx.moveTo(x,y-2.5);cx.lineTo(x+2.5,y+2);cx.lineTo(x-2.5,y+2);cx.closePath()}cx.fill()}
 cx.restore();
}
function label(x,y,text,color='#c9d9e9',large=false){cx.save();cx.font=(large?'600 10px ':'500 9px ')+ 'ui-monospace,Menlo,Consolas,monospace';const width=cx.measureText(text).width+16;const bx=x+13+width>W?x-13-width:x+13,by=y-18<12?y+16:y-18;cx.fillStyle='#0c1b28ed';cx.strokeStyle=large?'#55887a':'#2b4554';cx.lineWidth=.8;cx.beginPath();cx.roundRect(bx,by-10,width,20,5);cx.fill();cx.stroke();cx.fillStyle=color;cx.fillText(text,bx+8,by+4);cx.restore()}
// Label only the closest visible memory nodes across every enabled layer.
// Use full Euclidean camera distance, not z, screen position, or node kind.
function typedRelations(pmap){
 const palette={HAS_FACT:'#ffd299',HAS_STATE:'#eca7d6',PARTICIPATES_IN:'#80bdff',DEPENDS_ON:'#bba6ff',DERIVED_FROM:'#91afc7',APPLIES_TO:'#80bdff',SUPERSEDES:'#f4c487'};
 for(const r of relations){
  if(!['accepted','recorded',undefined].includes(r.status))continue;
  const a=pmap.get(r.source),b=pmap.get(r.target);if(!a||!b)continue;
  const picked=state.selected?.id===r.source||state.selected?.id===r.target;
  const col=palette[r.type]||'#91afc7';
  line(a,b,col,picked?.6:r.type==='DERIVED_FROM'?.06:.17,picked?1.2:.7,['DERIVED_FROM','DEPENDS_ON'].includes(r.type));
  if(picked&&r.type!=='DERIVED_FROM'){cx.save();cx.globalAlpha=.75;cx.fillStyle=col;cx.font='8px ui-monospace,Consolas,monospace';cx.fillText(r.type,(a.x+b.x)/2+3,(a.y+b.y)/2-4);cx.restore()}
 }
}
function floatingLabels(items){
 const limit=Math.max(2,Math.min(6,Math.floor(W/130)));
 const candidates=nearestLabelCandidates(items,W,H,limit);
 const boxes=[];
 for(const {n,p} of candidates){
  if(boxes.length>=limit)break;
  const primary=n.id===state.selected?.id,hovered=n.id===state.hoverId;
  const text=n.title.slice(0,30).toUpperCase()+(n.title.length>30?'…':'');
  cx.save();cx.font=(primary?'600 10px ':'500 9px ')+'ui-monospace,Menlo,Consolas,monospace';
  const width=cx.measureText(text).width+16;cx.restore();
  const spots=[[p.x+13,p.y-28],[p.x+13,p.y+12],[p.x-width-13,p.y-28],[p.x-width-13,p.y+12],[p.x-width/2,p.y-52],[p.x-width/2,p.y+36]];
  let box;
  for(const [left,top] of spots){
   const trial={left,top,right:left+width,bottom:top+20};
   if(left<12||trial.right>W-12||top<60||trial.bottom>H-32)continue;
   if(boxes.some(b=>trial.left<b.right+7&&trial.right>b.left-7&&trial.top<b.bottom+6&&trial.bottom>b.top-6))continue;
   box=trial;break;
  }
  if(!box)continue;
  const color=primary?'#fff1cc':n.kind==='concept'?'#cfbfff':n.kind==='episode'?'#a6ead8':'#a9c6d9';
  const endpoint={x:clamp(p.x,box.left,box.right),y:clamp(p.y,box.top,box.bottom)};
  line(p,endpoint,color,(primary||hovered)? .65 : .28,.65);
  label(box.left-13,box.top+28,text,color,primary);
  boxes.push(box);
  state.rendered.push({id:n.id,kind:n.kind,x:p.x,y:p.y,r:0,z:-p.z+10000,bounds:box});
 }
}
function hull(points){let pts=[...points].sort((a,b)=>a.x-b.x||a.y-b.y);if(pts.length<3)return pts;const cross=(a,b,c)=>(b.x-a.x)*(c.y-a.y)-(b.y-a.y)*(c.x-a.x);let lo=[],up=[];for(const p of pts){while(lo.length>=2&&cross(lo[lo.length-2],lo.at(-1),p)<=0)lo.pop();lo.push(p)}for(let i=pts.length-1;i>=0;i--){const p=pts[i];while(up.length>=2&&cross(up[up.length-2],up.at(-1),p)<=0)up.pop();up.push(p)}lo.pop();up.pop();return lo.concat(up)}
function drawHull(h,points,selected){const pts=hull(points);if(pts.length<3)return;const cxm=pts.reduce((a,p)=>a+p.x,0)/pts.length,cym=pts.reduce((a,p)=>a+p.y,0)/pts.length;cx.save();cx.beginPath();pts.forEach((p,i)=>{const x=cxm+(p.x-cxm)*1.16,y=cym+(p.y-cym)*1.16;i?cx.lineTo(x,y):cx.moveTo(x,y)});cx.closePath();cx.fillStyle=selected?'#c69cff11':'#8b78bb05';cx.strokeStyle=selected?'#caadff99':'#c6a1ff22';cx.lineWidth=selected?1.5:.6;cx.setLineDash(selected?[6,5]:[3,10]);cx.fill();cx.stroke();cx.restore()}
function paint(t){if(state.auto&&!state.drag)state.az+=.00105;cx.clearRect(0,0,W,H);const background=cx.createRadialGradient(W*.5,H*.5,0,W*.5,H*.5,Math.max(W,H)*.65);background.addColorStop(0,'#17344477');background.addColorStop(.7,'#0b1a2940');background.addColorStop(1,'#060d1400');cx.fillStyle=background;cx.fillRect(0,0,W,H);
 for(const [x,y,a] of stars)dot2(x*W,y*H,a*.86,'#96b8cd',a*.17);const radius=175*state.zoom*Math.min(W/880,H/670);cx.save();cx.translate(W*.5,H*.51);for(let i=0;i<4;i++){cx.beginPath();cx.ellipse(0,0,radius*(1+i*.42)*1.36,radius*(1+i*.42)*.48,state.el*.63,0,TAU);cx.strokeStyle='#37657b'+[26,22,15,10][i].toString(16).padStart(2,'0');cx.lineWidth=.8;cx.stroke()}cx.restore();
 const visible=[];const s=state.stage;for(const n of traces)if(s<2||state.showRaw)visible.push(n);for(const n of episodes)if(s>=1)visible.push(n);for(const n of concepts)if(s>=2)visible.push(n);const mlist=s>=5?state.metas:[];const pmap=new Map(visible.map(n=>[n.id,proj(n.pos)]));const selected=state.selected;
 if(!state.grounded&&s>=1&&s<=2){for(const e of episodes){const ep=pmap.get(e.id);if(!ep)continue;if(s===1){for(const id of e.sources){const tr=pmap.get(id);if(tr)line(tr,ep,'#78c1be',.06,.75)}}else{const c=pmap.get(e.concept);if(c)line(ep,c,'#99b8d4',.20,1.0)}}}
 // Query-driven semantic neighbors; actual 64D cosine determines the links.
 if(s>=3&&state.showEdges){for(const c of concepts){const a=pmap.get(c.id);if(!a)continue;const near=concepts.filter(n=>n!==c).map(n=>[n,dot(c.v,n.v)]).sort((x,y)=>y[1]-x[1]).slice(0,3);for(const [n,v] of near){if(n.id<c.id)continue;const b=pmap.get(n.id);line(a,b,state.active.has(c.id)&&state.active.has(n.id)?'#80ddd2':'#8194b6',state.active.has(c.id)&&state.active.has(n.id)?.42:.14,.6+Math.max(0,v)*.8)}}}
 const h=nearestHyper();if(s>=4){for(const edge of hypergraphs){const picked=edge.id===state.selectedHyperId,pts=edge.ids.map(id=>pmap.get(id)||((state.stage>=5&&state.metas.find(m=>m.id===id))?proj(conceptNode(id).pos):null)).filter(Boolean);if(pts.length>=3)drawHull(edge,pts,picked)}for(const edge of hypergraphs){const epos=proj(edge.pos);if(edge.id===state.selectedHyperId){for(const id of edge.ids){const pt=pmap.get(id)||(s>=5&&conceptNode(id)?.kind==='meta'?proj(conceptNode(id).pos):null);if(pt)line(pt,epos,'#e1c1ff',.68,1.6)}}else{for(const id of edge.ids){const pt=pmap.get(id)||(s>=5&&conceptNode(id)?.kind==='meta'?proj(conceptNode(id).pos):null);if(pt)line(pt,epos,'#ae91f0',.055,.8)}}if(edge.id===state.selectedHyperId)glow(epos.x,epos.y,52,'#c59bff',.19);dot2(epos.x,epos.y,edge.id===state.selectedHyperId?7:2.6,edge.id===state.selectedHyperId?'#ffd29a':'#9b84d2',edge.id===state.selectedHyperId?1:.55);if(edge.id===state.selectedHyperId&&s>=4)label(epos.x,epos.y,'HYPEREDGE · '+edge.ids.length+' CONCEPTS','#e1c7ff',true)}}
 if(state.grounded)typedRelations(pmap);
 // Materialized meta concepts persist and can be searched using their own embeddings.
 for(const m of mlist){const hh=hypergraphs.find(h=>h.id===m.hyperId);const p1=proj(m.pos);if(hh){const p0=proj(hh.pos);line(p0,p1,m.valid?'#ffd59b':'#ff829c',.78,1.75)}glow(p1.x,p1.y,51,m.valid?'#ffdea5':'#ff829c',.25);objectDot(m,p1,9,m.displayColor||(m.valid?'#ffe2ad':'#ff829c'),1);cx.save();cx.strokeStyle=m.valid?'#ffdfaa':'#ff91aa';cx.lineWidth=1.3;cx.beginPath();cx.arc(p1.x,p1.y,16+Math.sin(t*.002)*1.5,0,TAU);cx.stroke();cx.restore();}
 state.rendered=[];const sorted=visible.map(n=>({n,p:pmap.get(n.id)})).sort((a,b)=>b.p.z-a.p.z);for(const {n,p} of sorted){const active=state.active?.has(n.id),sel=selected?.id===n.id;let color=n.displayColor||(n.kind==='trace'?(GROUP_COLOR[n.group]||'#8baebe'):n.kind==='episode'?'#76d5bd':'#bba6ff');let opacity=n.kind==='trace'?(s===0?.7:s===1?.26:state.showRaw?.14:0):n.kind==='episode'?(s===1?.94:s===2?.48:.16):s===2?.9:s>=3?(active?1:.62):1;if(state.changed?.id===n.id){color='#ff8a9c';opacity=1}if(opacity<=0)continue;
 const rad=(n.kind==='trace'?1.9:n.kind==='episode'?3.4:active?5.7:4.0)*p.p*clamp(state.zoom,.68,1.45);if(n.kind!=='trace'){glow(p.x,p.y,rad*(active?5:3.1),color,opacity*.16)}dot2(p.x,p.y,rad+1.5,color,opacity*.12);objectDot(n,p,rad,color,opacity);if(sel||active&&n.kind==='concept'){cx.save();cx.strokeStyle=sel?'#fff0cf':color;cx.globalAlpha=sel?1:.55;cx.lineWidth=sel?1.5:1;cx.beginPath();cx.arc(p.x,p.y,rad+6+(sel?2:0),0,TAU);cx.stroke();cx.restore()}
 state.rendered.push({id:n.id,kind:n.kind,x:p.x,y:p.y,r:Math.max(10,rad+3),z:-p.z})}
 if(s>=4){for(const e of hypergraphs){const p=proj(e.pos);state.rendered.push({kind:'hyper',id:e.id,x:p.x,y:p.y,r:e.id===state.selectedHyperId?16:10,z:-p.z+30})}}for(const m of mlist){const p=proj(m.pos);state.rendered.push({kind:'meta',id:m.id,x:p.x,y:p.y,r:22,z:-p.z+50})}floatingLabels([...sorted,...mlist.map(n=>({n,p:proj(n.pos)}))]);requestAnimationFrame(paint)}

return {paint(frame,width,height,time){
 W=width;H=height;state=frame.state;relations=frame.relations||[];traces=frame.traces;episodes=frame.episodes;
 concepts=frame.concepts;hypergraphs=frame.hypergraphs;
 byId=Object.fromEntries(frame.all.map(node=>[node.id,node]));
 paint(time);
 return state.rendered;
}};
}
