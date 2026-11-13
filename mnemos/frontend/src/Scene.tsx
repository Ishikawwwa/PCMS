import { useEffect, useMemo, useRef } from 'react';
import { Graph } from './types';
import {groundedColors} from './Grounded';
import { Camera, Vec3, clamp } from './reference-canvas';
import { createReferenceRenderer, ReferenceFrame, ReferenceNode, ReferenceHit } from './reference-renderer';

export const colors: Record<string,string> = { trace: '#6b9db5', episode: '#76d5bd', concept: '#bba6ff', meta_concept: '#ffe2ad' };
type Props = { graph: Graph; selected: string; onSelect: (id:string)=>void; rotate: boolean; layers: number[]; showEdges: boolean; similarities: boolean; lineage: boolean; reset: number; onPerformance: (v: {fps:number;mean_ms:number;p95_ms:number;samples:number})=>void };
type Hit = ReferenceHit;
const initialCamera = ():Camera=>({az:-.28,el:.19,zoom:1});

export default function Scene(props:Props) {
  const canvas=useRef<HTMLCanvasElement>(null),latest=useRef(props),camera=useRef(initialCamera());
  latest.current=props;
  const positions=useMemo(()=>{
    const coords=props.graph.projection?.coordinates||{},result:Record<string,Vec3>={};
    // Original HTML projectPCA(): normalize each display axis by its own
    // maximum absolute coordinate, then multiply by 265. Stored PCA and
    // original-space vectors remain untouched.
    const projected=Object.values(coords);
    const limits=[0,1,2].map(k=>Math.max(.1,...projected.map(p=>Math.abs(p[k]||0))));
    props.graph.nodes.forEach((node,i)=>{
      if(coords[node.id]) result[node.id]=coords[node.id].map((v,k)=>v/limits[k]*265) as Vec3;
      else {const angle=i*2.399963,radius=66+Math.sqrt(i+1)*8.36;result[node.id]=[Math.cos(angle)*radius,((i%9)-4)*11,Math.sin(angle)*radius];}
    });
    return result;
  },[props.graph]);
  const currentPositions=useRef(positions);currentPositions.current=positions;
  useEffect(()=>{camera.current=initialCamera();},[props.reset]);
  useEffect(()=>{
    const view=canvas.current!,cx=view.getContext('2d',{alpha:true});
    if(!cx)return;
    const renderer=createReferenceRenderer(cx);
    let W=1,H=1,dpr=1,raf=0,last=0,hover='',hoverLabel=false,drag=false,moved=0,px=0,py=0;
    let hits:Hit[]=[],frames:number[]=[];
    function resize(){const rect=view.getBoundingClientRect();W=rect.width;H=rect.height;dpr=Math.min(devicePixelRatio||1,2);view.width=Math.round(W*dpr);view.height=Math.round(H*dpr);}
    const observer=new ResizeObserver(resize);observer.observe(view);resize();
    const hit=(x:number,y:number)=>hits.filter(n=>n.bounds?(x>=n.bounds.left&&x<=n.bounds.right&&y>=n.bounds.top&&y<=n.bounds.bottom):Math.hypot(x-n.x,y-n.y)<=n.r).sort((a,b)=>b.z-a.z)[0];
    function point(e:PointerEvent){const rect=view.getBoundingClientRect();return{x:e.clientX-rect.left,y:e.clientY-rect.top};}
    function down(e:PointerEvent){if(e.button!==0)return;view.setPointerCapture(e.pointerId);drag=true;moved=0;px=e.clientX;py=e.clientY;view.style.cursor='grabbing';}
    function move(e:PointerEvent){if(drag){const dx=e.clientX-px,dy=e.clientY-py;camera.current.az-=dx*.006;camera.current.el=clamp(camera.current.el+dy*.006,-1.37,1.37);moved+=Math.abs(dx)+Math.abs(dy);px=e.clientX;py=e.clientY;}else{const p=point(e);const target=hit(p.x,p.y);hover=target?.id||'';hoverLabel=!!target?.bounds;view.style.cursor=hover?'pointer':'grab';}}
    function up(e:PointerEvent){if(!drag)return;drag=false;view.style.cursor='grab';if(moved<10){const p=point(e);latest.current.onSelect(hit(p.x,p.y)?.id||'');}}
    function cancel(){drag=false;hover='';view.style.cursor='grab';}
    function leave(){if(!drag){hover='';view.style.cursor='grab';}}
    function wheel(e:WheelEvent){e.preventDefault();camera.current.zoom=clamp(camera.current.zoom*(e.deltaY>0?.90:1.10),.48,3.1);}
    function paint(t:number){
      if(!cx)return;
      const props=latest.current,positions=currentPositions.current,cam=camera.current;
      cx.setTransform(dpr,0,0,dpr,0,0);
      const visible=props.graph.nodes.filter(n=>props.layers.includes(Math.min(3,n.level)));
      const all:ReferenceNode[]=visible.map(n=>({
        id:n.id,title:n.title,objectType:n.type,status:n.status,displayColor:props.graph.schema_version?groundedColors[n.type]:undefined,sessionIndex:props.graph.schema_version?(n.type==='trace'?[...new Set(props.graph.nodes.filter(v=>v.type==='trace').map(v=>String(v.session_id||v.session||'source')))].indexOf(String(n.session_id||n.session||'source'))+1:4):undefined,layout:positions[n.id]&&props.graph.projection?.coordinates[n.id]?'semantic PCA':'structural',kind:n.level===0?'trace':n.level===1?'episode':n.type==='rule'||n.level>=3?'meta':n.level===2?'concept':'episode',
        pos:positions[n.id],v:n.vector||[],sources:n.evidence_ids as string[]||[],
        concept:props.graph.links.find(l=>l.target===n.id&&visible.some(parent=>parent.id===l.source&&parent.type==='concept'))?.source,
        valid:!['disputed','rejected','needs_revalidation'].includes(n.status),
        hyperId:props.graph.hyperedges.find(h=>(h.crystallized_concept_ids as string[]||[]).includes(n.id))?.id,
      }));
      const active=new Set([props.selected,hover]);
      for(const link of props.graph.links){if(link.source===props.selected)active.add(link.target);if(link.target===props.selected)active.add(link.source);}
      for(const edge of props.graph.hyperedges){if(edge.id===props.selected)edge.member_ids.forEach(id=>active.add(id));}
      const hypergraphs=props.showEdges?props.graph.hyperedges.filter(h=>h.member_ids.every(id=>positions[id]&&all.some(n=>n.id===id))).map(h=>({
        id:h.id,ids:h.member_ids,pos:h.member_ids.reduce<Vec3>((sum,id)=>sum.map((v,i)=>v+positions[id][i]/h.member_ids.length) as Vec3,[0,0,0]),
      })):[];
      const selected=all.find(n=>n.id===props.selected);
      const stage=all.some(n=>n.kind==='meta')?5:hypergraphs.length?4:all.some(n=>n.kind==='concept')?3:all.some(n=>n.kind==='episode')?1:0;
      const frame:ReferenceFrame={
        all,relations:props.graph.schema_version?props.graph.links:[],traces:all.filter(n=>n.kind==='trace'),episodes:all.filter(n=>n.kind==='episode'),concepts:all.filter(n=>n.kind==='concept'),hypergraphs,
        state:{...cam,stage,grounded:!!props.graph.schema_version,hoverId:hover,hoverLabel,auto:props.rotate,drag,showRaw:props.layers.includes(0),showEdges:props.similarities,
          selected:selected?{id:selected.id,kind:selected.kind}:null,selectedHyperId:hypergraphs.find(h=>h.id===props.selected)?.id||null,
          metas:all.filter(n=>n.kind==='meta'),changed:selected?.valid===false?{id:selected.id}:null,active,rendered:[]},
      };
      // The HTML's cosine-neighbor overlay is a display option, separate from
      // persisted scientific hyperedges and provenance.
      if(!props.lineage)frame.episodes=frame.episodes.map(n=>({...n,sources:[],concept:undefined}));
      hits=renderer.paint(frame,W,H,t);
      cam.az=frame.state.az;
      if(last){frames.push(t-last);if(frames.length===120){const times=[...frames].sort((a,b)=>a-b),mean=times.reduce((a,b)=>a+b,0)/times.length;props.onPerformance({fps:1000/mean,mean_ms:mean,p95_ms:times[Math.floor(times.length*.95)],samples:times.length});frames=[];}}
      last=t;raf=requestAnimationFrame(paint);
    }
    view.addEventListener('pointerdown',down);view.addEventListener('pointermove',move);view.addEventListener('pointerup',up);view.addEventListener('pointercancel',cancel);view.addEventListener('pointerleave',leave);view.addEventListener('wheel',wheel,{passive:false});
    raf=requestAnimationFrame(paint);
    return()=>{cancelAnimationFrame(raf);observer.disconnect();view.removeEventListener('pointerdown',down);view.removeEventListener('pointermove',move);view.removeEventListener('pointerup',up);view.removeEventListener('pointercancel',cancel);view.removeEventListener('pointerleave',leave);view.removeEventListener('wheel',wheel);};
  },[]);
  return <canvas ref={canvas} className="memory-canvas" role="img" aria-label="Interactive 3D semantic memory graph. Drag to orbit, scroll to zoom, select a node to inspect."/>;
}
