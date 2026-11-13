import { Camera, Vec3 } from './reference-canvas';
export type ReferenceHit = {id:string;kind:string;x:number;y:number;z:number;r:number;bounds?:{left:number;top:number;right:number;bottom:number}};
export type ReferenceNode = {id:string;title:string;kind:'trace'|'episode'|'concept'|'meta';pos:Vec3;v:number[];sources:string[];concept?:string;valid:boolean;hyperId?:string;objectType?:string;displayColor?:string;status?:string;sessionIndex?:number;layout?:string};
export type ReferenceEdge = {id:string;ids:string[];pos:Vec3};
export type ReferenceFrame = {
  relations?:{source:string;target:string;type:string;status?:string}[];
  all:ReferenceNode[];traces:ReferenceNode[];episodes:ReferenceNode[];concepts:ReferenceNode[];hypergraphs:ReferenceEdge[];
  state:Camera & {stage:number;grounded:boolean;hoverId:string;hoverLabel:boolean;auto:boolean;drag:boolean;showRaw:boolean;showEdges:boolean;selected:{id:string;kind:string}|null;selectedHyperId:string|null;metas:ReferenceNode[];changed:{id:string}|null;active:Set<string>;rendered:ReferenceHit[]};
};
export function createReferenceRenderer(context:CanvasRenderingContext2D):{paint(frame:ReferenceFrame,width:number,height:number,time:number):ReferenceHit[]};
