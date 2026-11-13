export type Camera = {az:number;el:number;zoom:number};
export type Vec3 = [number,number,number];
export const clamp = (x:number,a:number,b:number)=>Math.min(b,Math.max(a,x));
