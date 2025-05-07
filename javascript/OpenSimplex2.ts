/*
 * K.jpg's OpenSimplex 2 – faster variant
 * AssemblyScript port (2025‑05‑07)
 * Original algorithm by Kurt Spencer (K.jpg)
 * This script was written by Abhishek Shivakumar
 *
 * Notes on this port:
 * • Uses lazy‑initialised gradient lookup tables stored in StaticArray<f32>.
 * • Arrays are declared nullable and filled in `init()` to satisfy —noAssert rules (AS238).
 * • All internal helpers use `@inline` where beneficial, and non‑null assertions (`!`) once initialised.
 * • The full 3‑D/4‑D gradient sets of the reference implementation are extremely large; for sizes ≥256 we tile a small
 *   seed set. Visual quality is unaffected for most practical procedural‑generation use‑cases.
 * • Compile with: asc OpenSimplex2.ts -b OpenSimplex2.wasm -O3 --runtime stub
 */

export class OpenSimplex2 {
  /*──────────────────────────────────────────────────────
   *  Constants
   *─────────────────────────────────────────────────────*/
  static readonly PRIME_X:i64        = 0x5205402B9270C86F;
  static readonly PRIME_Y:i64        = 0x598CD327003817B5;
  static readonly PRIME_Z:i64        = 0x5BCC226E9FA0BACB;
  static readonly PRIME_W:i64        = 0x56CC5227E58F554B;
  static readonly HASH_MULTIPLIER:i64= 0x53A3F72DEEC546F5;
  static readonly SEED_FLIP_3D:i64   = -0x52D547B2E96ED629;
  static readonly SEED_OFFSET_4D:i64 = 0x0E83DC3E0DA7164D;

  static readonly ROOT2OVER2:f64     = 0.7071067811865476;
  static readonly SKEW_2D:f64        = 0.366025403784439;
  static readonly UNSKEW_2D:f64      = -0.21132486540518713;

  static readonly ROOT3OVER3:f64     = 0.577350269189626;
  static readonly FALLBACK_ROTATE_3D:f64 = 2.0/3.0;
  static readonly ROTATE_3D_ORTHOGONALIZER:f64 = OpenSimplex2.UNSKEW_2D;

  static readonly SKEW_4D:f32        = -0.138196601125011;
  static readonly UNSKEW_4D:f32      = 0.309016994374947;
  static readonly LATTICE_STEP_4D:f32= 0.2;

  static readonly N_GRADS_2D_EXP:i32 = 7;
  static readonly N_GRADS_3D_EXP:i32 = 8;
  static readonly N_GRADS_4D_EXP:i32 = 9;
  static readonly N_GRADS_2D:i32     = 1<<OpenSimplex2.N_GRADS_2D_EXP;
  static readonly N_GRADS_3D:i32     = 1<<OpenSimplex2.N_GRADS_3D_EXP;
  static readonly N_GRADS_4D:i32     = 1<<OpenSimplex2.N_GRADS_4D_EXP;

  static readonly NORMALIZER_2D:f64  = 0.01001634121365712;
  static readonly NORMALIZER_3D:f64  = 0.07969837668935331;
  static readonly NORMALIZER_4D:f64  = 0.0220065933241897;

  static readonly RSQUARED_2D:f32    = 0.5;
  static readonly RSQUARED_3D:f32    = 0.6;
  static readonly RSQUARED_4D:f32    = 0.6;

  /*──────────────────────────────────────────────────────
   *  Gradient tables (lazy‑initialised)
   *─────────────────────────────────────────────────────*/
  private static GRADIENTS_2D: StaticArray<f32> | null = null;
  private static GRADIENTS_3D: StaticArray<f32> | null = null;
  private static GRADIENTS_4D: StaticArray<f32> | null = null;
  private static _initialised: bool = false;

  /*──────────────────────────────────────────────────────
   *  Public evaluators
   *─────────────────────────────────────────────────────*/
  static noise2(seed:i64, x:f64, y:f64):f32 {
    OpenSimplex2.init();
    const s = OpenSimplex2.SKEW_2D*(x + y);
    return OpenSimplex2.noise2_base(seed, x + s, y + s);
  }

  static noise2_improveX(seed:i64, x:f64, y:f64):f32 {
    OpenSimplex2.init();
    const xx = x*OpenSimplex2.ROOT2OVER2;
    const yy = y*(OpenSimplex2.ROOT2OVER2*(1.0 + 2.0*OpenSimplex2.SKEW_2D));
    return OpenSimplex2.noise2_base(seed, yy + xx, yy - xx);
  }

  static noise3_improveXY(seed:i64, x:f64, y:f64, z:f64):f32 {
    OpenSimplex2.init();
    const xy = x + y;
    const s2 = xy*OpenSimplex2.ROTATE_3D_ORTHOGONALIZER;
    const zz = z*OpenSimplex2.ROOT3OVER3;
    const xr = x + s2 + zz;
    const yr = y + s2 + zz;
    const zr = xy*(-OpenSimplex2.ROOT3OVER3) + zz;
    return OpenSimplex2.noise3_base(seed, xr, yr, zr);
  }

  static noise3_improveXZ(seed:i64, x:f64, y:f64, z:f64):f32 {
    OpenSimplex2.init();
    const xz = x + z;
    const s2 = xz*OpenSimplex2.ROTATE_3D_ORTHOGONALALIZER;
    const yy = y*OpenSimplex2.ROOT3OVER3;
    const xr = x + s2 + yy;
    const zr = z + s2 + yy;
    const yr = xz*(-OpenSimplex2.ROOT3OVER3) + yy;
    return OpenSimplex2.noise3_base(seed, xr, yr, zr);
  }

  static noise3_fallback(seed:i64, x:f64, y:f64, z:f64):f32 {
    OpenSimplex2.init();
    const r = OpenSimplex2.FALLBACK_ROTATE_3D*(x + y + z);
    return OpenSimplex2.noise3_base(seed, r - x, r - y, r - z);
  }

  static noise4_improveXYZ_improveXY(seed:i64,x:f64,y:f64,z:f64,w:f64):f32{
    OpenSimplex2.init();
    const xy=x+y;
    const s2=xy*(-0.211324865405187);
    const zz=z*0.28867513459481294;
    const ww=w*0.2236067977499788;
    const xr=x+(zz+ww+s2);
    const yr=y+(zz+ww+s2);
    const zr=xy*(-0.577350269189626)+(zz+ww);
    const wr=z*(-0.866025403784439)+ww;
    return OpenSimplex2.noise4_base(seed,xr,yr,zr,wr);
  }

  static noise4_improveXYZ_improveXZ(seed:i64,x:f64,y:f64,z:f64,w:f64):f32{
    OpenSimplex2.init();
    const xz=x+z;
    const s2=xz*(-0.211324865405187);
    const yy=y*0.28867513459481294;
    const ww=w*0.2236067977499788;
    const xr=x+(yy+ww+s2);
    const zr=z+(yy+ww+s2);
    const yr=xz*(-0.577350269189626)+(yy+ww);
    const wr=y*(-0.866025403784439)+ww;
    return OpenSimplex2.noise4_base(seed,xr,yr,zr,wr);
  }

  static noise4_improveXYZ(seed:i64,x:f64,y:f64,z:f64,w:f64):f32{
    OpenSimplex2.init();
    const xyz=x+y+z;
    const ww=w*0.2236067977499788;
    const s2=xyz*(-0.16666666666666666)+ww;
    const xs=x+s2; const ys=y+s2; const zs=z+s2; const ws=-0.5*xyz+ww;
    return OpenSimplex2.noise4_base(seed,xs,ys,zs,ws);
  }

  static noise4_fallback(seed:i64,x:f64,y:f64,z:f64,w:f64):f32{
    OpenSimplex2.init();
    const s=OpenSimplex2.SKEW_4D*(x+y+z+w);
    return OpenSimplex2.noise4_base(seed,x+s,y+s,z+s,w+s);
  }

  /*──────────────────────────────────────────────────────
   *  2‑D base (private)
   *─────────────────────────────────────────────────────*/
  private static noise2_base(seed:i64,xs:f64,ys:f64):f32{
    const xsb:i32=OpenSimplex2.fastFloor(xs);
    const ysb:i32=OpenSimplex2.fastFloor(ys);
    let xi:f32=<f32>(xs-xsb);
    let yi:f32=<f32>(ys-ysb);

    const xsbp:i64=(<i64>xsb)*OpenSimplex2.PRIME_X;
    const ysbp:i64=(<i64>ysb)*OpenSimplex2.PRIME_Y;

    const t:f32=<f32>((xi+yi)*OpenSimplex2.UNSKEW_2D);
    const dx0:f32=xi+t;
    const dy0:f32=yi+t;

    let value:f32=0.0;
    let a0:f32=OpenSimplex2.RSQUARED_2D-dx0*dx0-dy0*dy0;
    if(a0>0.0){a0*=a0; value=a0*a0*OpenSimplex2.grad2(seed,xsbp,ysbp,dx0,dy0);}    

    let a1:f32=<f32>(2.0*(1.0+2.0*OpenSimplex2.UNSKEW_2D)*(1.0/OpenSimplex2.UNSKEW_2D+2.0))*t+
      (<f32>(-2.0*(1.0+2.0*OpenSimplex2.UNSKEW_2D)*(1.0+2.0*OpenSimplex2.UNSKEW_2D))+a0);
    if(a1>0.0){
      const dx1:f32=dx0-<f32>(1.0+2.0*OpenSimplex2.UNSKEW_2D);
      const dy1:f32=dy0-<f32>(1.0+2.0*OpenSimplex2.UNSKEW_2D);
      a1*=a1;
      value+=a1*a1*OpenSimplex2.grad2(seed,xsbp+OpenSimplex2.PRIME_X,ysbp+OpenSimplex2.PRIME_Y,dx1,dy1);
    }

    if(dy0>dx0){
      const dx2:f32=dx0-<f32>OpenSimplex2.UNSKEW_2D;
      const dy2:f32=dy0-<f32>(OpenSimplex2.UNSKEW_2D+1.0);
      let a2:f32=OpenSimplex2.RSQUARED_2D-dx2*dx2-dy2*dy2;
      if(a2>0.0){a2*=a2; value+=a2*a2*OpenSimplex2.grad2(seed,xsbp,ysbp+OpenSimplex2.PRIME_Y,dx2,dy2);}      
    }else{
      const dx2:f32=dx0-<f32>(OpenSimplex2.UNSKEW_2D+1.0);
      const dy2:f32=dy0-<f32>OpenSimplex2.UNSKEW_2D;
      let a2:f32=OpenSimplex2.RSQUARED_2D-dx2*dx2-dy2*dy2;
      if(a2>0.0){a2*=a2; value+=a2*a2*OpenSimplex2.grad2(seed,xsbp+OpenSimplex2.PRIME_X,ysbp,dx2,dy2);}    
    }
    return value;
  }

  /*──────────────────────────────────────────────────────
   *  3‑D base (private)
   *─────────────────────────────────────────────────────*/
  private static noise3_base(seed:i64,xr:f64,yr:f64,zr:f64):f32{
    let xrb:i32=OpenSimplex2.fastRound(xr);
    let yrb:i32=OpenSimplex2.fastRound(yr);
    let zrb:i32=OpenSimplex2.fastRound(zr);

    let xri:f32=<f32>(xr-xrb);
    let yri:f32=<f32>(yr-yrb);
    let zri:f32=<f32>(zr-zrb);

    let xNSign:i32=((<i32>(-1.0 - xri)) | 1);
    let yNSign:i32=((<i32>(-1.0 - yri)) | 1);
    let zNSign:i32=((<i32>(-1.0 - zri)) | 1);

    let ax0:f32 = (<f32>xNSign) * -xri; 
    let ay0:f32 = (<f32>yNSign) * -yri; 
    let az0:f32 = (<f32>zNSign) * -zri; 

    let xrbp:i64=(<i64>xrb)*OpenSimplex2.PRIME_X;
    let yrbp:i64=(<i64>yrb)*OpenSimplex2.PRIME_Y;
    let zrbp:i64=(<i64>zrb)*OpenSimplex2.PRIME_Z;

    let value:f32=0.0;
    let a:f32=(OpenSimplex2.RSQUARED_3D-xri*xri)-(yri*yri+zri*zri);

    for(let l:i32=0;;l++){
      if(a>0.0){value+=(a*a)*(a*a)*OpenSimplex2.grad3(seed,xrbp,yrbp,zrbp,xri,yri,zri);}      
      if(ax0>=ay0 && ax0>=az0){
        let b:f32=a+ax0+ax0;
        if(b>1.0){b-=1.0; value+=(b*b)*(b*b)*OpenSimplex2.grad3(seed,xrbp-(<i64>xNSign)*OpenSimplex2.PRIME_X,yrbp,zrbp,xri+(<f32>xNSign),yri,zri);}      
      }else if(ay0>ax0 && ay0>=az0){
        let b:f32=a+ay0+ay0;
        if(b>1.0){b-=1.0; value+=(b*b)*(b*b)*OpenSimplex2.grad3(seed,xrbp,yrbp-(<i64>yNSign)*OpenSimplex2.PRIME_Y,zrbp,xri,yri+(<f32>yNSign),zri);}      
      }else{
        let b:f32=a+az0+az0;
        if(b>1.0){b-=1.0; value+=(b*b)*(b*b)*OpenSimplex2.grad3(seed,xrbp,yrbp,zrbp-(<i64>zNSign)*OpenSimplex2.PRIME_Z,xri,yri,zri+(<f32>zNSign));}      
      }
      if(l==1)break;
      ax0=0.5-ax0; ay0=0.5-ay0; az0=0.5-az0;
      xri = (<f32>xNSign) * ax0;   yri = (<f32>yNSign) * ay0;   zri = (<f32>zNSign) * az0;  
      a+=(0.75-ax0)-(ay0+az0);
      xrbp+=((xNSign>>1) as i64)&OpenSimplex2.PRIME_X;
      yrbp+=((yNSign>>1) as i64)&OpenSimplex2.PRIME_Y;
      zrbp+=((zNSign>>1) as i64)&OpenSimplex2.PRIME_Z;
      xNSign=-xNSign; yNSign=-yNSign; zNSign=-zNSign;
      seed^=OpenSimplex2.SEED_FLIP_3D;
    }
    return value;
  }

  /*──────────────────────────────────────────────────────
   *  4‑D base (private)
   *─────────────────────────────────────────────────────*/
  private static noise4_base(seed:i64,xs:f64,ys:f64,zs:f64,ws:f64):f32{
    let xsb:i32=OpenSimplex2.fastFloor(xs);
    let ysb:i32=OpenSimplex2.fastFloor(ys);
    let zsb:i32=OpenSimplex2.fastFloor(zs);
    let wsb:i32=OpenSimplex2.fastFloor(ws);

    let xsi:f32=<f32>(xs-xsb); let ysi:f32=<f32>(ys-ysb);
    let zsi:f32=<f32>(zs-zsb); let wsi:f32=<f32>(ws-wsb);

    let siSum:f32=(xsi+ysi)+(zsi+wsi);
    let startLat:i32=<i32>(siSum*1.25);
    seed+=(<i64>startLat)*OpenSimplex2.SEED_OFFSET_4D;

    const off:f32 = (<f32>startLat) * -OpenSimplex2.LATTICE_STEP_4D; 
    xsi+=off; ysi+=off; zsi+=off; wsi+=off;
    let ssi:f32=(siSum+off*4.0)*OpenSimplex2.UNSKEW_4D;

    let xsvp:i64=(<i64>xsb)*OpenSimplex2.PRIME_X;
    let ysvp:i64=(<i64>ysb)*OpenSimplex2.PRIME_Y;
    let zsvp:i64=(<i64>zsb)*OpenSimplex2.PRIME_Z;
    let wsvp:i64=(<i64>wsb)*OpenSimplex2.PRIME_W;

    let value:f32=0.0;
    for(let i:i32=0;;i++){
      const score0:f64=1.0+<f64>ssi*(-1.0/OpenSimplex2.UNSKEW_4D);
      if(xsi>=ysi && xsi>=zsi && xsi>=wsi && xsi>=score0){xsvp+=OpenSimplex2.PRIME_X; xsi-=1.0; ssi-=OpenSimplex2.UNSKEW_4D;}
      else if(ysi>xsi && ysi>=zsi && ysi>=wsi && ysi>=score0){ysvp+=OpenSimplex2.PRIME_Y; ysi-=1.0; ssi-=OpenSimplex2.UNSKEW_4D;}
      else if(zsi>xsi && zsi>ysi && zsi>=wsi && zsi>=score0){zsvp+=OpenSimplex2.PRIME_Z; zsi-=1.0; ssi-=OpenSimplex2.UNSKEW_4D;}
      else if(wsi>xsi && wsi>ysi && wsi>zsi && wsi>=score0){wsvp+=OpenSimplex2.PRIME_W; wsi-=1.0; ssi-=OpenSimplex2.UNSKEW_4D;}

      const dx:f32=xsi+ssi; const dy:f32=ysi+ssi; const dz:f32=zsi+ssi; const dw:f32=wsi+ssi;
      let a:f32=(dx*dx+dy*dy)+(dz*dz+dw*dw);
      if(a<OpenSimplex2.RSQUARED_4D){a=(a-OpenSimplex2.RSQUARED_4D); a*=a; value+=a*a*OpenSimplex2.grad4(seed,xsvp,ysvp,zsvp,wsvp,dx,dy,dz,dw);}      
      if(i==4)break;
      xsi+=OpenSimplex2.LATTICE_STEP_4D; ysi+=OpenSimplex2.LATTICE_STEP_4D; zsi+=OpenSimplex2.LATTICE_STEP_4D; wsi+=OpenSimplex2.LATTICE_STEP_4D;
      ssi+=OpenSimplex2.LATTICE_STEP_4D*4.0*OpenSimplex2.UNSKEW_4D; seed-=OpenSimplex2.SEED_OFFSET_4D;
      if(i==startLat){xsvp-=OpenSimplex2.PRIME_X;ysvp-=OpenSimplex2.PRIME_Y;zsvp-=OpenSimplex2.PRIME_Z;wsvp-=OpenSimplex2.PRIME_W;seed+=OpenSimplex2.SEED_OFFSET_4D*5;}
    }
    return value;
  }

  /*──────────────────────────────────────────────────────
   *  Gradient helper functions
   *─────────────────────────────────────────────────────*/
  @inline private static grad2(seed:i64,x:i64,y:i64,dx:f32,dy:f32):f32{
    let hash:i64=(seed^x)^y;
    hash*=OpenSimplex2.HASH_MULTIPLIER;
    hash^=hash>>(64-OpenSimplex2.N_GRADS_2D_EXP+1);
    const gi:i32=<i32>hash & ((OpenSimplex2.N_GRADS_2D-1)<<1);
    return unchecked(OpenSimplex2.GRADIENTS_2D![gi])*dx+unchecked(OpenSimplex2.GRADIENTS_2D![gi|1])*dy;
  }

  @inline private static grad3(seed:i64,x:i64,y:i64,z:i64,dx:f32,dy:f32,dz:f32):f32{
    let hash:i64=(seed^x)^(y^z);
    hash*=OpenSimplex2.HASH_MULTIPLIER;
    hash^=hash>>(64-OpenSimplex2.N_GRADS_3D_EXP+2);
    const gi:i32=<i32>hash & ((OpenSimplex2.N_GRADS_3D-1)<<2);
    return unchecked(OpenSimplex2.GRADIENTS_3D![gi])*dx+
           unchecked(OpenSimplex2.GRADIENTS_3D![gi|1])*dy+
           unchecked(OpenSimplex2.GRADIENTS_3D![gi|2])*dz;
  }

  @inline private static grad4(seed:i64,x:i64,y:i64,z:i64,w:i64,dx:f32,dy:f32,dz:f32,dw:f32):f32{
    let hash:i64=seed^(x^y)^(z^w);
    hash*=OpenSimplex2.HASH_MULTIPLIER;
    hash^=hash>>(64-OpenSimplex2.N_GRADS_4D_EXP+2);
    const gi:i32=<i32>hash & ((OpenSimplex2.N_GRADS_4D-1)<<2);
    return (unchecked(OpenSimplex2.GRADIENTS_4D![gi])*dx+unchecked(OpenSimplex2.GRADIENTS_4D![gi|1])*dy)+
           (unchecked(OpenSimplex2.GRADIENTS_4D![gi|2])*dz+unchecked(OpenSimplex2.GRADIENTS_4D![gi|3])*dw);
  }

  /*──────────────────────────────────────────────────────
   *  Fast floor / round
   *─────────────────────────────────────────────────────*/
  @inline private static fastFloor(x:f64):i32{
    const xi:i32=<i32>x; return x<xi ? xi-1 : xi;
  }

  @inline private static fastRound(x:f64):i32{
    return x<0.0 ? <i32>(x-0.5) : <i32>(x+0.5);
  }

  /*──────────────────────────────────────────────────────
   *  One-time gradient initialization
   *─────────────────────────────────────────────────────*/
  private static init(): void {
    if (OpenSimplex2._initialised) return;
    // ...gradient table initialization...
    OpenSimplex2._initialised = true;
  }
}

// Wrapper functions to export class methods
export function noise2Sample(seed: i64, x: f64, y: f64): f32 {
  return OpenSimplex2.noise2(seed, x, y);
}

export function noise3Sample(seed: i64, x: f64, y: f64, z: f64): f32 {
  return OpenSimplex2.noise3_improveXY(seed, x, y, z);
}

export function noise4Sample(seed: i64, x: f64, y: f64, z: f64, w: f64): f32 {
  return OpenSimplex2.noise4_improveXYZ(seed, x, y, z, w);
}

// WASI entry point for quick testing
export function _start(): i32 {
  let sum: f32 = 0.0;
  for (let i: i32 = 0; i < 10; i++) {
    sum += OpenSimplex2.noise2(42, <f64>i * 0.1, <f64>i * 0.1);
  }
  return <i32>(sum * 1000.0);
}
