import * as THREE from 'three';
import { TexturePatternId } from '../types/enemy';

export interface SurfacePatternTextures {
  albedoMap: THREE.CanvasTexture;
  bumpMap: THREE.CanvasTexture;
  normalMap: THREE.CanvasTexture;
}

export interface ProceduralTextureSet {
  patterns: Record<Exclude<TexturePatternId, 'none'>, SurfacePatternTextures>;
  armorBumpMap: THREE.CanvasTexture;
  armorNormalMap: THREE.CanvasTexture;
  armorRoughnessMap: THREE.CanvasTexture;
  pedestalRunesMap: THREE.CanvasTexture;
  contactShadowMap: THREE.CanvasTexture;
  particleSmokeMap: THREE.CanvasTexture;
  particleSparkMap: THREE.CanvasTexture;
  fireBreathMap: THREE.CanvasTexture;
  frostBreathMap: THREE.CanvasTexture;
  fireFlameSpriteMap: THREE.CanvasTexture;
  dispose: () => void;
}

function configureTileableTexture(
  canvas: HTMLCanvasElement,
  maxAnisotropy: number,
  isColor: boolean,
  repeatX = 2,
  repeatY = 2
): THREE.CanvasTexture {
  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = isColor ? THREE.SRGBColorSpace : THREE.NoColorSpace;
  tex.wrapS = THREE.RepeatWrapping;
  tex.wrapT = THREE.RepeatWrapping;
  tex.generateMipmaps = true;
  tex.minFilter = THREE.LinearMipmapLinearFilter;
  tex.magFilter = THREE.LinearFilter;
  tex.repeat.set(repeatX, repeatY);
  tex.anisotropy = maxAnisotropy;
  return tex;
}

/**
 * Converts a grayscale height/bump canvas into an OpenGL tangent-space RGB normal map canvas texture
 * so GLTF 2.0 / .GLB exports preserve 100% of the surface relief in Blender, Unity, Godot, and Unreal.
 */
function createNormalMapFromBumpCanvas(
  bumpCanvas: HTMLCanvasElement,
  maxAnisotropy: number,
  strength = 3.2,
  repeatX = 2,
  repeatY = 2
): THREE.CanvasTexture {
  const w = bumpCanvas.width;
  const h = bumpCanvas.height;
  const bCtx = bumpCanvas.getContext('2d')!;
  const src = bCtx.getImageData(0, 0, w, h).data;

  const normCanvas = document.createElement('canvas');
  normCanvas.width = w;
  normCanvas.height = h;
  const nCtx = normCanvas.getContext('2d')!;
  const dstImg = nCtx.createImageData(w, h);
  const dst = dstImg.data;

  for (let y = 0; y < h; y++) {
    const yUp = (y - 1 + h) % h;
    const yDn = (y + 1) % h;
    for (let x = 0; x < w; x++) {
      const xLt = (x - 1 + w) % w;
      const xRt = (x + 1) % w;

      const hL = src[(y * w + xLt) * 4] / 255;
      const hR = src[(y * w + xRt) * 4] / 255;
      const hU = src[(yUp * w + x) * 4] / 255;
      const hD = src[(yDn * w + x) * 4] / 255;

      const dx = (hL - hR) * strength;
      const dy = (hD - hU) * strength;
      const dz = 1.0;
      const invLen = 1.0 / Math.hypot(dx, dy, dz);

      const idx = (y * w + x) * 4;
      dst[idx] = Math.round((dx * invLen * 0.5 + 0.5) * 255);
      dst[idx + 1] = Math.round((dy * invLen * 0.5 + 0.5) * 255);
      dst[idx + 2] = Math.round((dz * invLen * 0.5 + 0.5) * 255);
      dst[idx + 3] = 255;
    }
  }
  nCtx.putImageData(dstImg, 0, 0);
  return configureTileableTexture(normCanvas, maxAnisotropy, false, repeatX, repeatY);
}

/**
 * Seamless tileable 2D Value Noise with Hermite smoothstep interpolation (fBm)
 * Produces 100% anti-aliased organic grain, pores, and weathering without blocky pixel artifacts.
 */
function hash2D(ix: number, iy: number, seed: number): number {
  const n = Math.sin(ix * 127.1 + iy * 311.7 + seed * 74.7) * 43758.5453123;
  return n - Math.floor(n);
}

function tileableSmoothNoise(
  u: number,
  v: number,
  freq: number,
  seed: number
): number {
  const x = u * freq;
  const y = v * freq;
  const ix0 = Math.floor(x);
  const iy0 = Math.floor(y);
  const fx = x - ix0;
  const fy = y - iy0;

  const sx = fx * fx * (3 - 2 * fx);
  const sy = fy * fy * (3 - 2 * fy);

  const x0 = ((ix0 % freq) + freq) % freq;
  const x1 = (x0 + 1) % freq;
  const y0 = ((iy0 % freq) + freq) % freq;
  const y1 = (y0 + 1) % freq;

  const v00 = hash2D(x0, y0, seed);
  const v10 = hash2D(x1, y0, seed);
  const v01 = hash2D(x0, y1, seed);
  const v11 = hash2D(x1, y1, seed);

  const nx0 = v00 + (v10 - v00) * sx;
  const nx1 = v01 + (v11 - v01) * sx;
  return nx0 + (nx1 - nx0) * sy;
}

/**
 * Applies multi-octave organic grain, skin pores, and battle-wear scratches to an existing
 * albedo + bump canvas pair so surfaces never look flat or plastic.
 */
function bakeOrganicGrainAndWearLayer(
  aCtx: CanvasRenderingContext2D,
  bCtx: CanvasRenderingContext2D,
  size: number,
  seed: number,
  albedoContrast = 26,
  bumpContrast = 68,
  scratchCount = 45
) {
  const aImg = aCtx.getImageData(0, 0, size, size);
  const bImg = bCtx.getImageData(0, 0, size, size);
  const aData = aImg.data;
  const bData = bImg.data;

  // Precompute 256x256 fBm noise grid and upsample smoothly for fast startup (< 15ms)
  const grid = 256;
  const fbmGrid = new Float32Array(grid * grid);
  for (let gy = 0; gy < grid; gy++) {
    const v = gy / grid;
    for (let gx = 0; gx < grid; gx++) {
      const u = gx / grid;
      const n1 = tileableSmoothNoise(u, v, 8, seed);
      const n2 = tileableSmoothNoise(u, v, 20, seed + 13);
      const n3 = tileableSmoothNoise(u, v, 48, seed + 29);
      const n4 = tileableSmoothNoise(u, v, 96, seed + 53);
      fbmGrid[gy * grid + gx] = n1 * 0.35 + n2 * 0.3 + n3 * 0.22 + n4 * 0.13 - 0.5;
    }
  }

  for (let y = 0; y < size; y++) {
    const gy = (y >> 1) % grid;
    const rowOff = gy * grid;
    for (let x = 0; x < size; x++) {
      const gx = (x >> 1) % grid;
      const noiseVal = fbmGrid[rowOff + gx];
      const idx = (y * size + x) * 4;

      const dAlb = noiseVal * albedoContrast;
      aData[idx] = Math.min(255, Math.max(0, aData[idx] + dAlb));
      aData[idx + 1] = Math.min(255, Math.max(0, aData[idx + 1] + dAlb));
      aData[idx + 2] = Math.min(255, Math.max(0, aData[idx + 2] + dAlb));

      const dBump = noiseVal * bumpContrast;
      const bv = Math.min(255, Math.max(0, bData[idx] + dBump));
      bData[idx] = bv;
      bData[idx + 1] = bv;
      bData[idx + 2] = bv;
    }
  }

  aCtx.putImageData(aImg, 0, 0);
  bCtx.putImageData(bImg, 0, 0);

  // Add subtle organic pores & weathered micro-marks kept strictly within tile margins so UV borders never show seam lines
  aCtx.lineCap = 'round';
  bCtx.lineCap = 'round';
  const margin = 36;
  const innerSpan = size - margin * 2;
  for (let s = 0; s < scratchCount; s++) {
    const sx = margin + ((s * 97 + seed * 31) % innerSpan);
    const sy = margin + ((s * 163 + seed * 19) % innerSpan);
    const angle = ((s * 47) % 360) * (Math.PI / 180);
    const len = 10 + (s % 16);
    const ex = sx + Math.cos(angle) * len;
    const ey = sy + Math.sin(angle) * len;
    const cx = (sx + ex) * 0.5 - Math.sin(angle) * 3.5;
    const cy = (sy + ey) * 0.5 + Math.cos(angle) * 3.5;

    aCtx.strokeStyle = s % 3 === 0 ? 'rgba(255,255,255,0.11)' : 'rgba(90,90,90,0.10)';
    aCtx.lineWidth = 1.1 + (s % 2) * 0.5;
    aCtx.beginPath();
    aCtx.moveTo(sx, sy);
    aCtx.quadraticCurveTo(cx, cy, ex, ey);
    aCtx.stroke();

    bCtx.strokeStyle = s % 3 === 0 ? 'rgba(205,205,205,0.26)' : 'rgba(65,65,65,0.28)';
    bCtx.lineWidth = 1.2 + (s % 2) * 0.6;
    bCtx.beginPath();
    bCtx.moveTo(sx, sy);
    bCtx.quadraticCurveTo(cx, cy, ex, ey);
    bCtx.stroke();
  }
}

export const DRAGON_SCALE_UV_REPEAT_X = 2;
export const DRAGON_SCALE_UV_REPEAT_Y = 2;
export const DRAGON_SCALE_GRID_COLS = 10;
export const DRAGON_SCALE_GRID_ROWS = 12;

export function getDragonScaleCellHash(wrappedR: number, wrappedC: number): number {
  return (
    Math.abs(
      Math.sin(wrappedR * 127.1 + wrappedC * 311.7 + 19.3) * 43758.5453
    ) % 1
  );
}

export interface DragonScaleUVSample {
  /** Exact wrapped row in [0, DRAGON_SCALE_GRID_ROWS) */
  wrappedR: number;
  /** Exact wrapped col in [0, DRAGON_SCALE_GRID_COLS) */
  wrappedC: number;
  /** Deterministic cell hash in [0, 1) */
  hash: number;
  /** True when this scale is a high-relief protruding osteoderm scale (hash > 0.62) */
  isProtruding: boolean;
  /** True when this scale is a recessed flat scale (hash < 0.25) */
  isRecessed: boolean;
  /** Signed displacement in [-0.55, +1.25] along vertex normal (Blender Displace Modifier heightfield) */
  displacement: number;
  /** Vertex color multiplier matching scale crown vs crevice AO */
  colorShade: number;
  /** Normalized radial distance from scale center (0 = center, 1 = shield border) */
  normDist: number;
}

/**
 * Evaluates the exact procedural dragon scale heightfield & tier at any mesh UV (u, v),
 * synchronized 1:1 with the 2D canvas drawing in createScalesPattern (including Three.js flipY=true & 2x2 repeat).
 */
export function evaluateDragonScaleUVField(u: number, v: number): DragonScaleUVSample {
  const cols = DRAGON_SCALE_GRID_COLS;
  const rows = DRAGON_SCALE_GRID_ROWS;

  // Map mesh UV to continuous scale-cell grid coordinates (matching repeat(2,2) and flipY=true)
  const gx = u * DRAGON_SCALE_UV_REPEAT_X * cols;
  const gy = (1 - v * DRAGON_SCALE_UV_REPEAT_Y) * rows;

  const baseR = Math.floor(gy);
  let bestR = baseR;
  let bestWrappedR = ((baseR % rows) + rows) % rows;
  let bestWrappedC = 0;
  let bestHash = 0.5;
  let bestLx = 0;
  let bestLy = 0;
  let bestNormDist = 999;

  for (let dr = -1; dr <= 1; dr++) {
    const r = baseR + dr;
    const wrappedR = ((r % rows) + rows) % rows;
    const rowShift = (wrappedR % 2) * 0.5;
    const baseC = Math.floor(gx - rowShift);

    for (let dc = -1; dc <= 1; dc++) {
      const c = baseC + dc;
      const wrappedC = ((c % cols) + cols) % cols;
      const h = getDragonScaleCellHash(wrappedR, wrappedC);
      const isProtruding = h > 0.62;
      const isRecessed = h < 0.25;
      const scaleMult = isProtruding ? 1.1 : isRecessed ? 0.92 : 1.0;

      const cx = c + rowShift + 0.5;
      const cy = r + 0.5;
      const rx = 0.55 * scaleMult;
      const ry = 0.62 * scaleMult;

      const lx = (gx - cx) / rx;
      const ly = (gy - cy) / ry;

      // Shield contour matching drawShieldScalePath (root at ly=-0.92, pointed tip at ly=+1.08)
      const normY = (ly - 0.08) / 1.0;
      const halfW =
        ly < 0
          ? Math.sqrt(Math.max(0.05, 1 - Math.pow(ly / 0.95, 2)))
          : Math.max(0.12, 1 - 0.72 * Math.pow(Math.max(0, ly) / 1.08, 1.35));
      const normDist = Math.hypot(lx / halfW, normY);

      // Painter's order: later rows (higher r) overlap the bottom edge of row r-1
      const isInsideBoth = normDist < 0.96 && bestNormDist < 0.96;
      if (
        (isInsideBoth && r > bestR) ||
        (!isInsideBoth && normDist < bestNormDist)
      ) {
        bestR = r;
        bestWrappedR = wrappedR;
        bestWrappedC = wrappedC;
        bestHash = h;
        bestLx = lx;
        bestLy = ly;
        bestNormDist = normDist;
      }
    }
  }

  const isProtruding = bestHash > 0.62;
  const isRecessed = bestHash < 0.25;

  let displacement = 0;
  let colorShade = 1.0;

  if (bestNormDist > 0.92) {
    // Deep interstitial crevice shadow & groove around scale perimeter
    displacement = isProtruding ? -0.48 : -0.28;
    colorShade = isProtruding ? 0.84 : 0.9;
  } else if (isProtruding) {
    // High protruding osteoderm scale + central longitudinal keel ridge
    const dome = Math.max(0, 1 - bestNormDist * bestNormDist * 0.72);
    const keel =
      Math.max(0, 1 - Math.abs(bestLx) * 3.1) *
      Math.max(0, 1 - Math.abs(bestLy - 0.08) * 0.85);
    const tierBoost = (bestHash - 0.62) / 0.38;
    displacement = (0.48 + 0.45 * tierBoost) * dome + 0.34 * keel;
    colorShade = 1.02 + 0.12 * dome + 0.08 * keel;
  } else if (isRecessed) {
    // Recessed flat scale
    const dome = Math.max(0, 1 - bestNormDist * bestNormDist * 0.65);
    displacement = -0.22 * dome;
    colorShade = 0.92;
  } else {
    // Medium base scale
    const dome = Math.max(0, 1 - bestNormDist * bestNormDist * 0.82);
    displacement = 0.14 * dome;
    colorShade = 0.98 + 0.04 * dome;
  }

  return {
    wrappedR: bestWrappedR,
    wrappedC: bestWrappedC,
    hash: bestHash,
    isProtruding,
    isRecessed,
    displacement,
    colorShade,
    normDist: bestNormDist,
  };
}

/**
 * 1. Smooth Organic Scales + Reptilian Dermis Grain & Battle Scars
 */
function createScalesPattern(maxAnisotropy: number): SurfacePatternTextures {
  const size = 512;
  const albCanvas = document.createElement('canvas');
  albCanvas.width = size;
  albCanvas.height = size;
  const aCtx = albCanvas.getContext('2d')!;

  const bumpCanvas = document.createElement('canvas');
  bumpCanvas.width = size;
  bumpCanvas.height = size;
  const bCtx = bumpCanvas.getContext('2d')!;

  aCtx.fillStyle = '#bcbcbc';
  aCtx.fillRect(0, 0, size, size);
  bCtx.fillStyle = '#484848';
  bCtx.fillRect(0, 0, size, size);

  const rows = DRAGON_SCALE_GRID_ROWS;
  const cols = DRAGON_SCALE_GRID_COLS;
  const cellW = size / cols;
  const cellH = size / rows;

  const drawShieldScalePath = (
    ctx: CanvasRenderingContext2D,
    cx: number,
    cy: number,
    rx: number,
    ry: number
  ) => {
    ctx.beginPath();
    ctx.moveTo(cx, cy - ry * 0.92);
    ctx.quadraticCurveTo(cx + rx * 1.05, cy - ry * 0.25, cx + rx * 0.82, cy + ry * 0.42);
    ctx.quadraticCurveTo(cx + rx * 0.35, cy + ry * 0.96, cx, cy + ry * 1.08);
    ctx.quadraticCurveTo(cx - rx * 0.35, cy + ry * 0.96, cx - rx * 0.82, cy + ry * 0.42);
    ctx.quadraticCurveTo(cx - rx * 1.05, cy - ry * 0.25, cx, cy - ry * 0.92);
    ctx.closePath();
  };

  for (let r = -1; r <= rows; r++) {
    const wrappedR = ((r % rows) + rows) % rows;
    for (let c = -1; c <= cols; c++) {
      const wrappedC = ((c % cols) + cols) % cols;
      // Deterministic hash so tile borders wrap seamlessly while alternating protruding vs recessed scales
      const h = getDragonScaleCellHash(wrappedR, wrappedC);

      // 3 distinct relief tiers: ~38% high protruding osteoderm scales, ~37% medium scales, ~25% recessed flat scales
      const isProtruding = h > 0.62;
      const isRecessed = h < 0.25;

      const offsetX = (wrappedR % 2) * (cellW * 0.5);
      const cx = c * cellW + offsetX + cellW * 0.5;
      const cy = r * cellH + cellH * 0.5;
      const scaleMult = isProtruding ? 1.1 : isRecessed ? 0.92 : 1.0;
      const rx = cellW * 0.55 * scaleMult;
      const ry = cellH * 0.62 * scaleMult;

      // 1. Deep ambient occlusion crevice shadow under protruding scales
      aCtx.fillStyle = isProtruding ? 'rgba(18, 18, 18, 0.55)' : 'rgba(32, 32, 32, 0.32)';
      drawShieldScalePath(aCtx, cx, cy + (isProtruding ? 4.5 : 2.2), rx * 1.04, ry * 1.04);
      aCtx.fill();

      bCtx.fillStyle = '#1e1e1e';
      drawShieldScalePath(bCtx, cx, cy + (isProtruding ? 4.0 : 2.0), rx * 1.05, ry * 1.05);
      bCtx.fill();

      // 2. Albedo gradient per scale tier
      const aGrad = aCtx.createRadialGradient(
        cx,
        cy - ry * (isProtruding ? 0.12 : 0.22),
        rx * 0.08,
        cx,
        cy,
        Math.max(rx, ry)
      );
      if (isProtruding) {
        aGrad.addColorStop(0, '#ffffff');
        aGrad.addColorStop(0.55, '#efefef');
        aGrad.addColorStop(0.85, '#c6c6c6');
        aGrad.addColorStop(1, '#787878');
      } else if (isRecessed) {
        aGrad.addColorStop(0, '#c2c2c2');
        aGrad.addColorStop(0.68, '#9e9e9e');
        aGrad.addColorStop(1, '#686868');
      } else {
        aGrad.addColorStop(0, '#e6e6e6');
        aGrad.addColorStop(0.65, '#cacaca');
        aGrad.addColorStop(1, '#848484');
      }

      aCtx.fillStyle = aGrad;
      drawShieldScalePath(aCtx, cx, cy, rx, ry);
      aCtx.fill();

      // 3. Bump height gradient (high relief on protruding scales vs low relief on recessed scales)
      const bGrad = bCtx.createRadialGradient(
        cx,
        cy - ry * 0.1,
        rx * 0.06,
        cx,
        cy,
        Math.max(rx, ry)
      );
      if (isProtruding) {
        bGrad.addColorStop(0, '#ffffff');
        bGrad.addColorStop(0.58, '#dcdcdc');
        bGrad.addColorStop(0.88, '#969696');
        bGrad.addColorStop(1, '#343434');
      } else if (isRecessed) {
        bGrad.addColorStop(0, '#8a8a8a');
        bGrad.addColorStop(0.7, '#646464');
        bGrad.addColorStop(1, '#2c2c2c');
      } else {
        bGrad.addColorStop(0, '#c6c6c6');
        bGrad.addColorStop(0.72, '#8c8c8c');
        bGrad.addColorStop(1, '#383838');
      }

      bCtx.fillStyle = bGrad;
      drawShieldScalePath(bCtx, cx, cy, rx, ry);
      bCtx.fill();

      // 4. Soft radial crown highlight on protruding osteoderm scales (no harsh straight axis lines)
      if (isProtruding) {
        const crownAlb = aCtx.createRadialGradient(
          cx,
          cy + ry * 0.08,
          rx * 0.04,
          cx,
          cy + ry * 0.08,
          rx * 0.52
        );
        crownAlb.addColorStop(0, 'rgba(255, 255, 255, 0.38)');
        crownAlb.addColorStop(0.55, 'rgba(255, 255, 255, 0.14)');
        crownAlb.addColorStop(1, 'rgba(255, 255, 255, 0.0)');
        aCtx.fillStyle = crownAlb;
        drawShieldScalePath(aCtx, cx, cy, rx * 0.65, ry * 0.72);
        aCtx.fill();

        const crownBump = bCtx.createRadialGradient(
          cx,
          cy + ry * 0.08,
          rx * 0.04,
          cx,
          cy + ry * 0.08,
          rx * 0.56
        );
        crownBump.addColorStop(0, 'rgba(255, 255, 255, 0.55)');
        crownBump.addColorStop(0.6, 'rgba(240, 240, 240, 0.22)');
        crownBump.addColorStop(1, 'rgba(200, 200, 200, 0.0)');
        bCtx.fillStyle = crownBump;
        drawShieldScalePath(bCtx, cx, cy, rx * 0.68, ry * 0.75);
        bCtx.fill();
      }
    }
  }

  bakeOrganicGrainAndWearLayer(aCtx, bCtx, size, 11, 28, 72, 42);

  return {
    albedoMap: configureTileableTexture(albCanvas, maxAnisotropy, true, 2, 2),
    bumpMap: configureTileableTexture(bumpCanvas, maxAnisotropy, false, 2, 2),
    normalMap: createNormalMapFromBumpCanvas(bumpCanvas, maxAnisotropy, 3.8, 2, 2),
  };
}

/**
 * 2. Beast Fur, Hide Pores & Chitin Striations
 */
function createChitinFurPattern(maxAnisotropy: number): SurfacePatternTextures {
  const size = 512;
  const albCanvas = document.createElement('canvas');
  albCanvas.width = size;
  albCanvas.height = size;
  const aCtx = albCanvas.getContext('2d')!;

  const bumpCanvas = document.createElement('canvas');
  bumpCanvas.width = size;
  bumpCanvas.height = size;
  const bCtx = bumpCanvas.getContext('2d')!;

  aCtx.fillStyle = '#e4e4e4';
  aCtx.fillRect(0, 0, size, size);
  bCtx.fillStyle = '#808080';
  bCtx.fillRect(0, 0, size, size);

  aCtx.lineCap = 'round';
  bCtx.lineCap = 'round';

  for (let i = 0; i < 560; i++) {
    const x = (i * 73) % size;
    const y = (i * 137) % size;
    const len = 26 + (i % 26);
    const dx = ((i % 5) - 2) * 2.4;
    const isLight = i % 2 === 0;

    aCtx.strokeStyle = isLight ? 'rgba(255,255,255,0.28)' : 'rgba(150,150,150,0.26)';
    aCtx.lineWidth = 2.8;
    aCtx.beginPath();
    aCtx.moveTo(x, y);
    aCtx.quadraticCurveTo(x + dx * 0.5, y + len * 0.5, x + dx, y + len);
    aCtx.stroke();

    bCtx.strokeStyle = isLight ? 'rgba(195,195,195,0.48)' : 'rgba(75,75,75,0.48)';
    bCtx.lineWidth = 2.8;
    bCtx.beginPath();
    bCtx.moveTo(x, y);
    bCtx.quadraticCurveTo(x + dx * 0.5, y + len * 0.5, x + dx, y + len);
    bCtx.stroke();
  }

  bakeOrganicGrainAndWearLayer(aCtx, bCtx, size, 27, 32, 78, 50);

  return {
    albedoMap: configureTileableTexture(albCanvas, maxAnisotropy, true, 2, 2),
    bumpMap: configureTileableTexture(bumpCanvas, maxAnisotropy, false, 2, 2),
    normalMap: createNormalMapFromBumpCanvas(bumpCanvas, maxAnisotropy, 3.4, 2, 2),
  };
}

/**
 * 3. Brushed Damascus Forged Alloy + Battle Pitting & Wear Scratches
 */
function createDamascusPlatePattern(maxAnisotropy: number): SurfacePatternTextures {
  const size = 512;
  const albCanvas = document.createElement('canvas');
  albCanvas.width = size;
  albCanvas.height = size;
  const aCtx = albCanvas.getContext('2d')!;

  const bumpCanvas = document.createElement('canvas');
  bumpCanvas.width = size;
  bumpCanvas.height = size;
  const bCtx = bumpCanvas.getContext('2d')!;

  aCtx.fillStyle = '#e6e6e6';
  aCtx.fillRect(0, 0, size, size);
  bCtx.fillStyle = '#808080';
  bCtx.fillRect(0, 0, size, size);

  for (let y = 0; y < size; y += 10) {
    aCtx.strokeStyle = y % 20 === 0 ? 'rgba(255,255,255,0.28)' : 'rgba(175,175,175,0.26)';
    aCtx.lineWidth = 3;
    bCtx.strokeStyle = y % 20 === 0 ? 'rgba(185,185,185,0.42)' : 'rgba(90,90,90,0.42)';
    bCtx.lineWidth = 3;

    aCtx.beginPath();
    bCtx.beginPath();
    for (let x = 0; x <= size; x += 16) {
      const wy = y + Math.sin((x / size) * Math.PI * 2) * 8;
      if (x === 0) {
        aCtx.moveTo(x, wy);
        bCtx.moveTo(x, wy);
      } else {
        aCtx.lineTo(x, wy);
        bCtx.lineTo(x, wy);
      }
    }
    aCtx.stroke();
    bCtx.stroke();
  }

  bakeOrganicGrainAndWearLayer(aCtx, bCtx, size, 43, 28, 76, 65);

  return {
    albedoMap: configureTileableTexture(albCanvas, maxAnisotropy, true, 2, 2),
    bumpMap: configureTileableTexture(bumpCanvas, maxAnisotropy, false, 2, 2),
    normalMap: createNormalMapFromBumpCanvas(bumpCanvas, maxAnisotropy, 3.0, 2, 2),
  };
}

/**
 * 4. Volcanic Obsidian Veins + Mineral Grain & Fracture Micro-Relief
 */
function createRunicObsidianPattern(maxAnisotropy: number): SurfacePatternTextures {
  const size = 512;
  const albCanvas = document.createElement('canvas');
  albCanvas.width = size;
  albCanvas.height = size;
  const aCtx = albCanvas.getContext('2d')!;

  const bumpCanvas = document.createElement('canvas');
  bumpCanvas.width = size;
  bumpCanvas.height = size;
  const bCtx = bumpCanvas.getContext('2d')!;

  aCtx.fillStyle = '#e2e2e2';
  aCtx.fillRect(0, 0, size, size);
  bCtx.fillStyle = '#808080';
  bCtx.fillRect(0, 0, size, size);

  for (let i = 0; i < 42; i++) {
    const cx = (i * 97) % size;
    const cy = (i * 151) % size;
    const rad = 36 + (i % 44);
    const aGrad = aCtx.createRadialGradient(cx, cy, 4, cx, cy, rad);
    aGrad.addColorStop(0, 'rgba(255,255,255,0.26)');
    aGrad.addColorStop(0.6, 'rgba(195,195,195,0.16)');
    aGrad.addColorStop(1, 'rgba(226,226,226,0)');
    aCtx.fillStyle = aGrad;
    aCtx.beginPath();
    aCtx.arc(cx, cy, rad, 0, Math.PI * 2);
    aCtx.fill();

    const bGrad = bCtx.createRadialGradient(cx, cy, 4, cx, cy, rad);
    bGrad.addColorStop(0, 'rgba(190,190,190,0.45)');
    bGrad.addColorStop(0.65, 'rgba(85,85,85,0.35)');
    bGrad.addColorStop(1, 'rgba(128,128,128,0)');
    bCtx.fillStyle = bGrad;
    bCtx.beginPath();
    bCtx.arc(cx, cy, rad, 0, Math.PI * 2);
    bCtx.fill();
  }

  bakeOrganicGrainAndWearLayer(aCtx, bCtx, size, 61, 32, 82, 55);

  return {
    albedoMap: configureTileableTexture(albCanvas, maxAnisotropy, true, 2, 2),
    bumpMap: configureTileableTexture(bumpCanvas, maxAnisotropy, false, 2, 2),
    normalMap: createNormalMapFromBumpCanvas(bumpCanvas, maxAnisotropy, 3.5, 2, 2),
  };
}

/**
 * 0. Dense Multi-Layer Wolf / Beast Pelage (Overlapping Fur Clumps + 3,000 Directional Guard Hairs)
 */
function createWolfFurPattern(maxAnisotropy: number): SurfacePatternTextures {
  const size = 512;
  const albCanvas = document.createElement('canvas');
  albCanvas.width = size;
  albCanvas.height = size;
  const aCtx = albCanvas.getContext('2d')!;

  const bumpCanvas = document.createElement('canvas');
  bumpCanvas.width = size;
  bumpCanvas.height = size;
  const bCtx = bumpCanvas.getContext('2d')!;

  // Base undercoat tone
  aCtx.fillStyle = '#cfcfcf';
  aCtx.fillRect(0, 0, size, size);
  bCtx.fillStyle = '#686868';
  bCtx.fillRect(0, 0, size, size);

  // Layer 1: Overlapping V-shaped fur tuft locks (gives macro-volume to pelage)
  const tuftRows = 14;
  const tuftCols = 12;
  const cellW = size / tuftCols;
  const cellH = size / tuftRows;

  for (let r = -1; r <= tuftRows; r++) {
    for (let c = -1; c <= tuftCols; c++) {
      const jitterX = (hash2D(c + 10, r + 10, 71) - 0.5) * cellW * 0.45;
      const jitterY = (hash2D(c + 20, r + 20, 89) - 0.5) * cellH * 0.35;
      const cx = c * cellW + (r % 2) * (cellW * 0.5) + cellW * 0.5 + jitterX;
      const cy = r * cellH + cellH * 0.5 + jitterY;
      const w = cellW * 0.68;
      const h = cellH * 1.35;

      const aGrad = aCtx.createLinearGradient(cx, cy - h * 0.45, cx, cy + h * 0.65);
      aGrad.addColorStop(0, 'rgba(95, 95, 95, 0.55)');    // Deep root shadow
      aGrad.addColorStop(0.55, 'rgba(220, 220, 220, 0.5)'); // Mid fur lock
      aGrad.addColorStop(1, 'rgba(255, 255, 255, 0.72)');   // Guard hair tip highlight

      aCtx.fillStyle = aGrad;
      aCtx.beginPath();
      aCtx.moveTo(cx - w * 0.5, cy - h * 0.35);
      aCtx.quadraticCurveTo(cx, cy + h * 0.75, cx + w * 0.5, cy - h * 0.35);
      aCtx.closePath();
      aCtx.fill();

      const bGrad = bCtx.createLinearGradient(cx, cy - h * 0.45, cx, cy + h * 0.65);
      bGrad.addColorStop(0, 'rgba(35, 35, 35, 0.65)');
      bGrad.addColorStop(0.6, 'rgba(165, 165, 165, 0.65)');
      bGrad.addColorStop(1, 'rgba(245, 245, 245, 0.85)');

      bCtx.fillStyle = bGrad;
      bCtx.beginPath();
      bCtx.moveTo(cx - w * 0.5, cy - h * 0.35);
      bCtx.quadraticCurveTo(cx, cy + h * 0.75, cx + w * 0.5, cy - h * 0.35);
      bCtx.closePath();
      bCtx.fill();
    }
  }

  // Layer 2: 2,600 Fine Tapered Guard Hair Strands with Seamless Tile Wrapping
  aCtx.lineCap = 'round';
  bCtx.lineCap = 'round';

  const drawWrappedStrand = (
    ox: number,
    oy: number,
    ctrlDx: number,
    ctrlDy: number,
    endDx: number,
    endDy: number,
    albStyle: string,
    bumpStyle: string,
    width: number
  ) => {
    for (const wx of [-size, 0, size]) {
      for (const wy of [-size, 0, size]) {
        const sx = ox + wx;
        const sy = oy + wy;
        if (sx < -60 || sx > size + 60 || sy < -60 || sy > size + 60) continue;

        aCtx.strokeStyle = albStyle;
        aCtx.lineWidth = width;
        aCtx.beginPath();
        aCtx.moveTo(sx, sy);
        aCtx.quadraticCurveTo(sx + ctrlDx, sy + ctrlDy, sx + endDx, sy + endDy);
        aCtx.stroke();

        bCtx.strokeStyle = bumpStyle;
        bCtx.lineWidth = width * 1.15;
        bCtx.beginPath();
        bCtx.moveTo(sx, sy);
        bCtx.quadraticCurveTo(sx + ctrlDx, sy + ctrlDy, sx + endDx, sy + endDy);
        bCtx.stroke();
      }
    }
  };

  for (let i = 0; i < 2200; i++) {
    const x = (i * 79 + (i % 7) * 19) % size;
    const y = (i * 139 + (i % 11) * 23) % size;
    const wave = Math.sin((x / size) * Math.PI * 4 + (y / size) * Math.PI * 2);
    const len = 18 + (i % 24);
    const sway = wave * 5.5 + ((i % 5) - 2) * 2.1;
    const isGuardHighlight = i % 3 === 0;
    const isDarkUndercoat = i % 4 === 1;

    const albCol = isGuardHighlight
      ? 'rgba(255, 255, 255, 0.42)'
      : isDarkUndercoat
      ? 'rgba(85, 85, 85, 0.38)'
      : 'rgba(205, 205, 205, 0.34)';

    const bumpCol = isGuardHighlight
      ? 'rgba(245, 245, 245, 0.62)'
      : isDarkUndercoat
      ? 'rgba(40, 40, 40, 0.58)'
      : 'rgba(165, 165, 165, 0.48)';

    const w = isGuardHighlight ? 1.65 : 2.2;
    drawWrappedStrand(x, y, sway * 0.45, len * 0.5, sway, len, albCol, bumpCol, w);
  }

  bakeOrganicGrainAndWearLayer(aCtx, bCtx, size, 19, 24, 64, 24);

  return {
    albedoMap: configureTileableTexture(albCanvas, maxAnisotropy, true, 3, 3),
    bumpMap: configureTileableTexture(bumpCanvas, maxAnisotropy, false, 3, 3),
    normalMap: createNormalMapFromBumpCanvas(bumpCanvas, maxAnisotropy, 3.6, 3, 3),
  };
}

export function createProceduralTextureSet(maxAnisotropy: number): ProceduralTextureSet {
  const wolf_fur = createWolfFurPattern(maxAnisotropy);
  const scales = createScalesPattern(maxAnisotropy);
  const chitin_fur = createChitinFurPattern(maxAnisotropy);
  const damascus_plate = createDamascusPlatePattern(maxAnisotropy);
  const runic_obsidian = createRunicObsidianPattern(maxAnisotropy);

  // Smooth Multi-Octave Weathering & Roughness Map (256x256 fBm — zero blocky fillRects!)
  const roughCanvas = document.createElement('canvas');
  roughCanvas.width = 256;
  roughCanvas.height = 256;
  const rCtx = roughCanvas.getContext('2d')!;
  const rImg = rCtx.createImageData(256, 256);
  const rData = rImg.data;

  for (let y = 0; y < 256; y++) {
    const v = y / 256;
    for (let x = 0; x < 256; x++) {
      const u = x / 256;
      const n1 = tileableSmoothNoise(u, v, 6, 91);
      const n2 = tileableSmoothNoise(u, v, 18, 107);
      const n3 = tileableSmoothNoise(u, v, 42, 131);
      const val = Math.min(255, Math.max(90, Math.round((0.55 + (n1 * 0.45 + n2 * 0.35 + n3 * 0.2 - 0.5) * 0.75) * 255)));
      const idx = (y * 256 + x) * 4;
      rData[idx] = val;
      rData[idx + 1] = val;
      rData[idx + 2] = val;
      rData[idx + 3] = 255;
    }
  }
  rCtx.putImageData(rImg, 0, 0);

  const armorRoughnessMap = configureTileableTexture(roughCanvas, maxAnisotropy, false, 2, 2);

  // Tactical Forge Pedestal Dial & Runic Circle (512x512)
  const pedCanvas = document.createElement('canvas');
  pedCanvas.width = 512;
  pedCanvas.height = 512;
  const pCtx = pedCanvas.getContext('2d')!;
  const cx = 256;
  const cy = 256;

  pCtx.strokeStyle = 'rgba(245, 158, 11, 0.45)';
  pCtx.lineWidth = 2;
  pCtx.beginPath();
  pCtx.arc(cx, cy, 230, 0, Math.PI * 2);
  pCtx.stroke();

  pCtx.strokeStyle = 'rgba(148, 163, 184, 0.25)';
  pCtx.lineWidth = 1.5;
  pCtx.beginPath();
  pCtx.arc(cx, cy, 190, 0, Math.PI * 2);
  pCtx.stroke();

  pCtx.beginPath();
  pCtx.arc(cx, cy, 130, 0, Math.PI * 2);
  pCtx.stroke();

  for (let deg = 0; deg < 360; deg += 10) {
    const rad = (deg * Math.PI) / 180;
    const inner = deg % 30 === 0 ? 212 : 220;
    pCtx.strokeStyle =
      deg % 30 === 0 ? 'rgba(245, 158, 11, 0.65)' : 'rgba(148, 163, 184, 0.28)';
    pCtx.lineWidth = deg % 30 === 0 ? 2 : 1;
    pCtx.beginPath();
    pCtx.moveTo(cx + Math.cos(rad) * inner, cy + Math.sin(rad) * inner);
    pCtx.lineTo(cx + Math.cos(rad) * 230, cy + Math.sin(rad) * 230);
    pCtx.stroke();
  }

  const pedestalRunesMap = new THREE.CanvasTexture(pedCanvas);
  pedestalRunesMap.colorSpace = THREE.SRGBColorSpace;
  pedestalRunesMap.anisotropy = maxAnisotropy;

  // Soft Radial Contact Shadow Map (128x128)
  const shadowCanvas = document.createElement('canvas');
  shadowCanvas.width = 128;
  shadowCanvas.height = 128;
  const sCtx = shadowCanvas.getContext('2d')!;
  const grad = sCtx.createRadialGradient(64, 64, 6, 64, 64, 60);
  grad.addColorStop(0, 'rgba(0, 0, 0, 0.85)');
  grad.addColorStop(0.5, 'rgba(0, 0, 0, 0.42)');
  grad.addColorStop(1, 'rgba(0, 0, 0, 0.0)');
  sCtx.fillStyle = grad;
  sCtx.fillRect(0, 0, 128, 128);

  const contactShadowMap = new THREE.CanvasTexture(shadowCanvas);
  contactShadowMap.colorSpace = THREE.SRGBColorSpace;

  // Soft Billowing Volcanic Smoke Sprite Map (64x64)
  const smokeCanvas = document.createElement('canvas');
  smokeCanvas.width = 64;
  smokeCanvas.height = 64;
  const smCtx = smokeCanvas.getContext('2d')!;
  const smGrad = smCtx.createRadialGradient(32, 32, 2, 32, 32, 30);
  smGrad.addColorStop(0, 'rgba(255, 255, 255, 0.92)');
  smGrad.addColorStop(0.38, 'rgba(235, 225, 215, 0.62)');
  smGrad.addColorStop(0.72, 'rgba(180, 165, 155, 0.22)');
  smGrad.addColorStop(1, 'rgba(0, 0, 0, 0.0)');
  smCtx.fillStyle = smGrad;
  smCtx.fillRect(0, 0, 64, 64);
  const particleSmokeMap = new THREE.CanvasTexture(smokeCanvas);
  particleSmokeMap.colorSpace = THREE.SRGBColorSpace;

  // Crisp Starburst Arcane Spark Sprite Map (64x64)
  const sparkCanvas = document.createElement('canvas');
  sparkCanvas.width = 64;
  sparkCanvas.height = 64;
  const spCtx = sparkCanvas.getContext('2d')!;
  const spGrad = spCtx.createRadialGradient(32, 32, 1, 32, 32, 30);
  spGrad.addColorStop(0, 'rgba(255, 255, 255, 1.0)');
  spGrad.addColorStop(0.22, 'rgba(255, 255, 255, 0.85)');
  spGrad.addColorStop(0.55, 'rgba(255, 255, 255, 0.28)');
  spGrad.addColorStop(1, 'rgba(255, 255, 255, 0.0)');
  spCtx.fillStyle = spGrad;
  spCtx.fillRect(0, 0, 64, 64);

  // Subtle 4-point flare cross for arcane sparks
  spCtx.strokeStyle = 'rgba(255, 255, 255, 0.75)';
  spCtx.lineWidth = 2;
  spCtx.beginPath();
  spCtx.moveTo(32, 6);
  spCtx.lineTo(32, 58);
  spCtx.moveTo(6, 32);
  spCtx.lineTo(58, 32);
  spCtx.stroke();

  const particleSparkMap = new THREE.CanvasTexture(sparkCanvas);
  particleSparkMap.colorSpace = THREE.SRGBColorSpace;

  // Volumetric Turbulent Fire Plume Texture (256x256 tileable along X & scrolling along Y)
  const fireCanvas = document.createElement('canvas');
  fireCanvas.width = 256;
  fireCanvas.height = 256;
  const fCtx = fireCanvas.getContext('2d')!;
  const fImg = fCtx.createImageData(256, 256);
  const fData = fImg.data;

  // Frost Blizzard Plume Texture (256x256)
  const frostCanvas = document.createElement('canvas');
  frostCanvas.width = 256;
  frostCanvas.height = 256;
  const frCtx = frostCanvas.getContext('2d')!;
  const frImg = frCtx.createImageData(256, 256);
  const frData = frImg.data;

  for (let y = 0; y < 256; y++) {
    const v = y / 256; // 0 at mouth base, 1 at plume tip
    for (let x = 0; x < 256; x++) {
      const u = x / 256;
      // Multi-octave cellular flame tongues stretched along V
      const n1 = tileableSmoothNoise(u, v, 5, 301);
      const n2 = tileableSmoothNoise(u, v, 11, 317);
      const n3 = tileableSmoothNoise(u, v, 24, 349);
      const n4 = tileableSmoothNoise(u, v, 48, 389);
      const turb = n1 * 0.42 + n2 * 0.31 + n3 * 0.18 + n4 * 0.09;

      // Longitudinal flame filaments
      const filament = Math.pow(Math.max(0, Math.sin((u * 6.0 + (n1 - 0.5) * 1.4) * Math.PI * 2) * 0.5 + 0.5), 1.4);
      const heat = Math.min(1, Math.max(0, turb * 0.78 + filament * 0.36));

      // Blackbody fire palette: white-yellow core -> bright orange -> deep crimson -> dark smoke
      let r = 0;
      let g = 0;
      let b = 0;
      if (heat > 0.76) {
        const t = (heat - 0.76) / 0.24;
        r = 255;
        g = Math.round(225 + t * 30);
        b = Math.round(110 + t * 145);
      } else if (heat > 0.46) {
        const t = (heat - 0.46) / 0.30;
        r = 255;
        g = Math.round(95 + t * 130);
        b = Math.round(8 + t * 102);
      } else if (heat > 0.20) {
        const t = (heat - 0.20) / 0.26;
        r = Math.round(135 + t * 120);
        g = Math.round(15 + t * 80);
        b = Math.round(2 + t * 6);
      } else {
        const t = heat / 0.20;
        r = Math.round(t * 135);
        g = Math.round(t * 15);
        b = 0;
      }

      const alpha = Math.round(Math.pow(Math.max(0, (heat - 0.12) / 0.88), 1.15) * 255);
      const idx = (y * 256 + x) * 4;
      fData[idx] = r;
      fData[idx + 1] = g;
      fData[idx + 2] = b;
      fData[idx + 3] = alpha;

      // Glacial Frost Stream: crystalline white-cyan core -> deep boreal blue
      const frostIntensity = Math.min(1, Math.max(0, turb * 0.75 + filament * 0.38));
      frData[idx] = Math.round(85 + frostIntensity * 170);
      frData[idx + 1] = Math.round(195 + frostIntensity * 60);
      frData[idx + 2] = 255;
      frData[idx + 3] = Math.round(Math.pow(Math.max(0, (frostIntensity - 0.14) / 0.86), 1.1) * 255);
    }
  }
  fCtx.putImageData(fImg, 0, 0);
  frCtx.putImageData(frImg, 0, 0);

  const fireBreathMap = configureTileableTexture(fireCanvas, maxAnisotropy, true, 1, 1);
  const frostBreathMap = configureTileableTexture(frostCanvas, maxAnisotropy, true, 1, 1);

  // Organic Billowing Fireball / Flame Tongue Sprite (64x64)
  const flameSpriteCanvas = document.createElement('canvas');
  flameSpriteCanvas.width = 64;
  flameSpriteCanvas.height = 64;
  const fsCtx = flameSpriteCanvas.getContext('2d')!;
  const fsImg = fsCtx.createImageData(64, 64);
  const fsData = fsImg.data;
  for (let y = 0; y < 64; y++) {
    const dy = (y - 32) / 30;
    for (let x = 0; x < 64; x++) {
      const dx = (x - 32) / 30;
      const dist = Math.hypot(dx, dy);
      const ang = Math.atan2(dy, dx);
      // Organic multi-lobe flame edge perturbation
      const lobe =
        Math.sin(ang * 3 + 0.7) * 0.08 +
        Math.cos(ang * 5 - 1.2) * 0.06 +
        Math.sin(ang * 8 + 2.1) * 0.04;
      const effDist = dist + lobe * dist;
      const falloff = Math.max(0, 1 - effDist);
      const core = Math.pow(falloff, 1.35);
      const idx = (y * 64 + x) * 4;
      fsData[idx] = 255;
      fsData[idx + 1] = Math.round(215 + Math.pow(falloff, 0.6) * 40);
      fsData[idx + 2] = Math.round(160 + Math.pow(falloff, 0.4) * 95);
      fsData[idx + 3] = Math.round(core * 255);
    }
  }
  fsCtx.putImageData(fsImg, 0, 0);
  const fireFlameSpriteMap = new THREE.CanvasTexture(flameSpriteCanvas);
  fireFlameSpriteMap.colorSpace = THREE.SRGBColorSpace;

  return {
    patterns: {
      wolf_fur,
      scales,
      chitin_fur,
      damascus_plate,
      runic_obsidian,
    },
    armorBumpMap: damascus_plate.bumpMap,
    armorNormalMap: damascus_plate.normalMap,
    armorRoughnessMap,
    pedestalRunesMap,
    contactShadowMap,
    particleSmokeMap,
    particleSparkMap,
    fireBreathMap,
    frostBreathMap,
    fireFlameSpriteMap,
    dispose: () => {
      for (const p of Object.values({
        wolf_fur,
        scales,
        chitin_fur,
        damascus_plate,
        runic_obsidian,
      })) {
        p.albedoMap.dispose();
        p.bumpMap.dispose();
        p.normalMap.dispose();
      }
      armorRoughnessMap.dispose();
      pedestalRunesMap.dispose();
      contactShadowMap.dispose();
      particleSmokeMap.dispose();
      particleSparkMap.dispose();
      fireBreathMap.dispose();
      frostBreathMap.dispose();
      fireFlameSpriteMap.dispose();
    },
  };
}

