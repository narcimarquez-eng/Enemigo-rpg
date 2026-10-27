import * as THREE from 'three';
import { TexturePatternId } from '../types/enemy';

export interface SurfacePatternTextures {
  albedoMap: THREE.CanvasTexture;
  bumpMap: THREE.CanvasTexture;
}

export interface ProceduralTextureSet {
  patterns: Record<Exclude<TexturePatternId, 'none'>, SurfacePatternTextures>;
  armorBumpMap: THREE.CanvasTexture;
  armorRoughnessMap: THREE.CanvasTexture;
  pedestalRunesMap: THREE.CanvasTexture;
  contactShadowMap: THREE.CanvasTexture;
  dispose: () => void;
}

function configureTileableTexture(
  canvas: HTMLCanvasElement,
  maxAnisotropy: number,
  isColor: boolean,
  repeatX = 3,
  repeatY = 3
): THREE.CanvasTexture {
  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = isColor ? THREE.SRGBColorSpace : THREE.NoColorSpace;
  tex.wrapS = THREE.RepeatWrapping;
  tex.wrapT = THREE.RepeatWrapping;
  tex.repeat.set(repeatX, repeatY);
  tex.anisotropy = maxAnisotropy;
  return tex;
}

/**
 * 1. Dragon & Serpent Overlapping Scales (Albedo + Bump)
 */
function createScalesPattern(maxAnisotropy: number): SurfacePatternTextures {
  const size = 256;
  const albCanvas = document.createElement('canvas');
  albCanvas.width = size;
  albCanvas.height = size;
  const aCtx = albCanvas.getContext('2d')!;

  const bumpCanvas = document.createElement('canvas');
  bumpCanvas.width = size;
  bumpCanvas.height = size;
  const bCtx = bumpCanvas.getContext('2d')!;

  aCtx.fillStyle = '#525252';
  aCtx.fillRect(0, 0, size, size);
  bCtx.fillStyle = '#262626';
  bCtx.fillRect(0, 0, size, size);

  const rows = 8;
  const cols = 8;
  const cellW = size / cols;
  const cellH = size / rows;

  for (let r = -1; r <= rows; r++) {
    for (let c = -1; c <= cols; c++) {
      const offsetX = (r % 2) * (cellW * 0.5);
      const cx = c * cellW + offsetX + cellW * 0.5;
      const cy = r * cellH + cellH * 0.5;
      const rad = cellW * 0.58;

      // Albedo scale gradient (tints cleanly with material.color)
      const aGrad = aCtx.createRadialGradient(cx, cy - rad * 0.25, rad * 0.1, cx, cy, rad);
      aGrad.addColorStop(0, '#ffffff');
      aGrad.addColorStop(0.68, '#d4d4d4');
      aGrad.addColorStop(0.92, '#686868');
      aGrad.addColorStop(1, '#2a2a2a');

      aCtx.fillStyle = aGrad;
      aCtx.beginPath();
      aCtx.arc(cx, cy, rad, 0, Math.PI * 2);
      aCtx.fill();

      // Central scale keel ridge
      aCtx.strokeStyle = 'rgba(255,255,255,0.45)';
      aCtx.lineWidth = 2;
      aCtx.beginPath();
      aCtx.moveTo(cx, cy - rad * 0.45);
      aCtx.lineTo(cx, cy + rad * 0.45);
      aCtx.stroke();

      // Bump height map
      const bGrad = bCtx.createRadialGradient(cx, cy - rad * 0.2, rad * 0.08, cx, cy, rad);
      bGrad.addColorStop(0, '#ffffff');
      bGrad.addColorStop(0.75, '#a8a8a8');
      bGrad.addColorStop(1, '#181818');

      bCtx.fillStyle = bGrad;
      bCtx.beginPath();
      bCtx.arc(cx, cy, rad, 0, Math.PI * 2);
      bCtx.fill();
    }
  }

  return {
    albedoMap: configureTileableTexture(albCanvas, maxAnisotropy, true, 3, 3),
    bumpMap: configureTileableTexture(bumpCanvas, maxAnisotropy, false, 3, 3),
  };
}

/**
 * 2. Beast Fur & Chitin Hide Striations (Albedo + Bump)
 */
function createChitinFurPattern(maxAnisotropy: number): SurfacePatternTextures {
  const size = 256;
  const albCanvas = document.createElement('canvas');
  albCanvas.width = size;
  albCanvas.height = size;
  const aCtx = albCanvas.getContext('2d')!;

  const bumpCanvas = document.createElement('canvas');
  bumpCanvas.width = size;
  bumpCanvas.height = size;
  const bCtx = bumpCanvas.getContext('2d')!;

  aCtx.fillStyle = '#b8b8b8';
  aCtx.fillRect(0, 0, size, size);
  bCtx.fillStyle = '#707070';
  bCtx.fillRect(0, 0, size, size);

  // Layered directional fur / chitin fibers
  for (let i = 0; i < 650; i++) {
    const x = (i * 53) % size;
    const y = (i * 97) % size;
    const len = 14 + (i % 18);
    const dx = ((i % 7) - 3) * 1.4;
    const isLight = i % 2 === 0;

    aCtx.strokeStyle = isLight ? 'rgba(255,255,255,0.55)' : 'rgba(55,55,55,0.55)';
    aCtx.lineWidth = 1.6;
    aCtx.beginPath();
    aCtx.moveTo(x, y);
    aCtx.lineTo(x + dx, y + len);
    aCtx.stroke();

    bCtx.strokeStyle = isLight ? '#e8e8e8' : '#303030';
    bCtx.lineWidth = 1.8;
    bCtx.beginPath();
    bCtx.moveTo(x, y);
    bCtx.lineTo(x + dx, y + len);
    bCtx.stroke();
  }

  return {
    albedoMap: configureTileableTexture(albCanvas, maxAnisotropy, true, 3, 3),
    bumpMap: configureTileableTexture(bumpCanvas, maxAnisotropy, false, 3, 3),
  };
}

/**
 * 3. Damascus Forged Plate & Riveted Armor (Albedo + Bump)
 */
function createDamascusPlatePattern(maxAnisotropy: number): SurfacePatternTextures {
  const size = 256;
  const albCanvas = document.createElement('canvas');
  albCanvas.width = size;
  albCanvas.height = size;
  const aCtx = albCanvas.getContext('2d')!;

  const bumpCanvas = document.createElement('canvas');
  bumpCanvas.width = size;
  bumpCanvas.height = size;
  const bCtx = bumpCanvas.getContext('2d')!;

  aCtx.fillStyle = '#cccccc';
  aCtx.fillRect(0, 0, size, size);
  bCtx.fillStyle = '#808080';
  bCtx.fillRect(0, 0, size, size);

  // Damascus wave bands
  for (let y = 0; y < size; y += 6) {
    aCtx.strokeStyle = y % 12 === 0 ? 'rgba(255,255,255,0.42)' : 'rgba(70,70,70,0.38)';
    aCtx.lineWidth = 2;
    aCtx.beginPath();
    for (let x = 0; x <= size; x += 8) {
      const wy = y + Math.sin((x / size) * Math.PI * 4) * 5;
      if (x === 0) aCtx.moveTo(x, wy);
      else aCtx.lineTo(x, wy);
    }
    aCtx.stroke();
  }

  // Armor plate seams & rivets
  aCtx.strokeStyle = '#404040';
  aCtx.lineWidth = 2.5;
  bCtx.strokeStyle = '#282828';
  bCtx.lineWidth = 3;

  for (let r = 0; r < 2; r++) {
    for (let c = 0; c < 2; c++) {
      const px = c * 128 + 6;
      const py = r * 128 + 6;
      aCtx.strokeRect(px, py, 116, 116);
      bCtx.strokeRect(px, py, 116, 116);

      // Corner rivets
      for (const [rx, ry] of [
        [px + 10, py + 10],
        [px + 106, py + 10],
        [px + 10, py + 106],
        [px + 106, py + 106],
      ]) {
        aCtx.fillStyle = '#ffffff';
        aCtx.beginPath();
        aCtx.arc(rx, ry, 3.2, 0, Math.PI * 2);
        aCtx.fill();

        bCtx.fillStyle = '#ffffff';
        bCtx.beginPath();
        bCtx.arc(rx, ry, 3.5, 0, Math.PI * 2);
        bCtx.fill();
      }
    }
  }

  return {
    albedoMap: configureTileableTexture(albCanvas, maxAnisotropy, true, 2, 2),
    bumpMap: configureTileableTexture(bumpCanvas, maxAnisotropy, false, 2, 2),
  };
}

/**
 * 4. Runic Obsidian & Fractured Volcanic Stone (Albedo + Bump)
 */
function createRunicObsidianPattern(maxAnisotropy: number): SurfacePatternTextures {
  const size = 256;
  const albCanvas = document.createElement('canvas');
  albCanvas.width = size;
  albCanvas.height = size;
  const aCtx = albCanvas.getContext('2d')!;

  const bumpCanvas = document.createElement('canvas');
  bumpCanvas.width = size;
  bumpCanvas.height = size;
  const bCtx = bumpCanvas.getContext('2d')!;

  aCtx.fillStyle = '#b5b5b5';
  aCtx.fillRect(0, 0, size, size);
  bCtx.fillStyle = '#888888';
  bCtx.fillRect(0, 0, size, size);

  // Polygonal stone facets
  for (let i = 0; i < 60; i++) {
    const x = (i * 67) % size;
    const y = (i * 103) % size;
    const w = 24 + (i % 28);
    const h = 20 + ((i * 3) % 28);
    const shade = 145 + (i % 95);
    aCtx.fillStyle = `rgb(${shade},${shade},${shade})`;
    aCtx.fillRect(x, y, w, h);
    aCtx.strokeStyle = '#383838';
    aCtx.lineWidth = 2;
    aCtx.strokeRect(x, y, w, h);

    bCtx.fillStyle = `rgb(${shade},${shade},${shade})`;
    bCtx.fillRect(x, y, w, h);
    bCtx.strokeStyle = '#181818';
    bCtx.lineWidth = 2.5;
    bCtx.strokeRect(x, y, w, h);
  }

  return {
    albedoMap: configureTileableTexture(albCanvas, maxAnisotropy, true, 2, 2),
    bumpMap: configureTileableTexture(bumpCanvas, maxAnisotropy, false, 2, 2),
  };
}

export function createProceduralTextureSet(maxAnisotropy: number): ProceduralTextureSet {
  const scales = createScalesPattern(maxAnisotropy);
  const chitin_fur = createChitinFurPattern(maxAnisotropy);
  const damascus_plate = createDamascusPlatePattern(maxAnisotropy);
  const runic_obsidian = createRunicObsidianPattern(maxAnisotropy);

  // Fast Armor Roughness Variation Map (128x128)
  const roughCanvas = document.createElement('canvas');
  roughCanvas.width = 128;
  roughCanvas.height = 128;
  const rCtx = roughCanvas.getContext('2d')!;
  rCtx.fillStyle = '#999999';
  rCtx.fillRect(0, 0, 128, 128);

  for (let i = 0; i < 120; i++) {
    const x = (i * 37) % 128;
    const y = (i * 67) % 128;
    const v = 115 + (i % 85);
    rCtx.fillStyle = `rgba(${v},${v},${v},0.32)`;
    rCtx.fillRect(x, y, 8 + (i % 12), 8 + (i % 12));
  }

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

  return {
    patterns: {
      scales,
      chitin_fur,
      damascus_plate,
      runic_obsidian,
    },
    armorBumpMap: damascus_plate.bumpMap,
    armorRoughnessMap,
    pedestalRunesMap,
    contactShadowMap,
    dispose: () => {
      for (const p of Object.values({ scales, chitin_fur, damascus_plate, runic_obsidian })) {
        p.albedoMap.dispose();
        p.bumpMap.dispose();
      }
      armorRoughnessMap.dispose();
      pedestalRunesMap.dispose();
      contactShadowMap.dispose();
    },
  };
}
