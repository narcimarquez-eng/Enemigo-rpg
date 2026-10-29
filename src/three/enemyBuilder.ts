import * as THREE from 'three';
import {
  AnimationState,
  AuraParticleEffectId,
  EnemyConfig,
  HairStyleId,
  PartInspectionInfo,
  getEffectiveFurDensity,
  getEffectiveHairLength,
  getEffectiveHairStyle,
  getEffectiveMeshDensity,
  getEffectiveMuscleDefinition,
  getEffectiveTexturePattern,
  getEffectiveWeapon,
  isAnimalEnemy,
  isUndeadArchetype,
} from '../types/enemy';
import {
  DRAGON_SCALE_GRID_COLS,
  DRAGON_SCALE_GRID_ROWS,
  DRAGON_SCALE_UV_REPEAT_X,
  DRAGON_SCALE_UV_REPEAT_Y,
  ProceduralTextureSet,
  evaluateDragonScaleUVField,
  getDragonScaleCellHash,
} from './proceduralTextures';

// Pre-allocated scratch helpers for zero-GC in the animation loop
const _targetPos = new THREE.Vector3();
const _pelvisAnimOffset = new THREE.Vector3();
const _zeroVec = new THREE.Vector3();
const _mouthWorldPos = new THREE.Vector3();
const _mouthLocalPos = new THREE.Vector3();
const _mouthForwardDir = new THREE.Vector3();
const _headWorldQuat = new THREE.Quaternion();
const _rootWorldQuat = new THREE.Quaternion();
const _dummyObj = new THREE.Object3D();
const _shardColor = new THREE.Color();
const _anchorLocalPos = new THREE.Vector3();
const _particleBaseColor = new THREE.Color();
const _particleAccentColor = new THREE.Color();
const _particleColorOut = new THREE.Color();

export interface BuiltEnemyRig {
  rootGroup: THREE.Group;
  exportableGroup: THREE.Group; // Clean enemy hierarchy for .GLB export (no gizmos)
  hitboxGroup: THREE.Group;     // Toggleable hitboxes, SkeletonHelper, aggro rings & socket axes
  rootBone: THREE.Bone;
  skeleton: THREE.Skeleton;
  skinnedMeshes: THREE.SkinnedMesh[];
  skinnedMeshGroup: THREE.Group;
  explodedSourceMeshes: THREE.Mesh[];
  animations: THREE.AnimationClip[];
  materials: {
    primaryArmorMat: THREE.MeshPhysicalMaterial;
    secondaryTrimMat: THREE.MeshPhysicalMaterial;
    underbodyMat: THREE.MeshStandardMaterial;
    hairStrandMat: THREE.MeshPhysicalMaterial;
    emissiveRuneMat: THREE.MeshStandardMaterial;
    crystalCoreMat: THREE.MeshPhysicalMaterial;
    slashTrailMat: THREE.MeshStandardMaterial;
  };
  parts: {
    pelvisGroup: THREE.Bone;
    torsoGroup: THREE.Bone;
    neckGroup: THREE.Bone;
    headGroup: THREE.Bone;
    jawGroup: THREE.Bone;
    extraHeads: THREE.Bone[];
    leftShoulderGroup: THREE.Bone;
    rightShoulderGroup: THREE.Bone;
    leftArmGroup: THREE.Bone;
    rightArmGroup: THREE.Bone;
    leftElbowGroup: THREE.Bone;
    rightElbowGroup: THREE.Bone;
    leftLegGroup: THREE.Bone;
    rightLegGroup: THREE.Bone;
    leftKneeGroup: THREE.Bone;
    rightKneeGroup: THREE.Bone;
    leftWingGroup: THREE.Bone;
    rightWingGroup: THREE.Bone;
    tailGroup: THREE.Bone;
    tailTipBone: THREE.Bone;
    dorsalGroup: THREE.Bone;
    weaponGroup: THREE.Group;
    offhandGroup: THREE.Group;
    coreMesh: THREE.Mesh;
    breathConeMesh: THREE.Mesh;
    slashTrailMesh: THREE.Mesh;
    arrowProjectileMesh: THREE.Group;
    auraInstancedMesh: THREE.InstancedMesh | null;
  };
  sockets: {
    head: THREE.Object3D;
    weaponR: THREE.Object3D;
    shieldL: THREE.Object3D;
    core: THREE.Object3D;
  };
  explodableParts: THREE.Object3D[];
  shardData: Array<{
    angle: number;
    radius: number;
    yOffset: number;
    speed: number;
    scale: number;
  }>;
  auraParticleSystem: {
    points: THREE.Points;
    smokeMat: THREE.PointsMaterial;
    sparkMat: THREE.PointsMaterial;
    activeMode: AuraParticleEffectId;
    pool: Array<{
      age: number;
      lifetime: number;
      anchorIndex: number;
      vx: number;
      vy: number;
      vz: number;
      seed: number;
    }>;
  };
  mouthBreathSystem: {
    points: THREE.Points;
    fireMat: THREE.PointsMaterial;
    frostMat: THREE.PointsMaterial;
    breathLight: THREE.PointLight;
    fireBreathMap: THREE.CanvasTexture;
    frostBreathMap: THREE.CanvasTexture;
    pool: Array<{
      age: number;
      lifetime: number;
      vx: number;
      vy: number;
      vz: number;
      seed: number;
      isSpark: boolean;
    }>;
  };
  metrics: {
    totalVertices: number;
    totalTriangles: number;
    partCount: number;
    boneCount?: number;
    clipCount?: number;
  };
  dispose: () => void;
}

/**
 * Bakes smooth Ambient Occlusion + Dorsal-Ventral anatomical vertex color gradients
 * + longitudinal muscle-belly striations and fur-clump normal displacement
 * while preserving 100% smooth analytical normals and UV seams.
 */
function sculptAndShadeGeometry<T extends THREE.BufferGeometry>(
  geo: T,
  muscleAmplitude = 0.014,
  freq = 5.5,
  aoDarkenBottom = 0.24,
  furClumpAmplitude = 0.0
): T {
  const pos = geo.getAttribute('position');
  if (!pos) return geo;

  if (!geo.getAttribute('normal')) {
    geo.computeVertexNormals();
  }
  const norm = geo.getAttribute('normal');

  geo.computeBoundingBox();
  const bbox = geo.boundingBox!;
  const widthSpan = Math.max(0.001, bbox.max.x - bbox.min.x);
  const heightSpan = Math.max(0.001, bbox.max.y - bbox.min.y);
  const depthSpan = Math.max(0.001, bbox.max.z - bbox.min.z);
  const midX = (bbox.min.x + bbox.max.x) * 0.5;
  const midZ = (bbox.min.z + bbox.max.z) * 0.5;

  const count = pos.count;
  const colors = new Float32Array(count * 3);

  for (let i = 0; i < count; i++) {
    const x = pos.getX(i);
    const y = pos.getY(i);
    const z = pos.getZ(i);

    const normY = (y - bbox.min.y) / heightSpan; // 0 at bottom, 1 at top
    const normZ = (z - bbox.min.z) / depthSpan;  // 0 at back, 1 at front

    // Longitudinal muscle-belly envelope (0 at joint ends, 1 at mid-belly)
    const bellyEnv = Math.sin(normY * Math.PI);
    const theta = Math.atan2(x - midX, z - midZ);

    // Anatomical muscle bundles & inter-muscular sulcus grooves
    const primaryBelly = Math.cos(theta * 2.0) * 0.62 + Math.sin(theta * 3.0) * 0.38;
    const fiberStriation = Math.sin(theta * 6.0 + normY * freq) * 0.28;
    const muscleWave = bellyEnv * (primaryBelly + fiberStriation);

    // Directional fur clump ripples for pelage-covered animals
    const furWave =
      furClumpAmplitude > 0.0001
        ? bellyEnv *
          (Math.sin(x * 22.0 + y * 14.0) * Math.cos(z * 22.0 - y * 11.0) * 0.65 +
            Math.sin(theta * 9.0 - normY * 12.0) * 0.35)
        : 0.0;

    const totalDisp = muscleWave * muscleAmplitude + furWave * furClumpAmplitude;
    if (norm && Math.abs(totalDisp) > 1e-5 && widthSpan > 0.05) {
      const nx = norm.getX(i);
      const ny = norm.getY(i);
      const nz = norm.getZ(i);
      pos.setXYZ(i, x + nx * totalDisp, y + ny * totalDisp * 0.45, z + nz * totalDisp);
    }

    // Smooth Anatomical Vertex Color Shading (Dorsal highlight, Ventral/Joint AO shadow + Muscle Sulcus Grooves)
    const ridgeHighlight = bellyEnv * 0.1;
    const muscleTone = muscleAmplitude > 0.001 ? muscleWave * 0.11 : 0.0;
    const furTipTone = furClumpAmplitude > 0.001 ? furWave * 0.12 : 0.0;

    const aoFactor =
      1.0 -
      (1.0 - normY) * aoDarkenBottom +
      ridgeHighlight +
      (normZ - 0.5) * 0.05 +
      muscleTone +
      furTipTone;

    // Subtle warm/cool shift from top to bottom for organic skin/fur depth
    const r = Math.min(1.08, Math.max(0.52, aoFactor * 1.01));
    const g = Math.min(1.06, Math.max(0.50, aoFactor * 0.99));
    const b = Math.min(1.06, Math.max(0.48, aoFactor * 0.97));

    colors[i * 3] = r;
    colors[i * 3 + 1] = g;
    colors[i * 3 + 2] = b;
  }

  if (muscleAmplitude > 0.001 || furClumpAmplitude > 0.001) {
    pos.needsUpdate = true;
    geo.computeVertexNormals();
  }

  geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  return geo;
}

/**
 * Builds a 3D Multi-Strand Hair Tuft Cluster (7 curved, 3-segment tapered hair blades per tuft)
 * with root-to-tip vertex color gradient and soft bent normals so hair locks look like real hair strands.
 */
function createFurLockGeometry(
  baseRadius = 0.065,
  length = 0.26,
  curveBack = 0.08,
  strandCount = 7
): THREE.BufferGeometry {
  const vertsPerStrand = 7; // 3 segments: [rootL, rootR], [mid1L, mid1R], [mid2L, mid2R], [tip]
  const trisPerStrand = 5;
  const totalVerts = strandCount * vertsPerStrand;
  const totalIndices = strandCount * trisPerStrand * 3;

  const positions = new Float32Array(totalVerts * 3);
  const normals = new Float32Array(totalVerts * 3);
  const uvs = new Float32Array(totalVerts * 2);
  const colors = new Float32Array(totalVerts * 3);
  const indices = new Uint16Array(totalIndices);

  const levels = [0.0, 0.34, 0.70, 1.0];
  const widthFactors = [1.0, 0.78, 0.44, 0.0];

  for (let s = 0; s < strandCount; s++) {
    const vBase = s * vertsPerStrand;
    const iBase = s * trisPerStrand * 3;

    const ang = s * 2.3999632; // Golden angle distribution
    const radialFrac = s === 0 ? 0.0 : Math.sqrt(s / strandCount);
    const ox = Math.cos(ang) * baseRadius * 0.48 * radialFrac;
    const oz = Math.sin(ang) * baseRadius * 0.48 * radialFrac;

    const splayX = Math.cos(ang) * baseRadius * 0.65 * radialFrac;
    const splayZ = Math.sin(ang) * baseRadius * 0.45 * radialFrac;
    const bladeW = baseRadius * (0.28 + (s % 3) * 0.05);
    const bladeLen = length * (0.84 + ((s * 37) % 10) * 0.028);

    // Cross-axis of the hair ribbon rotated slightly per strand for 360-degree visibility
    const rx = Math.cos(ang * 0.5 + 0.35);
    const rz = -Math.sin(ang * 0.5 + 0.35);

    const shadeVar = 0.92 + ((s * 17) % 9) * 0.02;

    for (let segIdx = 0; segIdx < 4; segIdx++) {
      const t = levels[segIdx];
      const y = t * bladeLen;
      const bend = t * t * (curveBack * (0.85 + 0.12 * (s % 3)));
      const wave = Math.sin(t * Math.PI * 1.4 + s) * 0.014 * bladeLen;

      const cx = ox + splayX * t + wave;
      const cy = y;
      const cz = oz + splayZ * t - bend;

      // Soft bent hair normal pointing upward and outward
      let nx = splayX * 0.8 + rx * 0.15;
      let ny = 0.65;
      let nz = -0.55 + splayZ * 0.5;
      const nLen = Math.hypot(nx, ny, nz) || 1;
      nx /= nLen;
      ny /= nLen;
      nz /= nLen;

      // Root-to-tip hair pigment luminance (dark follicle root -> warm shaft -> sunlit tip)
      const lum = (0.36 + Math.pow(t, 0.85) * 0.98) * shadeVar;
      const cr = Math.min(1.48, lum * (0.98 + t * 0.08));
      const cg = Math.min(1.42, lum * (0.96 + t * 0.04));
      const cb = Math.min(1.36, lum * (0.92 - t * 0.02));

      if (segIdx < 3) {
        const w = bladeW * widthFactors[segIdx];
        const vL = vBase + segIdx * 2;
        const vR = vL + 1;

        positions[vL * 3] = cx - rx * w;
        positions[vL * 3 + 1] = cy;
        positions[vL * 3 + 2] = cz - rz * w;

        positions[vR * 3] = cx + rx * w;
        positions[vR * 3 + 1] = cy;
        positions[vR * 3 + 2] = cz + rz * w;

        normals[vL * 3] = nx;
        normals[vL * 3 + 1] = ny;
        normals[vL * 3 + 2] = nz;
        normals[vR * 3] = nx;
        normals[vR * 3 + 1] = ny;
        normals[vR * 3 + 2] = nz;

        uvs[vL * 2] = 0;
        uvs[vL * 2 + 1] = t;
        uvs[vR * 2] = 1;
        uvs[vR * 2 + 1] = t;

        colors[vL * 3] = cr;
        colors[vL * 3 + 1] = cg;
        colors[vL * 3 + 2] = cb;
        colors[vR * 3] = cr;
        colors[vR * 3 + 1] = cg;
        colors[vR * 3 + 2] = cb;
      } else {
        const vTip = vBase + 6;
        positions[vTip * 3] = cx;
        positions[vTip * 3 + 1] = cy;
        positions[vTip * 3 + 2] = cz;

        normals[vTip * 3] = nx;
        normals[vTip * 3 + 1] = ny;
        normals[vTip * 3 + 2] = nz;

        uvs[vTip * 2] = 0.5;
        uvs[vTip * 2 + 1] = 1.0;

        colors[vTip * 3] = cr;
        colors[vTip * 3 + 1] = cg;
        colors[vTip * 3 + 2] = cb;
      }
    }

    // 5 triangles per 3-segment tapered hair blade
    // Seg 0 -> Seg 1 quad
    indices[iBase + 0] = vBase + 0;
    indices[iBase + 1] = vBase + 1;
    indices[iBase + 2] = vBase + 2;
    indices[iBase + 3] = vBase + 1;
    indices[iBase + 4] = vBase + 3;
    indices[iBase + 5] = vBase + 2;
    // Seg 1 -> Seg 2 quad
    indices[iBase + 6] = vBase + 2;
    indices[iBase + 7] = vBase + 3;
    indices[iBase + 8] = vBase + 4;
    indices[iBase + 9] = vBase + 3;
    indices[iBase + 10] = vBase + 5;
    indices[iBase + 11] = vBase + 4;
    // Seg 2 -> Tip triangle
    indices[iBase + 12] = vBase + 4;
    indices[iBase + 13] = vBase + 5;
    indices[iBase + 14] = vBase + 6;
  }

  const geo = new THREE.BufferGeometry();
  geo.setIndex(new THREE.BufferAttribute(indices, 1));
  geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geo.setAttribute('normal', new THREE.BufferAttribute(normals, 3));
  geo.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
  geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  return geo;
}

interface SkinnedHairCoatParams {
  bodyGeos: THREE.BufferGeometry[];
  hairStyle: Exclude<HairStyleId, 'auto'>;
  furDensity: number;
  hairLength: number;
  lodMult: number;
  torsoBulk: number;
  heightScale: number;
  pelvisY: number;
  headBones: THREE.Bone[];
  neckIdx: number;
  torsoIdx: number;
  pelvisIdx: number;
  tailBaseIdx: number;
  tailTipIdx: number;
  boneToIndex: Map<THREE.Bone, number>;
  boneWorldPos: THREE.Vector3[];
}

/**
 * Generates thousands of 3-segment curved, tapered 3D hair strands across the creature's skinned body
 * AND/OR long flowing cranial hair & wild mane locks, with exact 4-bone skinIndex/skinWeight inheritance,
 * anatomical pelt combing, gravity droop, and root-to-tip vertex color shading.
 */
function buildSkinned3DHairCoatGeometry(
  params: SkinnedHairCoatParams
): THREE.BufferGeometry | null {
  const {
    bodyGeos,
    hairStyle,
    furDensity,
    hairLength,
    lodMult,
    torsoBulk,
    heightScale,
    pelvisY,
    headBones,
    neckIdx,
    torsoIdx,
    tailBaseIdx,
    tailTipIdx,
    boneToIndex,
    boneWorldPos,
  } = params;

  const hasBodyCoat =
    hairStyle === 'full_fur_coat' ||
    hairStyle === 'wild_mane_fur' ||
    hairStyle === 'dorsal_mohawk';
  const hasCranialHair =
    hairStyle === 'long_flowing_hair' || hairStyle === 'wild_mane_fur';

  // 1. Collect valid body triangles for surface-sprouted 3D hair strands
  interface FollicleTri {
    geoIdx: number;
    i0: number;
    i1: number;
    i2: number;
  }
  const tris: FollicleTri[] = [];
  const cdf: number[] = [];
  let totalWeightedArea = 0;

  const neckPos = boneWorldPos[neckIdx] || new THREE.Vector3(0, pelvisY + 0.5, 0.2);
  const headPositions = headBones.map((hb) => {
    const idx = boneToIndex.get(hb);
    return idx !== undefined && boneWorldPos[idx]
      ? boneWorldPos[idx]
      : new THREE.Vector3().setFromMatrixPosition(hb.matrixWorld);
  });

  if (hasBodyCoat && bodyGeos.length > 0) {
    for (let gIdx = 0; gIdx < bodyGeos.length; gIdx++) {
      const g = bodyGeos[gIdx];
      const posAttr = g.getAttribute('position');
      const normAttr = g.getAttribute('normal');
      const idxAttr = g.getIndex();
      if (!posAttr || !normAttr) continue;

      const indexArray = idxAttr ? idxAttr.array : null;
      const triCount = indexArray ? Math.floor(indexArray.length / 3) : Math.floor(posAttr.count / 3);

      for (let t = 0; t < triCount; t++) {
        const i0 = indexArray ? indexArray[t * 3] : t * 3;
        const i1 = indexArray ? indexArray[t * 3 + 1] : t * 3 + 1;
        const i2 = indexArray ? indexArray[t * 3 + 2] : t * 3 + 2;

        const x0 = posAttr.getX(i0), y0 = posAttr.getY(i0), z0 = posAttr.getZ(i0);
        const x1 = posAttr.getX(i1), y1 = posAttr.getY(i1), z1 = posAttr.getZ(i1);
        const x2 = posAttr.getX(i2), y2 = posAttr.getY(i2), z2 = posAttr.getZ(i2);

        const cy = (y0 + y1 + y2) / 3;
        // Skip ground-contact paw/foot soles so hair never pokes into the floor
        if (cy < 0.095) continue;

        const cx = (x0 + x1 + x2) / 3;
        const cz = (z0 + z1 + z2) / 3;
        const ny = (normAttr.getY(i0) + normAttr.getY(i1) + normAttr.getY(i2)) / 3;

        const ux = x1 - x0, uy = y1 - y0, uz = z1 - z0;
        const vx = x2 - x0, vy = y2 - y0, vz = z2 - z0;
        const crossX = uy * vz - uz * vy;
        const crossY = uz * vx - ux * vz;
        const crossZ = ux * vy - uy * vx;
        const area = 0.5 * Math.hypot(crossX, crossY, crossZ);
        if (area < 1e-6) continue;

        let weight = area;
        if (hairStyle === 'dorsal_mohawk') {
          const isDorsalCrest = Math.abs(cx) < 0.21 * torsoBulk && ny > 0.14;
          weight *= isDorsalCrest ? 9.5 : 0.35;
        } else if (hairStyle === 'wild_mane_fur') {
          const distNeck = Math.hypot(cx - neckPos.x, cy - neckPos.y, cz - neckPos.z);
          if (distNeck < 0.52 * torsoBulk) {
            weight *= 3.4;
          }
        }

        totalWeightedArea += weight;
        tris.push({ geoIdx: gIdx, i0, i1, i2 });
        cdf.push(totalWeightedArea);
      }
    }
  }

  const numBodyStrands =
    hasBodyCoat && tris.length > 0
      ? Math.min(
          4200,
          Math.max(
            240,
            Math.round(
              (hairStyle === 'dorsal_mohawk' ? 1900 : 2800) *
                Math.max(0.2, furDensity) *
                Math.min(1.45, lodMult)
            )
          )
        )
      : 0;

  const strandsPerHead = hasCranialHair
    ? Math.min(
        720,
        Math.max(
          160,
          Math.round(
            (hairStyle === 'long_flowing_hair' ? 520 : 360) *
              Math.max(0.3, furDensity) *
              Math.min(1.4, lodMult)
          )
        )
      )
    : 0;
  const numCranialStrands = strandsPerHead * headBones.length;

  const totalStrands = numBodyStrands + numCranialStrands;
  if (totalStrands === 0) return null;

  const vertsPerStrand = 7; // 3 segments: [rootL, rootR], [mid1L, mid1R], [mid2L, mid2R], [tip]
  const trisPerStrand = 5;
  const totalVerts = totalStrands * vertsPerStrand;
  const totalIndices = totalStrands * trisPerStrand * 3;

  const positions = new Float32Array(totalVerts * 3);
  const normals = new Float32Array(totalVerts * 3);
  const uvs = new Float32Array(totalVerts * 2);
  const colors = new Float32Array(totalVerts * 3);
  const skinIndices = new Uint16Array(totalVerts * 4);
  const skinWeights = new Float32Array(totalVerts * 4);
  const indices =
    totalVerts > 65535 ? new Uint32Array(totalIndices) : new Uint16Array(totalIndices);

  const levels = [0.0, 0.33, 0.68, 1.0];
  const widthFactors = [1.0, 0.78, 0.44, 0.0];

  const writeStrandRibbon = (
    strandIdx: number,
    ptsX: number[],
    ptsY: number[],
    ptsZ: number[],
    rx: number,
    ry: number,
    rz: number,
    nx: number,
    ny: number,
    nz: number,
    baseW: number,
    shadeVar: number,
    sIdx4: ArrayLike<number>,
    sW4: ArrayLike<number>,
    tipSkinIdx4?: ArrayLike<number>,
    tipSkinW4?: ArrayLike<number>
  ) => {
    const vBase = strandIdx * vertsPerStrand;
    const iBase = strandIdx * trisPerStrand * 3;

    for (let segIdx = 0; segIdx < 4; segIdx++) {
      const t = levels[segIdx];
      const cx = ptsX[segIdx];
      const cy = ptsY[segIdx];
      const cz = ptsZ[segIdx];

      // Root-to-tip hair pigment luminance (dark shadowed root -> rich mid shaft -> sunlit tip)
      const lum = (0.34 + Math.pow(t, 0.82) * 1.06) * shadeVar;
      const cr = Math.min(1.48, lum * (0.97 + t * 0.09));
      const cg = Math.min(1.44, lum * (0.95 + t * 0.05));
      const cb = Math.min(1.38, lum * (0.91 - t * 0.01));

      // Blend skinning from root bone to torso/shoulder bone for long cranial hair strands
      const useTipSkin = tipSkinIdx4 && tipSkinW4 && segIdx >= 2;
      const activeIdx = useTipSkin ? tipSkinIdx4 : sIdx4;
      const activeW = useTipSkin ? tipSkinW4 : sW4;

      if (segIdx < 3) {
        const w = baseW * widthFactors[segIdx];
        const vL = vBase + segIdx * 2;
        const vR = vL + 1;

        positions[vL * 3] = cx - rx * w;
        positions[vL * 3 + 1] = cy - ry * w;
        positions[vL * 3 + 2] = cz - rz * w;

        positions[vR * 3] = cx + rx * w;
        positions[vR * 3 + 1] = cy + ry * w;
        positions[vR * 3 + 2] = cz + rz * w;

        normals[vL * 3] = nx;
        normals[vL * 3 + 1] = ny;
        normals[vL * 3 + 2] = nz;
        normals[vR * 3] = nx;
        normals[vR * 3 + 1] = ny;
        normals[vR * 3 + 2] = nz;

        uvs[vL * 2] = 0;
        uvs[vL * 2 + 1] = t;
        uvs[vR * 2] = 1;
        uvs[vR * 2 + 1] = t;

        colors[vL * 3] = cr;
        colors[vL * 3 + 1] = cg;
        colors[vL * 3 + 2] = cb;
        colors[vR * 3] = cr;
        colors[vR * 3 + 1] = cg;
        colors[vR * 3 + 2] = cb;

        for (let k = 0; k < 4; k++) {
          skinIndices[vL * 4 + k] = activeIdx[k];
          skinWeights[vL * 4 + k] = activeW[k];
          skinIndices[vR * 4 + k] = activeIdx[k];
          skinWeights[vR * 4 + k] = activeW[k];
        }
      } else {
        const vTip = vBase + 6;
        positions[vTip * 3] = cx;
        positions[vTip * 3 + 1] = cy;
        positions[vTip * 3 + 2] = cz;

        normals[vTip * 3] = nx;
        normals[vTip * 3 + 1] = ny;
        normals[vTip * 3 + 2] = nz;

        uvs[vTip * 2] = 0.5;
        uvs[vTip * 2 + 1] = 1.0;

        colors[vTip * 3] = cr;
        colors[vTip * 3 + 1] = cg;
        colors[vTip * 3 + 2] = cb;

        for (let k = 0; k < 4; k++) {
          skinIndices[vTip * 4 + k] = activeIdx[k];
          skinWeights[vTip * 4 + k] = activeW[k];
        }
      }
    }

    indices[iBase + 0] = vBase + 0;
    indices[iBase + 1] = vBase + 1;
    indices[iBase + 2] = vBase + 2;
    indices[iBase + 3] = vBase + 1;
    indices[iBase + 4] = vBase + 3;
    indices[iBase + 5] = vBase + 2;

    indices[iBase + 6] = vBase + 2;
    indices[iBase + 7] = vBase + 3;
    indices[iBase + 8] = vBase + 4;
    indices[iBase + 9] = vBase + 3;
    indices[iBase + 10] = vBase + 5;
    indices[iBase + 11] = vBase + 4;

    indices[iBase + 12] = vBase + 4;
    indices[iBase + 13] = vBase + 5;
    indices[iBase + 14] = vBase + 6;
  };

  const ptsX = [0, 0, 0, 0];
  const ptsY = [0, 0, 0, 0];
  const ptsZ = [0, 0, 0, 0];
  const sIdx4 = [0, 0, 0, 0];
  const sW4 = [1, 0, 0, 0];

  // 2. Generate Surface-Sprouted 3D Body Pelage Hair Strands
  let strandCursor = 0;
  for (let s = 0; s < numBodyStrands; s++) {
    // Stratified area sampling along CDF
    const targetArea = ((s + 0.5) / numBodyStrands) * totalWeightedArea;
    let lo = 0;
    let hi = cdf.length - 1;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (cdf[mid] < targetArea) lo = mid + 1;
      else hi = mid;
    }

    const tri = tris[lo];
    const g = bodyGeos[tri.geoIdx];
    const posAttr = g.getAttribute('position');
    const normAttr = g.getAttribute('normal');
    const skinIdxAttr = g.getAttribute('skinIndex');
    const skinWAttr = g.getAttribute('skinWeight');

    // Low-discrepancy barycentric coordinates inside triangle
    const r1 = ((s * 0.61803398875) % 1 + 1) % 1;
    const r2 = ((s * 0.38196601125 + 0.17) % 1 + 1) % 1;
    const sqrtR1 = Math.sqrt(r1);
    const bU = 1 - sqrtR1;
    const bV = sqrtR1 * (1 - r2);
    const bW = sqrtR1 * r2;

    const { i0, i1, i2 } = tri;
    const px = bU * posAttr.getX(i0) + bV * posAttr.getX(i1) + bW * posAttr.getX(i2);
    const py = bU * posAttr.getY(i0) + bV * posAttr.getY(i1) + bW * posAttr.getY(i2);
    const pz = bU * posAttr.getZ(i0) + bV * posAttr.getZ(i1) + bW * posAttr.getZ(i2);

    let nx = bU * normAttr.getX(i0) + bV * normAttr.getX(i1) + bW * normAttr.getX(i2);
    let ny = bU * normAttr.getY(i0) + bV * normAttr.getY(i1) + bW * normAttr.getY(i2);
    let nz = bU * normAttr.getZ(i0) + bV * normAttr.getZ(i1) + bW * normAttr.getZ(i2);
    const nLen = Math.hypot(nx, ny, nz) || 1;
    nx /= nLen;
    ny /= nLen;
    nz /= nLen;

    if (skinIdxAttr && skinWAttr) {
      sIdx4[0] = skinIdxAttr.getX(i0);
      sIdx4[1] = skinIdxAttr.getY(i0);
      sIdx4[2] = skinIdxAttr.getZ(i0);
      sIdx4[3] = skinIdxAttr.getW(i0);
      sW4[0] = skinWAttr.getX(i0);
      sW4[1] = skinWAttr.getY(i0);
      sW4[2] = skinWAttr.getZ(i0);
      sW4[3] = skinWAttr.getW(i0);
    }

    const domBone = sIdx4[0];
    const sideSign = px >= 0 ? 1 : -1;

    // Determine anatomical pelt comb direction & regional hair length multiplier
    let fx = 0.16 * sideSign;
    let fy = -0.36;
    let fz = -0.88;
    let zoneLenMult = 1.05;

    // Check proximity to any head (snout vs cheeks/crown)
    let minHeadDist = Infinity;
    let nearestHeadZ = 0;
    for (let h = 0; h < headPositions.length; h++) {
      const hp = headPositions[h];
      const d = Math.hypot(px - hp.x, py - hp.y, pz - hp.z);
      if (d < minHeadDist) {
        minHeadDist = d;
        nearestHeadZ = pz - hp.z;
      }
    }

    const distNeck = Math.hypot(px - neckPos.x, py - neckPos.y, pz - neckPos.z);

    if (domBone === tailBaseIdx || domBone === tailTipIdx) {
      // Bushy tail hair plume flowing toward tail tip
      zoneLenMult = 1.65;
      fx = 0.22 * sideSign;
      fy = -0.18;
      fz = -0.95;
    } else if (minHeadDist < 0.42 * torsoBulk) {
      if (nearestHeadZ > 0.18) {
        // Front muzzle / snout: short sleek fur combed backward so eyes & fangs stay clean
        zoneLenMult = 0.36;
        fx = 0.28 * sideSign;
        fy = -0.12;
        fz = -0.94;
      } else if (Math.abs(px) > 0.11 * torsoBulk) {
        // Cheek & jowl ruffs flaring outward and backward
        zoneLenMult = 1.42;
        fx = 0.56 * sideSign;
        fy = -0.32;
        fz = -0.76;
      } else {
        // Crown & occiput swept back
        zoneLenMult = 1.38;
        fx = 0.12 * sideSign;
        fy = 0.18;
        fz = -0.96;
      }
    } else if (distNeck < 0.48 * torsoBulk || domBone === neckIdx) {
      // Neck ruff, collar & chest mane
      zoneLenMult = hairStyle === 'wild_mane_fur' ? 2.15 : 1.52;
      fx = 0.38 * sideSign;
      fy = -0.24;
      fz = -0.88;
    } else if (Math.abs(px) < 0.18 * torsoBulk && ny > 0.22) {
      // Dorsal spine hackles / mohawk crest
      zoneLenMult = hairStyle === 'dorsal_mohawk' ? 2.45 : 1.45;
      fx = 0.08 * sideSign;
      fy = hairStyle === 'dorsal_mohawk' ? 0.68 : 0.28;
      fz = -0.88;
    } else if (py < pelvisY + 0.14 && Math.abs(px) > 0.14 * torsoBulk) {
      // Limbs (arms/forelegs & thighs/calves): feathering longer on rear edge
      zoneLenMult = nz < -0.06 ? 1.45 : 0.92;
      fx = 0.1 * sideSign;
      fy = -0.86;
      fz = -0.48;
    }

    // Project flow direction partially onto the skin tangent plane so hair lies naturally in overlapping layers
    const dotFN = fx * nx + fy * ny + fz * nz;
    let tx = fx - dotFN * nx * 0.72;
    let ty = fy - dotFN * ny * 0.72;
    let tz = fz - dotFN * nz * 0.72;
    const tLen = Math.hypot(tx, ty, tz) || 1;
    tx /= tLen;
    ty /= tLen;
    tz /= tLen;

    const hash1 = ((s * 173) % 100) / 100;
    const hash2 = ((s * 293) % 100) / 100;
    const strandLen =
      (0.135 + hash1 * 0.075) * hairLength * zoneLenMult * (0.8 + 0.2 * torsoBulk);
    const baseW = (0.018 + hash2 * 0.009) * Math.sqrt(torsoBulk);

    // Golden-angle twist around the hair axis for full 3D volumetric visibility from any camera angle
    const twistAng = s * 2.3999632;
    let rx = ny * tz - nz * ty;
    let ry = nz * tx - nx * tz;
    let rz = nx * ty - ny * tx;
    const rLen = Math.hypot(rx, ry, rz) || 1;
    rx /= rLen;
    ry /= rLen;
    rz /= rLen;

    const cosTw = Math.cos(twistAng);
    const sinTw = Math.sin(twistAng) * 0.55;
    let twistRx = rx * cosTw + nx * sinTw;
    let twistRy = ry * cosTw + ny * sinTw;
    let twistRz = rz * cosTw + nz * sinTw;
    const trLen = Math.hypot(twistRx, twistRy, twistRz) || 1;
    twistRx /= trLen;
    twistRy /= trLen;
    twistRz /= trLen;

    const curlX = Math.sin(s * 1.7) * 0.045;
    const curlZ = Math.cos(s * 2.3) * 0.045;

    // 4 control points along the 3-segment curved hair strand
    ptsX[0] = px - nx * 0.006;
    ptsY[0] = py - ny * 0.006;
    ptsZ[0] = pz - nz * 0.006;

    ptsX[1] = px + strandLen * (0.22 * nx + 0.26 * tx);
    ptsY[1] = py + strandLen * (0.22 * ny + 0.26 * ty);
    ptsZ[1] = pz + strandLen * (0.22 * nz + 0.26 * tz);

    ptsX[2] = px + strandLen * (0.30 * nx + 0.62 * tx + curlX);
    ptsY[2] = py + strandLen * (0.30 * ny + 0.62 * ty - 0.12);
    ptsZ[2] = pz + strandLen * (0.30 * nz + 0.62 * tz + curlZ);

    ptsX[3] = px + strandLen * (0.28 * nx + 0.96 * tx + curlX * 1.65);
    ptsY[3] = py + strandLen * (0.28 * ny + 0.96 * ty - 0.26);
    ptsZ[3] = pz + strandLen * (0.28 * nz + 0.96 * tz + curlZ * 1.65);

    // Bent normal combining skin normal and upward sheen normal for velvety fur shading
    let bnx = nx * 0.72 + tx * 0.28;
    let bny = ny * 0.72 + 0.35;
    let bnz = nz * 0.72 + tz * 0.28;
    const bnLen = Math.hypot(bnx, bny, bnz) || 1;
    bnx /= bnLen;
    bny /= bnLen;
    bnz /= bnLen;

    const shadeVar = 0.88 + ((s * 43) % 25) * 0.01;
    writeStrandRibbon(
      strandCursor++,
      ptsX,
      ptsY,
      ptsZ,
      twistRx,
      twistRy,
      twistRz,
      bnx,
      bny,
      bnz,
      baseW,
      shadeVar,
      sIdx4,
      sW4
    );
  }

  // 3. Generate Long Flowing Cranial Scalp Hair & Wild Mane Locks
  if (hasCranialHair && strandsPerHead > 0) {
    const localPt = new THREE.Vector3();
    const localNorm = new THREE.Vector3();
    const normalMat = new THREE.Matrix3();
    const rootSkinIdx = [0, neckIdx, torsoIdx, 0];
    const rootSkinW = [0.88, 0.12, 0, 0];
    const tipSkinIdx = [torsoIdx, neckIdx, 0, 0];
    const tipSkinW = [0.58, 0.24, 0.18, 0];

    for (let h = 0; h < headBones.length; h++) {
      const hBone = headBones[h];
      const hBoneIdx = boneToIndex.get(hBone) ?? neckIdx;
      rootSkinIdx[0] = hBoneIdx;
      tipSkinIdx[2] = hBoneIdx;
      normalMat.getNormalMatrix(hBone.matrixWorld);

      for (let k = 0; k < strandsPerHead; k++) {
        // Golden-ratio distribution across the scalp crown, temples, sides, and nape (avoiding front face)
        const u = (k + 0.5) / strandsPerHead;
        const v = ((k * 0.61803398875) % 1 + 1) % 1;

        // theta in [-0.78*PI, +0.78*PI] centered at back of head (-Z) so face (+Z) stays open
        const theta = (v - 0.5) * 2.0 * Math.PI * 0.76;
        // phi from nape (-0.25) to top crown (+1.35)
        const phi = -0.22 + u * 1.52;

        const cosPhi = Math.cos(phi);
        const sinPhi = Math.sin(phi);
        const sinTheta = Math.sin(theta);
        const cosTheta = -Math.cos(theta); // -1 at back of skull, +0.7 at temples

        // Cranium ellipsoid in headGroup local coordinates (centered at (0, 0.20, 0.02))
        const sx = sinTheta * cosPhi;
        const sy = sinPhi;
        const sz = cosTheta * cosPhi;

        localPt.set(sx * 0.21, 0.20 + sy * 0.21, 0.02 + sz * 0.24);
        localPt.applyMatrix4(hBone.matrixWorld);

        localNorm.set(sx, sy, sz).applyMatrix3(normalMat).normalize();
        const nx = localNorm.x;
        const ny = localNorm.y;
        const nz = localNorm.z;

        const px = localPt.x;
        const py = localPt.y;
        const pz = localPt.z;

        const sideSign = sx >= 0 ? 1 : -1;
        const isLongStyle = hairStyle === 'long_flowing_hair';
        const hash1 = ((k * 137) % 100) / 100;
        const hash2 = ((k * 251) % 100) / 100;

        const longLen =
          (isLongStyle ? 0.62 + hash1 * 0.44 : 0.36 + hash1 * 0.26) *
          hairLength *
          heightScale;
        const baseW = (isLongStyle ? 0.024 : 0.022) * (0.85 + hash2 * 0.35) * Math.sqrt(torsoBulk);

        // Wave & outward shoulder drape so long hair cascades over the shoulders & back
        const waveX = Math.sin(k * 1.9) * 0.065 + sideSign * Math.abs(sx) * 0.24;
        const waveZ = Math.cos(k * 1.3) * 0.055 - 0.38;

        ptsX[0] = px - nx * 0.008;
        ptsY[0] = py - ny * 0.008;
        ptsZ[0] = pz - nz * 0.008;

        // Seg 1: Crown volume arch outward & slightly up before cascading down
        ptsX[1] = px + longLen * (0.26 * nx + sideSign * 0.08);
        ptsY[1] = py + longLen * (0.18 * Math.max(0, ny) + 0.04);
        ptsZ[1] = pz + longLen * (0.22 * nz - 0.16);

        // Seg 2: Mid-cascade draping past neck & shoulders
        ptsX[2] = px + longLen * (0.34 * nx + waveX * 0.65);
        ptsY[2] = py + longLen * (-0.42 + 0.08 * ny);
        ptsZ[2] = pz + longLen * (0.24 * nz + waveZ * 0.72);

        // Seg 3: Flowing wavy hair tip over back/shoulders
        ptsX[3] = px + longLen * (0.36 * nx + waveX * 1.15);
        ptsY[3] = py + longLen * (-0.88 + 0.05 * ny);
        ptsZ[3] = pz + longLen * (0.24 * nz + waveZ * 1.15);

        const twistAng = k * 2.3999632;
        let rx = Math.cos(theta + twistAng * 0.35);
        let ry = 0.15 * Math.sin(twistAng);
        let rz = Math.sin(theta + twistAng * 0.35);
        const rLen = Math.hypot(rx, ry, rz) || 1;
        rx /= rLen;
        ry /= rLen;
        rz /= rLen;

        let bnx = nx * 0.68 + sideSign * 0.15;
        let bny = Math.max(0.25, ny * 0.68 + 0.35);
        let bnz = nz * 0.68 - 0.32;
        const bnLen = Math.hypot(bnx, bny, bnz) || 1;
        bnx /= bnLen;
        bny /= bnLen;
        bnz /= bnLen;

        const shadeVar = 0.88 + ((k * 59) % 24) * 0.011;
        writeStrandRibbon(
          strandCursor++,
          ptsX,
          ptsY,
          ptsZ,
          rx,
          ry,
          rz,
          bnx,
          bny,
          bnz,
          baseW,
          shadeVar,
          rootSkinIdx,
          rootSkinW,
          tipSkinIdx,
          tipSkinW
        );
      }
    }
  }

  const outGeo = new THREE.BufferGeometry();
  outGeo.setIndex(new THREE.BufferAttribute(indices, 1));
  outGeo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  outGeo.setAttribute('normal', new THREE.BufferAttribute(normals, 3));
  outGeo.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
  outGeo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  outGeo.setAttribute('skinIndex', new THREE.Uint16BufferAttribute(skinIndices, 4));
  outGeo.setAttribute('skinWeight', new THREE.Float32BufferAttribute(skinWeights, 4));
  return outGeo;
}

interface SkinnedDragonScaleReliefParams {
  bodyGeo: THREE.BufferGeometry;
  lodMult: number;
  torsoBulk: number;
  dorsalScale: number;
  textureRelief: number;
  isDragon: boolean;
  tailBaseIdx: number;
  tailTipIdx: number;
  neckIdx: number;
  boneWorldPos: THREE.Vector3[];
}

/**
 * Generates UV-synchronized 3D protruding dragon osteoderm shield scales across the creature's unified SkinnedMesh
 * using a Blender-style UV-to-Mesh rasterizer so every 3D protruding scale sits at the exact UV center (centerGX, centerGY),
 * orientation (dP/dUV), and dimensions of an `isProtruding` (hash > 0.62) scale drawn in `createScalesPattern`.
 */
function buildSkinned3DDragonScaleReliefGeometry(
  params: SkinnedDragonScaleReliefParams
): THREE.BufferGeometry | null {
  const {
    bodyGeo,
    lodMult,
    torsoBulk,
    dorsalScale,
    textureRelief,
    isDragon,
    tailBaseIdx,
    tailTipIdx,
     boneWorldPos,
  } = params;

  const posAttr = bodyGeo.getAttribute('position');
  const normAttr = bodyGeo.getAttribute('normal');
  const uvAttr = bodyGeo.getAttribute('uv');
  const skinIdxAttr = bodyGeo.getAttribute('skinIndex');
  const skinWAttr = bodyGeo.getAttribute('skinWeight');
  const idxAttr = bodyGeo.getIndex();
  if (!posAttr || !normAttr || !uvAttr || !skinIdxAttr || !skinWAttr) return null;

  const indexArray = idxAttr ? idxAttr.array : null;
  const triCount = indexArray
    ? Math.floor(indexArray.length / 3)
    : Math.floor(posAttr.count / 3);

  const cols = DRAGON_SCALE_GRID_COLS;
  const rows = DRAGON_SCALE_GRID_ROWS;
  const totalColsUV = DRAGON_SCALE_UV_REPEAT_X * cols; // 20
  const totalRowsUV = DRAGON_SCALE_UV_REPEAT_Y * rows; // 24

  // Exact scale cell radii in grid units for protruding scales (scaleMult = 1.1 in createScalesPattern)
  const rxGrid = 0.55 * 1.1; // 0.605
  const ryGrid = 0.62 * 1.1; // 0.682

  const reliefFactor = Math.max(0.35, textureRelief);
  const bulkFactor = Math.pow(Math.max(0.6, torsoBulk), 0.45);
  const dorsalFactor = 0.8 + 0.32 * Math.max(0.4, dorsalScale);
  const maxScales = Math.min(
    2200,
    Math.max(360, Math.round((isDragon ? 1450 : 980) * Math.min(1.4, lodMult)))
  );

  interface EmittedScale {
    px: number;
    py: number;
    pz: number;
    nx: number;
    ny: number;
    nz: number;
    fx: number;
    fy: number;
    fz: number;
    rx: number;
    ry: number;
    rz: number;
    uvSign: number;
    centerGX: number;
    centerGY: number;
    scaleHalfW: number;
    scaleHalfL: number;
    scaleLift: number;
    shadeVar: number;
    sIdx: [number, number, number, number];
    sW: [number, number, number, number];
  }

  const emitted: EmittedScale[] = [];
  // 3D spatial hash so shared triangle edges or overlapping limb junctions never duplicate a scale at the same 3D spot
  const occupiedCells = new Set<string>();
  const minCellSpacing = 0.029 * Math.max(0.78, Math.sqrt(torsoBulk));

  for (let t = 0; t < triCount; t++) {
    if (emitted.length >= maxScales) break;

    const i0 = indexArray ? indexArray[t * 3] : t * 3;
    const i1 = indexArray ? indexArray[t * 3 + 1] : t * 3 + 1;
    const i2 = indexArray ? indexArray[t * 3 + 2] : t * 3 + 2;

    const x0 = posAttr.getX(i0), y0 = posAttr.getY(i0), z0 = posAttr.getZ(i0);
    const x1 = posAttr.getX(i1), y1 = posAttr.getY(i1), z1 = posAttr.getZ(i1);
    const x2 = posAttr.getX(i2), y2 = posAttr.getY(i2), z2 = posAttr.getZ(i2);

    const cy = (y0 + y1 + y2) / 3;
    // Skip ground-contact claws/soles
    if (cy < 0.095) continue;

    const cx = (x0 + x1 + x2) / 3;
    const triNx = (normAttr.getX(i0) + normAttr.getX(i1) + normAttr.getX(i2)) / 3;
    const triNy = (normAttr.getY(i0) + normAttr.getY(i1) + normAttr.getY(i2)) / 3;

    // Skip flat downward-facing soft belly plates so osteoderm scales concentrate on back, flanks, neck, tail & limbs
    if (triNy < -0.38 && Math.abs(cx) < 0.25 * torsoBulk) continue;
    // Skip ultra-thin wing patagium membrane interiors on dragons
    if (isDragon && Math.abs(cx) > 0.64 * torsoBulk && cy > 1.05 && Math.abs(triNx) < 0.34) continue;

    const ux = x1 - x0, uy = y1 - y0, uz = z1 - z0;
    const vx = x2 - x0, vy = y2 - y0, vz = z2 - z0;
    const area =
      0.5 *
      Math.hypot(
        uy * vz - uz * vy,
        uz * vx - ux * vz,
        ux * vy - uy * vx
      );
    if (area < 1e-6) continue;

    // Convert triangle vertex UVs to continuous scale-grid space (gx in [0..20], gy in [0..24])
    const u0 = uvAttr.getX(i0), v0 = uvAttr.getY(i0);
    const u1 = uvAttr.getX(i1), v1 = uvAttr.getY(i1);
    const u2 = uvAttr.getX(i2), v2 = uvAttr.getY(i2);

    // Skip UV-seam borders and polar pinch caps where primitive meridians converge along axes
    if (
      v0 < 0.08 || v0 > 0.92 ||
      v1 < 0.08 || v1 > 0.92 ||
      v2 < 0.08 || v2 > 0.92 ||
      u0 < 0.03 || u0 > 0.97 ||
      u1 < 0.03 || u1 > 0.97 ||
      u2 < 0.03 || u2 > 0.97
    ) {
      continue;
    }

    const gx0 = u0 * totalColsUV;
    const gy0 = (1 - v0 * DRAGON_SCALE_UV_REPEAT_Y) * rows;
    const gx1 = u1 * totalColsUV;
    const gy1 = (1 - v1 * DRAGON_SCALE_UV_REPEAT_Y) * rows;
    const gx2 = u2 * totalColsUV;
    const gy2 = (1 - v2 * DRAGON_SCALE_UV_REPEAT_Y) * rows;

    const dgx1 = gx1 - gx0;
    const dgy1 = gy1 - gy0;
    const dgx2 = gx2 - gx0;
    const dgy2 = gy2 - gy0;

    // Skip UV-seam wrap-around triangles where UV jumps across the [0,1] boundary
    if (
      Math.abs(dgx1) > 5 ||
      Math.abs(dgx2) > 5 ||
      Math.abs(dgy1) > 5 ||
      Math.abs(dgy2) > 5
    ) {
      continue;
    }

    const det = dgx1 * dgy2 - dgx2 * dgy1;
    if (Math.abs(det) < 1e-3) continue;
    const invDet = 1 / det;

    // World-space UV Jacobian vectors: dP/d(gx) (scale width axis) and dP/d(gy) (scale root-to-tip axis)
    const dPdgxX = (dgy2 * ux - dgy1 * vx) * invDet;
    const dPdgxY = (dgy2 * uy - dgy1 * vy) * invDet;
    const dPdgxZ = (dgy2 * uz - dgy1 * vz) * invDet;

    const dPdgyX = (-dgx2 * ux + dgx1 * vx) * invDet;
    const dPdgyY = (-dgx2 * uy + dgx1 * vy) * invDet;
    const dPdgyZ = (-dgx2 * uz + dgx1 * vz) * invDet;

    const rawHalfW = Math.hypot(dPdgxX, dPdgxY, dPdgxZ) * rxGrid;
    const rawHalfL = Math.hypot(dPdgyX, dPdgyY, dPdgyZ) * ryGrid;
    // Skip pinched UV poles or stretched seam triangles
    if (
      rawHalfW < 0.009 ||
      rawHalfL < 0.009 ||
      rawHalfW > 0.075 * bulkFactor ||
      rawHalfL > 0.085 * bulkFactor ||
      rawHalfW / rawHalfL > 2.4 ||
      rawHalfL / rawHalfW > 2.4
    ) {
      continue;
    }

    const minR = Math.floor(Math.min(gy0, gy1, gy2) - 0.5);
    const maxR = Math.ceil(Math.max(gy0, gy1, gy2) - 0.5);
    const minGX = Math.min(gx0, gx1, gx2);
    const maxGX = Math.max(gx0, gx1, gx2);

    for (let r = minR; r <= maxR; r++) {
      const wrappedR = ((r % rows) + rows) % rows;
      const rowShift = (wrappedR % 2) * 0.5;
      const centerGY = r + 0.5;

      const minC = Math.floor(minGX - rowShift - 0.5);
      const maxC = Math.ceil(maxGX - rowShift - 0.5);

      for (let c = minC; c <= maxC; c++) {
        const wrappedC = ((c % cols) + cols) % cols;
        const hashVal = getDragonScaleCellHash(wrappedR, wrappedC);

        // Only emit 3D raised scales for the exact `isProtruding` (h > 0.62) scales drawn in createScalesPattern!
        if (hashVal <= 0.62) continue;

        const centerGX = c + rowShift + 0.5;

        // Exact 2D UV barycentric coordinates (b0, b1, b2) of the drawn scale center (centerGX, centerGY)
        const relGX = centerGX - gx0;
        const relGY = centerGY - gy0;
        const b1 = (relGX * dgy2 - dgx2 * relGY) * invDet;
        const b2 = (dgx1 * relGY - relGX * dgy1) * invDet;
        const b0 = 1 - b1 - b2;

        // Check if this triangle strictly contains the center of scale (r, c)
        if (b0 < 0.02 || b1 < 0.02 || b2 < 0.02) continue;

        const px = b0 * x0 + b1 * x1 + b2 * x2;
        const py = b0 * y0 + b1 * y1 + b2 * y2;
        const pz = b0 * z0 + b1 * z1 + b2 * z2;
        if (py < 0.095) continue;

        // Skip interior overlapping sub-mesh triangles (e.g., inner half of muscle spheres buried inside torso/limbs)
        const domBoneIdx = skinIdxAttr.getX(i0);
        const boneOrigin = boneWorldPos[domBoneIdx];
        if (boneOrigin) {
          const rxBone = px - boneOrigin.x;
          const ryBone = py - boneOrigin.y;
          const rzBone = pz - boneOrigin.z;
          const rDist = Math.hypot(rxBone, ryBone, rzBone);
          if (rDist > 0.04) {
            const outwardDot = (rxBone * triNx + ryBone * triNy + rzBone * ((normAttr.getZ(i0) + normAttr.getZ(i1) + normAttr.getZ(i2)) / 3)) / rDist;
            if (outwardDot < 0.15) continue;
          }
        }

        const latX = Math.floor(px / minCellSpacing);
        const latY = Math.floor(py / minCellSpacing);
        const latZ = Math.floor(pz / minCellSpacing);
        const cellKey = `${latX},${latY},${latZ}`;
        if (occupiedCells.has(cellKey)) continue;
        occupiedCells.add(cellKey);

        let nx = b0 * normAttr.getX(i0) + b1 * normAttr.getX(i1) + b2 * normAttr.getX(i2);
        let ny = b0 * normAttr.getY(i0) + b1 * normAttr.getY(i1) + b2 * normAttr.getY(i2);
        let nz = b0 * normAttr.getZ(i0) + b1 * normAttr.getZ(i1) + b2 * normAttr.getZ(i2);
        const nLen = Math.hypot(nx, ny, nz) || 1;
        nx /= nLen;
        ny /= nLen;
        nz /= nLen;

        // Project the UV vertical axis dP/d(gy) (which points toward the drawn shield scale's tip!) onto the tangent plane
        const dotFN = dPdgyX * nx + dPdgyY * ny + dPdgyZ * nz;
        let fx = dPdgyX - dotFN * nx;
        let fy = dPdgyY - dotFN * ny;
        let fz = dPdgyZ - dotFN * nz;
        let fLen = Math.hypot(fx, fy, fz);

        if (fLen < 1e-5) {
          const domBone = skinIdxAttr.getX(i0);
          const sideSign = px >= 0 ? 1 : -1;
          fx = 0.14 * sideSign;
          fy = domBone === tailBaseIdx || domBone === tailTipIdx ? -0.14 : -0.35;
          fz = -0.92;
          const d2 = fx * nx + fy * ny + fz * nz;
          fx -= d2 * nx;
          fy -= d2 * ny;
          fz -= d2 * nz;
          fLen = Math.hypot(fx, fy, fz) || 1;
        }
        fx /= fLen;
        fy /= fLen;
        fz /= fLen;

        // Right tangent vector R = normalize(F x N) guarantees strictly outward-facing (+N) CCW winding for all top facets
        let rx = fy * nz - fz * ny;
        let ry = fz * nx - fx * nz;
        let rz = fx * ny - fy * nx;
        const rLen = Math.hypot(rx, ry, rz) || 1;
        rx /= rLen;
        ry /= rLen;
        rz /= rLen;

        // Preserve UV horizontal handedness so left/right vertices sample the exact left/right side of the drawn scale
        const uvSign = dPdgxX * rx + dPdgxY * ry + dPdgxZ * rz >= 0 ? 1 : -1;

        // Match the drawn scale's world-space dimensions on this triangle (clamped to prevent extreme stretch)
        const scaleHalfW = THREE.MathUtils.clamp(
          rawHalfW * 0.98,
          0.015 * bulkFactor,
          0.048 * bulkFactor
        );
        const scaleHalfL = THREE.MathUtils.clamp(
          rawHalfL * 0.98,
          0.018 * bulkFactor,
          0.058 * bulkFactor
        );

        const isDominantOsteoderm =
          hashVal > 0.82 || (Math.abs(px) < 0.24 * torsoBulk && ny > 0.25);
        const avgSpan = 0.5 * (scaleHalfW + scaleHalfL);
        const scaleLift =
          avgSpan *
          (isDominantOsteoderm ? 0.68 : 0.44) *
          dorsalFactor *
          (0.65 + 0.55 * reliefFactor);

        const shadeVar = isDominantOsteoderm
          ? 1.04 + (hashVal - 0.82) * 0.35
          : 0.96 + (hashVal - 0.62) * 0.25;

        // Pick dominant bone weights from closest vertex of the triangle
        const bestV = b0 >= b1 && b0 >= b2 ? i0 : b1 >= b2 ? i1 : i2;

        emitted.push({
          px,
          py,
          pz,
          nx,
          ny,
          nz,
          fx,
          fy,
          fz,
          rx,
          ry,
          rz,
          uvSign,
          centerGX,
          centerGY,
          scaleHalfW,
          scaleHalfL,
          scaleLift,
          shadeVar,
          sIdx: [
            skinIdxAttr.getX(bestV),
            skinIdxAttr.getY(bestV),
            skinIdxAttr.getZ(bestV),
            skinIdxAttr.getW(bestV),
          ],
          sW: [
            skinWAttr.getX(bestV),
            skinWAttr.getY(bestV),
            skinWAttr.getZ(bestV),
            skinWAttr.getW(bestV),
          ],
        });
      }
    }
  }

  if (emitted.length === 0) return null;

  // Each 3D protruding scale is a 7-vertex, 8-triangle curved shield hood matching `drawShieldScalePath`:
  // v0: Top root anchor (lx = 0, ly = -0.88)
  // v1: Upper-left shoulder (lx = -0.96, ly = -0.12)
  // v2: Upper-right shoulder (lx = +0.96, ly = -0.12)
  // v3: Raised central osteoderm keel crown (lx = 0, ly = +0.16, lifted by +scaleLift)
  // v4: Pointed overhanging shield tip (lx = 0, ly = +1.05, lifted by +0.68 * scaleLift)
  // v5: Lower-left tapered shield flank (lx = -0.72, ly = +0.52, lifted by +0.34 * scaleLift)
  // v6: Lower-right tapered shield flank (lx = +0.72, ly = +0.52, lifted by +0.34 * scaleLift)
  const vertsPerScale = 7;
  const trisPerScale = 8;
  const totalVerts = emitted.length * vertsPerScale;
  const totalIndices = emitted.length * trisPerScale * 3;

  const positions = new Float32Array(totalVerts * 3);
  const normals = new Float32Array(totalVerts * 3);
  const uvs = new Float32Array(totalVerts * 2);
  const colors = new Float32Array(totalVerts * 3);
  const skinIndices = new Uint16Array(totalVerts * 4);
  const skinWeights = new Float32Array(totalVerts * 4);
  const indices =
    totalVerts > 65535 ? new Uint32Array(totalIndices) : new Uint16Array(totalIndices);

  // Local shield contour coordinates (lx, ly, liftRatio) for v0..v6 matching drawShieldScalePath
  const localShieldVerts: ReadonlyArray<[number, number, number]> = [
    [0.0, -0.88, -0.08],  // v0: Root anchor sunk slightly into skin
    [-0.96, -0.12, 0.12], // v1: Upper-left shoulder
    [0.96, -0.12, 0.12],  // v2: Upper-right shoulder
    [0.0, 0.16, 1.0],     // v3: Central raised keel crown
    [0.0, 1.05, 0.68],    // v4: Pointed overhanging scale tip
    [-0.72, 0.52, 0.34],  // v5: Lower-left tapered flank
    [0.72, 0.52, 0.34],   // v6: Lower-right tapered flank
  ];

  // Vertex luminance profile (root AO -> shoulders -> bright keel crown -> sunlit tip -> flanks)
  const lumProfile = [0.76, 0.94, 0.94, 1.18, 1.12, 0.98, 0.98];

  for (let i = 0; i < emitted.length; i++) {
    const sc = emitted[i];
    const vBase = i * vertsPerScale;
    const iBase = i * trisPerScale * 3;
    const {
      px,
      py,
      pz,
      nx,
      ny,
      nz,
      fx,
      fy,
      fz,
      rx,
      ry,
      rz,
      uvSign,
      centerGX,
      centerGY,
      scaleHalfW,
      scaleHalfL,
      scaleLift,
      shadeVar,
      sIdx,
      sW,
    } = sc;

    for (let v = 0; v < vertsPerScale; v++) {
      const [lx, ly, liftFactor] = localShieldVerts[v];
      const worldOffR = lx * scaleHalfW;
      const worldOffF = ly * scaleHalfL;
      const worldOffN = liftFactor * scaleLift;

      positions[(vBase + v) * 3 + 0] = px + rx * worldOffR + fx * worldOffF + nx * worldOffN;
      positions[(vBase + v) * 3 + 1] = py + ry * worldOffR + fy * worldOffF + ny * worldOffN;
      positions[(vBase + v) * 3 + 2] = pz + rz * worldOffR + fz * worldOffF + nz * worldOffN;

      // Analytical outward-tilted facet normal so the central keel and shield bevels catch rim light cleanly
      let vnx = nx + lx * 0.52 * rx + ly * 0.38 * fx;
      let vny = ny + lx * 0.52 * ry + ly * 0.38 * fy;
      let vnz = nz + lx * 0.52 * rz + ly * 0.38 * fz;
      const vnLen = Math.hypot(vnx, vny, vnz) || 1;
      normals[(vBase + v) * 3 + 0] = vnx / vnLen;
      normals[(vBase + v) * 3 + 1] = vny / vnLen;
      normals[(vBase + v) * 3 + 2] = vnz / vnLen;

      // Exact UV coordinates locked to this specific scale cell (centerGX, centerGY) in the 2D texture!
      const gxV = centerGX + uvSign * lx * rxGrid * 0.92;
      const gyV = centerGY + ly * ryGrid * 0.92;
      uvs[(vBase + v) * 2 + 0] = gxV / totalColsUV;
      uvs[(vBase + v) * 2 + 1] = (1 - gyV / rows) / DRAGON_SCALE_UV_REPEAT_Y;

      const lum = lumProfile[v] * shadeVar;
      colors[(vBase + v) * 3 + 0] = Math.min(1.38, lum);
      colors[(vBase + v) * 3 + 1] = Math.min(1.36, lum * 0.99);
      colors[(vBase + v) * 3 + 2] = Math.min(1.34, lum * 0.98);

      for (let k = 0; k < 4; k++) {
        skinIndices[(vBase + v) * 4 + k] = sIdx[k];
        skinWeights[(vBase + v) * 4 + k] = sW[k];
      }
    }

    // 6 strictly CCW outward-facing (+N) top shield facets radiating around central keel crown v3:
    // Perimeter order in (R, F) plane: v0 -> v2 -> v6 -> v4 -> v5 -> v1 -> v0
    indices[iBase + 0] = vBase + 3;
    indices[iBase + 1] = vBase + 0;
    indices[iBase + 2] = vBase + 2;

    indices[iBase + 3] = vBase + 3;
    indices[iBase + 4] = vBase + 2;
    indices[iBase + 5] = vBase + 6;

    indices[iBase + 6] = vBase + 3;
    indices[iBase + 7] = vBase + 6;
    indices[iBase + 8] = vBase + 4;

    indices[iBase + 9] = vBase + 3;
    indices[iBase + 10] = vBase + 4;
    indices[iBase + 11] = vBase + 5;

    indices[iBase + 12] = vBase + 3;
    indices[iBase + 13] = vBase + 5;
    indices[iBase + 14] = vBase + 1;

    indices[iBase + 15] = vBase + 3;
    indices[iBase + 16] = vBase + 1;
    indices[iBase + 17] = vBase + 0;

    // 2 underside lip facets closing the overhanging posterior tip (v4, v6, v5) back to shoulders (v2, v1)
    indices[iBase + 18] = vBase + 4;
    indices[iBase + 19] = vBase + 6;
    indices[iBase + 20] = vBase + 2;

    indices[iBase + 21] = vBase + 5;
    indices[iBase + 22] = vBase + 4;
    indices[iBase + 23] = vBase + 1;
  }

  const outGeo = new THREE.BufferGeometry();
  outGeo.setIndex(new THREE.BufferAttribute(indices, 1));
  outGeo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  outGeo.setAttribute('normal', new THREE.BufferAttribute(normals, 3));
  outGeo.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
  outGeo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  outGeo.setAttribute('skinIndex', new THREE.Uint16BufferAttribute(skinIndices, 4));
  outGeo.setAttribute('skinWeight', new THREE.Float32BufferAttribute(skinWeights, 4));
  outGeo.computeBoundingBox();
  outGeo.computeBoundingSphere();
  return outGeo;
}

/**
 * Builds a 3D Crescent Weapon / Claw Slash Ribbon Geometry with tapered width and vertex color gradient
 * (100% GLTF 2.0 / .GLB compatible indexed BufferGeometry).
 */
function createWeaponSlashArcGeometry(isClaws: boolean, segments = 22): THREE.BufferGeometry {
  const ribbons = isClaws
    ? [
        { xOff: -0.14, innerR: 0.42, outerR: 0.66, arc: Math.PI * 0.78 },
        { xOff: 0.0, innerR: 0.46, outerR: 0.76, arc: Math.PI * 0.86 },
        { xOff: 0.14, innerR: 0.42, outerR: 0.66, arc: Math.PI * 0.78 },
      ]
    : [
        { xOff: 0.0, innerR: 0.52, outerR: 1.18, arc: Math.PI * 0.92 },
        { xOff: 0.025, innerR: 0.76, outerR: 1.24, arc: Math.PI * 0.76 },
      ];

  const vertsPerRibbon = (segments + 1) * 2;
  const totalVerts = ribbons.length * vertsPerRibbon;
  const trisPerRibbon = segments * 2;
  const totalIndices = ribbons.length * trisPerRibbon * 3;

  const positions = new Float32Array(totalVerts * 3);
  const normals = new Float32Array(totalVerts * 3);
  const uvs = new Float32Array(totalVerts * 2);
  const colors = new Float32Array(totalVerts * 3);
  const indices = new Uint16Array(totalIndices);

  let vCursor = 0;
  let iCursor = 0;

  for (let rIdx = 0; rIdx < ribbons.length; rIdx++) {
    const rb = ribbons[rIdx];
    const startAngle = -rb.arc * 0.45;
    const baseVertex = vCursor;

    for (let s = 0; s <= segments; s++) {
      const u = s / segments; // 0 = trailing tail, 1 = leading strike tip
      const angle = startAngle + u * rb.arc;
      // Taper width smoothly at both ends of the crescent arc
      const taper = Math.sin(u * Math.PI);
      const midR = (rb.innerR + rb.outerR) * 0.5;
      const halfSpan = (rb.outerR - rb.innerR) * 0.5 * Math.pow(taper, 0.65);

      const rInner = midR - halfSpan;
      const rOuter = midR + halfSpan;

      const sinA = Math.sin(angle);
      const cosA = Math.cos(angle);

      // Swept in YZ strike plane (+Z forward, Y vertical sweep)
      const yIn = sinA * rInner;
      const zIn = cosA * rInner + 0.22;
      const yOut = sinA * rOuter;
      const zOut = cosA * rOuter + 0.22;

      positions[vCursor * 3] = rb.xOff - (1 - u) * 0.02;
      positions[vCursor * 3 + 1] = yIn;
      positions[vCursor * 3 + 2] = zIn;

      normals[vCursor * 3] = 1;
      normals[vCursor * 3 + 1] = 0;
      normals[vCursor * 3 + 2] = 0;

      uvs[vCursor * 2] = u;
      uvs[vCursor * 2 + 1] = 0;

      // Inner edge intensity
      const innerGlow = Math.pow(u, 1.3) * 0.55;
      colors[vCursor * 3] = innerGlow;
      colors[vCursor * 3 + 1] = innerGlow;
      colors[vCursor * 3 + 2] = innerGlow;
      vCursor++;

      positions[vCursor * 3] = rb.xOff + (1 - u) * 0.02;
      positions[vCursor * 3 + 1] = yOut;
      positions[vCursor * 3 + 2] = zOut;

      normals[vCursor * 3] = 1;
      normals[vCursor * 3 + 1] = 0;
      normals[vCursor * 3 + 2] = 0;

      uvs[vCursor * 2] = u;
      uvs[vCursor * 2 + 1] = 1;

      // Outer leading blade edge: white-hot highlight near u = 0.7..0.95
      const outerGlow = Math.min(1.0, Math.pow(u, 0.85) * 1.15 * taper);
      colors[vCursor * 3] = outerGlow;
      colors[vCursor * 3 + 1] = outerGlow;
      colors[vCursor * 3 + 2] = outerGlow;
      vCursor++;
    }

    for (let s = 0; s < segments; s++) {
      const i0 = baseVertex + s * 2;
      const i1 = i0 + 1;
      const i2 = i0 + 2;
      const i3 = i0 + 3;

      indices[iCursor++] = i0;
      indices[iCursor++] = i1;
      indices[iCursor++] = i2;

      indices[iCursor++] = i2;
      indices[iCursor++] = i1;
      indices[iCursor++] = i3;
    }
  }

  const geo = new THREE.BufferGeometry();
  geo.setIndex(new THREE.BufferAttribute(indices, 1));
  geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geo.setAttribute('normal', new THREE.BufferAttribute(normals, 3));
  geo.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
  geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  geo.computeBoundingSphere();
  return geo;
}

function tagExplodablePart(
  obj: THREE.Object3D,
  explodeDir: THREE.Vector3,
  partInfo: PartInspectionInfo
) {
  obj.userData.basePos = obj.position.clone();
  obj.userData.explodeVec = explodeDir.clone();
  obj.userData.partInfo = partInfo;
  obj.traverse((child) => {
    if ((child as THREE.Mesh).isMesh) {
      child.userData.partInfo = partInfo;
      child.layers.enable(1);
    }
  });
}

function applySurfaceTexturesToMaterials(
  materials: BuiltEnemyRig['materials'],
  config: EnemyConfig,
  textures: ProceduralTextureSet
) {
  const patternId = getEffectiveTexturePattern(config);
  const relief = config.materials.textureRelief ?? 0.85;
  const grainWear = config.materials.grainWearIntensity ?? 0.85;
  const surf = patternId !== 'none' ? textures.patterns[patternId] : null;
  const isFurPattern = patternId === 'wolf_fur' || patternId === 'chitin_fur';

  const newAlbedo = surf ? surf.albedoMap : null;
  const newBump = surf ? surf.bumpMap : grainWear > 0.02 ? textures.armorBumpMap : null;
  const newNormal = surf ? surf.normalMap : grainWear > 0.02 ? textures.armorNormalMap : null;
  const newRoughMap = grainWear > 0.05 ? textures.armorRoughnessMap : null;

  // Combine structural pattern relief + high-frequency grain & battle-wear noise strength
  // Scales and fur pelage get stronger tactile bump relief so protruding osteoderms and hair strands catch rim light
  const isScalesPattern = patternId === 'scales';
  const patternMult = isScalesPattern ? 0.032 : isFurPattern ? 0.024 : 0.012;
  const primaryBump = (patternId !== 'none' ? patternMult * relief : 0) + 0.028 * grainWear;
  const secondaryBump = (patternId !== 'none' ? patternMult * 0.9 * relief : 0) + 0.022 * grainWear;
  const underbodyBump = (patternId !== 'none' ? patternMult * 0.85 * relief : 0) + 0.024 * grainWear;
  const normStrength = Math.min(1.4, Math.max(0.25, relief * 0.75 + grainWear * 0.35));

  if (materials.primaryArmorMat.map !== newAlbedo) {
    materials.primaryArmorMat.map = newAlbedo;
    materials.primaryArmorMat.needsUpdate = true;
  }
  if (materials.primaryArmorMat.bumpMap !== newBump) {
    materials.primaryArmorMat.bumpMap = newBump;
    materials.primaryArmorMat.needsUpdate = true;
  }
  if (materials.primaryArmorMat.normalMap !== newNormal) {
    materials.primaryArmorMat.normalMap = newNormal;
    materials.primaryArmorMat.needsUpdate = true;
  }
  if (materials.primaryArmorMat.roughnessMap !== newRoughMap) {
    materials.primaryArmorMat.roughnessMap = newRoughMap;
    materials.primaryArmorMat.needsUpdate = true;
  }
  materials.primaryArmorMat.bumpScale = primaryBump;
  materials.primaryArmorMat.normalScale.set(normStrength, normStrength);
  materials.primaryArmorMat.sheen = isFurPattern ? 0.88 : materials.primaryArmorMat.sheen;
  materials.primaryArmorMat.sheenRoughness = isFurPattern ? 0.42 : 0.55;
  materials.primaryArmorMat.clearcoatRoughness = Math.min(0.92, 0.2 + grainWear * 0.22);

  if (materials.secondaryTrimMat.map !== newAlbedo) {
    materials.secondaryTrimMat.map = newAlbedo;
    materials.secondaryTrimMat.needsUpdate = true;
  }
  if (materials.secondaryTrimMat.bumpMap !== newBump) {
    materials.secondaryTrimMat.bumpMap = newBump;
    materials.secondaryTrimMat.needsUpdate = true;
  }
  if (materials.secondaryTrimMat.normalMap !== newNormal) {
    materials.secondaryTrimMat.normalMap = newNormal;
    materials.secondaryTrimMat.needsUpdate = true;
  }
  if (materials.secondaryTrimMat.roughnessMap !== newRoughMap) {
    materials.secondaryTrimMat.roughnessMap = newRoughMap;
    materials.secondaryTrimMat.needsUpdate = true;
  }
  materials.secondaryTrimMat.bumpScale = secondaryBump;
  materials.secondaryTrimMat.normalScale.set(normStrength * 0.9, normStrength * 0.9);
  materials.secondaryTrimMat.sheen = isFurPattern ? 0.85 : materials.secondaryTrimMat.sheen;

  if (materials.underbodyMat.map !== newAlbedo) {
    materials.underbodyMat.map = newAlbedo;
    materials.underbodyMat.needsUpdate = true;
  }
  if (materials.underbodyMat.bumpMap !== newBump) {
    materials.underbodyMat.bumpMap = newBump;
    materials.underbodyMat.needsUpdate = true;
  }
  if (materials.underbodyMat.normalMap !== newNormal) {
    materials.underbodyMat.normalMap = newNormal;
    materials.underbodyMat.needsUpdate = true;
  }
  if (materials.underbodyMat.roughnessMap !== newRoughMap) {
    materials.underbodyMat.roughnessMap = newRoughMap;
    materials.underbodyMat.needsUpdate = true;
  }
  materials.underbodyMat.bumpScale = underbodyBump;
  materials.underbodyMat.normalScale.set(normStrength * 0.85, normStrength * 0.85);
}

/**
 * Updates materials, surface textures, grain/wear intensity, and RPG userData in-place (< 0.1ms) without rebuilding geometry.
 */
export function updateEnemyMaterialsInPlace(
  rig: BuiltEnemyRig,
  config: EnemyConfig,
  textures: ProceduralTextureSet,
  isSoftwareGl: boolean,
  wireframe: boolean
) {
  const { materials: matCfg } = config;
  const grainWear = matCfg.grainWearIntensity ?? 0.85;
  const wearRoughnessShift = (grainWear - 0.5) * 0.14;

  const {
    primaryArmorMat,
    secondaryTrimMat,
    underbodyMat,
    hairStrandMat,
    emissiveRuneMat,
    crystalCoreMat,
    slashTrailMat,
  } = rig.materials;

  primaryArmorMat.color.set(matCfg.primaryColor);
  primaryArmorMat.metalness = Math.max(0.02, matCfg.metalness - Math.max(0, grainWear - 0.8) * 0.12);
  primaryArmorMat.roughness = Math.min(0.95, Math.max(0.08, matCfg.roughness + wearRoughnessShift));
  primaryArmorMat.anisotropy = isSoftwareGl ? 0 : matCfg.anisotropy;
  primaryArmorMat.wireframe = wireframe;

  secondaryTrimMat.color.set(matCfg.secondaryColor);
  secondaryTrimMat.metalness = Math.min(1, matCfg.metalness + 0.12);
  secondaryTrimMat.roughness = Math.min(
    0.94,
    Math.max(0.08, ((matCfg.roughness + 0.32) % 0.85) + wearRoughnessShift)
  );
  secondaryTrimMat.wireframe = wireframe;

  underbodyMat.color.set(matCfg.underbodyColor);
  underbodyMat.metalness = Math.max(0.04, matCfg.metalness * 0.22);
  underbodyMat.roughness = Math.min(0.96, Math.max(0.15, matCfg.roughness + 0.38 + wearRoughnessShift));
  underbodyMat.wireframe = wireframe;

  hairStrandMat.color.set(matCfg.primaryColor).lerp(new THREE.Color(matCfg.secondaryColor), 0.34);
  hairStrandMat.sheenColor
    .set(matCfg.secondaryColor)
    .lerp(new THREE.Color('#ffffff'), 0.38);
  hairStrandMat.wireframe = wireframe;

  emissiveRuneMat.color.set(matCfg.emissiveColor);
  emissiveRuneMat.emissive.set(matCfg.emissiveColor);
  emissiveRuneMat.emissiveIntensity = matCfg.emissiveIntensity;
  emissiveRuneMat.wireframe = wireframe;

  crystalCoreMat.color.set(matCfg.emissiveColor).lerp(new THREE.Color('#ffffff'), 0.35);
  crystalCoreMat.emissive.set(matCfg.emissiveColor);
  crystalCoreMat.emissiveIntensity = matCfg.emissiveIntensity * 0.65;
  crystalCoreMat.wireframe = wireframe;

  slashTrailMat.color.set(matCfg.emissiveColor).lerp(new THREE.Color('#ffffff'), 0.45);
  slashTrailMat.emissive.set(matCfg.emissiveColor);
  slashTrailMat.emissiveIntensity = matCfg.emissiveIntensity * 1.45;
  slashTrailMat.wireframe = wireframe;

  applySurfaceTexturesToMaterials(rig.materials, config, textures);

  rig.exportableGroup.userData.name = config.name;
  rig.exportableGroup.userData.title = config.title;
  rig.exportableGroup.userData.rpgStats = { ...config.stats };
}

export function buildProceduralEnemy(
  config: EnemyConfig,
  textures: ProceduralTextureSet,
  isSoftwareGl: boolean,
  wireframe: boolean
): BuiltEnemyRig {
  const rootGroup = new THREE.Group();
  rootGroup.name = `${config.id}_root`;

  const exportableGroup = new THREE.Group();
  exportableGroup.name = config.id;
  exportableGroup.userData = {
    name: config.name,
    title: config.title,
    raceCategory: config.raceCategory,
    bodyPlan: config.bodyPlan,
    archetype: config.archetype,
    rpgStats: { ...config.stats },
    sockets: ['socket_head', 'socket_weapon_r', 'socket_shield_l', 'socket_core'],
    generator: 'AetherForge 3D RPG Studio',
  };
  rootGroup.add(exportableGroup);

  const geometriesToDispose: THREE.BufferGeometry[] = [];
  const materialsToDispose: THREE.Material[] = [];

  const hairStyle = getEffectiveHairStyle(config);
  const furDensity = getEffectiveFurDensity(config);
  const hairLength = getEffectiveHairLength(config);
  const muscleDef = getEffectiveMuscleDefinition(config);
  const isAnimal = isAnimalEnemy(config);
  const effectiveWeapon = getEffectiveWeapon(config);

  exportableGroup.userData.isAnimal = isAnimal;
  exportableGroup.userData.effectiveWeapon = effectiveWeapon;

  // Polygon Density (LOD) Multiplier — guaranteed smooth curvature even in low density
  const density = getEffectiveMeshDensity(config);
  const lodMult = density === 'low' ? 0.9 : density === 'high' ? 1.85 : 1.35;
  const seg = (base: number, min = 12) => Math.max(min, Math.round(base * lodMult));

  const trackGeo = <T extends THREE.BufferGeometry>(
    g: T,
    sculptAmp = 0.014,
    sculptFreq = 6.0,
    aoBottom = 0.26
  ): T => {
    sculptAndShadeGeometry(
      g,
      sculptAmp * (0.45 + muscleDef * 0.65),
      sculptFreq,
      aoBottom,
      furDensity * 0.0065
    );
    geometriesToDispose.push(g);
    return g;
  };
  const trackFurLock = (baseR: number, length: number, curveBack = 0.09): THREE.BufferGeometry => {
    const g = createFurLockGeometry(
      baseR,
      length * hairLength,
      curveBack,
      density === 'low' ? 5 : 7
    );
    geometriesToDispose.push(g);
    return g;
  };
  const trackMat = <T extends THREE.Material>(m: T): T => {
    materialsToDispose.push(m);
    return m;
  };

  // Safe defaults for every morphology parameter
  const rawM = config.morphology;
  const m = {
    heightScale: Number.isFinite(rawM.heightScale) ? rawM.heightScale : 1.0,
    torsoBulk: Number.isFinite(rawM.torsoBulk) ? rawM.torsoBulk : 1.0,
    shoulderWidth: Number.isFinite(rawM.shoulderWidth) ? rawM.shoulderWidth : 1.0,
    hornLength: Number.isFinite(rawM.hornLength) ? rawM.hornLength : 0.8,
    hornCurve: Number.isFinite(rawM.hornCurve) ? rawM.hornCurve : 0.6,
    pauldronSize: Number.isFinite(rawM.pauldronSize) ? rawM.pauldronSize : 1.0,
    weaponScale: Number.isFinite(rawM.weaponScale) ? rawM.weaponScale : 1.0,
    dorsalScale: Number.isFinite(rawM.dorsalScale) ? rawM.dorsalScale : 1.0,
    limbLength: Number.isFinite(rawM.limbLength) ? rawM.limbLength : 1.0,
    neckLength: Number.isFinite(rawM.neckLength) ? rawM.neckLength : 0.7,
    tailLength: Number.isFinite(rawM.tailLength) ? rawM.tailLength : 0.8,
    snoutLength: Number.isFinite(rawM.snoutLength) ? rawM.snoutLength : 0.8,
    headCount: Number.isFinite(rawM.headCount) ? rawM.headCount : 1,
    auraShardCount: Number.isFinite(rawM.auraShardCount) ? rawM.auraShardCount : 12,
    furDensity,
    hairLength,
    muscleDef,
  };

  const { materials: matCfg, bodyPlan } = config;
  const isQuadruped = bodyPlan === 'quadruped_beast' || bodyPlan === 'dragon_wyvern';
  const isDragon = bodyPlan === 'dragon_wyvern';
  const isArachnid = bodyPlan === 'arachnid_scorpion';
  const isSerpentine = bodyPlan === 'serpentine_naga';
  const isFloating = bodyPlan === 'floating_magical';
  const isCentaur =
    config.archetype === 'astral_centaur' ||
    config.archetype === 'aurora_ice_centaur' ||
    config.archetype === 'flayed_underworld_centaur' ||
    config.archetype === 'iron_crown_cataphract_skeleton' ||
    config.archetype === 'elite_centaur_steppe_khan' ||
    config.archetype === 'wow_centaur_kolkar_marauder' ||
    config.archetype === 'wow_mannoroth_pit_lord' ||
    config.archetype === 'wow_magnataur_northrend_behemoth' ||
    config.archetype === 'archer_species_7_centaur_khan_horsebow';
  const isBeholder =
    config.archetype === 'arcane_beholder' ||
    config.archetype === 'kraken_deep_watcher' ||
    config.archetype === 'skull_tyrant_overlord' ||
    config.archetype === 'wow_cthun_old_god_eye';
  const isUndeadSkeleton = isUndeadArchetype(config);
  const isOrganicCreature =
    !isUndeadSkeleton &&
    (config.raceCategory === 'starter_animals' ||
      config.raceCategory === 'beasts' ||
      config.raceCategory === 'dragons' ||
      config.raceCategory === 'mythological' ||
      isAnimalEnemy(config));

  // --- STUDIO PBR MATERIALS WITH VERTEX COLORS, ORGANIC SHEEN & PROCEDURAL TEXTURES ---
  const primaryArmorMat = trackMat(
    new THREE.MeshPhysicalMaterial({
      name: 'Mat_PrimarySurface',
      color: new THREE.Color(matCfg.primaryColor),
      vertexColors: true,
      metalness: matCfg.metalness,
      roughness: matCfg.roughness,
      roughnessMap: textures.armorRoughnessMap,
      anisotropy: isSoftwareGl ? 0 : matCfg.anisotropy,
      anisotropyRotation: Math.PI / 4,
      sheen: isOrganicCreature || furDensity > 0.2 ? 0.82 : 0.15,
      sheenRoughness: furDensity > 0.2 ? 0.42 : 0.55,
      sheenColor: new THREE.Color(matCfg.secondaryColor).lerp(new THREE.Color('#ffffff'), 0.28),
      clearcoat: isDragon ? 0.38 : furDensity > 0.3 ? 0.04 : 0.12,
      clearcoatRoughness: 0.32,
      wireframe,
    })
  );

  const secondaryTrimMat = trackMat(
    new THREE.MeshPhysicalMaterial({
      name: 'Mat_SecondaryScalesOrTrim',
      color: new THREE.Color(matCfg.secondaryColor),
      vertexColors: true,
      metalness: isOrganicCreature ? Math.min(0.35, matCfg.metalness) : Math.min(1, matCfg.metalness + 0.12),
      roughness: Math.max(0.12, (matCfg.roughness + 0.28) % 0.85),
      sheen: isOrganicCreature || furDensity > 0.2 ? 0.78 : 0.1,
      sheenRoughness: 0.5,
      anisotropy: isSoftwareGl ? 0 : matCfg.anisotropy * 0.8,
      wireframe,
    })
  );

  const underbodyMat = trackMat(
    new THREE.MeshStandardMaterial({
      name: 'Mat_UnderbodyHide',
      color: new THREE.Color(matCfg.underbodyColor),
      vertexColors: true,
      metalness: Math.max(0.04, matCfg.metalness * 0.2),
      roughness: Math.min(0.92, matCfg.roughness + 0.38),
      wireframe,
    })
  );

  const hairStrandMat = trackMat(
    new THREE.MeshPhysicalMaterial({
      name: 'Mat_3DHairStrands',
      color: new THREE.Color(matCfg.primaryColor).lerp(new THREE.Color(matCfg.secondaryColor), 0.34),
      vertexColors: true,
      metalness: 0.04,
      roughness: 0.48,
      sheen: 0.95,
      sheenRoughness: 0.32,
      sheenColor: new THREE.Color(matCfg.secondaryColor).lerp(new THREE.Color('#ffffff'), 0.38),
      anisotropy: isSoftwareGl ? 0 : 0.82,
      anisotropyRotation: Math.PI / 2,
      bumpMap: textures.patterns.wolf_fur.bumpMap,
      bumpScale: 0.024,
      normalMap: textures.patterns.wolf_fur.normalMap,
      normalScale: new THREE.Vector2(0.85, 0.85),
      side: THREE.DoubleSide,
      wireframe,
    })
  );

  const ivoryFangMat = trackMat(
    new THREE.MeshStandardMaterial({
      name: 'Mat_IvoryFangsAndHorns',
      color: new THREE.Color('#e8e0cc'),
      vertexColors: true,
      metalness: 0.06,
      roughness: 0.36,
      bumpMap: textures.armorBumpMap,
      bumpScale: 0.004,
      normalMap: textures.armorNormalMap,
      normalScale: new THREE.Vector2(0.35, 0.35),
      wireframe,
    })
  );

  const bladeAlloyMat = trackMat(
    new THREE.MeshPhysicalMaterial({
      name: 'Mat_WeaponBlade',
      color: new THREE.Color('#dce5f2'),
      vertexColors: true,
      metalness: 0.94,
      roughness: 0.14,
      map: textures.patterns.damascus_plate.albedoMap,
      bumpMap: textures.patterns.damascus_plate.bumpMap,
      bumpScale: 0.008,
      normalMap: textures.patterns.damascus_plate.normalMap,
      normalScale: new THREE.Vector2(0.55, 0.55),
      anisotropy: isSoftwareGl ? 0 : 0.75,
      clearcoat: 0.4,
      wireframe,
    })
  );

  const emissiveRuneMat = trackMat(
    new THREE.MeshStandardMaterial({
      name: 'Mat_RunicEmissive',
      color: new THREE.Color(matCfg.emissiveColor),
      emissive: new THREE.Color(matCfg.emissiveColor),
      emissiveIntensity: matCfg.emissiveIntensity,
      roughness: 0.15,
      metalness: 0.1,
      wireframe,
    })
  );

  const crystalCoreMat = trackMat(
    new THREE.MeshPhysicalMaterial({
      name: 'Mat_ArcaneCrystal',
      color: new THREE.Color(matCfg.emissiveColor).lerp(new THREE.Color('#ffffff'), 0.35),
      emissive: new THREE.Color(matCfg.emissiveColor),
      emissiveIntensity: matCfg.emissiveIntensity * 0.65,
      metalness: 0.1,
      roughness: 0.08,
      transmission: isSoftwareGl ? 0 : matCfg.crystalTransmission,
      transparent: true,
      opacity: isSoftwareGl ? 0.78 : 1.0,
      ior: 1.75,
      thickness: 1.2,
      clearcoat: 1.0,
      wireframe,
    })
  );

  const slashTrailMat = trackMat(
    new THREE.MeshStandardMaterial({
      name: 'Mat_WeaponSlashTrail',
      color: new THREE.Color(matCfg.emissiveColor).lerp(new THREE.Color('#ffffff'), 0.45),
      emissive: new THREE.Color(matCfg.emissiveColor),
      emissiveIntensity: matCfg.emissiveIntensity * 1.45,
      vertexColors: true,
      roughness: 0.12,
      metalness: 0.08,
      transparent: true,
      opacity: 0.88,
      side: THREE.DoubleSide,
      depthWrite: false,
      wireframe,
    })
  );

  applySurfaceTexturesToMaterials(
    {
      primaryArmorMat,
      secondaryTrimMat,
      underbodyMat,
      hairStrandMat,
      emissiveRuneMat,
      crystalCoreMat,
      slashTrailMat,
    },
    config,
    textures
  );

  const explodableParts: THREE.Object3D[] = [];

  const legHeight = isSerpentine
    ? 0.55 * m.limbLength
    : isArachnid
    ? 0.68 * m.limbLength
    : isQuadruped
    ? 0.92 * m.limbLength
    : 1.05 * m.limbLength;
  const pelvisY = isFloating ? legHeight + 0.38 : legHeight;

  const makeBone = (name: string): THREE.Bone => {
    const bone = new THREE.Bone();
    bone.name = name;
    return bone;
  };

  // Root Bone of the Skeletal Hierarchy (y = 0 ground pivot)
  const rootBone = makeBone('Root_Bone');
  exportableGroup.add(rootBone);

  // 1. PELVIS / ANATOMICAL RIBCAGE & ABDOMEN / SERPENT COIL
  const pelvisGroup = makeBone('Pelvis_Chassis');
  pelvisGroup.position.set(0, pelvisY, isQuadruped ? -0.36 : 0);
  rootBone.add(pelvisGroup);

  if (isSerpentine) {
    const coilCurve = new THREE.CatmullRomCurve3([
      new THREE.Vector3(0, 0.24, 0),
      new THREE.Vector3(0.08 * m.torsoBulk, -0.06, 0.08),
      new THREE.Vector3(0.32 * m.torsoBulk, -legHeight * 0.45, 0.25),
      new THREE.Vector3(0, -legHeight + 0.14, 0.48),
      new THREE.Vector3(-0.42 * m.torsoBulk, -legHeight + 0.12, 0.05),
      new THREE.Vector3(-0.15 * m.torsoBulk, -legHeight + 0.1, -0.55),
      new THREE.Vector3(0.35 * m.torsoBulk, -legHeight + 0.08, -0.88),
      new THREE.Vector3(0.12 * m.torsoBulk, -legHeight + 0.05, -1.18),
    ]);
    if (isUndeadSkeleton) {
      // Exposed 3D Serpentine Vertebral Column & Coiled Rib Arches for Undead Naga / Bone Serpents
      const serpDarkMat = trackMat(
        new THREE.MeshStandardMaterial({
          name: 'Mat_SerpSkeletonCord',
          color: new THREE.Color('#08090d'),
          roughness: 0.92,
          metalness: 0.05,
          wireframe,
        })
      );
      const cordGeo = trackGeo(
        new THREE.TubeGeometry(coilCurve, seg(28, 18), 0.045 * m.torsoBulk, seg(8, 6), false),
        0.006
      );
      pelvisGroup.add(new THREE.Mesh(cordGeo, serpDarkMat));

      const serpVertCount = 18;
      for (let sv = 0; sv < serpVertCount; sv++) {
        const u = sv / (serpVertCount - 1);
        const pt = coilCurve.getPoint(u);
        const tang = coilCurve.getTangent(u);
        const vScale = (1.0 - u * 0.45) * m.torsoBulk;

        const vGeo = trackGeo(
          new THREE.CylinderGeometry(0.075 * vScale, 0.082 * vScale, 0.08, seg(10, 8)),
          0.006
        );
        vGeo.rotateX(Math.PI / 2);
        const vMesh = new THREE.Mesh(vGeo, ivoryFangMat);
        vMesh.position.copy(pt);
        vMesh.lookAt(pt.clone().add(tang));
        pelvisGroup.add(vMesh);

        if (sv < 14) {
          const ribRingGeo = trackGeo(
            new THREE.TorusGeometry(0.16 * vScale, 0.022 * vScale, 6, 12, Math.PI * 1.45),
            0.004
          );
          const ribRing = new THREE.Mesh(ribRingGeo, ivoryFangMat);
          ribRing.position.copy(pt);
          ribRing.lookAt(pt.clone().add(tang));
          pelvisGroup.add(ribRing);
        }
      }
    } else {
      const coilGeo = trackGeo(
        new THREE.TubeGeometry(coilCurve, seg(32, 20), 0.24 * m.torsoBulk, seg(16, 12), false),
        0.022
      );
      const coilMesh = new THREE.Mesh(coilGeo, primaryArmorMat);
      coilMesh.castShadow = true;
      coilMesh.receiveShadow = true;
      pelvisGroup.add(coilMesh);

      const waistCollarPts = [
        new THREE.Vector2(0.24 * m.torsoBulk, -0.08),
        new THREE.Vector2(0.32 * m.torsoBulk, 0.12),
        new THREE.Vector2(0.35 * m.torsoBulk, 0.28),
        new THREE.Vector2(0.29 * m.torsoBulk, 0.42),
      ];
      const waistCollarGeo = trackGeo(new THREE.LatheGeometry(waistCollarPts, seg(18, 12)), 0.018);
      const waistCollarMesh = new THREE.Mesh(waistCollarGeo, secondaryTrimMat);
      waistCollarMesh.position.y = 0.02;
      pelvisGroup.add(waistCollarMesh);

      // Blender-Sculpted Serpentine Coil-to-Torso Muscular Transition Sleeve
      const serpBridgeCurve = new THREE.CatmullRomCurve3([
        new THREE.Vector3(0.06 * m.torsoBulk, -0.05, 0.06),
        new THREE.Vector3(0, 0.14, 0.02),
        new THREE.Vector3(0, 0.36, 0),
      ]);
      const serpBridgeGeo = trackGeo(
        new THREE.TubeGeometry(serpBridgeCurve, seg(14, 10), 0.27 * m.torsoBulk, seg(16, 12), false),
        0.018
      );
      const serpBridgeMesh = new THREE.Mesh(serpBridgeGeo, primaryArmorMat);
      pelvisGroup.add(serpBridgeMesh);
    }
  } else if (isQuadruped) {
    if (isUndeadSkeleton) {
      // BLENDER-SCULPTED QUADRUPED SKELETAL PELVIC GIRDLE, SACRUM & HORIZONTAL LUMBAR VERTEBRAL COLUMN (100% EXPOSED BONE!)
      const quadDarkMat = trackMat(
        new THREE.MeshStandardMaterial({
          name: 'Mat_QuadSkeletonSocketVoid',
          color: new THREE.Color('#08090d'),
          roughness: 0.92,
          metalness: 0.05,
          wireframe,
        })
      );

      // Horizontal Sacrum & Lumbar Spinal Cord Core
      const quadSpineCoreGeo = trackGeo(
        new THREE.CylinderGeometry(0.038 * m.torsoBulk, 0.044 * m.torsoBulk, 0.78, seg(10, 8)),
        0.006
      );
      quadSpineCoreGeo.rotateX(Math.PI / 2);
      const quadSpineCore = new THREE.Mesh(quadSpineCoreGeo, quadDarkMat);
      quadSpineCore.position.set(0, 0.14, 0.16);
      pelvisGroup.add(quadSpineCore);

      // 5 Exposed Horizontal Lumbar & Sacral Vertebrae with Dorsal Spinous Spikes & Transverse Ribs
      for (let lv = 0; lv < 5; lv++) {
        const lz = -0.18 + lv * 0.16;
        const ly = 0.12 + Math.sin((lv / 4) * Math.PI) * 0.04;
        const centrumGeo = trackGeo(
          new THREE.CylinderGeometry(0.078 * m.torsoBulk, 0.082 * m.torsoBulk, 0.085, seg(12, 8)),
          0.008
        );
        centrumGeo.rotateX(Math.PI / 2);
        const centrum = new THREE.Mesh(centrumGeo, ivoryFangMat);
        centrum.position.set(0, ly, lz);
        pelvisGroup.add(centrum);

        const discGeo = trackGeo(
          new THREE.CylinderGeometry(0.07 * m.torsoBulk, 0.07 * m.torsoBulk, 0.028, seg(10, 8)),
          0.002
        );
        discGeo.rotateX(Math.PI / 2);
        const disc = new THREE.Mesh(discGeo, quadDarkMat);
        disc.position.set(0, ly, lz + 0.055);
        pelvisGroup.add(disc);

        const transGeo = trackGeo(
          new THREE.CapsuleGeometry(0.022 * m.torsoBulk, 0.24 * m.torsoBulk, 6, 8),
          0.006
        );
        transGeo.rotateZ(Math.PI / 2);
        const trans = new THREE.Mesh(transGeo, ivoryFangMat);
        trans.position.set(0, ly, lz);
        pelvisGroup.add(trans);

        const spinousGeo = trackGeo(
          new THREE.ConeGeometry(0.032 * m.torsoBulk, 0.16 * m.torsoBulk, 6),
          0.006
        );
        spinousGeo.rotateX(-0.32);
        const spinous = new THREE.Mesh(spinousGeo, ivoryFangMat);
        spinous.position.set(0, ly + 0.11 * m.torsoBulk, lz);
        pelvisGroup.add(spinous);
      }

      // Paired Skeletal Iliac Blades & Acetabular Hip Sockets for Quadruped Hind Legs
      for (const side of [-1, 1]) {
        const iliumCurve = new THREE.CatmullRomCurve3([
          new THREE.Vector3(side * 0.07 * m.torsoBulk, 0.16, -0.18),
          new THREE.Vector3(side * 0.28 * m.torsoBulk, 0.22, -0.04),
          new THREE.Vector3(side * 0.26 * m.torsoBulk, 0.02, 0.06),
          new THREE.Vector3(side * 0.12 * m.torsoBulk, -0.08, -0.12),
        ]);
        const iliumGeo = trackGeo(
          new THREE.TubeGeometry(iliumCurve, seg(14, 10), 0.044 * m.torsoBulk, seg(8, 6), false),
          0.008
        );
        pelvisGroup.add(new THREE.Mesh(iliumGeo, ivoryFangMat));
      }
    } else {
      // Rear Croup, Pelvis & Tucked Abdominal Loin (Overlaps seamlessly into Torso_Main ribcage at z = +0.44)
      const croupPts = [
        new THREE.Vector2(0.01, -0.38),
        new THREE.Vector2(0.26 * m.torsoBulk, -0.28), // Rounded rump / tail base
        new THREE.Vector2(0.37 * m.torsoBulk, -0.06), // Muscular hind haunch / pelvis
        new THREE.Vector2(0.33 * m.torsoBulk, 0.22),  // Tucked athletic abdominal waist
        new THREE.Vector2(0.41 * m.torsoBulk, 0.5),   // Deep overlap into rear ribcage
        new THREE.Vector2(0.32 * m.torsoBulk, 0.68),
        new THREE.Vector2(0.01, 0.74),
      ];
      const croupGeo = trackGeo(new THREE.LatheGeometry(croupPts, seg(20, 14)), 0.02);
      croupGeo.rotateX(Math.PI / 2); // +Y maps to +Z forward!
      croupGeo.scale(0.92, 1.04, 1.0);
      const croupMesh = new THREE.Mesh(croupGeo, primaryArmorMat);
      croupMesh.position.set(0, 0.04, -0.08);
      croupMesh.castShadow = true;
      pelvisGroup.add(croupMesh);

      // Blender-Sculpted Lumbar-Thoracic Muscular Saddle & Flank Oblique Bridges (Eliminates any waist gap on quadrupeds & dragons)
      const lumbarBridgeCurve = new THREE.CatmullRomCurve3([
        new THREE.Vector3(0, 0.08, -0.02),
        new THREE.Vector3(0, 0.09, 0.24),
        new THREE.Vector3(0, 0.12, 0.52),
      ]);
      const lumbarBridgeGeo = trackGeo(
        new THREE.TubeGeometry(lumbarBridgeCurve, seg(14, 10), 0.34 * m.torsoBulk, seg(16, 12), false),
        0.02
      );
      lumbarBridgeGeo.scale(0.92, 0.96, 1.0);
      const lumbarBridgeMesh = new THREE.Mesh(lumbarBridgeGeo, primaryArmorMat);
      pelvisGroup.add(lumbarBridgeMesh);

      // Paired Gluteal / Iliac Croup Muscle Bellies & External Oblique Flank Straps on Quadruped Rump
      if (muscleDef > 0.15) {
        const mScale = 0.8 + muscleDef * 0.32;
        for (const side of [-1, 1]) {
          const gluteGeo = trackGeo(
            new THREE.SphereGeometry(0.19 * m.torsoBulk * mScale, seg(12, 8), seg(10, 8)),
            0.022
          );
          gluteGeo.scale(0.86, 1.08, 1.45);
          const gluteMesh = new THREE.Mesh(gluteGeo, primaryArmorMat);
          gluteMesh.position.set(side * 0.16 * m.torsoBulk, 0.14, -0.14);
          gluteMesh.rotation.y = -side * 0.12;
          pelvisGroup.add(gluteMesh);

          // External Oblique Flank Strap bridging hind haunch to ribcage
          const obliqueFlankGeo = trackGeo(
            new THREE.SphereGeometry(0.14 * m.torsoBulk * mScale, seg(12, 8), seg(10, 8)),
            0.018
          );
          obliqueFlankGeo.scale(0.75, 0.95, 1.85);
          const obliqueFlankMesh = new THREE.Mesh(obliqueFlankGeo, primaryArmorMat);
          obliqueFlankMesh.position.set(side * 0.24 * m.torsoBulk, 0.08, 0.22);
          obliqueFlankMesh.rotation.y = side * 0.08;
          pelvisGroup.add(obliqueFlankMesh);
        }
      }

      // Ventral Underbelly Hide / Scute Plates along the abdomen
      const bellyRingCount = 5;
      for (let b = 0; b < bellyRingCount; b++) {
        const frac = b / (bellyRingCount - 1);
        const zOff = -0.24 + frac * 0.72;
        const rad = (0.24 + Math.sin(frac * Math.PI) * 0.08) * m.torsoBulk;
        const scuteGeo = trackGeo(new THREE.SphereGeometry(rad, seg(14, 10), seg(10, 8)));
        scuteGeo.scale(0.84, 0.55, 0.85);
        const scuteMesh = new THREE.Mesh(scuteGeo, underbodyMat);
        scuteMesh.position.set(0, -0.06 + frac * 0.03, zOff);
        pelvisGroup.add(scuteMesh);
      }
    }
  } else if (isArachnid) {
    if (isUndeadSkeleton) {
      // Exposed 3D Skeletal Ossuary Rib-Cage Abdomen for Undead Arachnids
      for (let r = 0; r < 6; r++) {
        const rFrac = r / 5;
        const archRad = (0.44 - Math.abs(rFrac - 0.45) * 0.22) * m.torsoBulk;
        const archGeo = trackGeo(
          new THREE.TorusGeometry(archRad, 0.034 * m.torsoBulk, seg(8, 6), seg(16, 12), Math.PI * 1.4)
        );
        const archMesh = new THREE.Mesh(archGeo, ivoryFangMat);
        archMesh.position.set(0, 0.12, -0.06 - r * 0.14);
        archMesh.rotation.x = -0.18;
        archMesh.rotation.z = -Math.PI * 0.2;
        pelvisGroup.add(archMesh);
      }
      const spineBarGeo = trackGeo(
        new THREE.CylinderGeometry(0.055 * m.torsoBulk, 0.045 * m.torsoBulk, 0.92, seg(10, 8))
      );
      spineBarGeo.rotateX(Math.PI / 2);
      const spineBar = new THREE.Mesh(spineBarGeo, ivoryFangMat);
      spineBar.position.set(0, 0.22 * m.torsoBulk, -0.34);
      pelvisGroup.add(spineBar);
    } else {
      const abdomenGeo = trackGeo(
        new THREE.SphereGeometry(0.48 * m.torsoBulk, seg(20, 14), seg(14, 10))
      );
      abdomenGeo.scale(1.05, 0.76, 1.48);
      const abdomenMesh = new THREE.Mesh(abdomenGeo, primaryArmorMat);
      abdomenMesh.position.set(0, 0.1, -0.36);
      abdomenMesh.castShadow = true;
      pelvisGroup.add(abdomenMesh);

      for (let r = 0; r < 3; r++) {
        const archGeo = trackGeo(
          new THREE.TorusGeometry(
            (0.42 - r * 0.05) * m.torsoBulk,
            0.038,
            seg(10, 8),
            seg(18, 12),
            Math.PI
          )
        );
        const archMesh = new THREE.Mesh(archGeo, secondaryTrimMat);
        archMesh.position.set(0, 0.12, -0.14 - r * 0.22);
        archMesh.rotation.x = -0.2;
        pelvisGroup.add(archMesh);
      }

      // Blender-Sculpted Chitinous Pedicel Articulation Bridge (Seamlessly unites abdomen, cephalothorax & upper torso)
      const pedicelCurve = new THREE.CatmullRomCurve3([
        new THREE.Vector3(0, 0.08, -0.22),
        new THREE.Vector3(0, 0.07, 0.02),
        new THREE.Vector3(0, 0.14, 0.18),
      ]);
      const pedicelGeo = trackGeo(
        new THREE.TubeGeometry(pedicelCurve, seg(12, 8), 0.26 * m.torsoBulk, seg(14, 10), false),
        0.018
      );
      pedicelGeo.scale(1.12, 0.78, 1.0);
      const pedicelMesh = new THREE.Mesh(pedicelGeo, underbodyMat);
      pelvisGroup.add(pedicelMesh);

      const cephaloGeo = trackGeo(
        new THREE.SphereGeometry(0.38 * m.torsoBulk, seg(16, 12), seg(12, 10))
      );
      cephaloGeo.scale(1.15, 0.72, 1.32);
      const cephaloMesh = new THREE.Mesh(cephaloGeo, secondaryTrimMat);
      cephaloMesh.position.set(0, 0.06, 0.12);
      pelvisGroup.add(cephaloMesh);
    }
  } else if (isUndeadSkeleton) {
    // BLENDER-SCULPTED 3D ANATOMICAL SKELETAL PELVIS, SACRUM, COCCYX & LUMBAR VERTEBRAL COLUMN (100% EXPOSED!)
    const darkBoneCavityMat = trackMat(
      new THREE.MeshStandardMaterial({
        name: 'Mat_SkullSocketVoid',
        color: new THREE.Color('#08090d'),
        roughness: 0.92,
        metalness: 0.05,
        wireframe,
      })
    );

    // 1) Central Sacrum Wedge & Articulated Coccyx Tailbone at Posterior Pelvis
    const sacrumGeo = trackGeo(
      new THREE.ConeGeometry(0.145 * m.torsoBulk, 0.28, seg(10, 8)),
      0.012
    );
    sacrumGeo.scale(1.18, 1.0, 0.58);
    const sacrumMesh = new THREE.Mesh(sacrumGeo, ivoryFangMat);
    sacrumMesh.position.set(0, 0.02, -0.14 * m.torsoBulk);
    sacrumMesh.rotation.x = Math.PI - 0.18;
    sacrumMesh.castShadow = true;
    pelvisGroup.add(sacrumMesh);

    // 2) 4 Exposed 3D Lumbar Vertebrae (L1-L4) rising through the open waist from Sacrum to Ribcage
    const spinalCordGeo = trackGeo(
      new THREE.CylinderGeometry(0.036 * m.torsoBulk, 0.042 * m.torsoBulk, 0.52, seg(10, 8)),
      0.006
    );
    const spinalCordMesh = new THREE.Mesh(spinalCordGeo, darkBoneCavityMat);
    spinalCordMesh.position.set(0, 0.24, -0.12 * m.torsoBulk);
    pelvisGroup.add(spinalCordMesh);

    for (let lv = 0; lv < 4; lv++) {
      const ly = 0.08 + lv * 0.115;
      const lz = (-0.125 + lv * 0.006) * m.torsoBulk;

      // Cylindrical Lumbar Vertebral Body (Centrum)
      const centrumGeo = trackGeo(
        new THREE.CylinderGeometry(
          0.078 * m.torsoBulk,
          0.082 * m.torsoBulk,
          0.068,
          seg(14, 10)
        ),
        0.008
      );
      centrumGeo.scale(1.12, 1.0, 0.92);
      const centrumMesh = new THREE.Mesh(centrumGeo, ivoryFangMat);
      centrumMesh.position.set(0, ly, lz);
      centrumMesh.castShadow = true;
      pelvisGroup.add(centrumMesh);

      // Dark Intervertebral Fibrocartilage Disc between lumbar vertebrae
      const discGeo = trackGeo(
        new THREE.CylinderGeometry(
          0.072 * m.torsoBulk,
          0.072 * m.torsoBulk,
          0.026,
          seg(12, 8)
        ),
        0.002
      );
      const discMesh = new THREE.Mesh(discGeo, darkBoneCavityMat);
      discMesh.position.set(0, ly + 0.046, lz);
      pelvisGroup.add(discMesh);

      // Paired Horizontal Transverse Processes jutting out left & right from each lumbar vertebra
      const transGeo = trackGeo(
        new THREE.CapsuleGeometry(0.022 * m.torsoBulk, 0.16 * m.torsoBulk, 6, 8),
        0.006
      );
      transGeo.rotateZ(Math.PI / 2);
      const transMesh = new THREE.Mesh(transGeo, ivoryFangMat);
      transMesh.position.set(0, ly, lz - 0.035 * m.torsoBulk);
      pelvisGroup.add(transMesh);

      // Posterior Dorsal Spinous Process Spike protruding backward
      const spinousGeo = trackGeo(
        new THREE.ConeGeometry(0.028 * m.torsoBulk, 0.11 * m.torsoBulk, 6),
        0.006
      );
      spinousGeo.rotateX(-Math.PI * 0.58);
      const spinousMesh = new THREE.Mesh(spinousGeo, ivoryFangMat);
      spinousMesh.position.set(0, ly - 0.01, lz - 0.085 * m.torsoBulk);
      pelvisGroup.add(spinousMesh);
    }

    // 3) Paired Butterfly Iliac Pelvic Wings, Ischial Lower Rings, Acetabular Hip Sockets & Pubic Symphysis
    for (const side of [-1, 1]) {
      // Upper Butterfly Ilium Crest & Iliac Ala
      const iliumCurve = new THREE.CatmullRomCurve3([
        new THREE.Vector3(side * 0.08 * m.torsoBulk, 0.08, -0.14 * m.torsoBulk),
        new THREE.Vector3(side * 0.34 * m.torsoBulk, 0.19, -0.02 * m.torsoBulk),
        new THREE.Vector3(side * 0.36 * m.torsoBulk, 0.06, 0.14 * m.torsoBulk),
        new THREE.Vector3(side * 0.24 * m.torsoBulk, -0.06, 0.12 * m.torsoBulk),
        new THREE.Vector3(side * 0.04 * m.torsoBulk, -0.11, 0.19 * m.torsoBulk),
      ]);
      const iliumGeo = trackGeo(
        new THREE.TubeGeometry(iliumCurve, seg(16, 12), 0.046 * m.torsoBulk, seg(10, 8), false),
        0.01
      );
      const iliumMesh = new THREE.Mesh(iliumGeo, ivoryFangMat);
      iliumMesh.castShadow = true;
      pelvisGroup.add(iliumMesh);

      // Sculpted Iliac Wing Plate (Fossa Iliaca)
      const alaGeo = trackGeo(
        new THREE.SphereGeometry(0.135 * m.torsoBulk, seg(12, 8), seg(10, 8)),
        0.012
      );
      alaGeo.scale(0.88, 0.95, 0.38);
      const alaMesh = new THREE.Mesh(alaGeo, ivoryFangMat);
      alaMesh.position.set(side * 0.24 * m.torsoBulk, 0.09, 0.02 * m.torsoBulk);
      alaMesh.rotation.y = side * 0.48;
      alaMesh.rotation.x = 0.22;
      pelvisGroup.add(alaMesh);

      // Lower Ischial Ramus & Obturator Foramen Loop
      const ischiumCurve = new THREE.CatmullRomCurve3([
        new THREE.Vector3(side * 0.24 * m.torsoBulk, -0.04, 0.04 * m.torsoBulk),
        new THREE.Vector3(side * 0.21 * m.torsoBulk, -0.17, -0.04 * m.torsoBulk),
        new THREE.Vector3(side * 0.05 * m.torsoBulk, -0.13, 0.18 * m.torsoBulk),
      ]);
      const ischiumGeo = trackGeo(
        new THREE.TubeGeometry(ischiumCurve, seg(10, 8), 0.032 * m.torsoBulk, seg(8, 6), false),
        0.008
      );
      pelvisGroup.add(new THREE.Mesh(ischiumGeo, ivoryFangMat));

      // Acetabular Hip Socket Cup where the Femoral Head articulates
      const acetabulumGeo = trackGeo(
        new THREE.TorusGeometry(0.072 * m.torsoBulk, 0.024 * m.torsoBulk, seg(8, 6), seg(14, 10)),
        0.006
      );
      acetabulumGeo.rotateY(side * Math.PI * 0.42);
      const acetabulumMesh = new THREE.Mesh(acetabulumGeo, ivoryFangMat);
      acetabulumMesh.position.set(side * 0.24 * m.torsoBulk, -0.04, 0.02 * m.torsoBulk);
      pelvisGroup.add(acetabulumMesh);
    }

    // Anterior Pubic Symphysis Joint bridging Left & Right Pubic Bones
    const symphysisGeo = trackGeo(
      new THREE.CylinderGeometry(0.038 * m.torsoBulk, 0.038 * m.torsoBulk, 0.09, seg(10, 8)),
      0.006
    );
    const symphysisMesh = new THREE.Mesh(symphysisGeo, ivoryFangMat);
    symphysisMesh.position.set(0, -0.11, 0.19 * m.torsoBulk);
    pelvisGroup.add(symphysisMesh);

    // Open-Front Royal Necromancer / Sovereign Mantle Skirt (leaves front pelvis & leg bones 100% visible!)
    if (
      config.archetype === 'arch_lich' ||
      config.archetype === 'bone_crown_lich_king' ||
      config.archetype === 'wraith_soul_reaper' ||
      config.archetype === 'skeletal_archbishop_pontiff' ||
      config.archetype === 'emerald_crown_necromancer_lord' ||
      config.archetype === 'blood_crown_reaper_marquis' ||
      config.archetype === 'star_crown_astral_lich' ||
      config.archetype === 'crowned_catacomb_pharaoh' ||
      config.archetype === 'elite_plague_doctor_lich_skeleton' ||
      config.archetype === 'elite_caged_rib_jailer_skeleton' ||
      config.archetype === 'elite_spiked_wheel_bone_martyr' ||
      config.archetype === 'elite_twin_skull_ettin_skeleton' ||
      config.archetype === 'elite_mammoth_ossuary_behemoth'
    ) {
      const robePts = [
        new THREE.Vector2(0.27 * m.torsoBulk, 0.06),
        new THREE.Vector2(0.38 * m.torsoBulk, -0.45),
        new THREE.Vector2(0.52 * m.torsoBulk, -0.92),
      ];
      // Open-front mantle (back & flanks only) so skeletal pelvis, femurs & tibias are never hidden
      const robeGeo = trackGeo(
        new THREE.LatheGeometry(robePts, seg(16, 12), Math.PI * 0.38, Math.PI * 1.24)
      );
      const robeMesh = new THREE.Mesh(robeGeo, primaryArmorMat);
      robeMesh.castShadow = true;
      pelvisGroup.add(robeMesh);
    }
  } else {
    const pelvisPts = [
      new THREE.Vector2(0.01, -0.24),
      new THREE.Vector2(0.27 * m.torsoBulk, -0.16),
      new THREE.Vector2(0.38 * m.torsoBulk, 0.02),
      new THREE.Vector2(0.33 * m.torsoBulk, 0.18),
      new THREE.Vector2(0.36 * m.torsoBulk, 0.36),
      new THREE.Vector2(0.32 * m.torsoBulk, 0.48),
      new THREE.Vector2(0.01, 0.52),
    ];
    const pelvisGeo = trackGeo(new THREE.LatheGeometry(pelvisPts, seg(18, 12)));
    pelvisGeo.scale(1.12, 1.0, 0.86);
    const pelvisMesh = new THREE.Mesh(pelvisGeo, underbodyMat);
    pelvisMesh.castShadow = true;
    pelvisGroup.add(pelvisMesh);

    // Blender-Sculpted Iliac Crest & External Oblique Hip-to-Waist Bridges
    for (const side of [-1, 1]) {
      // Muscular External Oblique & Gluteal Medius Waist-to-Hip Bridge
      const obliqueHipGeo = trackGeo(
        new THREE.SphereGeometry(0.145 * m.torsoBulk, seg(12, 8), seg(10, 8)),
        0.018
      );
      obliqueHipGeo.scale(0.82, 1.45, 1.12);
      const obliqueHipMesh = new THREE.Mesh(
        obliqueHipGeo,
        isOrganicCreature ? primaryArmorMat : underbodyMat
      );
      obliqueHipMesh.position.set(side * 0.25 * m.torsoBulk, 0.16, 0.02);
      obliqueHipMesh.rotation.z = -side * 0.18;
      pelvisGroup.add(obliqueHipMesh);
    }

    for (let w = 0; w < 2; w++) {
      const cinctureGeo = trackGeo(
        new THREE.TorusGeometry((0.31 + w * 0.02) * m.torsoBulk, 0.036, seg(10, 8), seg(18, 12))
      );
      cinctureGeo.rotateX(Math.PI / 2);
      cinctureGeo.scale(1.12, 0.86, 1.0);
      const cinctureMesh = new THREE.Mesh(cinctureGeo, secondaryTrimMat);
      cinctureMesh.position.y = 0.08 + w * 0.11;
      pelvisGroup.add(cinctureMesh);
    }

    const beltShape = new THREE.Shape();
    beltShape.moveTo(-0.24 * m.torsoBulk, 0.12);
    beltShape.lineTo(0.24 * m.torsoBulk, 0.12);
    beltShape.lineTo(0.16 * m.torsoBulk, -0.26);
    beltShape.lineTo(0, -0.36);
    beltShape.lineTo(-0.16 * m.torsoBulk, -0.26);
    beltShape.closePath();

    const beltGeo = trackGeo(
      new THREE.ExtrudeGeometry(beltShape, {
        depth: 0.07,
        bevelEnabled: true,
        bevelThickness: 0.02,
        bevelSize: 0.02,
        bevelSegments: 2,
      })
    );
    const frontTasset = new THREE.Mesh(beltGeo, secondaryTrimMat);
    frontTasset.position.set(0, 0.02, 0.25 * m.torsoBulk);
    frontTasset.castShadow = true;
    pelvisGroup.add(frontTasset);

    if (
      config.archetype === 'solar_phoenix' ||
      config.archetype === 'void_djinn' ||
      config.archetype === 'zephyr_storm_elemental' ||
      config.archetype === 'tempest_sky_djinn' ||
      config.archetype === 'cyclone_air_archon' ||
      config.archetype === 'plague_hierophant_mummy' ||
      config.archetype === 'banshee_frozen_throne'
    ) {
      const robePts = [
        new THREE.Vector2(0.26 * m.torsoBulk, 0.12),
        new THREE.Vector2(0.38 * m.torsoBulk, -0.45),
        new THREE.Vector2(0.54 * m.torsoBulk, -0.95),
      ];
      const robeGeo = trackGeo(new THREE.LatheGeometry(robePts, seg(20, 14)));
      const robeMesh = new THREE.Mesh(robeGeo, primaryArmorMat);
      robeMesh.castShadow = true;
      pelvisGroup.add(robeMesh);
    }
  }

  tagExplodablePart(pelvisGroup, new THREE.Vector3(0, -0.35, -0.25), {
    name: isSerpentine
      ? 'Anillos Serpentinos Esculpidos'
      : isQuadruped
      ? 'Grupa Muscular, Pelvis y Cintura Abdominal'
      : isArachnid
      ? 'Abdomen y Cefalotórax Quitinoso'
      : isUndeadSkeleton
      ? 'Pelvis Ósea, Sacro y Columna Lumbar 3D'
      : 'Pelvis, Abdomen y Faldón',
    category: 'Primary Silhouette',
    vertexCount: 420,
    triangleCount: 360,
    materialSummary: 'Geometría continua suavizada + musculatura y sombreado AO por vértice',
  });
  explodableParts.push(pelvisGroup);

  // 2. HIERARCHICAL 2-BONE LEGS WITH ANATOMICAL QUADS/HAMSTRINGS/CALVES, FUR FEATHERING & SCULPTED PAWS
  const buildLeg = (isLeft: boolean): { legGroup: THREE.Bone; kneeGroup: THREE.Bone } => {
    const side = isLeft ? 1 : -1;
    const legGroup = makeBone(isLeft ? 'Leg_L' : 'Leg_R');
    const zPos = isQuadruped ? -0.44 : 0;
    legGroup.position.set(side * 0.25 * m.torsoBulk, pelvisY, zPos);
    rootBone.add(legGroup);

    const kneeZ = isQuadruped ? 0.08 : 0.04;
    const kneeY = -legHeight * 0.48;
    const kneeGroup = makeBone(isLeft ? 'Knee_Hock_L' : 'Knee_Hock_R');
    kneeGroup.position.set(0, kneeY, kneeZ);
    legGroup.add(kneeGroup);

    if (isSerpentine) {
      return { legGroup, kneeGroup };
    }

    if (isFloating && isBeholder) {
      const tendrilCurve = new THREE.CatmullRomCurve3([
        new THREE.Vector3(0, 0.1, 0),
        new THREE.Vector3(side * 0.12, -0.35, -0.1),
        new THREE.Vector3(side * 0.05, -0.7, 0.08),
      ]);
      const tendrilGeo = trackGeo(
        new THREE.TubeGeometry(tendrilCurve, seg(14, 10), 0.052, seg(10, 8), false)
      );
      legGroup.add(new THREE.Mesh(tendrilGeo, underbodyMat));
    } else if (isArachnid) {
      for (let k = 0; k < 4; k++) {
        const zOff = (k - 1.5) * 0.26;
        const reachX = side * (0.82 + (k % 2) * 0.16) * m.limbLength;

        const coxaGeo = trackGeo(
          new THREE.SphereGeometry(0.1 * m.torsoBulk, seg(10, 8), seg(8, 6))
        );
        const coxaMesh = new THREE.Mesh(coxaGeo, secondaryTrimMat);
        coxaMesh.position.set(-side * 0.04, 0.04, zOff * 0.52);
        legGroup.add(coxaMesh);

        const curve = new THREE.CatmullRomCurve3([
          new THREE.Vector3(-side * 0.04, 0.04, zOff * 0.52),
          new THREE.Vector3(reachX * 0.55, 0.36, zOff * 0.9),
          new THREE.Vector3(reachX * 1.02, -legHeight * 0.35, zOff * 1.22),
          new THREE.Vector3(reachX * 0.92, -legHeight + 0.02, zOff * 1.32),
        ]);
        const spiderLegGeo = trackGeo(
          new THREE.TubeGeometry(curve, seg(16, 12), 0.058, seg(10, 8), false)
        );
        const spiderMesh = new THREE.Mesh(spiderLegGeo, primaryArmorMat);
        spiderMesh.castShadow = true;
        legGroup.add(spiderMesh);

        const apexJointGeo = trackGeo(
          new THREE.SphereGeometry(0.078, seg(10, 8), seg(8, 6))
        );
        const apexJoint = new THREE.Mesh(apexJointGeo, secondaryTrimMat);
        apexJoint.position.set(reachX * 0.55, 0.36, zOff * 0.9);
        legGroup.add(apexJoint);
      }
    } else if (isUndeadSkeleton) {
      // BLENDER-SCULPTED 3D SKELETAL LEG BONES: FEMUR, PATELLA, DUAL TIBIA + FIBULA & 5-RAY METATARSAL/PHALANGEAL FOOT
      // 1) Spherical Femoral Head seated in Pelvic Acetabulum + Angled Femoral Neck + Greater Trochanter
      const femoralHeadGeo = trackGeo(
        new THREE.SphereGeometry(0.068 * m.torsoBulk, seg(12, 8), seg(10, 8)),
        0.008
      );
      const femoralHeadMesh = new THREE.Mesh(femoralHeadGeo, ivoryFangMat);
      femoralHeadMesh.position.set(-side * 0.045 * m.torsoBulk, 0.04, 0.01);
      legGroup.add(femoralHeadMesh);

      const trochanterGeo = trackGeo(
        new THREE.SphereGeometry(0.064 * m.torsoBulk, seg(10, 8), seg(8, 6)),
        0.012
      );
      trochanterGeo.scale(0.95, 1.35, 1.05);
      const trochanterMesh = new THREE.Mesh(trochanterGeo, ivoryFangMat);
      trochanterMesh.position.set(side * 0.048 * m.torsoBulk, 0.02, -0.01);
      legGroup.add(trochanterMesh);

      // 2) Anatomical Femur Diaphysis Shaft + Distal Bicondylar Knee Epiphysis
      const femurCurve = new THREE.CatmullRomCurve3([
        new THREE.Vector3(-side * 0.035 * m.torsoBulk, 0.04, 0.01),
        new THREE.Vector3(side * 0.032 * m.torsoBulk, 0.0, 0.01),
        new THREE.Vector3(side * 0.012 * m.torsoBulk, kneeY * 0.5, kneeZ * 0.55 + 0.015),
        new THREE.Vector3(0, kneeY, kneeZ),
      ]);
      const femurShaftGeo = trackGeo(
        new THREE.TubeGeometry(
          femurCurve,
          seg(14, 10),
          0.054 * m.torsoBulk,
          seg(12, 8),
          false
        ),
        0.01
      );
      const femurShaftMesh = new THREE.Mesh(femurShaftGeo, ivoryFangMat);
      femurShaftMesh.castShadow = true;
      legGroup.add(femurShaftMesh);

      // Wide Medial & Lateral Femoral Condyles at the Knee Joint
      const condyleGeo = trackGeo(
        new THREE.CapsuleGeometry(0.058 * m.torsoBulk, 0.075 * m.torsoBulk, seg(8, 6), seg(10, 8)),
        0.008
      );
      condyleGeo.rotateZ(Math.PI / 2);
      const femoralCondyles = new THREE.Mesh(condyleGeo, ivoryFangMat);
      femoralCondyles.position.set(0, 0.025, 0);
      kneeGroup.add(femoralCondyles);

      // 3) Sculpted Patella (Kneecap Sesamoid Bone) floating anteriorly over the Knee Condyles
      const patellaGeo = trackGeo(
        new THREE.SphereGeometry(0.054 * m.torsoBulk, seg(10, 8), seg(8, 6)),
        0.008
      );
      patellaGeo.scale(1.05, 1.22, 0.55);
      const patellaMesh = new THREE.Mesh(patellaGeo, ivoryFangMat);
      patellaMesh.position.set(0, 0.02, 0.068 * m.torsoBulk);
      kneeGroup.add(patellaMesh);

      // 4) Dual Lower Leg Bones: Medial Tibia (Shinbone) + Lateral Fibula (Calf Bone) with Visible Interosseous Gap!
      const shinLen = legHeight * 0.52;
      const ankleZ = isQuadruped ? -0.05 : 0.0;
      const footY = -shinLen + 0.05;

      // Proximal Tibial Plateau
      const tibialPlateauGeo = trackGeo(
        new THREE.CylinderGeometry(
          0.068 * m.torsoBulk,
          0.048 * m.torsoBulk,
          0.065,
          seg(12, 8)
        ),
        0.008
      );
      const tibialPlateau = new THREE.Mesh(tibialPlateauGeo, ivoryFangMat);
      tibialPlateau.position.set(-side * 0.014 * m.torsoBulk, -0.03, 0.008);
      kneeGroup.add(tibialPlateau);

      // Medial Tibia Bone Shaft (Thicker anterior-medial shinbone)
      const tibiaCurve = new THREE.CatmullRomCurve3([
        new THREE.Vector3(-side * 0.022 * m.torsoBulk, -0.02, 0.015),
        new THREE.Vector3(-side * 0.026 * m.torsoBulk, -shinLen * 0.48, ankleZ * 0.4 + 0.022),
        new THREE.Vector3(-side * 0.02 * m.torsoBulk, footY + 0.01, ankleZ + 0.005),
      ]);
      const tibiaGeo = trackGeo(
        new THREE.TubeGeometry(
          tibiaCurve,
          seg(12, 8),
          0.044 * m.torsoBulk,
          seg(10, 8),
          false
        ),
        0.008
      );
      const tibiaMesh = new THREE.Mesh(tibiaGeo, ivoryFangMat);
      tibiaMesh.castShadow = true;
      kneeGroup.add(tibiaMesh);

      // Lateral Fibula Bone Shaft (Slender outer calf bone separated by a visible gap from the Tibia!)
      const fibulaCurve = new THREE.CatmullRomCurve3([
        new THREE.Vector3(side * 0.048 * m.torsoBulk, -0.04, -0.015),
        new THREE.Vector3(side * 0.052 * m.torsoBulk, -shinLen * 0.5, ankleZ * 0.4 - 0.012),
        new THREE.Vector3(side * 0.042 * m.torsoBulk, footY + 0.005, ankleZ - 0.01),
      ]);
      const fibulaGeo = trackGeo(
        new THREE.TubeGeometry(
          fibulaCurve,
          seg(12, 8),
          0.024 * m.torsoBulk,
          seg(8, 6),
          false
        ),
        0.006
      );
      const fibulaMesh = new THREE.Mesh(fibulaGeo, ivoryFangMat);
      fibulaMesh.castShadow = true;
      kneeGroup.add(fibulaMesh);

      // 5) Skeletal Ankle Malleoli, Calcaneus Heel Bone, 5 Metatarsal Bones & 5 Clawed Toe Phalanges
      const talusGeo = trackGeo(
        new THREE.CapsuleGeometry(0.046 * m.torsoBulk, 0.062 * m.torsoBulk, 6, 10),
        0.006
      );
      talusGeo.rotateZ(Math.PI / 2);
      const talusMesh = new THREE.Mesh(talusGeo, ivoryFangMat);
      talusMesh.position.set(0, footY + 0.01, ankleZ);
      kneeGroup.add(talusMesh);

      // Posterior Calcaneus (Heel Bone spur)
      const calcaneusGeo = trackGeo(
        new THREE.SphereGeometry(0.048 * m.torsoBulk, seg(10, 8), seg(8, 6)),
        0.008
      );
      calcaneusGeo.scale(0.85, 0.78, 1.45);
      const calcaneusMesh = new THREE.Mesh(calcaneusGeo, ivoryFangMat);
      calcaneusMesh.position.set(0, footY - 0.015, ankleZ - 0.055 * m.torsoBulk);
      kneeGroup.add(calcaneusMesh);

      // 5 Individual Metatarsal Foot Bones + Proximal & Distal Toe Phalanges
      for (let t = -2; t <= 2; t++) {
        const toeX = t * 0.032 * m.torsoBulk;
        const toeSpread = t * 0.08;
        const metaLen = (0.135 - Math.abs(t) * 0.01) * m.torsoBulk;

        const metatarsalCurve = new THREE.CatmullRomCurve3([
          new THREE.Vector3(toeX * 0.65, footY + 0.005, ankleZ + 0.015),
          new THREE.Vector3(toeX, footY - 0.022, ankleZ + 0.015 + metaLen),
        ]);
        const metatarsalGeo = trackGeo(
          new THREE.TubeGeometry(
            metatarsalCurve,
            6,
            0.016 * m.torsoBulk,
            6,
            false
          ),
          0.003
        );
        kneeGroup.add(new THREE.Mesh(metatarsalGeo, ivoryFangMat));

        // Proximal + Distal Toe Phalanx Claw
        const toePhalanxGeo = trackGeo(
          new THREE.ConeGeometry(0.017 * m.torsoBulk, 0.095 * m.torsoBulk, 6),
          0.003
        );
        toePhalanxGeo.rotateX(Math.PI / 2.15);
        const toePhalanx = new THREE.Mesh(toePhalanxGeo, ivoryFangMat);
        toePhalanx.position.set(
          toeX * 1.08,
          footY - 0.028,
          ankleZ + 0.02 + metaLen + 0.042 * m.torsoBulk
        );
        toePhalanx.rotation.y = toeSpread;
        kneeGroup.add(toePhalanx);
      }
    } else {
      // 1) Gluteal / Hip Haunch Muscle + Blender-Sculpted Acetabular Hip-to-Pelvis Bridge
      const hipHaunchGeo = trackGeo(
        new THREE.SphereGeometry(0.245 * m.torsoBulk, seg(16, 12), seg(12, 10)),
        0.022
      );
      hipHaunchGeo.scale(0.96, 1.24, 1.3);
      const hipHaunchMesh = new THREE.Mesh(
        hipHaunchGeo,
        isQuadruped ? primaryArmorMat : secondaryTrimMat
      );
      hipHaunchMesh.position.set(-side * 0.045, 0.045, isQuadruped ? 0.04 : 0);
      hipHaunchMesh.castShadow = true;
      legGroup.add(hipHaunchMesh);

      // Acetabular Hip Socket & Gluteal-Femoral Tendon Bridge (Eliminates any gap between Pelvis_Chassis and Thigh)
      const hipBridgeCurve = new THREE.CatmullRomCurve3([
        new THREE.Vector3(-side * 0.16 * m.torsoBulk, 0.14, isQuadruped ? 0.08 : -0.02),
        new THREE.Vector3(-side * 0.05 * m.torsoBulk, 0.04, 0.01),
        new THREE.Vector3(0, kneeY * 0.28, kneeZ * 0.35),
      ]);
      const hipBridgeGeo = trackGeo(
        new THREE.TubeGeometry(
          hipBridgeCurve,
          seg(12, 8),
          0.148 * m.torsoBulk,
          seg(12, 8),
          false
        ),
        0.018
      );
      const hipBridgeMesh = new THREE.Mesh(
        hipBridgeGeo,
        isQuadruped || isOrganicCreature ? primaryArmorMat : underbodyMat
      );
      legGroup.add(hipBridgeMesh);

      // 2) Continuous Upper Thigh / Femur Curve (Starts deep inside hip (0, +0.09, 0) and ends deep inside knee (0, kneeY - 0.05, kneeZ))
      const thighCurve = new THREE.CatmullRomCurve3([
        new THREE.Vector3(-side * 0.03, 0.09, 0),
        new THREE.Vector3(side * 0.01, kneeY * 0.5, kneeZ * 0.65 + 0.03),
        new THREE.Vector3(0, kneeY - 0.05, kneeZ),
      ]);
      const thighGeo = trackGeo(
        new THREE.TubeGeometry(
          thighCurve,
          seg(14, 10),
          0.168 * m.torsoBulk,
          seg(14, 10),
          false
        ),
        0.024
      );
      const thighMesh = new THREE.Mesh(thighGeo, isQuadruped ? primaryArmorMat : underbodyMat);
      thighMesh.castShadow = true;
      legGroup.add(thighMesh);

      // 2B) Sculpted Quadriceps (Rectus Femoris + Vastus Lateralis) & Hamstring Muscle Bellies
      if (muscleDef > 0.15) {
        const mScale = 0.78 + muscleDef * 0.34;
        // Anterior-outer Quadriceps teardrop belly
        const quadGeo = trackGeo(
          new THREE.SphereGeometry(0.135 * m.torsoBulk * mScale, seg(12, 8), seg(10, 8)),
          0.025
        );
        quadGeo.scale(0.84, 1.75, 1.08);
        const quadMesh = new THREE.Mesh(
          quadGeo,
          isQuadruped || isOrganicCreature ? primaryArmorMat : underbodyMat
        );
        quadMesh.position.set(side * 0.045 * m.torsoBulk, kneeY * 0.44, kneeZ * 0.5 + 0.065 * m.torsoBulk);
        quadMesh.rotation.x = 0.14;
        quadMesh.rotation.z = side * 0.08;
        legGroup.add(quadMesh);

        // Posterior Biceps Femoris / Hamstring sweep
        const hamGeo = trackGeo(
          new THREE.SphereGeometry(0.125 * m.torsoBulk * mScale, seg(12, 8), seg(10, 8)),
          0.022
        );
        hamGeo.scale(0.82, 1.65, 1.12);
        const hamMesh = new THREE.Mesh(
          hamGeo,
          isQuadruped || isOrganicCreature ? primaryArmorMat : underbodyMat
        );
        hamMesh.position.set(side * 0.02 * m.torsoBulk, kneeY * 0.38, -0.06 * m.torsoBulk);
        hamMesh.rotation.x = -0.15;
        legGroup.add(hamMesh);
      }

      // 2C) 3D Haunch Hair Strands along outer/back thigh on furry animals
      if (furDensity > 0.15 && hairStyle !== 'none') {
        const fScale = 0.75 + furDensity * 0.38;
        for (let t = 0; t < 3; t++) {
          const tuftGeo = trackFurLock(0.068 * m.torsoBulk * fScale, (0.24 - t * 0.03) * fScale, 0.09);
          const tuftMesh = new THREE.Mesh(tuftGeo, hairStrandMat);
          tuftMesh.position.set(
            side * (0.14 - t * 0.02) * m.torsoBulk,
            -0.02 + kneeY * (0.2 + t * 0.25),
            -0.08 * m.torsoBulk
          );
          tuftMesh.rotation.x = -Math.PI * 0.68 - t * 0.08;
          tuftMesh.rotation.z = -side * 0.32;
          legGroup.add(tuftMesh);
        }
      }

      // 3) Blender-Sculpted Bicondylar Knee / Hock Joint Capsule + Patellar Tendon Bridge
      const kneeSphereGeo = trackGeo(
        new THREE.CapsuleGeometry(0.145 * m.torsoBulk, 0.11 * m.torsoBulk, seg(10, 8), seg(14, 10)),
        0.016
      );
      kneeSphereGeo.scale(1.04, 1.12, 1.14);
      const kneeSphere = new THREE.Mesh(
        kneeSphereGeo,
        isQuadruped ? primaryArmorMat : secondaryTrimMat
      );
      kneeGroup.add(kneeSphere);

      // Patellar Tendon & Popliteal Flexor Bridge spanning across the knee hinge
      const patellarCurve = new THREE.CatmullRomCurve3([
        new THREE.Vector3(0, 0.11, 0.055 * m.torsoBulk),
        new THREE.Vector3(0, 0.0, 0.082 * m.torsoBulk),
        new THREE.Vector3(0, -0.12, 0.048 * m.torsoBulk),
      ]);
      const patellarGeo = trackGeo(
        new THREE.TubeGeometry(patellarCurve, seg(10, 8), 0.068 * m.torsoBulk, seg(10, 8), false),
        0.012
      );
      const patellarMesh = new THREE.Mesh(
        patellarGeo,
        isOrganicCreature ? primaryArmorMat : secondaryTrimMat
      );
      kneeGroup.add(patellarMesh);

      if (!isOrganicCreature) {
        const kneeGuardGeo = trackGeo(
          new THREE.SphereGeometry(0.135 * m.torsoBulk, seg(12, 8), seg(10, 8))
        );
        kneeGuardGeo.scale(0.9, 1.25, 0.85);
        const kneeMesh = new THREE.Mesh(kneeGuardGeo, secondaryTrimMat);
        kneeMesh.position.set(0, 0.02, 0.07);
        kneeGroup.add(kneeMesh);
      }

      // 4) Continuous Lower Leg / Shin Curve (Starts deep inside knee (0, +0.07, 0) and ends deep inside ankle/paw)
      const shinLen = legHeight * 0.52;
      const ankleZ = isQuadruped ? -0.05 : 0.0;
      const footY = -shinLen + 0.05;

      const shinCurve = new THREE.CatmullRomCurve3([
        new THREE.Vector3(0, 0.07, 0),
        new THREE.Vector3(0, -shinLen * 0.48, ankleZ * 0.4 + 0.02),
        new THREE.Vector3(0, footY - 0.02, ankleZ),
      ]);
      const greaveGeo = trackGeo(
        new THREE.TubeGeometry(
          shinCurve,
          seg(14, 10),
          0.134 * m.torsoBulk,
          seg(14, 10),
          false
        ),
        0.022
      );
      const greaveMesh = new THREE.Mesh(greaveGeo, primaryArmorMat);
      greaveMesh.castShadow = true;
      kneeGroup.add(greaveMesh);

      // 4B) Sculpted Gastrocnemius (Calf / Gaskin) Muscle Belly on Upper Lower Leg
      if (muscleDef > 0.15) {
        const mScale = 0.78 + muscleDef * 0.34;
        const calfGeo = trackGeo(
          new THREE.SphereGeometry(0.112 * m.torsoBulk * mScale, seg(12, 8), seg(10, 8)),
          0.022
        );
        calfGeo.scale(0.88, 1.55, 1.15);
        const calfMesh = new THREE.Mesh(calfGeo, primaryArmorMat);
        calfMesh.position.set(side * 0.015 * m.torsoBulk, -shinLen * 0.32, -0.055 * m.torsoBulk);
        calfMesh.rotation.x = 0.18;
        kneeGroup.add(calfMesh);
      }

      // 4C) 3D Hind Hock Hair Feathering Strands
      if (furDensity > 0.15 && hairStyle !== 'none') {
        const fScale = 0.72 + furDensity * 0.36;
        for (let t = 0; t < 2; t++) {
          const hockTuftGeo = trackFurLock(0.052 * m.torsoBulk * fScale, 0.2 * fScale, 0.08);
          const hockTuft = new THREE.Mesh(hockTuftGeo, hairStrandMat);
          hockTuft.position.set(
            side * 0.02 * m.torsoBulk,
            -shinLen * (0.22 + t * 0.28),
            -0.08 * m.torsoBulk
          );
          hockTuft.rotation.x = -Math.PI * 0.65;
          kneeGroup.add(hockTuft);
        }
      }

      // 5) Blender-Sculpted Malleolar Ankle Joint + Tarsal Instep Bridge + Clawed Foot/Paw
      const ankleSphereGeo = trackGeo(
        new THREE.CapsuleGeometry(0.115 * m.torsoBulk, 0.075 * m.torsoBulk, seg(8, 6), seg(12, 8)),
        0.014
      );
      const ankleSphere = new THREE.Mesh(ankleSphereGeo, primaryArmorMat);
      ankleSphere.position.set(0, footY + 0.015, ankleZ);
      kneeGroup.add(ankleSphere);

      // Tarsal-Instep Bridge smoothly uniting the lower shin with the dorsal foot/paw
      const instepBridgeCurve = new THREE.CatmullRomCurve3([
        new THREE.Vector3(0, footY + 0.06, ankleZ - 0.01),
        new THREE.Vector3(0, footY + 0.01, ankleZ + 0.04),
        new THREE.Vector3(0, footY - 0.01, ankleZ + 0.11),
      ]);
      const instepBridgeGeo = trackGeo(
        new THREE.TubeGeometry(instepBridgeCurve, seg(10, 8), 0.105 * m.torsoBulk, seg(10, 8), false),
        0.014
      );
      const instepBridgeMesh = new THREE.Mesh(instepBridgeGeo, primaryArmorMat);
      kneeGroup.add(instepBridgeMesh);

      const pawGeo = trackGeo(
        new THREE.SphereGeometry(0.155 * m.torsoBulk, seg(14, 10), seg(10, 8))
      );
      pawGeo.scale(0.96, 0.55, 1.45);
      const pawMesh = new THREE.Mesh(pawGeo, secondaryTrimMat);
      pawMesh.position.set(0, footY - 0.01, ankleZ + 0.08);
      kneeGroup.add(pawMesh);

      // 3 Sculpted Metatarsal Toes + Curved Claws overlapping the front of pawMesh
      for (let c = -1; c <= 1; c++) {
        const toeGeo = trackGeo(
          new THREE.SphereGeometry(0.058 * m.torsoBulk, seg(10, 8), seg(8, 6))
        );
        toeGeo.scale(0.8, 0.68, 1.45);
        const toeMesh = new THREE.Mesh(toeGeo, primaryArmorMat);
        toeMesh.position.set(c * 0.065 * m.torsoBulk, footY - 0.02, ankleZ + 0.19);
        kneeGroup.add(toeMesh);

        if (isDragon || isQuadruped || isOrganicCreature) {
          const clawGeo = trackGeo(new THREE.ConeGeometry(0.032, 0.16, seg(10, 8)));
          clawGeo.rotateX(Math.PI / 2.15);
          const clawMesh = new THREE.Mesh(clawGeo, ivoryFangMat);
          clawMesh.position.set(c * 0.068 * m.torsoBulk, footY - 0.025, ankleZ + 0.27);
          kneeGroup.add(clawMesh);
        }
      }
    }

    tagExplodablePart(legGroup, new THREE.Vector3(side * 0.5, -0.45, -0.15), {
      name: isLeft ? 'Extremidad Inferior Izquierda' : 'Extremidad Inferior Derecha',
      category: 'Primary Silhouette',
      vertexCount: 480,
      triangleCount: 420,
      materialSummary: 'Cuádriceps, isquiotibiales y gemelo esculpidos + mechones 3D + zarpa',
    });
    explodableParts.push(legGroup);

    return { legGroup, kneeGroup };
  };

  const { legGroup: leftLegGroup, kneeGroup: leftKneeGroup } = buildLeg(true);
  const { legGroup: rightLegGroup, kneeGroup: rightKneeGroup } = buildLeg(false);

  // 3. TORSO / QUADRUPED THORACIC RIBCAGE / DRAGON CHEST / BEHOLDER ORB
  const torsoGroup = makeBone('Torso_Main');
  if (isQuadruped && !isCentaur) {
    // Positioned at the middle of the spine so the front ribcage overlaps the rear croup seamlessly
    torsoGroup.position.set(0, 0.06, 0.38);
  } else if (isCentaur) {
    torsoGroup.position.set(0, 0.22, 0.56);
  } else {
    torsoGroup.position.set(0, 0.15, 0);
  }
  pelvisGroup.add(torsoGroup);

  const torsoHeight = 0.92 * m.heightScale;

  if (isBeholder) {
    const orbBodyGeo = trackGeo(
      new THREE.SphereGeometry(0.62 * m.torsoBulk, seg(24, 16), seg(18, 12))
    );
    const orbBodyMesh = new THREE.Mesh(orbBodyGeo, primaryArmorMat);
    orbBodyMesh.position.y = torsoHeight * 0.55;
    orbBodyMesh.castShadow = true;
    torsoGroup.add(orbBodyMesh);

    const browGeo = trackGeo(
      new THREE.TorusGeometry(0.44 * m.torsoBulk, 0.08, seg(12, 8), seg(24, 16))
    );
    const browMesh = new THREE.Mesh(browGeo, secondaryTrimMat);
    browMesh.position.set(0, torsoHeight * 0.58, 0.42 * m.torsoBulk);
    torsoGroup.add(browMesh);
  } else if (isQuadruped && !isCentaur) {
    // Continuous Quadruped Thoracic Ribcage, Brisket & Muscular Shoulder Withers
    // Overlaps pelvisGroup croup at z = -0.25 and extends forward to z = +0.58 where forelegs & neck attach!
    if (!isUndeadSkeleton) {
      const thoraxCurve = new THREE.CatmullRomCurve3([
        new THREE.Vector3(0, -0.02, -0.24),
        new THREE.Vector3(0, 0.06 * m.heightScale, 0.08),
        new THREE.Vector3(0, 0.16 * m.heightScale, 0.36),
        new THREE.Vector3(0, 0.24 * m.heightScale, 0.54),
      ]);
      const thoraxGeo = trackGeo(
        new THREE.TubeGeometry(
          thoraxCurve,
          seg(18, 12),
          0.41 * m.torsoBulk,
          seg(18, 12),
          false
        ),
        0.024
      );
      thoraxGeo.scale(0.92 * m.shoulderWidth, 1.06, 1.0);
      const thoraxMesh = new THREE.Mesh(thoraxGeo, primaryArmorMat);
      thoraxMesh.castShadow = true;
      torsoGroup.add(thoraxMesh);

      // Pectoral Brisket / Front Chest Sphere closing the front of the ribcage smoothly
      const brisketGeo = trackGeo(
        new THREE.SphereGeometry(0.39 * m.torsoBulk, seg(18, 12), seg(14, 10)),
        0.022
      );
      brisketGeo.scale(0.94 * m.shoulderWidth, 1.08, 1.15);
      const brisketMesh = new THREE.Mesh(brisketGeo, primaryArmorMat);
      brisketMesh.position.set(0, 0.16 * m.heightScale, 0.42);
      brisketMesh.castShadow = true;
      torsoGroup.add(brisketMesh);

      // 3A) Quadruped Anatomical Musculature: Paired Pectoral Brisket Lobes, Scapular Withers & Serratus Rib Bands
      if (muscleDef > 0.15) {
        const mScale = 0.78 + muscleDef * 0.34;
        for (const side of [-1, 1]) {
          // Paired Anterior Pectoral Chest Lobes (creates distinct sternal cleavage on front chest)
          const pecLobeGeo = trackGeo(
            new THREE.SphereGeometry(0.21 * m.torsoBulk * mScale, seg(12, 8), seg(10, 8)),
            0.025
          );
          pecLobeGeo.scale(0.88, 1.05, 1.25);
          const pecLobe = new THREE.Mesh(pecLobeGeo, primaryArmorMat);
          pecLobe.position.set(
            side * 0.145 * m.torsoBulk * m.shoulderWidth,
            0.11 * m.heightScale,
            0.49
          );
          torsoGroup.add(pecLobe);

          // Muscular Trapezius / Scapular Withers Crest above shoulders
          const withersGeo = trackGeo(
            new THREE.SphereGeometry(0.22 * m.torsoBulk * mScale, seg(12, 8), seg(10, 8)),
            0.024
          );
          withersGeo.scale(0.82, 1.18, 1.48);
          const withersMesh = new THREE.Mesh(withersGeo, primaryArmorMat);
          withersMesh.position.set(
            side * 0.13 * m.torsoBulk * m.shoulderWidth,
            0.32 * m.heightScale,
            0.26
          );
          withersMesh.rotation.x = 0.18;
          torsoGroup.add(withersMesh);

          // 3 Lateral Serratus / Intercostal Ribcage Muscle Bands along the flanks
          for (let r = 0; r < 3; r++) {
            const ribMuscleGeo = trackGeo(
              new THREE.SphereGeometry(0.11 * m.torsoBulk * mScale, seg(10, 8), seg(8, 6)),
              0.018
            );
            ribMuscleGeo.scale(0.58, 1.55, 0.85);
            const ribMuscle = new THREE.Mesh(ribMuscleGeo, primaryArmorMat);
            ribMuscle.position.set(
              side * 0.34 * m.torsoBulk * m.shoulderWidth,
              (0.06 + r * 0.04) * m.heightScale,
              0.22 - r * 0.15
            );
            ribMuscle.rotation.x = -0.45;
            ribMuscle.rotation.z = -side * 0.22;
            torsoGroup.add(ribMuscle);
          }
        }
      }

      // Feral Neck Ruff / Chest Mane or Draconic Ventral Breastplate
      const maneGeo = trackGeo(
        new THREE.SphereGeometry(0.36 * m.torsoBulk, seg(16, 12), seg(12, 10)),
        0.024
      );
      maneGeo.scale(1.04, 1.05, 1.22);
      const maneMesh = new THREE.Mesh(maneGeo, secondaryTrimMat);
      maneMesh.position.set(0, 0.22 * m.heightScale, 0.48);
      maneMesh.castShadow = true;
      torsoGroup.add(maneMesh);
    }

    // 3B) Multi-Layered 3D Hair Mane Crown & Hackle Tufts on Furry Quadrupeds
    if (!isUndeadSkeleton && furDensity > 0.15 && hairStyle !== 'none') {
      const fScale = 0.78 + furDensity * 0.4;
      const maneLockCount = density === 'low' ? 10 : 14;
      for (let i = 0; i < maneLockCount; i++) {
        const ang = (i / maneLockCount) * Math.PI * 2;
        const lockLen = (0.32 + (i % 3) * 0.07) * m.torsoBulk * fScale;
        const lockGeo = trackFurLock(0.095 * m.torsoBulk * fScale, lockLen, 0.11);
        const lockMesh = new THREE.Mesh(lockGeo, hairStrandMat);

        const rx = Math.sin(ang) * 0.31 * m.torsoBulk * m.shoulderWidth;
        const ry = 0.21 * m.heightScale + Math.cos(ang) * 0.26 * m.torsoBulk;
        const rz = 0.42 - Math.cos(ang) * 0.08;
        lockMesh.position.set(rx, ry, rz);

        // Point hair locks outward and swept backward toward the withers
        lockMesh.rotation.z = -Math.sin(ang) * 0.85;
        lockMesh.rotation.x = -Math.PI * 0.58 + Math.cos(ang) * 0.35;
        lockMesh.castShadow = true;
        torsoGroup.add(lockMesh);
      }
    }

    // Continuous Dorsal Spine Scutes / Hair Hackles from Croup to Withers
    const spineCount = 7;
    for (let s = 0; s < spineCount; s++) {
      const frac = s / (spineCount - 1);
      const zOff = -0.52 + frac * 0.98;
      const yOff = (0.36 + frac * 0.18 * m.heightScale) * m.torsoBulk;
      const ridgeH = (0.16 + Math.sin(frac * Math.PI) * 0.1) * m.dorsalScale;
      const useHair = !isUndeadSkeleton && furDensity > 0.2 && hairStyle !== 'none';
      const ridgeGeo = useHair
        ? trackFurLock(0.068 * m.torsoBulk, ridgeH * (0.85 + furDensity * 0.3), 0.09)
        : trackGeo(new THREE.ConeGeometry(0.055 * m.torsoBulk, ridgeH, seg(10, 8)));
      if (!useHair) {
        ridgeGeo.rotateX(-0.38);
      }
      const ridgeMesh = new THREE.Mesh(
        ridgeGeo,
        isUndeadSkeleton ? ivoryFangMat : useHair ? hairStrandMat : secondaryTrimMat
      );
      ridgeMesh.position.set(0, yOff - (useHair ? 0.06 : 0), zOff);
      if (useHair) {
        ridgeMesh.rotation.x = -0.52;
      }
      torsoGroup.add(ridgeMesh);
    }

    // Exposed 3D Skeletal Dragon / Quadruped Spine, Sternal Keel, Scapulae & Costal Rib Arches when isUndeadSkeleton is true
    if (isUndeadSkeleton) {
      const qDiscMat = trackMat(
        new THREE.MeshStandardMaterial({
          name: 'Mat_SkullSocketVoid',
          color: new THREE.Color('#07080c'),
          roughness: 0.94,
          metalness: 0.04,
          wireframe,
        })
      );
      // 9-Vertebra Horizontal Spinal Column with Dark Intervertebral Discs & Tall Spinous Processes
      for (let v = 0; v < 9; v++) {
        const vFrac = v / 8;
        const vz = -0.32 + vFrac * 0.88;
        const vy = (0.04 + vFrac * 0.22) * m.heightScale + 0.24 * m.torsoBulk;
        const vertGeo = trackGeo(
          new THREE.CylinderGeometry(
            0.068 * m.torsoBulk,
            0.072 * m.torsoBulk,
            0.078,
            seg(10, 8)
          )
        );
        vertGeo.rotateX(Math.PI * 0.5);
        const vertMesh = new THREE.Mesh(vertGeo, ivoryFangMat);
        vertMesh.position.set(0, vy, vz);
        torsoGroup.add(vertMesh);

        const discGeo = trackGeo(
          new THREE.CylinderGeometry(
            0.052 * m.torsoBulk,
            0.052 * m.torsoBulk,
            0.025,
            seg(8, 6)
          )
        );
        discGeo.rotateX(Math.PI * 0.5);
        const discMesh = new THREE.Mesh(discGeo, qDiscMat);
        discMesh.position.set(0, vy - 0.008, vz + 0.048);
        torsoGroup.add(discMesh);
      }

      // Ventral Sternal Keel Bone connecting the ventral tips of the ribs
      const keelCurve = new THREE.CatmullRomCurve3([
        new THREE.Vector3(0, -0.18 * m.torsoBulk, -0.12),
        new THREE.Vector3(0, -0.12 * m.torsoBulk, 0.18),
        new THREE.Vector3(0, -0.02 * m.torsoBulk, 0.48),
      ]);
      const keelGeo = trackGeo(
        new THREE.TubeGeometry(keelCurve, seg(10, 8), 0.042 * m.torsoBulk, seg(8, 6), false)
      );
      torsoGroup.add(new THREE.Mesh(keelGeo, ivoryFangMat));

      // Paired Scapular Blades & 7 Pairs of Open 3D Costal Rib Arches
      for (const side of [-1, 1]) {
        const scapGeo = trackGeo(
          new THREE.BoxGeometry(0.05 * m.torsoBulk, 0.34 * m.torsoBulk, 0.22 * m.torsoBulk),
          0.01
        );
        const scapMesh = new THREE.Mesh(scapGeo, ivoryFangMat);
        scapMesh.position.set(
          side * 0.34 * m.torsoBulk * m.shoulderWidth,
          0.18 * m.heightScale,
          0.38
        );
        scapMesh.rotation.z = -side * 0.25;
        scapMesh.rotation.x = 0.28;
        torsoGroup.add(scapMesh);

        for (let r = 0; r < 7; r++) {
          const rFrac = r / 6;
          const rz = -0.22 + rFrac * 0.68;
          const ry = (0.04 + rFrac * 0.18) * m.heightScale;
          const rw = (0.45 - Math.abs(rFrac - 0.45) * 0.1) * m.torsoBulk * m.shoulderWidth;
          const qRibCurve = new THREE.CatmullRomCurve3([
            new THREE.Vector3(side * 0.04 * m.torsoBulk, ry + 0.26 * m.torsoBulk, rz),
            new THREE.Vector3(side * rw, ry + 0.02 * m.torsoBulk, rz + 0.03),
            new THREE.Vector3(side * 0.06 * m.torsoBulk, ry - 0.22 * m.torsoBulk, rz + 0.06),
          ]);
          const qRibGeo = trackGeo(
            new THREE.TubeGeometry(qRibCurve, seg(12, 8), 0.036 * m.torsoBulk, seg(8, 6), false),
            0.008
          );
          torsoGroup.add(new THREE.Mesh(qRibGeo, ivoryFangMat));
        }
      }
    }
  } else {
    // Bipedal / Centaur / Serpentine / Arachnid Upper Torso
    const isSkeletalTorso = isUndeadSkeleton;

    if (isSkeletalTorso) {
      // 3D BLENDER-SCULPTED OPEN SKELETAL RIBCAGE, STERNUM, CLAVICLES, SCAPULAE & THORACIC VERTEBRAL COLUMN
      // (No solid waistBridgeMesh or cuirassMesh to swallow the ribs or vertebrae!)
      const darkThoracicMat = trackMat(
        new THREE.MeshStandardMaterial({
          name: 'Mat_SkullSocketVoid',
          color: new THREE.Color('#07080c'),
          roughness: 0.94,
          metalness: 0.04,
          wireframe,
        })
      );

      // Recessed dark thoracic inner shadow core (only inside the upper ribcage, small radius so ribs protrude in full 3D relief!)
      const innerShadowGeo = trackGeo(
        new THREE.CylinderGeometry(
          0.165 * m.torsoBulk,
          0.115 * m.torsoBulk,
          torsoHeight * 0.52,
          seg(14, 10)
        ),
        0.008
      );
      innerShadowGeo.scale(1.12, 1.0, 0.72);
      const innerShadowMesh = new THREE.Mesh(innerShadowGeo, darkThoracicMat);
      innerShadowMesh.position.set(0, torsoHeight * 0.58, 0.02 * m.torsoBulk);
      torsoGroup.add(innerShadowMesh);

      // Dark Spinal Cord Rod running through the center of all Thoracic & Lumbar Vertebrae
      const dorsalCordCurve = new THREE.CatmullRomCurve3([
        new THREE.Vector3(0, -0.06, -0.11 * m.torsoBulk),
        new THREE.Vector3(0, torsoHeight * 0.48, -0.175 * m.torsoBulk),
        new THREE.Vector3(0, torsoHeight * 0.94, -0.08 * m.torsoBulk),
      ]);
      const dorsalCordGeo = trackGeo(
        new THREE.TubeGeometry(dorsalCordCurve, seg(14, 10), 0.035 * m.torsoBulk, seg(8, 6), false),
        0.004
      );
      torsoGroup.add(new THREE.Mesh(dorsalCordGeo, darkThoracicMat));

      // 1) 8 Distinct 3D Thoracic Vertebrae (T1-T8) with Centrum, Intervertebral Disc, Transverse Processes & Spinous Spikes
      const thoracicCount = 8;
      for (let v = 0; v < thoracicCount; v++) {
        const vFrac = v / (thoracicCount - 1);
        const pt = dorsalCordCurve.getPoint(0.06 + vFrac * 0.88);
        const vScale = (1.05 - vFrac * 0.18) * m.torsoBulk;

        // Cylindrical Vertebral Centrum Body
        const vBodyGeo = trackGeo(
          new THREE.CylinderGeometry(0.072 * vScale, 0.076 * vScale, 0.064, seg(12, 8)),
          0.007
        );
        vBodyGeo.scale(1.12, 1.0, 0.92);
        const vBodyMesh = new THREE.Mesh(vBodyGeo, ivoryFangMat);
        vBodyMesh.position.copy(pt);
        vBodyMesh.castShadow = true;
        torsoGroup.add(vBodyMesh);

        // Dark Intervertebral Fibrocartilage Disc between each vertebra
        const vDiscGeo = trackGeo(
          new THREE.CylinderGeometry(0.066 * vScale, 0.066 * vScale, 0.022, seg(10, 8)),
          0.002
        );
        const vDiscMesh = new THREE.Mesh(vDiscGeo, darkThoracicMat);
        vDiscMesh.position.set(pt.x, pt.y + 0.042, pt.z);
        torsoGroup.add(vDiscMesh);

        // Paired Horizontal Transverse Processes (Rib Articulation Tubercles)
        const vTransGeo = trackGeo(
          new THREE.CapsuleGeometry(0.02 * vScale, 0.155 * vScale, 6, 8),
          0.005
        );
        vTransGeo.rotateZ(Math.PI / 2);
        const vTransMesh = new THREE.Mesh(vTransGeo, ivoryFangMat);
        vTransMesh.position.set(pt.x, pt.y + 0.005, pt.z - 0.032 * m.torsoBulk);
        torsoGroup.add(vTransMesh);

        // Prominent Posterior Dorsal Spinous Process Spike jutting out along the back
        const vSpineGeo = trackGeo(
          new THREE.ConeGeometry(0.026 * vScale, 0.125 * vScale, 6),
          0.005
        );
        vSpineGeo.rotateX(-Math.PI * 0.62);
        const vSpineMesh = new THREE.Mesh(vSpineGeo, ivoryFangMat);
        vSpineMesh.position.set(pt.x, pt.y - 0.015, pt.z - 0.088 * m.torsoBulk);
        vSpineMesh.castShadow = true;
        torsoGroup.add(vSpineMesh);
      }

      // 2) Sculpted 3D Anterior Sternum Bone (Manubrium + Gladiolus Body + Xiphoid Process Tip)
      const manubriumGeo = trackGeo(
        new THREE.CylinderGeometry(
          0.082 * m.torsoBulk,
          0.058 * m.torsoBulk,
          0.12,
          6
        ),
        0.008
      );
      manubriumGeo.scale(1.15, 1.0, 0.45);
      const manubriumMesh = new THREE.Mesh(manubriumGeo, ivoryFangMat);
      manubriumMesh.position.set(0, torsoHeight * 0.84, 0.255 * m.torsoBulk);
      manubriumMesh.rotation.x = -0.22;
      torsoGroup.add(manubriumMesh);

      const sternumCurve = new THREE.CatmullRomCurve3([
        new THREE.Vector3(0, torsoHeight * 0.82, 0.265 * m.torsoBulk),
        new THREE.Vector3(0, torsoHeight * 0.62, 0.315 * m.torsoBulk),
        new THREE.Vector3(0, torsoHeight * 0.41, 0.285 * m.torsoBulk),
      ]);
      const sternumGeo = trackGeo(
        new THREE.TubeGeometry(sternumCurve, seg(12, 8), 0.044 * m.torsoBulk, seg(10, 8), false),
        0.01
      );
      sternumGeo.scale(1.38, 1.0, 0.58);
      const sternumMesh = new THREE.Mesh(sternumGeo, ivoryFangMat);
      sternumMesh.castShadow = true;
      torsoGroup.add(sternumMesh);

      // Sharp Xiphoid Process Dagger Tip at bottom of Sternum
      const xiphoidGeo = trackGeo(
        new THREE.ConeGeometry(0.032 * m.torsoBulk, 0.095, 5),
        0.005
      );
      xiphoidGeo.scale(1.1, 1.0, 0.5);
      const xiphoidMesh = new THREE.Mesh(xiphoidGeo, ivoryFangMat);
      xiphoidMesh.position.set(0, torsoHeight * 0.35, 0.275 * m.torsoBulk);
      xiphoidMesh.rotation.x = Math.PI + 0.15;
      torsoGroup.add(xiphoidMesh);

      // 3) Paired S-Curved Clavicles (Collarbones), Dorsal Scapulae (Shoulder Blades) & 7 Pairs of High-Relief 3D Costal Ribs
      for (const side of [-1, 1]) {
        // S-Curved Anatomical Clavicle from Manubrium to Acromion Shoulder Joint
        const clavicleCurve = new THREE.CatmullRomCurve3([
          new THREE.Vector3(side * 0.045 * m.torsoBulk, torsoHeight * 0.86, 0.26 * m.torsoBulk),
          new THREE.Vector3(
            side * 0.21 * m.torsoBulk * m.shoulderWidth,
            torsoHeight * 0.89,
            0.23 * m.torsoBulk
          ),
          new THREE.Vector3(
            side * 0.36 * m.torsoBulk * m.shoulderWidth,
            torsoHeight * 0.85,
            0.09 * m.torsoBulk
          ),
          new THREE.Vector3(
            side * 0.44 * m.torsoBulk * m.shoulderWidth,
            torsoHeight * 0.80,
            0.01 * m.torsoBulk
          ),
        ]);
        const clavicleGeo = trackGeo(
          new THREE.TubeGeometry(
            clavicleCurve,
            seg(14, 10),
            0.032 * m.torsoBulk,
            seg(10, 8),
            false
          ),
          0.008
        );
        const clavicleMesh = new THREE.Mesh(clavicleGeo, ivoryFangMat);
        clavicleMesh.castShadow = true;
        torsoGroup.add(clavicleMesh);

        // Triangular Dorsal Scapula (Shoulder Blade) + Raised Scapular Spine on Upper Back
        const scapulaGeo = trackGeo(
          new THREE.ConeGeometry(0.155 * m.torsoBulk, 0.32 * torsoHeight, 5),
          0.01
        );
        scapulaGeo.scale(1.12, 1.0, 0.28);
        const scapulaMesh = new THREE.Mesh(scapulaGeo, ivoryFangMat);
        scapulaMesh.position.set(
          side * 0.23 * m.torsoBulk * m.shoulderWidth,
          torsoHeight * 0.68,
          -0.165 * m.torsoBulk
        );
        scapulaMesh.rotation.x = Math.PI - 0.14;
        scapulaMesh.rotation.z = side * 0.24;
        scapulaMesh.rotation.y = -side * 0.28;
        torsoGroup.add(scapulaMesh);

        // 7 Pairs of Individual 3D Costal Rib Arches (5 True/False Sternocostal Ribs + 2 Lower Floating Ribs)
        const ribCount = 7;
        for (let r = 0; r < ribCount; r++) {
          const rFrac = r / (ribCount - 1);
          const ribYBack = torsoHeight * (0.82 - rFrac * 0.56);
          const ribYMid = ribYBack - 0.035 * torsoHeight;
          const ribYFront = ribYBack - 0.065 * torsoHeight;
          // Bell-shaped thoracic cage width: widest at ribs 3-5, tapering at top and bottom
          const archBell = Math.sin(rFrac * Math.PI * 0.82 + 0.25);
          const ribW =
            (0.35 + archBell * 0.115) * m.torsoBulk * (0.88 + 0.12 * m.shoulderWidth);
          const ribZBack = (-0.155 - Math.sin(rFrac * Math.PI) * 0.025) * m.torsoBulk;
          const ribZFront = (0.305 - Math.abs(rFrac - 0.35) * 0.06) * m.torsoBulk;
          const isFloatingRib = r >= 5;

          const ribPts = isFloatingRib
            ? [
                new THREE.Vector3(side * 0.045 * m.torsoBulk, ribYBack, ribZBack),
                new THREE.Vector3(side * ribW * 0.78, ribYBack - 0.01, ribZBack * 0.35),
                new THREE.Vector3(
                  side * ribW * (r === 5 ? 0.86 : 0.72),
                  ribYMid,
                  (r === 5 ? 0.08 : -0.01) * m.torsoBulk
                ),
              ]
            : [
                new THREE.Vector3(side * 0.045 * m.torsoBulk, ribYBack, ribZBack),
                new THREE.Vector3(side * ribW * 0.76, ribYBack - 0.008, ribZBack * 0.32),
                new THREE.Vector3(side * ribW, ribYMid, 0.04 * m.torsoBulk),
                new THREE.Vector3(side * ribW * 0.62, ribYFront, ribZFront * 0.78),
                new THREE.Vector3(
                  side * 0.048 * m.torsoBulk,
                  Math.max(torsoHeight * 0.38, ribYFront + (r >= 3 ? (r - 2) * 0.028 : 0)),
                  ribZFront
                ),
              ];

          const ribCurve = new THREE.CatmullRomCurve3(ribPts);
          const ribRadius = (0.032 - rFrac * 0.005) * m.torsoBulk;
          const ribGeo = trackGeo(
            new THREE.TubeGeometry(ribCurve, seg(16, 12), ribRadius, seg(10, 8), false),
            0.008
          );
          // Slightly flatten each rib cross-section like a real anatomical costal bone
          ribGeo.scale(1.0, 1.18, 0.92);
          const ribMesh = new THREE.Mesh(ribGeo, ivoryFangMat);
          ribMesh.castShadow = true;
          torsoGroup.add(ribMesh);
        }
      }
    } else {
      // Blender-Sculpted Anatomical Lumbar-Thoracic Core Sleeve (Deeply bridges Pelvis_Chassis and Torso_Main)
      const waistBridgePts = [
        new THREE.Vector2(0.01, -0.26),
        new THREE.Vector2(0.35 * m.torsoBulk, -0.18),
        new THREE.Vector2(0.32 * m.torsoBulk, 0.0),
        new THREE.Vector2(0.36 * m.torsoBulk, 0.18),
        new THREE.Vector2(0.01, 0.26),
      ];
      const waistBridgeGeo = trackGeo(new THREE.LatheGeometry(waistBridgePts, seg(18, 12)), 0.018);
      waistBridgeGeo.scale(1.08, 1.0, 0.88);
      const waistBridgeMesh = new THREE.Mesh(waistBridgeGeo, underbodyMat);
      waistBridgeMesh.position.set(0, -0.02, 0);
      torsoGroup.add(waistBridgeMesh);

      const cuirassPts = [
        new THREE.Vector2(0.01, -0.18),
        new THREE.Vector2(0.28 * m.torsoBulk, -0.1),
        new THREE.Vector2(0.34 * m.torsoBulk, 0.08),
        new THREE.Vector2(0.43 * m.torsoBulk, torsoHeight * 0.42),
        new THREE.Vector2(0.5 * m.torsoBulk * m.shoulderWidth * 0.85, torsoHeight * 0.78),
        new THREE.Vector2(0.34 * m.torsoBulk, torsoHeight * 0.96),
        new THREE.Vector2(0.22 * m.torsoBulk, torsoHeight * 1.04),
        new THREE.Vector2(0.01, torsoHeight * 1.06),
      ];
      const cuirassGeo = trackGeo(new THREE.LatheGeometry(cuirassPts, seg(20, 14)), 0.022);
      cuirassGeo.scale(1.12, 1.0, 0.86);
      const cuirassMesh = new THREE.Mesh(cuirassGeo, primaryArmorMat);
      cuirassMesh.castShadow = true;
      torsoGroup.add(cuirassMesh);

      // 3C) Bipedal / Titan Anatomical Musculature: Paired Pectoralis Major, Latissimus Dorsi V-Taper & 6-Pack Abdominals
      if (muscleDef > 0.15) {
        const mScale = 0.78 + muscleDef * 0.34;
        for (const side of [-1, 1]) {
          // Sculpted Pectoralis Major Chest Plate
          const pecGeo = trackGeo(
            new THREE.SphereGeometry(0.185 * m.torsoBulk * mScale, seg(14, 10), seg(10, 8)),
            0.022
          );
          pecGeo.scale(1.18, 0.82, 0.88);
          const pecMesh = new THREE.Mesh(pecGeo, primaryArmorMat);
          pecMesh.position.set(
            side * 0.155 * m.torsoBulk * m.shoulderWidth,
            torsoHeight * 0.68,
            0.24 * m.torsoBulk
          );
          pecMesh.rotation.z = -side * 0.12;
          pecMesh.castShadow = true;
          torsoGroup.add(pecMesh);

          // Sculpted Latissimus Dorsi V-Taper Sweep on Back/Flanks
          const latGeo = trackGeo(
            new THREE.SphereGeometry(0.175 * m.torsoBulk * mScale, seg(12, 8), seg(10, 8)),
            0.022
          );
          latGeo.scale(0.85, 1.65, 1.05);
          const latMesh = new THREE.Mesh(latGeo, primaryArmorMat);
          latMesh.position.set(
            side * 0.28 * m.torsoBulk * m.shoulderWidth,
            torsoHeight * 0.54,
            -0.06 * m.torsoBulk
          );
          latMesh.rotation.z = -side * 0.24;
          torsoGroup.add(latMesh);

          // 3 Pairs of Rectus Abdominis (6-Pack) Muscle Bellies
          for (let a = 0; a < 3; a++) {
            const abGeo = trackGeo(
              new THREE.SphereGeometry(0.088 * m.torsoBulk * mScale, seg(10, 8), seg(8, 6)),
              0.018
            );
            abGeo.scale(1.15, 0.72, 0.78);
            const abMesh = new THREE.Mesh(
              abGeo,
              isOrganicCreature ? primaryArmorMat : underbodyMat
            );
            abMesh.position.set(
              side * 0.082 * m.torsoBulk,
              torsoHeight * (0.46 - a * 0.13),
              (0.28 - a * 0.015) * m.torsoBulk
            );
            torsoGroup.add(abMesh);
          }
        }
      }

      const isFeralAnimal =
        config.raceCategory === 'starter_animals' ||
        config.raceCategory === 'beasts' ||
        furDensity > 0.25;
      if (isFeralAnimal) {
        const maneGeo = trackGeo(
          new THREE.SphereGeometry(0.37 * m.torsoBulk, seg(16, 12), seg(12, 10)),
          0.025
        );
        maneGeo.scale(1.08, 0.96, 1.22);
        const maneMesh = new THREE.Mesh(maneGeo, secondaryTrimMat);
        maneMesh.position.set(0, torsoHeight * 0.7, 0.14 * m.torsoBulk);
        maneMesh.castShadow = true;
        torsoGroup.add(maneMesh);

        // 3D Layered Hair Collar Locks around Biped/Bear/Minotaur Chest & Neck
        if (furDensity > 0.15 && hairStyle !== 'none') {
          const fScale = 0.75 + furDensity * 0.38;
          const collarLocks = density === 'low' ? 8 : 10;
          for (let i = 0; i < collarLocks; i++) {
            const ang = (i / collarLocks) * Math.PI * 2;
            const lockGeo = trackFurLock(0.088 * m.torsoBulk * fScale, 0.28 * m.torsoBulk * fScale, 0.1);
            const lockMesh = new THREE.Mesh(lockGeo, hairStrandMat);
            lockMesh.position.set(
              Math.sin(ang) * 0.32 * m.torsoBulk,
              torsoHeight * 0.8 + Math.cos(ang) * 0.06,
              Math.cos(ang) * 0.26 * m.torsoBulk
            );
            lockMesh.rotation.z = -Math.sin(ang) * 0.95;
            lockMesh.rotation.x = Math.cos(ang) * 0.85 - 0.35;
            torsoGroup.add(lockMesh);
          }
        }
      } else {
        const gorgetGeo = trackGeo(
          new THREE.TorusGeometry(0.26 * m.torsoBulk, 0.065 * m.torsoBulk, seg(10, 8), seg(18, 12))
        );
        gorgetGeo.rotateX(Math.PI / 2 - 0.15);
        const gorgetMesh = new THREE.Mesh(gorgetGeo, secondaryTrimMat);
        gorgetMesh.position.set(0, torsoHeight * 0.93, 0.03);
        torsoGroup.add(gorgetMesh);

        const chestShape = new THREE.Shape();
        chestShape.moveTo(-0.28 * m.torsoBulk, torsoHeight * 0.78);
        chestShape.lineTo(0.28 * m.torsoBulk, torsoHeight * 0.78);
        chestShape.lineTo(0.2 * m.torsoBulk, torsoHeight * 0.22);
        chestShape.lineTo(0, torsoHeight * 0.04);
        chestShape.lineTo(-0.2 * m.torsoBulk, torsoHeight * 0.22);
        chestShape.closePath();

        const chestPlateGeo = trackGeo(
          new THREE.ExtrudeGeometry(chestShape, {
            depth: 0.13 * m.torsoBulk,
            bevelEnabled: true,
            bevelThickness: 0.025,
            bevelSize: 0.02,
            bevelSegments: 2,
          })
        );
        const chestPlateMesh = new THREE.Mesh(chestPlateGeo, secondaryTrimMat);
        chestPlateMesh.position.set(0, 0, 0.26 * m.torsoBulk);
        chestPlateMesh.castShadow = true;
        torsoGroup.add(chestPlateMesh);
      }
    }
  }

  const socketCore = new THREE.Object3D();
  socketCore.name = 'socket_core';
  socketCore.position.set(
    0,
    isQuadruped && !isCentaur ? 0.18 * m.heightScale : torsoHeight * 0.56,
    isQuadruped && !isCentaur ? 0.62 * m.torsoBulk : (isBeholder ? 0.55 : 0.44) * m.torsoBulk
  );
  torsoGroup.add(socketCore);

  const coreGeo = trackGeo(
    new THREE.OctahedronGeometry((isBeholder ? 0.32 : 0.15) * m.torsoBulk, 2)
  );
  const coreMesh = new THREE.Mesh(coreGeo, crystalCoreMat);
  coreMesh.name = 'Runic_Core_Crystal';
  coreMesh.position.copy(socketCore.position);
  coreMesh.visible = !isOrganicCreature && !isDragon && !isAnimal;
  torsoGroup.add(coreMesh);

  tagExplodablePart(coreMesh, new THREE.Vector3(0, 0.15, 0.8), {
    name: isBeholder ? 'Iris Central del Ojo Tirano' : 'Núcleo Cristalino Elemental',
    category: 'Accent Detail',
    vertexCount: 128,
    triangleCount: 160,
    materialSummary: 'Cristal PBR con transmisión óptica y emisión',
    socketName: 'socket_core',
  });
  explodableParts.push(coreMesh);

  // 4. CONTINUOUS ARTICULATED NECK & MULTI-HEAD BUILDER
  const neckGroup = makeBone('Neck_Assembly');
  if (isQuadruped && !isCentaur) {
    neckGroup.position.set(0, 0.26 * m.heightScale, 0.48);
  } else {
    neckGroup.position.set(0, torsoHeight * 0.9, 0.04);
  }
  torsoGroup.add(neckGroup);

  const neckLen = Math.max(0.22, 0.56 * m.neckLength);
  const isLongNeck = isDragon || isSerpentine || isQuadruped || m.neckLength > 0.72;
  const headPos = isBeholder
    ? new THREE.Vector3(0, -0.1, 0.04)
    : isQuadruped && !isCentaur
    ? new THREE.Vector3(0, neckLen * 0.62, neckLen * 0.52)
    : isLongNeck
    ? new THREE.Vector3(0, neckLen * 0.88, 0.18 * m.neckLength)
    : new THREE.Vector3(0, neckLen * 0.48, 0.05);

  if (!isBeholder) {
    const neckCurve = new THREE.CatmullRomCurve3([
      new THREE.Vector3(0, -0.16, isQuadruped && !isCentaur ? -0.12 : -0.05),
      new THREE.Vector3(0, headPos.y * 0.45, headPos.z * 0.45 - 0.01),
      new THREE.Vector3(0, headPos.y + 0.14, headPos.z - 0.02),
    ]);

    if (isUndeadSkeleton) {
      // EXPOSED 3D CERVICAL VERTEBRAL COLUMN (C1 ATLAS - C7) WITHOUT FLESHY NECK TUBE OCCLUSION!
      const darkCervicalMat = trackMat(
        new THREE.MeshStandardMaterial({
          name: 'Mat_SkullSocketVoid',
          color: new THREE.Color('#07080c'),
          roughness: 0.92,
          metalness: 0.04,
          wireframe,
        })
      );
      const cervicalCordGeo = trackGeo(
        new THREE.TubeGeometry(neckCurve, seg(12, 8), 0.034 * m.torsoBulk, seg(8, 6), false),
        0.004
      );
      neckGroup.add(new THREE.Mesh(cervicalCordGeo, darkCervicalMat));

      const cCount = 5;
      for (let cv = 0; cv < cCount; cv++) {
        const cFrac = cv / (cCount - 1);
        const pt = neckCurve.getPoint(0.12 + cFrac * 0.78);
        const cScale = (0.98 - cFrac * 0.12) * m.torsoBulk;

        // Cervical Vertebral Body + Atlas/Axis Ring
        const cBodyGeo = trackGeo(
          new THREE.CylinderGeometry(0.068 * cScale, 0.072 * cScale, 0.052, seg(12, 8)),
          0.006
        );
        const cBodyMesh = new THREE.Mesh(cBodyGeo, ivoryFangMat);
        cBodyMesh.position.copy(pt);
        cBodyMesh.rotation.x = 0.18;
        cBodyMesh.castShadow = true;
        neckGroup.add(cBodyMesh);

        // Transverse Process Wings on each Cervical Vertebra
        const cTransGeo = trackGeo(
          new THREE.CapsuleGeometry(0.018 * cScale, 0.125 * cScale, 5, 8),
          0.004
        );
        cTransGeo.rotateZ(Math.PI / 2);
        const cTransMesh = new THREE.Mesh(cTransGeo, ivoryFangMat);
        cTransMesh.position.set(pt.x, pt.y, pt.z - 0.02 * m.torsoBulk);
        neckGroup.add(cTransMesh);

        // Posterior Cervical Spinous Process Spike
        const cSpineGeo = trackGeo(
          new THREE.ConeGeometry(0.022 * cScale, 0.095 * cScale, 5),
          0.004
        );
        cSpineGeo.rotateX(-Math.PI * 0.56);
        const cSpineMesh = new THREE.Mesh(cSpineGeo, ivoryFangMat);
        cSpineMesh.position.set(pt.x, pt.y - 0.005, pt.z - 0.068 * m.torsoBulk);
        neckGroup.add(cSpineMesh);
      }
    } else {
      // Blender-Sculpted Trapezius & Sternocleidomastoid Cervical Collar Flare at Base of Neck
      const cervicalFlarePts = [
        new THREE.Vector2(0.01, -0.22),
        new THREE.Vector2((isQuadruped ? 0.31 : 0.27) * m.torsoBulk, -0.14),
        new THREE.Vector2((isQuadruped ? 0.23 : 0.20) * m.torsoBulk, 0.02),
        new THREE.Vector2((isQuadruped ? 0.18 : 0.16) * m.torsoBulk, 0.14),
        new THREE.Vector2(0.01, 0.18),
      ];
      const cervicalFlareGeo = trackGeo(new THREE.LatheGeometry(cervicalFlarePts, seg(16, 12)), 0.018);
      cervicalFlareGeo.scale(1.14, 1.0, 0.95);
      const cervicalFlareMesh = new THREE.Mesh(
        cervicalFlareGeo,
        isOrganicCreature || isQuadruped ? primaryArmorMat : underbodyMat
      );
      cervicalFlareMesh.position.set(0, -0.02, isQuadruped && !isCentaur ? -0.04 : 0);
      neckGroup.add(cervicalFlareMesh);

      const neckTubeGeo = trackGeo(
        new THREE.TubeGeometry(
          neckCurve,
          seg(14, 10),
          (isQuadruped ? 0.185 : 0.165) * m.torsoBulk,
          seg(14, 10),
          false
        ),
        0.022
      );
      const neckMesh = new THREE.Mesh(neckTubeGeo, primaryArmorMat);
      neckMesh.castShadow = true;
      neckGroup.add(neckMesh);

      if (muscleDef > 0.15) {
        for (const side of [-1, 1]) {
          const scmCurve = new THREE.CatmullRomCurve3([
            new THREE.Vector3(side * 0.16 * m.torsoBulk, -0.12, 0.06),
            new THREE.Vector3(side * 0.12 * m.torsoBulk, headPos.y * 0.48, headPos.z * 0.45 + 0.04),
            new THREE.Vector3(side * 0.09 * m.torsoBulk, headPos.y + 0.08, headPos.z - 0.02),
          ]);
          const scmGeo = trackGeo(
            new THREE.TubeGeometry(scmCurve, seg(10, 8), 0.045 * m.torsoBulk, seg(8, 6), false),
            0.012
          );
          const scmMesh = new THREE.Mesh(
            scmGeo,
            isOrganicCreature || isQuadruped ? primaryArmorMat : underbodyMat
          );
          neckGroup.add(scmMesh);
        }
      }
    }

    if (isDragon || isSerpentine || isQuadruped) {
      const neckSpineCount = 3;
      const useHair = furDensity > 0.2 && hairStyle !== 'none';
      for (let ns = 1; ns <= neckSpineCount; ns++) {
        const pt = neckCurve.getPoint(ns / (neckSpineCount + 1));
        const nSpineGeo = useHair
          ? trackFurLock(0.058 * m.torsoBulk, 0.22 * (0.8 + furDensity * 0.35), 0.08)
          : trackGeo(new THREE.ConeGeometry(0.042, 0.14, seg(8, 6)));
        if (!useHair) {
          nSpineGeo.rotateX(-0.55);
        }
        const nSpine = new THREE.Mesh(
          nSpineGeo,
          isUndeadSkeleton ? ivoryFangMat : useHair ? hairStrandMat : secondaryTrimMat
        );
        nSpine.position.set(0, pt.y + 0.06, pt.z - 0.12 * m.torsoBulk);
        if (useHair) {
          nSpine.rotation.x = -0.65;
        }
        neckGroup.add(nSpine);
      }
    }
  }

  // 5. ORGANIC SCULPTED HEAD, SNOUT, MUZZLE, JAW, CHEEK FUR RUFFS, EARS/HORNS & BREATH EMITTER
  const headGroup = makeBone('Head_Helm_Assembly');
  headGroup.position.copy(headPos);
  neckGroup.add(headGroup);

  // Blender-Sculpted Suboccipital-Atlas Cranial Socket Bridge (Seamlessly unites Neck_Assembly with Head base)
  if (!isBeholder) {
    const cranialSocketGeo = trackGeo(
      new THREE.CapsuleGeometry(
        isUndeadSkeleton ? 0.072 : 0.142,
        isUndeadSkeleton ? 0.055 : 0.09,
        seg(10, 8),
        seg(14, 10)
      ),
      0.012
    );
    cranialSocketGeo.scale(1.06, 1.0, 1.12);
    const cranialSocketMesh = new THREE.Mesh(
      cranialSocketGeo,
      isUndeadSkeleton
        ? ivoryFangMat
        : isOrganicCreature || isQuadruped
        ? primaryArmorMat
        : underbodyMat
    );
    // Keep occipital condyles tucked at the back-base of the skull on skeletons so the mouth & teeth remain 100% unobstructed
    cranialSocketMesh.position.set(0, isUndeadSkeleton ? 0.135 : 0.09, isUndeadSkeleton ? -0.055 : 0.01);
    cranialSocketMesh.rotation.x = 0.22;
    headGroup.add(cranialSocketMesh);
  }

  const snoutLen = 0.34 * m.snoutLength;

  const socketHead = new THREE.Object3D();
  socketHead.name = 'socket_head';
  socketHead.position.set(0, 0.2, 0.24 + snoutLen * 0.5);
  headGroup.add(socketHead);

  const jawGroup = makeBone('Articulated_Lower_Jaw');
  jawGroup.position.set(0, 0.09, 0.06);
  headGroup.add(jawGroup);

  const isHumanoidBullHorned =
    config.archetype === 'balrog_archdemon' ||
    config.archetype === 'lesser_fire_imp' ||
    config.archetype === 'briar_wood_spriggan';

  const isLivingCrownedHumanoid =
    config.archetype === 'vampire_lord' ||
    config.archetype === 'gorgon_medusa' ||
    config.archetype === 'treant_ancient';

  const isSkeletalSkullFace =
    isUndeadSkeleton ||
    config.archetype === 'flesh_bone_abomination' ||
    config.archetype === 'plague_hierophant_mummy' ||
    config.archetype === 'necrotic_archdragon' ||
    (config.helm === 'crowned_skull' && !isLivingCrownedHumanoid);

  if (
    (config.helm === 'draconic_snout' ||
      config.helm === 'lupine_maw' ||
      (config.helm === 'tauren_bull_horns' && !isHumanoidBullHorned)) &&
    !isSkeletalSkullFace
  ) {
    // 1) Organic Sculpted Skull Cranium (Anchored around (0, 0.18, 0.04) overlapping neck end)
    const craniumGeo = trackGeo(
      new THREE.SphereGeometry(0.25, seg(18, 12), seg(14, 10)),
      0.018
    );
    craniumGeo.scale(0.98, 0.88, 1.24);
    const craniumMesh = new THREE.Mesh(craniumGeo, primaryArmorMat);
    craniumMesh.position.set(0, 0.18, 0.04);
    craniumMesh.castShadow = true;
    headGroup.add(craniumMesh);

    // 2) Sculpted Supraorbital Brow & Zygomatic Cheekbone Ridges + Masseter Jaw Muscles
    for (const side of [-1, 1]) {
      const browGeo = trackGeo(new THREE.SphereGeometry(0.09, seg(12, 8), seg(10, 8)));
      browGeo.scale(0.88, 0.58, 1.9);
      const browRidge = new THREE.Mesh(browGeo, secondaryTrimMat);
      browRidge.position.set(side * 0.13, 0.25, 0.15);
      browRidge.rotation.y = side * 0.2;
      browRidge.rotation.z = -side * 0.16;
      headGroup.add(browRidge);

      // Masseter Cheek / Jaw Muscle Bulge
      if (muscleDef > 0.15) {
        const masseterGeo = trackGeo(new THREE.SphereGeometry(0.095, seg(10, 8), seg(8, 6)));
        masseterGeo.scale(0.85, 0.95, 1.35);
        const masseterMesh = new THREE.Mesh(masseterGeo, primaryArmorMat);
        masseterMesh.position.set(side * 0.155, 0.13, 0.06);
        headGroup.add(masseterMesh);
      }

      // 3D Lupine / Feral Cheek & Jowl Hair Ruff Tufts
      if (furDensity > 0.15 && hairStyle !== 'none') {
        const fScale = 0.75 + furDensity * 0.36;
        for (let c = 0; c < 3; c++) {
          const cheekTuftGeo = trackFurLock(0.055 * fScale, (0.22 - c * 0.03) * fScale, 0.07);
          const cheekTuft = new THREE.Mesh(cheekTuftGeo, hairStrandMat);
          cheekTuft.position.set(side * 0.18, 0.19 - c * 0.055, 0.03 - c * 0.02);
          cheekTuft.rotation.z = -side * (1.15 - c * 0.12);
          cheekTuft.rotation.x = -0.58;
          headGroup.add(cheekTuft);
        }
      }
    }

    // 3) Continuous Forward-Pointing Tapered Snout / Muzzle Bridge
    const muzzleLen = snoutLen + 0.26;
    const snoutPts = [
      new THREE.Vector2(0.01, -0.04),
      new THREE.Vector2(0.195, 0.0),                // Wide base embedded inside cranium
      new THREE.Vector2(0.155, muzzleLen * 0.45),   // Mid nasal bridge
      new THREE.Vector2(0.11, muzzleLen * 0.88),    // Muzzle tip
      new THREE.Vector2(0.065, muzzleLen),
      new THREE.Vector2(0.01, muzzleLen + 0.02),
    ];
    const upperSnoutGeo = trackGeo(new THREE.LatheGeometry(snoutPts, seg(18, 12)));
    upperSnoutGeo.rotateX(Math.PI / 2); // +Y -> +Z forward
    upperSnoutGeo.scale(0.96, 0.72, 1.0);
    const upperSnoutMesh = new THREE.Mesh(upperSnoutGeo, primaryArmorMat);
    upperSnoutMesh.position.set(0, 0.16, 0.06);
    upperSnoutMesh.castShadow = true;
    headGroup.add(upperSnoutMesh);

    const snoutTipZ = 0.06 + muzzleLen;

    // 4) Dark Feral Rhinarium (Nose Tip) anchored at snoutTipZ
    const noseTipGeo = trackGeo(new THREE.SphereGeometry(0.065, seg(12, 8), seg(10, 8)));
    noseTipGeo.scale(1.12, 0.78, 0.9);
    const noseTipMesh = new THREE.Mesh(noseTipGeo, underbodyMat);
    noseTipMesh.position.set(0, 0.175, snoutTipZ);
    headGroup.add(noseTipMesh);

    if (config.helm === 'tauren_bull_horns' && config.raceCategory !== 'starter_animals') {
      const noseRingGeo = trackGeo(
        new THREE.TorusGeometry(0.082, 0.018, seg(8, 6), seg(14, 10))
      );
      const noseRingMesh = new THREE.Mesh(noseRingGeo, secondaryTrimMat);
      noseRingMesh.position.set(0, 0.11, snoutTipZ - 0.04);
      noseRingMesh.rotation.x = 0.3;
      headGroup.add(noseRingMesh);
    }

    // 5) Glowing Predatory Eyes recessed under brow ridges
    for (const side of [-1, 1]) {
      const eyeGeo = trackGeo(new THREE.SphereGeometry(0.046, seg(10, 8), seg(8, 6)));
      eyeGeo.scale(0.7, 0.82, 1.3);
      const eyeMesh = new THREE.Mesh(eyeGeo, emissiveRuneMat);
      eyeMesh.position.set(side * 0.145, 0.22, 0.17);
      headGroup.add(eyeMesh);
    }

    // 6) Continuous Lower Mandible + Interlocking Fangs (Anchored at jawGroup hinge (0, 0.09, 0.06) and extending +Z forward)
    const jawLen = snoutLen + 0.22;
    const jawPts = [
      new THREE.Vector2(0.01, -0.04),
      new THREE.Vector2(0.155, 0.0),               // Wide mandibular hinge inside skull
      new THREE.Vector2(0.125, jawLen * 0.5),
      new THREE.Vector2(0.085, jawLen * 0.9),
      new THREE.Vector2(0.01, jawLen + 0.02),
    ];
    const lowerJawGeo = trackGeo(new THREE.LatheGeometry(jawPts, seg(16, 12)));
    lowerJawGeo.rotateX(Math.PI / 2); // +Y -> +Z forward
    lowerJawGeo.scale(0.92, 0.54, 1.0);
    const lowerJawMesh = new THREE.Mesh(lowerJawGeo, underbodyMat);
    lowerJawMesh.position.set(0, 0.0, 0.0);
    jawGroup.add(lowerJawMesh);

    for (const side of [-1, 1]) {
      const isBoar =
        config.archetype === 'rabid_razorback_boar' ||
        config.archetype === 'iron_boar_juggernaut' ||
        config.archetype === 'phantom_armored_boar';
      // Lower canines / tusks anchored on front of lowerJawMesh
      const fangGeo = trackGeo(
        new THREE.ConeGeometry(isBoar ? 0.038 : 0.026, isBoar ? 0.22 : 0.11, seg(10, 8))
      );
      const fangMesh = new THREE.Mesh(fangGeo, ivoryFangMat);
      fangMesh.position.set(
        side * (isBoar ? 0.095 : 0.068),
        isBoar ? 0.08 : 0.045,
        jawLen * 0.82
      );
      if (isBoar) {
        fangMesh.rotation.x = 0.32;
        fangMesh.rotation.z = -side * 0.26;
      }
      jawGroup.add(fangMesh);

      // Upper predatory canines hanging down from upper snout
      const upperFangGeo = trackGeo(new THREE.ConeGeometry(0.028, 0.12, seg(10, 8)));
      upperFangGeo.rotateX(Math.PI);
      const upperFang = new THREE.Mesh(upperFangGeo, ivoryFangMat);
      upperFang.position.set(side * 0.082, 0.1, 0.06 + muzzleLen * 0.72);
      headGroup.add(upperFang);
    }
  } else if (config.helm === 'avian_phoenix_beak') {
    const avianSkullGeo = trackGeo(
      new THREE.SphereGeometry(0.24, seg(18, 14), seg(14, 10)),
      0.016
    );
    avianSkullGeo.scale(0.96, 0.92, 1.18);
    const avianSkull = new THREE.Mesh(avianSkullGeo, primaryArmorMat);
    avianSkull.position.set(0, 0.19, 0.04);
    avianSkull.castShadow = true;
    headGroup.add(avianSkull);

    // Sculpted Raptor Supraorbital Brow Ridges + Rabid Predatory Raptor Eyes
    const raptorPupilMat = trackMat(
      new THREE.MeshStandardMaterial({
        name: 'Mat_RaptorPupil',
        color: new THREE.Color('#090505'),
        roughness: 0.08,
        metalness: 0.2,
        wireframe,
      })
    );
    for (const side of [-1, 1]) {
      const browGeo = trackGeo(new THREE.SphereGeometry(0.085, seg(12, 8), seg(10, 8)), 0.012);
      browGeo.scale(0.85, 0.52, 1.85);
      const browMesh = new THREE.Mesh(browGeo, primaryArmorMat);
      browMesh.position.set(side * 0.135, 0.255, 0.14);
      browMesh.rotation.y = side * 0.22;
      browMesh.rotation.z = -side * 0.22;
      headGroup.add(browMesh);

      const eyeGeo = trackGeo(new THREE.SphereGeometry(0.048, seg(12, 8), seg(10, 8)), 0.004);
      eyeGeo.scale(0.78, 0.92, 1.18);
      const eyeMesh = new THREE.Mesh(eyeGeo, emissiveRuneMat);
      eyeMesh.position.set(side * 0.135, 0.215, 0.165);
      headGroup.add(eyeMesh);

      const pupilGeo = trackGeo(new THREE.SphereGeometry(0.022, 8, 6), 0);
      pupilGeo.scale(0.55, 1.15, 0.65);
      const pupilMesh = new THREE.Mesh(pupilGeo, raptorPupilMat);
      pupilMesh.position.set(side * 0.155, 0.215, 0.198);
      headGroup.add(pupilMesh);
    }

    // Curved Hooked Upper Raptor Beak starting inside skull
    const beakCurve = new THREE.CatmullRomCurve3([
      new THREE.Vector3(0, 0.185, 0.08),
      new THREE.Vector3(0, 0.185, 0.24 + snoutLen * 0.48),
      new THREE.Vector3(0, 0.095, 0.35 + snoutLen * 0.85),
    ]);
    const beakGeo = trackGeo(
      new THREE.TubeGeometry(beakCurve, seg(14, 10), 0.084, seg(12, 8), false),
      0.012
    );
    beakGeo.scale(0.86, 1.05, 1.0);
    const beakMesh = new THREE.Mesh(beakGeo, secondaryTrimMat);
    beakMesh.castShadow = true;
    headGroup.add(beakMesh);

    // Sharp Hooked Upper Beak Tip + Serrated Tomia Fang Notches
    const beakTipGeo = trackGeo(new THREE.ConeGeometry(0.056, 0.14, seg(10, 8)), 0.006);
    const beakTip = new THREE.Mesh(beakTipGeo, ivoryFangMat);
    beakTip.position.set(0, 0.065, 0.36 + snoutLen * 0.86);
    beakTip.rotation.x = Math.PI - 0.32;
    headGroup.add(beakTip);

    // Articulated Lower Raptor Beak on jawGroup + Saw-Tooth Beak Serrations
    const lowerBeakLen = 0.22 + snoutLen * 0.62;
    const lowerBeakCurve = new THREE.CatmullRomCurve3([
      new THREE.Vector3(0, 0.0, 0.0),
      new THREE.Vector3(0, -0.01, lowerBeakLen * 0.55),
      new THREE.Vector3(0, 0.015, lowerBeakLen),
    ]);
    const lowerBeakGeo = trackGeo(
      new THREE.TubeGeometry(lowerBeakCurve, seg(12, 8), 0.058, seg(10, 8), false),
      0.01
    );
    lowerBeakGeo.scale(0.84, 0.72, 1.0);
    const lowerBeakMesh = new THREE.Mesh(lowerBeakGeo, secondaryTrimMat);
    jawGroup.add(lowerBeakMesh);

    for (const side of [-1, 1]) {
      for (let t = 0; t < 3; t++) {
        const tomiaGeo = trackGeo(new THREE.ConeGeometry(0.016, 0.052, 6), 0.003);
        const tomiaTooth = new THREE.Mesh(tomiaGeo, ivoryFangMat);
        tomiaTooth.position.set(side * 0.038, 0.025, 0.06 + t * (lowerBeakLen * 0.25));
        tomiaTooth.rotation.x = 0.25;
        jawGroup.add(tomiaTooth);
      }
    }
  } else if (
    config.helm === 'goblin_feral_face' ||
    config.archetype === 'forest_goblin_scavenger'
  ) {
    // BLENDER-SCULPTED GOBLIN ("TRASGO") HEAD & FACE:
    // Prominent cheekbones, deep scowling brow ridges, long hooked warty nose,
    // rabid bloodshot eyes with slit pupils, and a wide snarling jaw packed with huge jagged teeth & tusks!
    const rabidScleraMat = trackMat(
      new THREE.MeshStandardMaterial({
        name: 'Mat_GoblinRabidSclera',
        color: new THREE.Color('#fef08a'),
        emissive: new THREE.Color('#dc2626'),
        emissiveIntensity: 1.9,
        roughness: 0.14,
        metalness: 0.05,
        wireframe,
      })
    );
    const rabidIrisMat = trackMat(
      new THREE.MeshStandardMaterial({
        name: 'Mat_GoblinRabidIris',
        color: new THREE.Color('#ff2a00'),
        emissive: new THREE.Color('#ff4500'),
        emissiveIntensity: 3.4,
        roughness: 0.1,
        metalness: 0.08,
        wireframe,
      })
    );
    const slitPupilMat = trackMat(
      new THREE.MeshStandardMaterial({
        name: 'Mat_GoblinSlitPupil',
        color: new THREE.Color('#090505'),
        roughness: 0.08,
        metalness: 0.2,
        wireframe,
      })
    );

    // 1) Blender-Sculpted Goblin Cranium (Wide temporal skull, recessed eye sockets, flared zygomatic cheekbones)
    const craniumGeo = new THREE.SphereGeometry(0.245, seg(24, 18), seg(20, 14));
    const cPos = craniumGeo.getAttribute('position');
    for (let i = 0; i < cPos.count; i++) {
      let x = cPos.getX(i);
      let y = cPos.getY(i);
      let z = cPos.getZ(i);
      const ax = Math.abs(x);

      // Flare out wide goblin zygomatic cheekbones
      if (y > -0.08 && y < 0.06 && z > 0.02 && ax > 0.09) {
        const cheekBoost = Math.exp(-Math.pow((y + 0.01) / 0.06, 2));
        x *= 1.0 + 0.28 * cheekBoost;
        z += 0.045 * cheekBoost;
      }
      // Carve deep orbital eye sockets under the brow
      if (y > 0.0 && y < 0.11 && z > 0.1 && ax > 0.04 && ax < 0.16) {
        z -= 0.042;
      }
      // Protrude fierce furrowed glabella / supraorbital brow ridge
      if (y >= 0.07 && y < 0.16 && z > 0.08) {
        const browV = 1.0 - Math.min(1.0, ax / 0.18) * 0.35;
        z += 0.058 * browV;
        y -= (1.0 - Math.min(1.0, ax / 0.14)) * 0.028; // Angry V-scowl in center
      }
      // Narrow lower temples into a wide snarling upper maxilla
      if (y < -0.06 && z > 0.04) {
        z += 0.055;
        x *= 1.12;
      }

      cPos.setXYZ(i, x, y, z);
    }
    craniumGeo.scale(1.06, 0.94, 1.14);
    trackGeo(craniumGeo, 0.02, 7.5, 0.28);
    const craniumMesh = new THREE.Mesh(craniumGeo, primaryArmorMat);
    craniumMesh.position.set(0, 0.19, 0.04);
    craniumMesh.castShadow = true;
    headGroup.add(craniumMesh);

    // 2) Heavy Scowling V-Shaped Supraorbital Brow Ridges & Wrinkled Glabella
    for (const side of [-1, 1]) {
      const browCurve = new THREE.CatmullRomCurve3([
        new THREE.Vector3(side * 0.02, 0.235, 0.245),
        new THREE.Vector3(side * 0.11, 0.268, 0.235),
        new THREE.Vector3(side * 0.195, 0.255, 0.155),
      ]);
      const browGeo = trackGeo(
        new THREE.TubeGeometry(browCurve, seg(12, 8), 0.044, seg(10, 8), false),
        0.016
      );
      const browMesh = new THREE.Mesh(browGeo, primaryArmorMat);
      headGroup.add(browMesh);

      // Prominent Zygomatic Cheekbone & Snarling Nasolabial Fold
      const cheekGeo = trackGeo(new THREE.SphereGeometry(0.088, seg(12, 8), seg(10, 8)), 0.018);
      cheekGeo.scale(1.18, 0.72, 1.25);
      const cheekMesh = new THREE.Mesh(cheekGeo, primaryArmorMat);
      cheekMesh.position.set(side * 0.155, 0.145, 0.155);
      cheekMesh.rotation.y = side * 0.28;
      cheekMesh.rotation.z = -side * 0.18;
      headGroup.add(cheekMesh);

      // Scavenger Studded Leather/Iron Skullcap Rim on top-back of head (leaves 100% of face uncovered!)
      const templeStudGeo = trackGeo(new THREE.ConeGeometry(0.026, 0.065, 6), 0.005);
      const templeStud = new THREE.Mesh(templeStudGeo, secondaryTrimMat);
      templeStud.position.set(side * 0.17, 0.33, -0.02);
      templeStud.rotation.z = -side * 0.45;
      headGroup.add(templeStud);
    }

    // Open Scavenger Leather Skullcap Band sitting high on the back crown
    const capBandGeo = trackGeo(
      new THREE.SphereGeometry(0.242, seg(16, 12), seg(12, 8), 0, Math.PI * 2, 0, Math.PI * 0.42),
      0.014
    );
    capBandGeo.scale(1.07, 0.92, 1.12);
    const capBandMesh = new THREE.Mesh(capBandGeo, secondaryTrimMat);
    capBandMesh.position.set(0, 0.21, 0.01);
    capBandMesh.rotation.x = -0.36; // Tilted backward so forehead, brows, eyes, nose & mouth are 100% exposed
    headGroup.add(capBandMesh);

    // 3) Long Hooked Warty Goblin Nose (Protruding crookedly forward and hooking down over the upper lip)
    const noseLen = 0.18 + snoutLen * 0.42;
    const noseCurve = new THREE.CatmullRomCurve3([
      new THREE.Vector3(0, 0.23, 0.18),
      new THREE.Vector3(0, 0.205, 0.26 + noseLen * 0.45),
      new THREE.Vector3(0, 0.155, 0.28 + noseLen * 0.92),
      new THREE.Vector3(0, 0.12, 0.26 + noseLen * 1.02),
    ]);
    const noseGeo = trackGeo(
      new THREE.TubeGeometry(noseCurve, seg(14, 10), 0.048, seg(12, 8), false),
      0.016
    );
    noseGeo.scale(0.86, 1.05, 1.0);
    const noseMesh = new THREE.Mesh(noseGeo, primaryArmorMat);
    noseMesh.castShadow = true;
    headGroup.add(noseMesh);

    // Hooked bulbous goblin nose tip + flared nostrils + dorsal nose wart
    const noseTipGeo = trackGeo(new THREE.SphereGeometry(0.046, seg(10, 8), seg(8, 6)), 0.014);
    noseTipGeo.scale(0.95, 0.85, 1.25);
    const noseTipMesh = new THREE.Mesh(noseTipGeo, primaryArmorMat);
    noseTipMesh.position.set(0, 0.132, 0.27 + noseLen * 0.96);
    headGroup.add(noseTipMesh);

    const wartGeo = trackGeo(new THREE.SphereGeometry(0.022, 8, 6), 0.008);
    const wartMesh = new THREE.Mesh(wartGeo, underbodyMat);
    wartMesh.position.set(0.022, 0.2, 0.24 + noseLen * 0.55);
    headGroup.add(wartMesh);

    for (const side of [-1, 1]) {
      const nostrilGeo = trackGeo(new THREE.SphereGeometry(0.028, 8, 6), 0.01);
      nostrilGeo.scale(1.1, 0.8, 1.2);
      const nostrilMesh = new THREE.Mesh(nostrilGeo, primaryArmorMat);
      nostrilMesh.position.set(side * 0.042, 0.15, 0.24 + noseLen * 0.72);
      headGroup.add(nostrilMesh);
    }

    // 4) OJOS RABIOSOS (Large Bulging Rabid Bloodshot Eyes with Glowing Fiery Iris & Frenzied Slit Pupils)
    for (const side of [-1, 1]) {
      const eyeX = side * 0.105;
      const eyeY = 0.212;
      const eyeZ = 0.215;

      // Dark fleshy orbital socket rim
      const orbitRimGeo = trackGeo(
        new THREE.TorusGeometry(0.056, 0.015, seg(8, 6), seg(14, 10)),
        0.008
      );
      const orbitRim = new THREE.Mesh(orbitRimGeo, underbodyMat);
      orbitRim.position.set(eyeX, eyeY, eyeZ - 0.008);
      orbitRim.rotation.y = side * 0.24;
      orbitRim.rotation.z = -side * 0.22;
      headGroup.add(orbitRim);

      // Bulging bloodshot yellow-red rabid eyeball (Sclera)
      const scleraGeo = trackGeo(new THREE.SphereGeometry(0.052, seg(14, 10), seg(12, 8)), 0.004);
      scleraGeo.scale(1.05, 0.96, 1.12);
      const scleraMesh = new THREE.Mesh(scleraGeo, rabidScleraMat);
      scleraMesh.position.set(eyeX, eyeY, eyeZ);
      scleraMesh.rotation.y = side * 0.22;
      headGroup.add(scleraMesh);

      // Intense glowing crimson-orange rabid iris ring protruding at front of eyeball
      const irisGeo = trackGeo(new THREE.SphereGeometry(0.035, seg(12, 8), seg(10, 8)), 0.002);
      irisGeo.scale(1.0, 0.95, 0.65);
      const irisMesh = new THREE.Mesh(irisGeo, rabidIrisMat);
      irisMesh.position.set(eyeX + side * 0.006, eyeY, eyeZ + 0.038);
      irisMesh.rotation.y = side * 0.22;
      headGroup.add(irisMesh);

      // Contracted vertical predatory slit pupil + pinpoint rabid highlight
      const pupilGeo = trackGeo(new THREE.SphereGeometry(0.02, 10, 8), 0);
      pupilGeo.scale(0.38, 1.25, 0.55);
      const pupilMesh = new THREE.Mesh(pupilGeo, slitPupilMat);
      pupilMesh.position.set(eyeX + side * 0.008, eyeY, eyeZ + 0.052);
      pupilMesh.rotation.y = side * 0.22;
      pupilMesh.rotation.z = -side * 0.12;
      headGroup.add(pupilMesh);
    }

    // 5) GRANDES DIENTES Y FAUCES DE TRASGO (Wide Snarling Upper Lip + Protruding Underslung Lower Jaw + Huge Jagged Fangs & Tusks)
    // Upper snarling gum/maxilla arch
    const upperMawGeo = trackGeo(
      new THREE.CylinderGeometry(0.145, 0.165, 0.075, seg(16, 12), 2),
      0.014
    );
    upperMawGeo.scale(1.15, 1.0, 0.92);
    const upperMawMesh = new THREE.Mesh(upperMawGeo, primaryArmorMat);
    upperMawMesh.position.set(0, 0.115, 0.155);
    headGroup.add(upperMawMesh);

    // Dark oral cavity inside the open mouth so the white ivory teeth pop with high contrast
    const mouthCavityGeo = trackGeo(new THREE.SphereGeometry(0.135, seg(12, 8), seg(10, 8)), 0.008);
    mouthCavityGeo.scale(1.12, 0.65, 0.95);
    const mouthCavityMesh = new THREE.Mesh(mouthCavityGeo, slitPupilMat);
    mouthCavityMesh.position.set(0, 0.095, 0.155);
    headGroup.add(mouthCavityMesh);

    // Upper Row of 6 Oversized Jagged Goblin Teeth + 2 Huge Upper Canines
    const upperTeethSpecs = [
      { x: -0.125, r: 0.028, h: 0.115, rotZ: -0.18, rotX: Math.PI - 0.18 }, // Left huge upper canine
      { x: -0.082, r: 0.022, h: 0.088, rotZ: -0.08, rotX: Math.PI - 0.22 },
      { x: -0.044, r: 0.024, h: 0.096, rotZ: -0.04, rotX: Math.PI - 0.26 },
      { x: 0.0, r: 0.021, h: 0.085, rotZ: 0.02, rotX: Math.PI - 0.25 },
      { x: 0.044, r: 0.025, h: 0.098, rotZ: 0.05, rotX: Math.PI - 0.26 },
      { x: 0.082, r: 0.022, h: 0.088, rotZ: 0.08, rotX: Math.PI - 0.22 },
      { x: 0.125, r: 0.028, h: 0.115, rotZ: 0.18, rotX: Math.PI - 0.18 },  // Right huge upper canine
    ];
    for (let t = 0; t < upperTeethSpecs.length; t++) {
      const sp = upperTeethSpecs[t];
      const zCurve = Math.cos((sp.x / 0.145) * (Math.PI * 0.42)) * 0.135;
      const toothGeo = trackGeo(new THREE.ConeGeometry(sp.r, sp.h, seg(10, 8)), 0.005);
      const toothMesh = new THREE.Mesh(toothGeo, ivoryFangMat);
      toothMesh.position.set(sp.x, 0.082, 0.135 + zCurve);
      toothMesh.rotation.x = sp.rotX;
      toothMesh.rotation.z = sp.rotZ;
      headGroup.add(toothMesh);
    }

    // Articulated Underslung Goblin Lower Mandible & Chin (Anchored on jawGroup at (0, 0.09, 0.06))
    const jawLen = 0.24 + snoutLen * 0.22;
    const jawPts = [
      new THREE.Vector2(0.01, -0.04),
      new THREE.Vector2(0.168, 0.0),
      new THREE.Vector2(0.162, jawLen * 0.55),
      new THREE.Vector2(0.128, jawLen * 0.92),
      new THREE.Vector2(0.01, jawLen + 0.03),
    ];
    const lowerJawGeo = trackGeo(new THREE.LatheGeometry(jawPts, seg(18, 12)), 0.018);
    lowerJawGeo.rotateX(Math.PI / 2); // +Y -> +Z forward
    lowerJawGeo.scale(1.08, 0.58, 1.0);
    const lowerJawMesh = new THREE.Mesh(lowerJawGeo, primaryArmorMat);
    lowerJawMesh.position.set(0, -0.035, 0.0);
    lowerJawMesh.castShadow = true;
    jawGroup.add(lowerJawMesh);

    // Jutting Goblin Chin Boss
    const chinGeo = trackGeo(new THREE.SphereGeometry(0.068, seg(12, 8), seg(10, 8)), 0.015);
    chinGeo.scale(1.15, 0.85, 1.25);
    const chinMesh = new THREE.Mesh(chinGeo, underbodyMat);
    chinMesh.position.set(0, -0.055, jawLen * 0.92);
    jawGroup.add(chinMesh);

    // Lower Row of Massive Protruding Tusks & Jagged Lower Teeth anchored on jawGroup
    const lowerTeethSpecs = [
      { x: -0.138, r: 0.034, h: 0.155, rotZ: 0.24, rotX: 0.28 },  // Left massive outer goblin tusk!
      { x: -0.096, r: 0.026, h: 0.112, rotZ: 0.12, rotX: 0.22 },
      { x: -0.058, r: 0.024, h: 0.098, rotZ: 0.06, rotX: 0.25 },
      { x: -0.02, r: 0.022, h: 0.092, rotZ: 0.02, rotX: 0.28 },
      { x: 0.02, r: 0.022, h: 0.092, rotZ: -0.02, rotX: 0.28 },
      { x: 0.058, r: 0.024, h: 0.098, rotZ: -0.06, rotX: 0.25 },
      { x: 0.096, r: 0.026, h: 0.112, rotZ: -0.12, rotX: 0.22 },
      { x: 0.138, r: 0.034, h: 0.155, rotZ: -0.24, rotX: 0.28 }, // Right massive outer goblin tusk!
    ];
    for (let t = 0; t < lowerTeethSpecs.length; t++) {
      const sp = lowerTeethSpecs[t];
      const zCurve = Math.cos((sp.x / 0.155) * (Math.PI * 0.42)) * (jawLen * 0.86);
      const fangGeo = trackGeo(new THREE.ConeGeometry(sp.r, sp.h, seg(10, 8)), 0.005);
      const fangMesh = new THREE.Mesh(fangGeo, ivoryFangMat);
      fangMesh.position.set(sp.x, 0.022 + (Math.abs(sp.x) > 0.11 ? 0.022 : 0), 0.025 + zCurve);
      fangMesh.rotation.x = sp.rotX;
      fangMesh.rotation.z = sp.rotZ;
      jawGroup.add(fangMesh);
    }
  } else if (isSkeletalSkullFace) {
    // =========================================================================================
    // BLENDER-SCULPTED 3D ANATOMICAL HUMANOID SKULL (Esqueletos, Reyes Exánimes, Segadores, Momias y No Muertos)
    // Features vertex-sculpted cranial vault, temporal fossae, prominent 3D zygomatic cheekbone arches
    // with hollow sub-zygomatic cheek cavities, deep black orbital sockets with glowing rabid soul-fire eyes,
    // triangular piriform nasal cavity with septum, exposed maxillary alveolar ridge with 8 individual 3D upper teeth,
    // and an articulated skeletal lower mandible (jawGroup) with ascending rami, chin symphysis & 8 lower teeth!
    // =========================================================================================
    const isBoneWhiteSkull =
      isUndeadSkeleton ||
      config.materials.preset === 'ancient_bone';

    const skullBoneMat = isBoneWhiteSkull
      ? trackMat(
          new THREE.MeshStandardMaterial({
            name: 'Mat_SculptedCraniumBone',
            color: new THREE.Color('#dfd7c6').lerp(new THREE.Color(matCfg.primaryColor), 0.25),
            vertexColors: true,
            roughness: 0.56,
            metalness: 0.12,
            bumpMap: textures.armorBumpMap,
            bumpScale: 0.024,
            wireframe,
          })
        )
      : primaryArmorMat;

    const darkSocketMat = trackMat(
      new THREE.MeshStandardMaterial({
        name: 'Mat_SkullSocketVoid',
        color: new THREE.Color('#070506'),
        roughness: 0.92,
        metalness: 0.05,
        wireframe,
      })
    );

    const skullRabidEyeMat = trackMat(
      new THREE.MeshStandardMaterial({
        name: 'Mat_SkullRabidSoulEye',
        color: new THREE.Color('#fef08a').lerp(new THREE.Color(matCfg.emissiveColor), 0.55),
        emissive: new THREE.Color(matCfg.emissiveColor),
        emissiveIntensity: 3.6,
        roughness: 0.1,
        metalness: 0.08,
        wireframe,
      })
    );

    const toothEnamelMat = trackMat(
      new THREE.MeshStandardMaterial({
        name: 'Mat_ToothEnamel',
        color: new THREE.Color('#fefcf2'),
        roughness: 0.16,
        metalness: 0.04,
        wireframe,
      })
    );

    // 1) Blender-Sculpted Anatomical Human Skull Cranium & Maxilla
    const skullCraniumGeo = new THREE.SphereGeometry(0.245, seg(26, 20), seg(22, 16));
    const sPos = skullCraniumGeo.getAttribute('position');
    for (let i = 0; i < sPos.count; i++) {
      let x = sPos.getX(i);
      let y = sPos.getY(i);
      let z = sPos.getZ(i);
      const ax = Math.abs(x);

      // Narrow temporal fossae behind the brow on the sides of the cranium
      if (y > -0.02 && y < 0.13 && ax > 0.13 && z > -0.06 && z < 0.11) {
        x *= 0.87;
      }
      // Flare out sharp 3D zygomatic cheekbones (malars)
      if (y > -0.055 && y < 0.045 && z > 0.03 && ax > 0.085) {
        const zygoFactor = Math.exp(-Math.pow((y + 0.005) / 0.052, 2));
        x *= 1.0 + 0.26 * zygoFactor;
        z += 0.044 * zygoFactor;
      }
      // Carve deep sunken sub-zygomatic cheek cavities (maxillary hollows below cheekbones)
      if (y <= -0.038 && ax > 0.082 && z > -0.04) {
        x *= 0.72;
        z -= 0.038;
      }
      // Carve deep recessed orbital eye sockets
      if (y > -0.01 && y < 0.105 && z > 0.105 && ax > 0.034 && ax < 0.158) {
        z -= 0.064;
      }
      // Protrude fierce furrowed supraorbital brow ridge & glabella
      if (y >= 0.075 && y < 0.158 && z > 0.09) {
        const browW = 1.0 - Math.min(1.0, ax / 0.18) * 0.32;
        z += 0.054 * browW;
        y -= (1.0 - Math.min(1.0, ax / 0.13)) * 0.025; // Angry skeletal V-scowl
      }
      // Recess & flatten the bottom of the cranium below the upper maxillary gum line so it NEVER covers the teeth or mouth!
      if (y < -0.062 && z > -0.05) {
        y = -0.062 + (y + 0.062) * 0.15;
        z *= 0.82;
      }

      sPos.setXYZ(i, x, y, z);
    }
    skullCraniumGeo.scale(0.98, 1.02, 1.12);
    trackGeo(skullCraniumGeo, 0.016, 7.5, 0.28);
    const skullCraniumMesh = new THREE.Mesh(skullCraniumGeo, skullBoneMat);
    skullCraniumMesh.position.set(0, 0.212, 0.032);
    skullCraniumMesh.castShadow = true;
    headGroup.add(skullCraniumMesh);

    // 2) Deep Hollow Orbital Sockets, 3D Zygomatic Cheekbone Bridges & Rabid Soul-Fire Eyes
    for (const side of [-1, 1]) {
      const eyeX = side * 0.094;
      const eyeY = 0.214;
      const eyeZ = 0.202;

      // Deep black recessed orbital socket interior void
      const orbitVoidGeo = trackGeo(new THREE.SphereGeometry(0.062, seg(12, 8), seg(10, 8)), 0.004);
      orbitVoidGeo.scale(1.06, 0.94, 0.78);
      const orbitVoidMesh = new THREE.Mesh(orbitVoidGeo, darkSocketMat);
      orbitVoidMesh.position.set(eyeX, eyeY, eyeZ - 0.012);
      orbitVoidMesh.rotation.y = side * 0.2;
      headGroup.add(orbitVoidMesh);

      // Sculpted Bony Orbital Socket Rim
      const orbitRimGeo = trackGeo(
        new THREE.TorusGeometry(0.062, 0.017, seg(10, 8), seg(16, 12)),
        0.012
      );
      const orbitRimMesh = new THREE.Mesh(orbitRimGeo, skullBoneMat);
      orbitRimMesh.position.set(eyeX, eyeY, eyeZ - 0.004);
      orbitRimMesh.rotation.y = side * 0.22;
      orbitRimMesh.rotation.z = -side * 0.22;
      headGroup.add(orbitRimMesh);

      // Angry Supraorbital Brow Ridge Arch above the socket
      const browCurve = new THREE.CatmullRomCurve3([
        new THREE.Vector3(side * 0.018, 0.238, 0.242),
        new THREE.Vector3(side * 0.102, 0.272, 0.232),
        new THREE.Vector3(side * 0.178, 0.252, 0.165),
      ]);
      const browGeo = trackGeo(
        new THREE.TubeGeometry(browCurve, seg(12, 8), 0.036, seg(10, 8), false),
        0.014
      );
      const browMesh = new THREE.Mesh(browGeo, skullBoneMat);
      headGroup.add(browMesh);

      // 3D Zygomatic Arch (Cheekbone Bridge from infraorbital rim to temporal bone)
      const zygoCurve = new THREE.CatmullRomCurve3([
        new THREE.Vector3(side * 0.065, 0.152, 0.225),
        new THREE.Vector3(side * 0.165, 0.155, 0.168),
        new THREE.Vector3(side * 0.192, 0.168, 0.045),
      ]);
      const zygoGeo = trackGeo(
        new THREE.TubeGeometry(zygoCurve, seg(12, 8), 0.032, seg(10, 8), false),
        0.012
      );
      const zygoMesh = new THREE.Mesh(zygoGeo, skullBoneMat);
      zygoMesh.castShadow = true;
      headGroup.add(zygoMesh);

      // Dark sunken sub-zygomatic cheek hollow below the cheekbone
      const cheekHollowGeo = trackGeo(new THREE.SphereGeometry(0.054, 10, 8), 0.006);
      cheekHollowGeo.scale(0.65, 0.85, 1.15);
      const cheekHollowMesh = new THREE.Mesh(cheekHollowGeo, darkSocketMat);
      cheekHollowMesh.position.set(side * 0.125, 0.118, 0.135);
      headGroup.add(cheekHollowMesh);

      // Rabid Necromantic Soul-Fire Eye burning deep inside the dark skull socket
      const soulEyeGeo = trackGeo(new THREE.SphereGeometry(0.038, seg(12, 8), seg(10, 8)), 0.002);
      soulEyeGeo.scale(1.0, 0.92, 0.85);
      const soulEyeMesh = new THREE.Mesh(soulEyeGeo, skullRabidEyeMat);
      soulEyeMesh.position.set(eyeX, eyeY, eyeZ + 0.012);
      headGroup.add(soulEyeMesh);

      // Fierce contracted vertical slit pupil inside the burning eye
      const pupilGeo = trackGeo(new THREE.SphereGeometry(0.018, 8, 6), 0);
      pupilGeo.scale(0.38, 1.25, 0.55);
      const pupilMesh = new THREE.Mesh(pupilGeo, darkSocketMat);
      pupilMesh.position.set(eyeX + side * 0.005, eyeY, eyeZ + 0.036);
      headGroup.add(pupilMesh);
    }

    // 3) Triangular Piriform Nasal Cavity (Cavidad Nasal de Calavera + Tabique Óseo)
    const nasalBridgeGeo = trackGeo(new THREE.ConeGeometry(0.034, 0.085, 6), 0.008);
    const nasalBridgeMesh = new THREE.Mesh(nasalBridgeGeo, skullBoneMat);
    nasalBridgeMesh.position.set(0, 0.205, 0.242);
    nasalBridgeMesh.rotation.x = 0.38;
    headGroup.add(nasalBridgeMesh);

    const nasalVoidGeo = trackGeo(new THREE.ConeGeometry(0.044, 0.082, 4), 0.004);
    nasalVoidGeo.scale(0.95, 1.0, 0.65);
    const nasalVoidMesh = new THREE.Mesh(nasalVoidGeo, darkSocketMat);
    nasalVoidMesh.position.set(0, 0.162, 0.238);
    nasalVoidMesh.rotation.y = Math.PI * 0.25;
    headGroup.add(nasalVoidMesh);

    // Thin vertical bony nasal septum dividing the piriform nasal cavity
    const septumGeo = trackGeo(new THREE.BoxGeometry(0.01, 0.068, 0.035), 0.002);
    const septumMesh = new THREE.Mesh(septumGeo, skullBoneMat);
    septumMesh.position.set(0, 0.158, 0.244);
    headGroup.add(septumMesh);

    // 4) Exposed Upper Maxillary Alveolar Arch + Deep Black Oral Void + 8 High-Contrast 3D Upper Teeth
    const maxillaArchCurve = new THREE.CatmullRomCurve3([
      new THREE.Vector3(-0.125, 0.126, 0.115),
      new THREE.Vector3(-0.085, 0.122, 0.235),
      new THREE.Vector3(0, 0.12, 0.268),
      new THREE.Vector3(0.085, 0.122, 0.235),
      new THREE.Vector3(0.125, 0.126, 0.115),
    ]);
    const upperMaxillaGeo = trackGeo(
      new THREE.TubeGeometry(maxillaArchCurve, seg(18, 12), 0.026, seg(10, 8), false),
      0.008
    );
    upperMaxillaGeo.scale(1.0, 1.15, 1.0);
    const upperMaxillaMesh = new THREE.Mesh(upperMaxillaGeo, skullBoneMat);
    headGroup.add(upperMaxillaMesh);

    // Pitch-black recessed oral cavity backdrop so the bright white upper & lower teeth stand out with maximum contrast!
    const skullMouthVoidGeo = trackGeo(
      new THREE.CylinderGeometry(0.105, 0.108, 0.095, seg(14, 10)),
      0.004
    );
    skullMouthVoidGeo.scale(1.05, 1.0, 0.92);
    const skullMouthVoid = new THREE.Mesh(skullMouthVoidGeo, darkSocketMat);
    skullMouthVoid.position.set(0, 0.072, 0.145);
    headGroup.add(skullMouthVoid);

    // 8 Prominent 3D Sculpted Upper Skeletal Teeth (Central/Lateral Incisors, Long Sharp Canines, Premolars)
    // Positioned cleanly in front of skullMouthVoid (z up to 0.282) hanging down from the maxillary arch!
    const skullUpperTeeth = [
      { x: -0.108, r: 0.017, h: 0.064, isCanine: false, rotZ: -0.12 },
      { x: -0.078, r: 0.021, h: 0.088, isCanine: true, rotZ: -0.06 },  // Left upper canine fang
      { x: -0.048, r: 0.017, h: 0.068, isCanine: false, rotZ: -0.03 },
      { x: -0.017, r: 0.018, h: 0.072, isCanine: false, rotZ: -0.01 }, // Left central incisor
      { x: 0.017, r: 0.018, h: 0.072, isCanine: false, rotZ: 0.01 },   // Right central incisor
      { x: 0.048, r: 0.017, h: 0.068, isCanine: false, rotZ: 0.03 },
      { x: 0.078, r: 0.021, h: 0.088, isCanine: true, rotZ: 0.06 },   // Right upper canine fang
      { x: 0.108, r: 0.017, h: 0.064, isCanine: false, rotZ: 0.12 },
    ];
    for (let t = 0; t < skullUpperTeeth.length; t++) {
      const sp = skullUpperTeeth[t];
      const zCurve = Math.cos((sp.x / 0.128) * (Math.PI * 0.44)) * 0.132;
      const toothZ = 0.146 + zCurve;

      // Sculpted Bony Alveolar Root Bulge (Juga Alveolaria) gripping the root of each tooth
      const rootBulgeGeo = trackGeo(
        new THREE.CylinderGeometry(sp.r * 0.85, sp.r * 1.05, 0.034, 7),
        0.003
      );
      const rootBulge = new THREE.Mesh(rootBulgeGeo, skullBoneMat);
      rootBulge.position.set(sp.x, 0.114, toothZ - 0.004);
      headGroup.add(rootBulge);

      // Bright Polished Enamel Tooth Crown / Canine Fang
      const toothGeo = sp.isCanine
        ? trackGeo(new THREE.ConeGeometry(sp.r, sp.h, seg(10, 8)), 0.002)
        : trackGeo(
            new THREE.CylinderGeometry(sp.r * 0.68, sp.r * 0.94, sp.h, seg(8, 6)),
            0.002
          );
      toothGeo.scale(1.0, 1.0, 0.78);
      const toothMesh = new THREE.Mesh(toothGeo, toothEnamelMat);
      toothMesh.position.set(sp.x, 0.082 - (sp.isCanine ? 0.008 : 0), toothZ);
      toothMesh.rotation.x = sp.isCanine ? Math.PI - 0.08 : 0.06;
      toothMesh.rotation.z = sp.rotZ;
      toothMesh.castShadow = true;
      headGroup.add(toothMesh);
    }

    // 5) Articulated Skeletal Lower Mandible on jawGroup (U-Shaped Jawbone + Ascending Rami + Chin + 8 Lower Teeth)
    const jawLen = 0.22 + snoutLen * 0.14;
    const jawCurve = new THREE.CatmullRomCurve3([
      new THREE.Vector3(-0.142, 0.055, 0.0),
      new THREE.Vector3(-0.122, -0.052, jawLen * 0.44),
      new THREE.Vector3(0, -0.068, jawLen * 0.96),
      new THREE.Vector3(0.122, -0.052, jawLen * 0.44),
      new THREE.Vector3(0.142, 0.055, 0.0),
    ]);
    const mandibleGeo = trackGeo(
      new THREE.TubeGeometry(jawCurve, seg(20, 14), 0.032, seg(10, 8), false),
      0.01
    );
    mandibleGeo.scale(1.0, 1.15, 1.0);
    const mandibleMesh = new THREE.Mesh(mandibleGeo, skullBoneMat);
    mandibleMesh.castShadow = true;
    jawGroup.add(mandibleMesh);

    // Prominent Bony Chin Symphysis (Protuberancia Mentoniana) at bottom of mandible
    const chinBoneGeo = trackGeo(new THREE.SphereGeometry(0.044, seg(10, 8), seg(8, 6)), 0.01);
    chinBoneGeo.scale(1.15, 0.88, 1.12);
    const chinBoneMesh = new THREE.Mesh(chinBoneGeo, skullBoneMat);
    chinBoneMesh.position.set(0, -0.078, jawLen * 0.96);
    jawGroup.add(chinBoneMesh);

    // 8 Prominent 3D Lower Skeletal Teeth anchored on jawGroup (protruding cleanly upward above the mandible rim!)
    const skullLowerTeeth = [
      { x: -0.102, r: 0.016, h: 0.062, isCanine: false, rotZ: 0.1 },
      { x: -0.074, r: 0.02, h: 0.086, isCanine: true, rotZ: 0.06 },   // Left lower canine fang
      { x: -0.045, r: 0.016, h: 0.064, isCanine: false, rotZ: 0.03 },
      { x: -0.016, r: 0.016, h: 0.066, isCanine: false, rotZ: 0.01 },
      { x: 0.016, r: 0.016, h: 0.066, isCanine: false, rotZ: -0.01 },
      { x: 0.045, r: 0.016, h: 0.064, isCanine: false, rotZ: -0.03 },
      { x: 0.074, r: 0.02, h: 0.086, isCanine: true, rotZ: -0.06 },  // Right lower canine fang
      { x: 0.102, r: 0.016, h: 0.062, isCanine: false, rotZ: -0.1 },
    ];
    for (let t = 0; t < skullLowerTeeth.length; t++) {
      const sp = skullLowerTeeth[t];
      const zCurve = Math.cos((sp.x / 0.122) * (Math.PI * 0.44)) * (jawLen * 0.88);
      const toothGeo = sp.isCanine
        ? trackGeo(new THREE.ConeGeometry(sp.r, sp.h, seg(10, 8)), 0.002)
        : trackGeo(
            new THREE.CylinderGeometry(sp.r * 0.68, sp.r * 0.94, sp.h, seg(8, 6)),
            0.002
          );
      toothGeo.scale(1.0, 1.0, 0.78);
      const toothMesh = new THREE.Mesh(toothGeo, toothEnamelMat);
      toothMesh.position.set(sp.x, -0.02 + (sp.isCanine ? 0.01 : 0), 0.022 + zCurve);
      toothMesh.rotation.x = 0.08;
      toothMesh.rotation.z = sp.rotZ;
      toothMesh.castShadow = true;
      jawGroup.add(toothMesh);
    }

    // 6) BLENDER-SCULPTED ROYAL SKELETAL CROWNS, IMPERIAL ARCHES & VALKYRIE DIADEMS (leaves 100% of skull face exposed!)
    const hasRoyalSkullCrown =
      (config.helm === 'crowned_skull' &&
        config.archetype !== 'ruins_skeleton_pawn' &&
        config.archetype !== 'cursed_skeleton_berserker') ||
      config.archetype === 'bone_crown_lich_king' ||
      config.archetype === 'arch_lich' ||
      config.archetype === 'wraith_soul_reaper' ||
      config.archetype === 'death_knight_colossus' ||
      config.archetype === 'plague_hierophant_mummy' ||
      config.archetype === 'ash_bone_naga_siren' ||
      config.archetype === 'necrotic_archdragon' ||
      config.archetype === 'cursed_diadem_valkyrie_skeleton';

    if (hasRoyalSkullCrown) {
      // Lower & Upper Royal Circlet Bands hugging the sculpted skull brow
      const crownRingGeo = trackGeo(
        new THREE.TorusGeometry(0.218, 0.028, seg(10, 8), seg(24, 16)),
        0.008
      );
      crownRingGeo.rotateX(Math.PI / 2 - 0.16);
      const crownRingMesh = new THREE.Mesh(crownRingGeo, secondaryTrimMat);
      crownRingMesh.position.set(0, 0.332, 0.038);
      crownRingMesh.castShadow = true;
      headGroup.add(crownRingMesh);

      const upperFiligreeGeo = trackGeo(
        new THREE.TorusGeometry(0.224, 0.016, seg(8, 6), seg(22, 14)),
        0.006
      );
      upperFiligreeGeo.rotateX(Math.PI / 2 - 0.16);
      const upperFiligreeMesh = new THREE.Mesh(upperFiligreeGeo, ivoryFangMat);
      upperFiligreeMesh.position.set(0, 0.365, 0.036);
      headGroup.add(upperFiligreeMesh);

      // 7 Multi-Tiered Imperial Crown Spikes around the Royal Circlet
      const spikeCount = 7;
      for (let s = 0; s < spikeCount; s++) {
        const ang = ((s / (spikeCount - 1)) - 0.5) * Math.PI * 1.12;
        const isCenter = s === 3;
        const isMajor = s === 1 || s === 5;
        const spikeH = isCenter ? 0.24 : isMajor ? 0.185 : 0.135;
        const spikeR = isCenter ? 0.034 : isMajor ? 0.028 : 0.022;
        const spikeGeo = trackGeo(new THREE.ConeGeometry(spikeR, spikeH, 5), 0.004);
        const spikeMesh = new THREE.Mesh(
          spikeGeo,
          s % 2 === 0 ? ivoryFangMat : secondaryTrimMat
        );
        spikeMesh.position.set(
          Math.sin(ang) * 0.215,
          0.358 + spikeH * 0.44,
          0.038 + Math.cos(ang) * 0.215
        );
        spikeMesh.rotation.x = Math.cos(ang) * 0.18;
        spikeMesh.rotation.z = -Math.sin(ang) * 0.22;
        spikeMesh.castShadow = true;
        headGroup.add(spikeMesh);
      }

      // Imperial Arched Sovereign Dome (for Emperors, Kings, Pharaohs, Pontiffs, Star Liches & Elite Royal Skeletons)
      const hasImperialArches =
        config.archetype === 'skeleton_emperor_goldcrown' ||
        config.archetype === 'ossuary_king_leoric' ||
        config.archetype === 'crowned_catacomb_pharaoh' ||
        config.archetype === 'skeletal_archbishop_pontiff' ||
        config.archetype === 'glacial_crown_revenant_king' ||
        config.archetype === 'star_crown_astral_lich' ||
        config.archetype === 'abyssal_crown_ossuary_titan' ||
        config.archetype === 'bone_crown_lich_king' ||
        config.archetype === 'elite_pirate_captain_skeleton' ||
        config.archetype === 'elite_bone_scorpion_construct' ||
        config.archetype === 'elite_plague_doctor_lich_skeleton' ||
        config.archetype === 'elite_caged_rib_jailer_skeleton' ||
        config.archetype === 'elite_spiked_wheel_bone_martyr' ||
        config.archetype === 'elite_horned_minotaur_skeleton' ||
        config.archetype === 'elite_four_armed_bone_reaver' ||
        config.archetype === 'elite_twin_skull_ettin_skeleton' ||
        config.archetype === 'elite_mammoth_ossuary_behemoth';

      if (hasImperialArches) {
        for (let a = 0; a < 2; a++) {
          const archGeo = trackGeo(
            new THREE.TorusGeometry(0.21, 0.018, seg(8, 6), seg(18, 12), Math.PI),
            0.005
          );
          const archMesh = new THREE.Mesh(archGeo, secondaryTrimMat);
          archMesh.position.set(0, 0.345, 0.035);
          archMesh.rotation.y = a === 0 ? 0 : Math.PI * 0.5;
          headGroup.add(archMesh);
        }

        // Sovereign Globus Cruciger / Necromantic Apex Orb atop the Crown Arches
        const apexOrbGeo = trackGeo(new THREE.OctahedronGeometry(0.052, 1), 0);
        const apexOrb = new THREE.Mesh(apexOrbGeo, crystalCoreMat);
        apexOrb.position.set(0, 0.585, 0.035);
        headGroup.add(apexOrb);
      }

      // Swept Wing Plumes on Valkyrie Diadem Skeleton
      if (config.helm === 'valkyrie_crest' || config.archetype === 'cursed_diadem_valkyrie_skeleton') {
        for (const side of [-1, 1]) {
          for (let w = 0; w < 3; w++) {
            const wingBladeGeo = trackGeo(
              new THREE.ConeGeometry(0.036 - w * 0.005, 0.28 - w * 0.04, 4),
              0.004
            );
            wingBladeGeo.scale(0.38, 1.0, 1.35);
            const wingBlade = new THREE.Mesh(
              wingBladeGeo,
              w === 0 ? secondaryTrimMat : ivoryFangMat
            );
            wingBlade.position.set(
              side * (0.215 + w * 0.022),
              0.34 + w * 0.055,
              0.02 - w * 0.06
            );
            wingBlade.rotation.x = -0.72 - w * 0.14;
            wingBlade.rotation.z = -side * (0.35 + w * 0.1);
            headGroup.add(wingBlade);
          }
        }
      }

      // Forehead & Lateral Phylactery Soul Gems on the Royal Crown
      const crownGemGeo = trackGeo(new THREE.OctahedronGeometry(0.046, 1), 0);
      const crownGem = new THREE.Mesh(crownGemGeo, crystalCoreMat);
      crownGem.position.set(0, 0.345, 0.258);
      headGroup.add(crownGem);

      for (const side of [-1, 1]) {
        const sideGemGeo = trackGeo(new THREE.OctahedronGeometry(0.028, 0), 0);
        const sideGem = new THREE.Mesh(sideGemGeo, emissiveRuneMat);
        sideGem.position.set(side * 0.165, 0.345, 0.19);
        headGroup.add(sideGem);
      }
    }
  } else {
    // =========================================================================================
    // BLENDER-SCULPTED 3D HUMANOID / WARRIOR / VAMPIRE / VALKYRIE / BANDIT / ONI / DEMON FACE
    // Replaces the old closed cylinder faceplate with a true sculpted 3D humanoid head & face:
    // - Sculpted cheekbones, temporal planes, angry V-furrowed brow ridges & humanoid nose
    // - Rabid/glowing eyes with sclera, iris & slit/pinpoint pupils in deep orbital sockets
    // - Articulated lower mandible (jawGroup) with strong chin, dark oral cavity & full upper/lower teeth & fangs
    // - Signature open-faced helmet / Valkyrie winged tiara / Oni mask / Crown leaving the face 100% visible!
    // =========================================================================================
    const isVampireOrPaleUndead =
      config.archetype === 'vampire_lord' ||
      config.archetype === 'crimson_nosferatu_elder' ||
      config.archetype === 'grave_fallen_valkyrie' ||
      config.archetype === 'banshee_frozen_throne';

    const isHumanFleshTone =
      config.archetype === 'outlaw_bandit_cutthroat' ||
      config.archetype === 'solar_valkyrie' ||
      config.archetype === 'valkyrie_of_the_gale' ||
      config.archetype === 'drake_berserker' ||
      config.archetype === 'abyssal_knight' ||
      config.archetype === 'astral_centaur' ||
      config.archetype === 'aurora_ice_centaur';

    const isFangedHumanoid =
      isVampireOrPaleUndead ||
      config.archetype === 'samurai_oni' ||
      config.archetype === 'balrog_archdemon' ||
      config.archetype === 'lesser_fire_imp' ||
      config.archetype === 'drake_berserker' ||
      config.archetype === 'gorgon_medusa' ||
      config.archetype === 'triton_abyss_warden' ||
      config.helm === 'arachnid_mandibles' ||
      config.helm === 'drake_fang_helm';

    const humanoidFaceMat = isVampireOrPaleUndead
      ? trackMat(
          new THREE.MeshStandardMaterial({
            name: 'Mat_HumanoidPaleVampireSkin',
            color: new THREE.Color('#d8d4dc').lerp(new THREE.Color(matCfg.primaryColor), 0.22),
            vertexColors: true,
            roughness: 0.48,
            metalness: 0.08,
            bumpMap: textures.armorBumpMap,
            bumpScale: 0.018,
            wireframe,
          })
        )
      : isHumanFleshTone
      ? trackMat(
          new THREE.MeshStandardMaterial({
            name: 'Mat_HumanoidWarriorSkin',
            color: new THREE.Color('#c69076').lerp(new THREE.Color(matCfg.primaryColor), 0.28),
            vertexColors: true,
            roughness: 0.55,
            metalness: 0.08,
            bumpMap: textures.armorBumpMap,
            bumpScale: 0.018,
            wireframe,
          })
        )
      : primaryArmorMat;

    const rabidScleraMat = trackMat(
      new THREE.MeshStandardMaterial({
        name: 'Mat_HumanoidRabidSclera',
        color: new THREE.Color('#fef08a'),
        emissive: new THREE.Color(matCfg.emissiveColor),
        emissiveIntensity: 1.6,
        roughness: 0.14,
        metalness: 0.05,
        wireframe,
      })
    );
    const rabidIrisMat = trackMat(
      new THREE.MeshStandardMaterial({
        name: 'Mat_HumanoidRabidIris',
        color: new THREE.Color(matCfg.emissiveColor),
        emissive: new THREE.Color(matCfg.emissiveColor),
        emissiveIntensity: 3.4,
        roughness: 0.1,
        metalness: 0.08,
        wireframe,
      })
    );
    const darkPupilMat = trackMat(
      new THREE.MeshStandardMaterial({
        name: 'Mat_HumanoidDarkPupil',
        color: new THREE.Color('#080506'),
        roughness: 0.1,
        metalness: 0.18,
        wireframe,
      })
    );

    // 1) Blender-Sculpted Humanoid Cranium, Zygomatic Cheekbones, Brow Scowl & Maxilla
    const craniumGeo = new THREE.SphereGeometry(0.242, seg(24, 18), seg(20, 14));
    const cPos = craniumGeo.getAttribute('position');
    for (let i = 0; i < cPos.count; i++) {
      let x = cPos.getX(i);
      let y = cPos.getY(i);
      let z = cPos.getZ(i);
      const ax = Math.abs(x);

      // Flatten temporal sides slightly for athletic humanoid skull proportions
      if (y > -0.04 && y < 0.12 && ax > 0.15 && z < 0.08) {
        x *= 0.91;
      }
      // Sculpt prominent humanoid zygomatic cheekbones
      if (y > -0.07 && y < 0.05 && z > 0.03 && ax > 0.085) {
        const cheekBoost = Math.exp(-Math.pow((y + 0.01) / 0.055, 2));
        x *= 1.0 + 0.2 * cheekBoost;
        z += 0.038 * cheekBoost;
      }
      // Carve deep orbital eye sockets under the brow
      if (y > 0.0 && y < 0.105 && z > 0.105 && ax > 0.038 && ax < 0.155) {
        z -= 0.042;
      }
      // Protrude fierce furrowed supraorbital brow ridge & angry V-glabella
      if (y >= 0.072 && y < 0.155 && z > 0.085) {
        const browV = 1.0 - Math.min(1.0, ax / 0.18) * 0.34;
        z += 0.05 * browV;
        y -= (1.0 - Math.min(1.0, ax / 0.13)) * 0.024;
      }
      // Sculpt upper maxillary lip ridge
      if (y < -0.055 && ax < 0.12 && z > 0.05) {
        z += 0.045;
      }

      cPos.setXYZ(i, x, y, z);
    }
    craniumGeo.scale(0.98, 1.02, 1.12);
    trackGeo(craniumGeo, 0.018, 7.5, 0.28);
    const craniumMesh = new THREE.Mesh(craniumGeo, humanoidFaceMat);
    craniumMesh.position.set(0, 0.195, 0.035);
    craniumMesh.castShadow = true;
    headGroup.add(craniumMesh);

    // 2) Sculpted Angry Supraorbital Eyebrow Ridges, Cheekbones & Lateral Humanoid/Elven Ears
    const hasPointedElvenOrDemonEars =
      isVampireOrPaleUndead ||
      config.archetype === 'balrog_archdemon' ||
      config.archetype === 'lesser_fire_imp' ||
      config.archetype === 'samurai_oni' ||
      config.archetype === 'tidal_naiad_queen' ||
      config.archetype === 'triton_abyss_warden' ||
      config.archetype === 'gorgon_medusa' ||
      config.archetype === 'briar_wood_spriggan';

    for (const side of [-1, 1]) {
      // Fierce V-slanted supraorbital brow ridge
      const browCurve = new THREE.CatmullRomCurve3([
        new THREE.Vector3(side * 0.018, 0.236, 0.242),
        new THREE.Vector3(side * 0.102, 0.266, 0.232),
        new THREE.Vector3(side * 0.182, 0.252, 0.158),
      ]);
      const browGeo = trackGeo(
        new THREE.TubeGeometry(browCurve, seg(12, 8), 0.036, seg(10, 8), false),
        0.014
      );
      const browMesh = new THREE.Mesh(browGeo, humanoidFaceMat);
      headGroup.add(browMesh);

      // Sculpted Zygomatic Cheekbone & Nasolabial Snarl Fold
      const cheekGeo = trackGeo(new THREE.SphereGeometry(0.078, seg(12, 8), seg(10, 8)), 0.016);
      cheekGeo.scale(1.12, 0.74, 1.22);
      const cheekMesh = new THREE.Mesh(cheekGeo, humanoidFaceMat);
      cheekMesh.position.set(side * 0.142, 0.15, 0.158);
      cheekMesh.rotation.y = side * 0.26;
      cheekMesh.rotation.z = -side * 0.16;
      headGroup.add(cheekMesh);

      // Lateral Humanoid or Pointed Vampire/Demon/Elven Ears
      const earLen = hasPointedElvenOrDemonEars ? 0.22 : 0.11;
      const earGeo = trackGeo(
        new THREE.ConeGeometry(0.052, earLen, seg(10, 8)),
        0.012
      );
      earGeo.scale(0.48, 1.0, 1.25);
      const earMesh = new THREE.Mesh(earGeo, humanoidFaceMat);
      earMesh.position.set(
        side * (hasPointedElvenOrDemonEars ? 0.235 : 0.205),
        0.195 + (hasPointedElvenOrDemonEars ? 0.025 : 0),
        0.01
      );
      earMesh.rotation.z = -side * (hasPointedElvenOrDemonEars ? 1.05 : 1.35);
      earMesh.rotation.x = -0.25;
      headGroup.add(earMesh);
    }

    // 3) Blender-Sculpted Humanoid Nose (Nasal Bridge + Nose Tip + Flared Nostril Wings)
    const noseBridgeLen = 0.12 + snoutLen * 0.22;
    const noseCurve = new THREE.CatmullRomCurve3([
      new THREE.Vector3(0, 0.232, 0.185),
      new THREE.Vector3(0, 0.195, 0.245 + noseBridgeLen * 0.48),
      new THREE.Vector3(0, 0.158, 0.262 + noseBridgeLen * 0.85),
    ]);
    const noseGeo = trackGeo(
      new THREE.TubeGeometry(noseCurve, seg(12, 8), 0.036, seg(10, 8), false),
      0.014
    );
    noseGeo.scale(0.82, 1.04, 1.0);
    const noseMesh = new THREE.Mesh(noseGeo, humanoidFaceMat);
    noseMesh.castShadow = true;
    headGroup.add(noseMesh);

    const noseTipGeo = trackGeo(new THREE.SphereGeometry(0.036, seg(10, 8), seg(8, 6)), 0.012);
    noseTipGeo.scale(0.96, 0.84, 1.16);
    const noseTipMesh = new THREE.Mesh(noseTipGeo, humanoidFaceMat);
    noseTipMesh.position.set(0, 0.154, 0.265 + noseBridgeLen * 0.86);
    headGroup.add(noseTipMesh);

    for (const side of [-1, 1]) {
      const nostrilAlaGeo = trackGeo(new THREE.SphereGeometry(0.024, 8, 6), 0.008);
      nostrilAlaGeo.scale(1.1, 0.78, 1.15);
      const nostrilAla = new THREE.Mesh(nostrilAlaGeo, humanoidFaceMat);
      nostrilAla.position.set(side * 0.034, 0.148, 0.248 + noseBridgeLen * 0.68);
      headGroup.add(nostrilAla);
    }

    // 4) OJOS RABIOSOS / EXPRESIVOS HUMANOIDES (Deep Orbital Eyelid Rims + Sclera + Glowing Iris + Slit/Pinpoint Pupils)
    for (const side of [-1, 1]) {
      const eyeX = side * 0.095;
      const eyeY = 0.214;
      const eyeZ = 0.212;

      const orbitRimGeo = trackGeo(
        new THREE.TorusGeometry(0.048, 0.013, seg(8, 6), seg(14, 10)),
        0.006
      );
      const orbitRim = new THREE.Mesh(orbitRimGeo, underbodyMat);
      orbitRim.position.set(eyeX, eyeY, eyeZ - 0.006);
      orbitRim.rotation.y = side * 0.22;
      orbitRim.rotation.z = -side * 0.2;
      headGroup.add(orbitRim);

      const scleraGeo = trackGeo(new THREE.SphereGeometry(0.045, seg(14, 10), seg(12, 8)), 0.004);
      scleraGeo.scale(1.06, 0.92, 1.08);
      const scleraMesh = new THREE.Mesh(scleraGeo, rabidScleraMat);
      scleraMesh.position.set(eyeX, eyeY, eyeZ);
      scleraMesh.rotation.y = side * 0.2;
      headGroup.add(scleraMesh);

      const irisGeo = trackGeo(new THREE.SphereGeometry(0.03, seg(12, 8), seg(10, 8)), 0.002);
      irisGeo.scale(1.0, 0.94, 0.65);
      const irisMesh = new THREE.Mesh(irisGeo, rabidIrisMat);
      irisMesh.position.set(eyeX + side * 0.005, eyeY, eyeZ + 0.032);
      irisMesh.rotation.y = side * 0.2;
      headGroup.add(irisMesh);

      const pupilGeo = trackGeo(new THREE.SphereGeometry(0.017, 10, 8), 0);
      pupilGeo.scale(isFangedHumanoid ? 0.38 : 0.68, isFangedHumanoid ? 1.22 : 0.95, 0.55);
      const pupilMesh = new THREE.Mesh(pupilGeo, darkPupilMat);
      pupilMesh.position.set(eyeX + side * 0.007, eyeY, eyeZ + 0.045);
      pupilMesh.rotation.y = side * 0.2;
      headGroup.add(pupilMesh);
    }

    // 5) FAUCES Y DIENTES HUMANOIDES / COLMILLOS DE VAMPIRO, ONI Y GUERRERO
    const upperMawGeo = trackGeo(
      new THREE.CylinderGeometry(0.132, 0.148, 0.068, seg(16, 12), 2),
      0.012
    );
    upperMawGeo.scale(1.08, 1.0, 0.92);
    const upperMawMesh = new THREE.Mesh(upperMawGeo, humanoidFaceMat);
    upperMawMesh.position.set(0, 0.115, 0.152);
    headGroup.add(upperMawMesh);

    const mouthCavityGeo = trackGeo(new THREE.SphereGeometry(0.125, seg(12, 8), seg(10, 8)), 0.006);
    mouthCavityGeo.scale(1.08, 0.64, 0.94);
    const mouthCavityMesh = new THREE.Mesh(mouthCavityGeo, darkPupilMat);
    mouthCavityMesh.position.set(0, 0.094, 0.152);
    headGroup.add(mouthCavityMesh);

    // Upper Row of 7 Sculpted 3D Humanoid Teeth & Prominent Canine Fangs
    const fangMult = isFangedHumanoid ? 1.32 : 1.0;
    const upperTeethSpecs = [
      { x: -0.108, r: 0.022 * fangMult, h: 0.095 * fangMult, isFang: true, rotZ: -0.14 }, // Left upper canine fang
      { x: -0.072, r: 0.018, h: 0.072, isFang: false, rotZ: -0.06 },
      { x: -0.036, r: 0.019, h: 0.076, isFang: false, rotZ: -0.03 },
      { x: 0.0, r: 0.019, h: 0.076, isFang: false, rotZ: 0.0 },
      { x: 0.036, r: 0.019, h: 0.076, isFang: false, rotZ: 0.03 },
      { x: 0.072, r: 0.018, h: 0.072, isFang: false, rotZ: 0.06 },
      { x: 0.108, r: 0.022 * fangMult, h: 0.095 * fangMult, isFang: true, rotZ: 0.14 },  // Right upper canine fang
    ];
    for (let t = 0; t < upperTeethSpecs.length; t++) {
      const sp = upperTeethSpecs[t];
      const zCurve = Math.cos((sp.x / 0.132) * (Math.PI * 0.42)) * 0.128;
      const toothGeo = trackGeo(new THREE.ConeGeometry(sp.r, sp.h, seg(10, 8)), 0.004);
      const toothMesh = new THREE.Mesh(toothGeo, ivoryFangMat);
      toothMesh.position.set(sp.x, 0.082 - (sp.isFang ? 0.008 : 0), 0.132 + zCurve);
      toothMesh.rotation.x = Math.PI - 0.2;
      toothMesh.rotation.z = sp.rotZ;
      headGroup.add(toothMesh);
    }

    // Articulated Humanoid Lower Mandible, Chin & Lower Teeth Row on jawGroup
    const jawLen = 0.215 + snoutLen * 0.18;
    const jawPts = [
      new THREE.Vector2(0.01, -0.04),
      new THREE.Vector2(0.152, 0.0),
      new THREE.Vector2(0.144, jawLen * 0.55),
      new THREE.Vector2(0.112, jawLen * 0.92),
      new THREE.Vector2(0.01, jawLen + 0.025),
    ];
    const lowerJawGeo = trackGeo(new THREE.LatheGeometry(jawPts, seg(18, 12)), 0.016);
    lowerJawGeo.rotateX(Math.PI / 2);
    lowerJawGeo.scale(1.02, 0.56, 1.0);
    const lowerJawMesh = new THREE.Mesh(lowerJawGeo, humanoidFaceMat);
    lowerJawMesh.position.set(0, -0.032, 0.0);
    lowerJawMesh.castShadow = true;
    jawGroup.add(lowerJawMesh);

    const chinGeo = trackGeo(new THREE.SphereGeometry(0.058, seg(12, 8), seg(10, 8)), 0.014);
    chinGeo.scale(1.12, 0.86, 1.2);
    const chinMesh = new THREE.Mesh(chinGeo, humanoidFaceMat);
    chinMesh.position.set(0, -0.05, jawLen * 0.92);
    jawGroup.add(chinMesh);

    const lowerTeethSpecs = [
      { x: -0.114, r: 0.024 * fangMult, h: 0.11 * fangMult, rotZ: 0.18, rotX: 0.26 }, // Left lower canine tusk
      { x: -0.078, r: 0.019, h: 0.078, rotZ: 0.09, rotX: 0.22 },
      { x: -0.044, r: 0.018, h: 0.074, rotZ: 0.04, rotX: 0.24 },
      { x: -0.014, r: 0.018, h: 0.074, rotZ: 0.01, rotX: 0.25 },
      { x: 0.014, r: 0.018, h: 0.074, rotZ: -0.01, rotX: 0.25 },
      { x: 0.044, r: 0.018, h: 0.074, rotZ: -0.04, rotX: 0.24 },
      { x: 0.078, r: 0.019, h: 0.078, rotZ: -0.09, rotX: 0.22 },
      { x: 0.114, r: 0.024 * fangMult, h: 0.11 * fangMult, rotZ: -0.18, rotX: 0.26 }, // Right lower canine tusk
    ];
    for (let t = 0; t < lowerTeethSpecs.length; t++) {
      const sp = lowerTeethSpecs[t];
      const zCurve = Math.cos((sp.x / 0.135) * (Math.PI * 0.42)) * (jawLen * 0.84);
      const fangGeo = trackGeo(new THREE.ConeGeometry(sp.r, sp.h, seg(10, 8)), 0.004);
      const fangMesh = new THREE.Mesh(fangGeo, ivoryFangMat);
      fangMesh.position.set(sp.x, 0.016 + (Math.abs(sp.x) > 0.095 ? 0.015 : 0), 0.022 + zCurve);
      fangMesh.rotation.x = sp.rotX;
      fangMesh.rotation.z = sp.rotZ;
      jawGroup.add(fangMesh);
    }

    // 6) OPEN-FACED SIGNATURE HEADGEAR (Leaves 100% of the sculpted eyes, nose, cheekbones & toothed mouth visible!)
    if (config.helm === 'valkyrie_crest') {
      // Open-Faced Valkyrie / Celestial Diadem Tiara + Forehead Gem + Swept Wing Plumes
      const tiaraGeo = trackGeo(
        new THREE.TorusGeometry(0.215, 0.024, seg(10, 8), seg(20, 14), Math.PI * 1.15),
        0.008
      );
      tiaraGeo.rotateX(Math.PI / 2 - 0.15);
      tiaraGeo.rotateZ(-Math.PI * 0.075);
      const tiaraMesh = new THREE.Mesh(tiaraGeo, secondaryTrimMat);
      tiaraMesh.position.set(0, 0.305, 0.04);
      headGroup.add(tiaraMesh);

      const foreheadGemGeo = trackGeo(new THREE.OctahedronGeometry(0.044, 1), 0);
      const foreheadGem = new THREE.Mesh(foreheadGemGeo, crystalCoreMat);
      foreheadGem.position.set(0, 0.315, 0.255);
      headGroup.add(foreheadGem);

      for (const side of [-1, 1]) {
        for (let w = 0; w < 3; w++) {
          const wingBladeGeo = trackGeo(
            new THREE.ConeGeometry(0.036 - w * 0.005, 0.28 - w * 0.04, 4),
            0.004
          );
          wingBladeGeo.scale(0.38, 1.0, 1.35);
          const wingBlade = new THREE.Mesh(
            wingBladeGeo,
            w === 0 ? secondaryTrimMat : bladeAlloyMat
          );
          wingBlade.position.set(
            side * (0.21 + w * 0.02),
            0.32 + w * 0.055,
            0.02 - w * 0.06
          );
          wingBlade.rotation.x = -0.72 - w * 0.14;
          wingBlade.rotation.z = -side * (0.35 + w * 0.1);
          headGroup.add(wingBlade);
        }
      }
    } else if (config.helm === 'arachnid_mandibles') {
      // Chitinous Predator Carapace Crown + 4 Extra Forehead Ocelli Eyes + Lateral Chelicerae Pincers
      const carapaceDomeGeo = trackGeo(
        new THREE.SphereGeometry(0.248, seg(16, 12), seg(12, 8), 0, Math.PI * 2, 0, Math.PI * 0.44),
        0.014
      );
      carapaceDomeGeo.scale(1.04, 0.95, 1.14);
      const carapaceDome = new THREE.Mesh(carapaceDomeGeo, secondaryTrimMat);
      carapaceDome.position.set(0, 0.205, 0.025);
      carapaceDome.rotation.x = -0.25;
      headGroup.add(carapaceDome);

      for (const side of [-1, 1]) {
        // 2 Extra Arachnid Ocelli Eyes per side above the main rabid eyes
        for (let e = 0; e < 2; e++) {
          const ocellusGeo = trackGeo(new THREE.SphereGeometry(0.022 - e * 0.004, 8, 6), 0);
          const ocellusMesh = new THREE.Mesh(ocellusGeo, rabidIrisMat);
          ocellusMesh.position.set(
            side * (0.065 + e * 0.058),
            0.285 + e * 0.015,
            0.225 - e * 0.025
          );
          headGroup.add(ocellusMesh);
        }

        // Curved Lateral Chelicerae Mandible Pincers flanking the fanged mouth
        const pincerCurve = new THREE.CatmullRomCurve3([
          new THREE.Vector3(side * 0.155, 0.115, 0.12),
          new THREE.Vector3(side * 0.205, 0.085, 0.24),
          new THREE.Vector3(side * 0.095, 0.055, 0.34),
        ]);
        const pincerGeo = trackGeo(
          new THREE.TubeGeometry(pincerCurve, seg(12, 8), 0.034, seg(10, 8), false),
          0.01
        );
        const pincerMesh = new THREE.Mesh(pincerGeo, secondaryTrimMat);
        headGroup.add(pincerMesh);

        const pincerTipGeo = trackGeo(new THREE.ConeGeometry(0.026, 0.1, 6), 0.004);
        const pincerTip = new THREE.Mesh(pincerTipGeo, ivoryFangMat);
        pincerTip.position.set(side * 0.075, 0.05, 0.355);
        pincerTip.rotation.z = side * 1.15;
        pincerTip.rotation.x = Math.PI * 0.42;
        headGroup.add(pincerTip);
      }
    } else if (config.helm === 'monolith_visor') {
      // Faceted Runic Titan Crown + Lateral Cheek Monoliths framing the carved stone face
      const monolithCrownGeo = trackGeo(new THREE.OctahedronGeometry(0.24, 1), 0.008);
      monolithCrownGeo.scale(1.06, 0.65, 1.08);
      const monolithCrown = new THREE.Mesh(monolithCrownGeo, secondaryTrimMat);
      monolithCrown.position.set(0, 0.34, 0.02);
      headGroup.add(monolithCrown);

      const runeCoreGeo = trackGeo(new THREE.OctahedronGeometry(0.048, 1), 0);
      const runeCore = new THREE.Mesh(runeCoreGeo, crystalCoreMat);
      runeCore.position.set(0, 0.31, 0.25);
      headGroup.add(runeCore);
    } else {
      // Open-Visored Battle Barbute / War Crown (horned_greathelm, drake_fang_helm, Vampire/Gorgon crown)
      const openHelmDomeGeo = trackGeo(
        new THREE.SphereGeometry(0.252, seg(18, 14), seg(12, 10), 0, Math.PI * 2, 0, Math.PI * 0.45),
        0.014
      );
      openHelmDomeGeo.scale(1.04, 1.02, 1.14);
      const openHelmDome = new THREE.Mesh(openHelmDomeGeo, secondaryTrimMat);
      openHelmDome.position.set(0, 0.21, 0.02);
      openHelmDome.rotation.x = -0.28; // Tilted backward so brow, rabid eyes, nose & toothed mouth are 100% visible!
      openHelmDome.castShadow = true;
      headGroup.add(openHelmDome);

      // Central Dorsal War-Crest Ridge atop the Open Helm
      const crestGeo = trackGeo(new THREE.BoxGeometry(0.042, 0.095, 0.28), 0.008);
      const crestMesh = new THREE.Mesh(crestGeo, secondaryTrimMat);
      crestMesh.position.set(0, 0.39, 0.02);
      crestMesh.rotation.x = -0.15;
      headGroup.add(crestMesh);

      // Swept Cheek-Guards framing the sides of the jaw without covering the face
      for (const side of [-1, 1]) {
        const cheekGuardCurve = new THREE.CatmullRomCurve3([
          new THREE.Vector3(side * 0.21, 0.24, 0.04),
          new THREE.Vector3(side * 0.205, 0.14, 0.09),
          new THREE.Vector3(side * 0.165, 0.06, 0.15),
        ]);
        const cheekGuardGeo = trackGeo(
          new THREE.TubeGeometry(cheekGuardCurve, seg(10, 8), 0.028, seg(8, 6), false),
          0.008
        );
        cheekGuardGeo.scale(0.65, 1.0, 1.35);
        const cheekGuard = new THREE.Mesh(cheekGuardGeo, secondaryTrimMat);
        headGroup.add(cheekGuard);
      }
    }
  }

  // Curved Horns / Pointed Lupine & Keltir Ears / Goblin Lateral Ears / Branching Stag Antlers
  const skipDefaultHorns =
    (config.archetype === 'ruins_skeleton_pawn' ||
      config.archetype === 'outlaw_bandit_cutthroat' ||
      config.helm === 'valkyrie_crest') &&
    m.hornLength <= 1.3;

  if (m.hornLength > 0.05 && !isBeholder && !skipDefaultHorns) {
    const isLupineEars = config.helm === 'lupine_maw';
    const isGoblinEars =
      config.helm === 'goblin_feral_face' ||
      config.archetype === 'forest_goblin_scavenger';
    const isStagAntlers =
      config.archetype === 'furious_antler_stag' ||
      config.archetype === 'hailstorm_stag_spirit' ||
      config.archetype === 'briar_wood_spriggan' ||
      config.archetype === 'treant_ancient';

    for (const side of [-1, 1]) {
      const hLen = 0.44 * m.hornLength;
      const hCurve = m.hornCurve;

      if (isGoblinEars) {
        // Long Swept-Back Pointed Goblin / Bat-Like Lateral Pinnae + Gold Ear Ring
        const earSpan = 0.36 + hLen * 0.45;
        const earCurve = new THREE.CatmullRomCurve3([
          new THREE.Vector3(side * 0.19, 0.2, 0.04),
          new THREE.Vector3(side * (0.28 + earSpan * 0.48), 0.23, -0.03),
          new THREE.Vector3(side * (0.36 + earSpan * 0.95), 0.27 + hLen * 0.12, -0.12),
        ]);
        const earGeo = trackGeo(
          new THREE.ConeGeometry(0.078, earSpan, seg(14, 10)),
          0.016
        );
        earGeo.scale(0.52, 1.0, 1.38);
        const earMesh = new THREE.Mesh(earGeo, primaryArmorMat);
        const midEar = earCurve.getPoint(0.45);
        earMesh.position.copy(midEar);
        earMesh.rotation.z = -side * 1.18; // Swept sharply outward to the sides
        earMesh.rotation.x = -0.28;        // Swept slightly backward
        earMesh.castShadow = true;
        headGroup.add(earMesh);

        // Inner ear cartilage ridge
        const innerEarGeo = trackGeo(
          new THREE.ConeGeometry(0.046, earSpan * 0.82, seg(10, 8)),
          0.01
        );
        innerEarGeo.scale(0.38, 1.0, 1.15);
        const innerEar = new THREE.Mesh(innerEarGeo, underbodyMat);
        innerEar.position.set(side * (0.26 + earSpan * 0.36), 0.215, -0.005);
        innerEar.rotation.z = -side * 1.18;
        innerEar.rotation.x = -0.28;
        headGroup.add(innerEar);

        // Goblin Scavenger Pierced Ear Ring
        const earringGeo = trackGeo(new THREE.TorusGeometry(0.036, 0.009, 6, 12), 0.002);
        const earringMesh = new THREE.Mesh(earringGeo, secondaryTrimMat);
        earringMesh.position.set(side * (0.27 + earSpan * 0.35), 0.165, -0.02);
        earringMesh.rotation.y = side * 0.4;
        headGroup.add(earringMesh);
      } else if (isLupineEars) {
        const earGeo = trackGeo(
          new THREE.ConeGeometry(0.082, 0.26 + hLen * 0.4, seg(12, 8))
        );
        earGeo.scale(0.65, 1.0, 1.25);
        const earMesh = new THREE.Mesh(earGeo, primaryArmorMat);
        earMesh.position.set(side * 0.15, 0.31 + hLen * 0.08, -0.02);
        earMesh.rotation.z = -side * 0.32;
        earMesh.rotation.x = -0.22;
        headGroup.add(earMesh);
      } else {
        const isSweepBack =
          config.helm === 'draconic_snout' || config.helm === 'avian_phoenix_beak';

        const socketRingGeo = trackGeo(
          new THREE.SphereGeometry(0.068, seg(10, 8), seg(8, 6))
        );
        const socketRing = new THREE.Mesh(socketRingGeo, secondaryTrimMat);
        socketRing.position.set(side * 0.14, 0.26, 0.02);
        headGroup.add(socketRing);

        const hornPath = new THREE.CatmullRomCurve3([
          new THREE.Vector3(side * 0.12, 0.24, 0.02),
          new THREE.Vector3(
            side * (0.32 + hLen * 0.45),
            0.35 + hLen * 0.35,
            isSweepBack ? -0.25 * hCurve : 0.08 * hCurve
          ),
          new THREE.Vector3(
            side * (0.25 + hLen * 0.55),
            0.5 + hLen * 0.85,
            isSweepBack ? -0.56 * hCurve : 0.32 * hCurve
          ),
        ]);
        const hornGeo = trackGeo(
          new THREE.TubeGeometry(hornPath, seg(14, 10), 0.048, seg(10, 8), false)
        );
        const hornMat =
          isDragon || config.raceCategory === 'mythological' || isStagAntlers
            ? ivoryFangMat
            : secondaryTrimMat;
        const hornMesh = new THREE.Mesh(hornGeo, hornMat);
        headGroup.add(hornMesh);

        if (isStagAntlers) {
          for (let t = 1; t <= 2; t++) {
            const branchGeo = trackGeo(new THREE.ConeGeometry(0.034, 0.26, seg(10, 8)));
            const branchMesh = new THREE.Mesh(branchGeo, ivoryFangMat);
            branchMesh.position.set(
              side * (0.26 + t * 0.11),
              0.46 + t * 0.16,
              0.06 + t * 0.07
            );
            branchMesh.rotation.z = -side * 0.42;
            branchMesh.rotation.x = 0.42;
            headGroup.add(branchMesh);
          }
        }
      }
    }
  }

  // Clone side necks & heads for Multi-Headed Bosses (Hydra, Cerberus, or user slider!)
  const extraHeads: THREE.Bone[] = [];
  const headCount = Math.max(1, Math.min(3, Math.round(m.headCount || 1)));
  if (headCount > 1 && !isBeholder) {
    const sides = headCount === 2 ? [-1] : [-1, 1];
    for (const side of sides) {
      const sideNeck = neckGroup.clone(true) as THREE.Bone;
      sideNeck.name = side === 1 ? 'Neck_Head_Left' : 'Neck_Head_Right';
      sideNeck.traverse((c) => {
        if ((c as THREE.Bone).isBone && c !== sideNeck) {
          c.name = `${c.name}_${side === 1 ? 'L' : 'R'}`;
        }
      });
      sideNeck.position.x = side * 0.22 * m.torsoBulk;
      sideNeck.rotation.z = -side * 0.28;
      sideNeck.rotation.y = side * 0.22;
      sideNeck.scale.setScalar(0.88);
      torsoGroup.add(sideNeck);
      extraHeads.push(sideNeck);
    }
  }

  // Helper: Builds an organic, multi-segmented billowing flame/frost plume cylinder
  // with vertex-color blackbody gradient & feathered alpha tip (zero hard cone edges!)
  const createVolumetricBreathPlumeGeo = (
    throatRadius: number,
    tipRadius: number,
    length: number,
    radialSegs: number,
    heightSegs: number,
    isFrostPlume: boolean,
    isInnerCore: boolean,
    wavePhaseOffset = 0
  ): THREE.BufferGeometry => {
    const g = new THREE.CylinderGeometry(
      tipRadius,
      throatRadius,
      length,
      radialSegs,
      heightSegs,
      true
    );
    // Orient cylinder along +Z (0 at mouth throat -> +length at tip)
    g.rotateX(Math.PI / 2);
    g.translate(0, 0, length * 0.5);

    const pos = g.getAttribute('position');
    const col = new Float32Array(pos.count * 3);
    const cTmp = new THREE.Color();

    for (let i = 0; i < pos.count; i++) {
      const x = pos.getX(i);
      const y = pos.getY(i);
      const z = pos.getZ(i);
      const zFrac = Math.min(1, Math.max(0, z / Math.max(0.001, length)));
      const ang = Math.atan2(y, x);

      // Organic billowing flame tongue expansion along the plume
      const billow =
        1.0 +
        Math.sin(zFrac * 10.5 + ang * 3.0 + wavePhaseOffset) * 0.14 * zFrac +
        Math.cos(zFrac * 16.0 - ang * 2.0 + wavePhaseOffset * 1.7) * 0.08 * zFrac;
      // Slight thermal rise (+Y) toward the end of the fire stream
      const thermalLift = isFrostPlume ? -0.05 * zFrac * zFrac : 0.09 * zFrac * zFrac;
      pos.setXYZ(i, x * billow, y * billow + thermalLift, z);

      // Smooth fade at very base (0..0.06) and feathered dissipation at tip (0.68..1.0)
      const rootFade = Math.min(1, zFrac / 0.06);
      const tipFade = Math.pow(Math.max(0, 1.0 - Math.pow(zFrac, 1.45)), 1.35);
      const intensity = rootFade * tipFade;

      if (isFrostPlume) {
        if (isInnerCore || zFrac < 0.25) {
          cTmp.setRGB(0.92, 0.98, 1.0);
        } else if (zFrac < 0.65) {
          cTmp.setRGB(0.38, 0.82, 1.0);
        } else {
          cTmp.setRGB(0.08, 0.45, 0.88);
        }
      } else {
        // Real Blackbody Combustion Spectrum: White-Yellow Plasma -> Gold-Orange -> Crimson -> Dark Fade
        if (isInnerCore) {
          if (zFrac < 0.35) {
            cTmp.setRGB(1.0, 0.98, 0.86);
          } else {
            cTmp.setRGB(1.0, 0.68, 0.16);
          }
        } else if (zFrac < 0.22) {
          cTmp.setRGB(1.0, 0.95, 0.68);
        } else if (zFrac < 0.56) {
          const u = (zFrac - 0.22) / 0.34;
          cTmp.setRGB(1.0, 0.82 - u * 0.44, 0.22 - u * 0.19);
        } else {
          const u = (zFrac - 0.56) / 0.44;
          cTmp.setRGB(0.95 - u * 0.55, 0.28 * (1 - u), 0.02 * (1 - u));
        }
      }

      col[i * 3] = cTmp.r * intensity;
      col[i * 3 + 1] = cTmp.g * intensity;
      col[i * 3 + 2] = cTmp.b * intensity;
    }

    g.setAttribute('color', new THREE.BufferAttribute(col, 3));
    g.computeVertexNormals();
    geometriesToDispose.push(g);
    return g;
  };

  // Mouth & Head Attack Emitter (Volumetric Fire Stream, Glacial Frost Plume, Fang Bite Snap, or Ram Shockwave)
  // Kept visible with 0.0001 rest scale so GLTFExporter embeds it and its children in .GLB!
  const buildMouthOrChargeEffect = (): THREE.Mesh => {
    if (effectiveWeapon === 'frost_breath_maw') {
      // SKILL DE HIELO POR LA BOCA: Multi-layer Volumetric Glacial Blizzard Plume + Crystalline Ice Lances
      const frostOuterMat = trackMat(
        new THREE.MeshBasicMaterial({
          name: 'Mat_MouthFrostPlumeOuter',
          map: textures.frostBreathMap,
          vertexColors: true,
          transparent: true,
          opacity: 0.92,
          blending: THREE.AdditiveBlending,
          side: THREE.DoubleSide,
          depthWrite: false,
          wireframe,
        })
      );
      const frostCoreMat = trackMat(
        new THREE.MeshBasicMaterial({
          name: 'Mat_MouthFrostPlumeCore',
          vertexColors: true,
          transparent: true,
          opacity: 0.95,
          blending: THREE.AdditiveBlending,
          side: THREE.DoubleSide,
          depthWrite: false,
          wireframe,
        })
      );
      const iceCrystalMat = trackMat(
        new THREE.MeshPhysicalMaterial({
          name: 'Mat_MouthIceCrystals',
          color: new THREE.Color('#e0f2fe'),
          emissive: new THREE.Color('#38bdf8'),
          emissiveIntensity: 2.2,
          roughness: 0.08,
          metalness: 0.1,
          clearcoat: 1.0,
          transparent: true,
          opacity: 0.92,
          wireframe,
        })
      );

      const outerPlumeGeo = createVolumetricBreathPlumeGeo(
        0.07,
        0.54,
        2.65,
        seg(18, 12),
        14,
        true,
        false,
        0
      );
      const coneRoot = new THREE.Mesh(outerPlumeGeo, frostOuterMat);

      const midPlumeGeo = createVolumetricBreathPlumeGeo(
        0.05,
        0.34,
        2.35,
        seg(14, 10),
        12,
        true,
        false,
        2.1
      );
      const midPlumeMesh = new THREE.Mesh(midPlumeGeo, frostOuterMat);
      midPlumeMesh.name = 'Breath_SubLayer_Mid';
      coneRoot.add(midPlumeMesh);

      const corePlumeGeo = createVolumetricBreathPlumeGeo(
        0.035,
        0.16,
        1.85,
        seg(12, 8),
        10,
        true,
        true,
        4.2
      );
      const corePlumeMesh = new THREE.Mesh(corePlumeGeo, frostCoreMat);
      corePlumeMesh.name = 'Breath_SubLayer_Core';
      coneRoot.add(corePlumeMesh);

      // 5 Razor Crystalline Ice Shards suspended inside the blizzard jet
      for (let i = 0; i < 5; i++) {
        const ang = (i / 5) * Math.PI * 2;
        const distZ = 0.45 + i * 0.36;
        const rad = 0.06 + i * 0.035;
        const shardGeo = trackGeo(new THREE.ConeGeometry(0.048, 0.48, 5), 0);
        shardGeo.rotateX(Math.PI / 2);
        const shardMesh = new THREE.Mesh(shardGeo, iceCrystalMat);
        shardMesh.position.set(Math.cos(ang) * rad, Math.sin(ang) * rad, distZ);
        shardMesh.rotation.z = ang;
        coneRoot.add(shardMesh);
      }

      return coneRoot;
    } else if (effectiveWeapon === 'dragon_breath_maw') {
      // FUEGO REAL POR LA BOCA: 3-Layer Volumetric Billowing Flame Jet + Crossed Axial Fire Tongues
      const fireOuterMat = trackMat(
        new THREE.MeshBasicMaterial({
          name: 'Mat_MouthVolumetricFireOuter',
          map: textures.fireBreathMap,
          vertexColors: true,
          transparent: true,
          opacity: 0.96,
          blending: THREE.AdditiveBlending,
          side: THREE.DoubleSide,
          depthWrite: false,
          wireframe,
        })
      );
      const fireCoreMat = trackMat(
        new THREE.MeshBasicMaterial({
          name: 'Mat_MouthVolumetricFireCore',
          map: textures.fireBreathMap,
          vertexColors: true,
          transparent: true,
          opacity: 0.98,
          blending: THREE.AdditiveBlending,
          side: THREE.DoubleSide,
          depthWrite: false,
          wireframe,
        })
      );

      // 1) Outer Billowing Crimson-Orange Flame Plume
      const outerFireGeo = createVolumetricBreathPlumeGeo(
        0.075,
        0.58,
        2.75,
        seg(20, 14),
        16,
        false,
        false,
        0
      );
      const coneRoot = new THREE.Mesh(outerFireGeo, fireOuterMat);

      // 2) Middle Turbulent Gold-Orange Flame Tongue Shell
      const midFireGeo = createVolumetricBreathPlumeGeo(
        0.055,
        0.38,
        2.4,
        seg(16, 12),
        14,
        false,
        false,
        2.35
      );
      const midFireMesh = new THREE.Mesh(midFireGeo, fireOuterMat);
      midFireMesh.name = 'Breath_SubLayer_Mid';
      coneRoot.add(midFireMesh);

      // 3) Inner White-Hot Incandescent Plasma Core Jet emerging directly from the throat
      const innerFireGeo = createVolumetricBreathPlumeGeo(
        0.038,
        0.19,
        1.85,
        seg(12, 10),
        12,
        false,
        true,
        4.7
      );
      const innerFireMesh = new THREE.Mesh(innerFireGeo, fireCoreMat);
      innerFireMesh.name = 'Breath_SubLayer_Core';
      coneRoot.add(innerFireMesh);

      // 4) Crossed Axial Flame Ribbons (X-Planes) so the interior of the flame jet is dense from all angles
      for (let p = 0; p < 2; p++) {
        const ribbonGeo = new THREE.PlaneGeometry(0.56, 2.45, 4, 12);
        ribbonGeo.rotateX(Math.PI / 2);
        ribbonGeo.rotateZ(p === 0 ? Math.PI * 0.25 : -Math.PI * 0.25);
        ribbonGeo.translate(0, 0, 1.22);
        const rPos = ribbonGeo.getAttribute('position');
        const rCol = new Float32Array(rPos.count * 3);
        for (let i = 0; i < rPos.count; i++) {
          const z = rPos.getZ(i);
          const zFrac = Math.min(1, Math.max(0, z / 2.45));
          const x = rPos.getX(i);
          const y = rPos.getY(i);
          // Taper width near the mouth and flare outward with organic wave
          const taper = 0.18 + zFrac * 0.88;
          const wave = Math.sin(zFrac * 11.0 + p * 2.1) * 0.06 * zFrac;
          rPos.setXYZ(i, x * taper + wave, y * taper + 0.08 * zFrac * zFrac, z);
          const fade =
            Math.min(1, zFrac / 0.08) *
            Math.pow(Math.max(0, 1 - Math.pow(zFrac, 1.4)), 1.5);
          rCol[i * 3] = 1.0 * fade;
          rCol[i * 3 + 1] = Math.max(0.15, 0.88 - zFrac * 0.65) * fade;
          rCol[i * 3 + 2] = Math.max(0.0, 0.42 - zFrac * 0.4) * fade;
        }
        ribbonGeo.setAttribute('color', new THREE.BufferAttribute(rCol, 3));
        geometriesToDispose.push(ribbonGeo);
        const ribbonMesh = new THREE.Mesh(ribbonGeo, fireCoreMat);
        ribbonMesh.name = `Breath_Flame_Ribbon_${p}`;
        coneRoot.add(ribbonMesh);
      }

      return coneRoot;
    } else if (effectiveWeapon === 'ram_charge') {
      // EMBESTIDA BRUTAL: Kinetic Horn/Head Charge Shockwave Cone & Impact Rings
      const ramShockMat = trackMat(
        new THREE.MeshStandardMaterial({
          name: 'Mat_RamChargeShockwave',
          color: new THREE.Color('#fef08a'),
          emissive: new THREE.Color('#f59e0b'),
          emissiveIntensity: 2.6,
          roughness: 0.15,
          transparent: true,
          opacity: 0.85,
          side: THREE.DoubleSide,
          depthWrite: false,
          wireframe,
        })
      );

      // Sonic-boom cone with apex at the snout/horns flaring backward over the charging head
      const shockConeGeo = trackGeo(new THREE.ConeGeometry(0.56, 1.05, seg(16, 10), 1, true), 0);
      shockConeGeo.rotateX(Math.PI / 2);
      shockConeGeo.translate(0, 0, 0.18);
      const coneRoot = new THREE.Mesh(shockConeGeo, ramShockMat);

      for (let r = 0; r < 2; r++) {
        const ringGeo = trackGeo(
          new THREE.TorusGeometry(0.32 + r * 0.18, 0.026, 6, seg(16, 12)),
          0
        );
        const ringMesh = new THREE.Mesh(ringGeo, ramShockMat);
        ringMesh.position.set(0, 0, 0.32 - r * 0.24);
        coneRoot.add(ringMesh);
      }

      return coneRoot;
    } else {
      // MORDISCO SALVAJE / SALTO CON MORDIDA (Feral Bite Jaws Snap Effect for animals & predators)
      const biteShockMat = trackMat(
        new THREE.MeshStandardMaterial({
          name: 'Mat_FeralBiteSnap',
          color: new THREE.Color('#fca5a5'),
          emissive: new THREE.Color('#ef4444'),
          emissiveIntensity: 2.8,
          roughness: 0.15,
          transparent: true,
          opacity: 0.9,
          side: THREE.DoubleSide,
          depthWrite: false,
          wireframe,
        })
      );

      const ringGeo = trackGeo(
        new THREE.TorusGeometry(0.3, 0.034, 6, seg(14, 10), Math.PI * 0.88),
        0
      );
      const upperJawArc = new THREE.Mesh(ringGeo, biteShockMat);
      upperJawArc.rotation.z = Math.PI * 0.06;

      const lowerJawArc = new THREE.Mesh(ringGeo, biteShockMat);
      lowerJawArc.rotation.z = Math.PI * 1.06;
      upperJawArc.add(lowerJawArc);

      // 6 Energy Fang Spikes snapping inward at the bite point
      for (const side of [-1, 0, 1]) {
        const fangSpikeGeo = trackGeo(
          new THREE.ConeGeometry(side === 0 ? 0.028 : 0.04, side === 0 ? 0.14 : 0.21, 4),
          0
        );
        const upperSpike = new THREE.Mesh(fangSpikeGeo, biteShockMat);
        upperSpike.position.set(side * 0.11, 0.22, 0.08);
        upperSpike.rotation.x = Math.PI;
        upperJawArc.add(upperSpike);

        const lowerSpike = new THREE.Mesh(fangSpikeGeo, biteShockMat);
        lowerSpike.position.set(side * 0.1, -0.22, 0.08);
        upperJawArc.add(lowerSpike);
      }

      return upperJawArc;
    }
  };

  const breathConeMesh = buildMouthOrChargeEffect();
  breathConeMesh.name = 'Elemental_Breath_Cone';
  breathConeMesh.position.set(0, 0.12, 0.24 + snoutLen);
  breathConeMesh.scale.setScalar(0.0001);
  breathConeMesh.visible = true;
  headGroup.add(breathConeMesh);

  // Dynamic Flickering Mouth Fire / Frost Light (casts real illumination on teeth, snout, scales & floor during breath)
  const breathLight = new THREE.PointLight(
    effectiveWeapon === 'frost_breath_maw' ? 0x38bdf8 : 0xff6a00,
    0,
    6.5,
    1.6
  );
  breathLight.name = 'Mouth_Breath_Dynamic_Light';
  breathLight.position.set(0, 0.14, 0.36 + snoutLen);
  headGroup.add(breathLight);

  tagExplodablePart(headGroup, new THREE.Vector3(0, 0.85, 0.35), {
    name:
      headCount > 1
        ? `Cabezas Múltiples (${headCount}x Hidra/Cerbero)`
        : 'Cráneo Orgánico, Hocico y Fauces',
    category: 'Primary Silhouette',
    vertexCount: 540 * headCount,
    triangleCount: 480 * headCount,
    materialSummary: 'Cráneo continuo + hocico orientado hacia adelante + fauces articuladas',
    socketName: 'socket_head',
  });
  explodableParts.push(headGroup);

  // 6. PAULDRONS / ORGANIC SCAPULAR MUSCULATURE & SHOULDER FUR TUFTS
  const buildShoulder = (isLeft: boolean): THREE.Bone => {
    const side = isLeft ? 1 : -1;
    const shoulderGroup = makeBone(isLeft ? 'Pauldron_L' : 'Pauldron_R');
    if (isQuadruped && !isCentaur) {
      shoulderGroup.position.set(
        side * 0.29 * m.torsoBulk * m.shoulderWidth,
        0.18 * m.heightScale,
        0.38
      );
    } else {
      shoulderGroup.position.set(
        side * 0.44 * m.torsoBulk * m.shoulderWidth,
        torsoHeight * 0.78,
        0
      );
    }
    torsoGroup.add(shoulderGroup);

    if (!isBeholder) {
      if (!isUndeadSkeleton && (isOrganicCreature || m.pauldronSize <= 0.45)) {
        // Organic Feral Scapular Muscle & Fur Hump bridging ribcage and foreleg
        const furShoulderGeo = trackGeo(
          new THREE.SphereGeometry(0.24 * m.torsoBulk, seg(16, 12), seg(12, 10)),
          0.024
        );
        furShoulderGeo.scale(0.94, 1.18, 1.28);
        const furShoulderMesh = new THREE.Mesh(furShoulderGeo, primaryArmorMat);
        furShoulderMesh.castShadow = true;
        shoulderGroup.add(furShoulderMesh);

        // 3D Scapular Shoulder Hair Locks
        if (furDensity > 0.15 && hairStyle !== 'none') {
          const fScale = 0.75 + furDensity * 0.38;
          for (let f = 0; f < 3; f++) {
            const shTuftGeo = trackFurLock(0.072 * m.torsoBulk * fScale, (0.26 - f * 0.03) * fScale, 0.09);
            const shTuft = new THREE.Mesh(shTuftGeo, hairStrandMat);
            shTuft.position.set(side * 0.1 * m.torsoBulk, 0.08 - f * 0.06, -0.04 * f);
            shTuft.rotation.z = -side * (0.75 + f * 0.14);
            shTuft.rotation.x = -0.52;
            shoulderGroup.add(shTuft);
          }
        }
      } else {
        const pShape = new THREE.Shape();
        pShape.moveTo(-side * 0.06, 0.22 * m.pauldronSize);
        pShape.lineTo(side * 0.36 * m.pauldronSize, 0.08 * m.pauldronSize);
        pShape.lineTo(side * 0.42 * m.pauldronSize, -0.2 * m.pauldronSize);
        pShape.lineTo(-side * 0.04, -0.14 * m.pauldronSize);
        pShape.closePath();

        const pauldronGeo = trackGeo(
          new THREE.ExtrudeGeometry(pShape, {
            depth: 0.34 * m.pauldronSize,
            bevelEnabled: true,
            bevelThickness: 0.025,
            bevelSize: 0.02,
            bevelSegments: 2,
          })
        );
        pauldronGeo.center();
        const pauldronMesh = new THREE.Mesh(pauldronGeo, secondaryTrimMat);
        pauldronMesh.castShadow = true;
        shoulderGroup.add(pauldronMesh);
      }
    }

    tagExplodablePart(shoulderGroup, new THREE.Vector3(side * 0.75, 0.45, 0), {
      name: isLeft ? 'Omóplato / Hombrera Izquierda' : 'Omóplato / Hombrera Derecha',
      category: 'Secondary Read',
      vertexCount: 310,
      triangleCount: 270,
      materialSummary: 'Musculatura escapular continua + mechones de pelaje 3D',
    });
    explodableParts.push(shoulderGroup);

    return shoulderGroup;
  };

  const leftShoulderGroup = buildShoulder(true);
  const rightShoulderGroup = buildShoulder(false);

  // Strict Weapon & Offhand Eligibility: ONLY creatures with grasping humanoid hands can hold weapons or shields!
  const hasGraspingHands =
    !isAnimal &&
    !isBeholder &&
    !isArachnid &&
    (!isQuadruped || isCentaur) &&
    config.archetype !== 'leviathan_abyssal' &&
    config.archetype !== 'coral_sea_serpent' &&
    config.archetype !== 'horned_field_viper' &&
    config.archetype !== 'marsh_leech_worm' &&
    config.archetype !== 'putrid_bog_hydra' &&
    config.archetype !== 'cryo_hydra_trident' &&
    config.archetype !== 'abyssal_hydra';

  const isNaturalUnarmed =
    !hasGraspingHands ||
    effectiveWeapon === 'feral_claws_fangs' ||
    effectiveWeapon === 'leap_fang_bite' ||
    effectiveWeapon === 'ram_charge' ||
    effectiveWeapon === 'dragon_breath_maw' ||
    effectiveWeapon === 'frost_breath_maw';

  const hasLeftHandItem =
    hasGraspingHands &&
    (config.offhand === 'tower_shield' ||
      config.offhand === 'runic_buckler' ||
      config.offhand === 'phylactery_orb' ||
      config.offhand === 'dual_blade');

  exportableGroup.userData.hasGraspingHands = hasGraspingHands;
  exportableGroup.userData.isNaturalUnarmed = isNaturalUnarmed;

  // 7. HIERARCHICAL 2-BONE ARMS / FORE-LEGS WITH EXPOSED SKELETAL HUMERUS, RADIUS+ULNA, CARPALS, METACARPALS & PHALANGES
  const buildArm = (
    isLeft: boolean
  ): {
    armGroup: THREE.Bone;
    elbowGroup: THREE.Bone;
    handSocket: THREE.Object3D;
  } => {
    const side = isLeft ? 1 : -1;
    const isGrippingHandle = hasGraspingHands && (isLeft ? hasLeftHandItem : !isNaturalUnarmed);
    const armGroup = makeBone(isLeft ? 'Arm_Or_Foreleg_L' : 'Arm_Or_Foreleg_R');

    if (isQuadruped && !isCentaur) {
      // Positioned directly under the front thoracic brisket & scapular shoulder muscles
      armGroup.position.set(side * 0.27 * m.torsoBulk * m.shoulderWidth, pelvisY, 0.38);
      rootBone.add(armGroup);
    } else {
      armGroup.position.set(
        side * 0.44 * m.torsoBulk * m.shoulderWidth,
        torsoHeight * 0.75,
        0
      );
      torsoGroup.add(armGroup);

      if (isCentaur) {
        const equineHaunch = new THREE.Mesh(
          trackGeo(new THREE.SphereGeometry(0.22 * m.torsoBulk, seg(14, 10), seg(10, 8))),
          isUndeadSkeleton ? ivoryFangMat : primaryArmorMat
        );
        equineHaunch.position.set(side * 0.25 * m.torsoBulk, 0.02, 0.52);
        pelvisGroup.add(equineHaunch);

        const foreLegCurve = new THREE.CatmullRomCurve3([
          new THREE.Vector3(side * 0.25 * m.torsoBulk, 0.04, 0.52),
          new THREE.Vector3(side * 0.26 * m.torsoBulk, -legHeight * 0.48, 0.5),
          new THREE.Vector3(side * 0.26 * m.torsoBulk, -legHeight + 0.04, 0.54),
        ]);
        const foreLeg = new THREE.Mesh(
          trackGeo(
            new THREE.TubeGeometry(
              foreLegCurve,
              seg(14, 10),
              (isUndeadSkeleton ? 0.068 : 0.125) * m.torsoBulk,
              seg(12, 8),
              false
            )
          ),
          isUndeadSkeleton ? ivoryFangMat : primaryArmorMat
        );
        pelvisGroup.add(foreLeg);
      }
    }

    const armLen = isQuadruped && !isCentaur ? legHeight : 0.78 * m.limbLength;
    const elbowY = -armLen * 0.48;
    const elbowZ = isQuadruped && !isCentaur ? -0.05 : -0.02;

    const elbowGroup = makeBone(isLeft ? 'Elbow_Joint_L' : 'Elbow_Joint_R');
    elbowGroup.position.set(0, elbowY, elbowZ);
    armGroup.add(elbowGroup);

    const forearmLen = armLen * 0.52;
    const handY = -forearmLen + 0.05;
    const handZ = isQuadruped && !isCentaur ? 0.05 : 0.08;

    if (isUndeadSkeleton) {
      // =========================================================================================
      // BLENDER-SCULPTED 3D SKELETAL ARM & HAND ANATOMY:
      // 1) Humeral Head + Greater Tubercle + Humerus Diaphysis Shaft + Distal Epicondyles
      // 2) Olecranon Elbow Spur + Dual Forearm Bones (Radius + Ulna with visible interosseous gap!)
      // 3) 8 Carpal Wrist Bones + 5 Individual Metacarpal Palm Bones + 14 Multi-Jointed Finger Phalanges
      //    curled tightly around the weapon/shield handle (or spread as 3D skeletal claws when unarmed!)
      // =========================================================================================
      const humeralHeadGeo = trackGeo(
        new THREE.SphereGeometry(0.068 * m.torsoBulk, seg(12, 8), seg(10, 8)),
        0.008
      );
      const humeralHeadMesh = new THREE.Mesh(humeralHeadGeo, ivoryFangMat);
      humeralHeadMesh.position.set(-side * 0.04 * m.torsoBulk, 0.04, 0);
      armGroup.add(humeralHeadMesh);

      const tubercleGeo = trackGeo(
        new THREE.SphereGeometry(0.054 * m.torsoBulk, seg(10, 8), seg(8, 6)),
        0.008
      );
      const tubercleMesh = new THREE.Mesh(tubercleGeo, ivoryFangMat);
      tubercleMesh.position.set(side * 0.032 * m.torsoBulk, 0.025, 0.015);
      armGroup.add(tubercleMesh);

      // Slender Anatomical Humerus Bone Shaft
      const humerusCurve = new THREE.CatmullRomCurve3([
        new THREE.Vector3(-side * 0.025 * m.torsoBulk, 0.04, 0),
        new THREE.Vector3(side * 0.015 * m.torsoBulk, 0.0, 0.005),
        new THREE.Vector3(side * 0.008 * m.torsoBulk, elbowY * 0.52, elbowZ * 0.5),
        new THREE.Vector3(0, elbowY, elbowZ),
      ]);
      const humerusGeo = trackGeo(
        new THREE.TubeGeometry(
          humerusCurve,
          seg(14, 10),
          0.045 * m.torsoBulk,
          seg(10, 8),
          false
        ),
        0.008
      );
      const humerusMesh = new THREE.Mesh(humerusGeo, ivoryFangMat);
      humerusMesh.castShadow = true;
      armGroup.add(humerusMesh);

      // Distal Humeral Medial & Lateral Epicondyles + Trochlea Spool at Elbow
      const epicondyleGeo = trackGeo(
        new THREE.CapsuleGeometry(0.046 * m.torsoBulk, 0.072 * m.torsoBulk, 6, 10),
        0.006
      );
      epicondyleGeo.rotateZ(Math.PI / 2);
      const epicondyleMesh = new THREE.Mesh(epicondyleGeo, ivoryFangMat);
      epicondyleMesh.position.set(0, 0.015, 0);
      elbowGroup.add(epicondyleMesh);

      // Posterior Olecranon Process (Funny-Bone Elbow Spur of the Ulna)
      const olecranonGeo = trackGeo(
        new THREE.SphereGeometry(0.046 * m.torsoBulk, seg(10, 8), seg(8, 6)),
        0.008
      );
      olecranonGeo.scale(0.85, 1.25, 1.15);
      const olecranonMesh = new THREE.Mesh(olecranonGeo, ivoryFangMat);
      olecranonMesh.position.set(-side * 0.018 * m.torsoBulk, 0.01, -0.042 * m.torsoBulk);
      elbowGroup.add(olecranonMesh);

      // Dual Forearm Bones: Lateral Radius + Medial Ulna with a Real See-Through Interosseous Gap!
      const radiusCurve = new THREE.CatmullRomCurve3([
        new THREE.Vector3(side * 0.032 * m.torsoBulk, -0.01, 0.012),
        new THREE.Vector3(side * 0.038 * m.torsoBulk, -forearmLen * 0.48, handZ * 0.48 + 0.016),
        new THREE.Vector3(side * 0.028 * m.torsoBulk, handY + 0.02, handZ + 0.012),
      ]);
      const radiusGeo = trackGeo(
        new THREE.TubeGeometry(
          radiusCurve,
          seg(12, 8),
          0.031 * m.torsoBulk,
          seg(8, 6),
          false
        ),
        0.006
      );
      const radiusMesh = new THREE.Mesh(radiusGeo, ivoryFangMat);
      radiusMesh.castShadow = true;
      elbowGroup.add(radiusMesh);

      const ulnaCurve = new THREE.CatmullRomCurve3([
        new THREE.Vector3(-side * 0.028 * m.torsoBulk, 0.005, -0.022),
        new THREE.Vector3(-side * 0.034 * m.torsoBulk, -forearmLen * 0.48, handZ * 0.45 - 0.014),
        new THREE.Vector3(-side * 0.026 * m.torsoBulk, handY + 0.02, handZ - 0.008),
      ]);
      const ulnaGeo = trackGeo(
        new THREE.TubeGeometry(
          ulnaCurve,
          seg(12, 8),
          0.028 * m.torsoBulk,
          seg(8, 6),
          false
        ),
        0.006
      );
      const ulnaMesh = new THREE.Mesh(ulnaGeo, ivoryFangMat);
      ulnaMesh.castShadow = true;
      elbowGroup.add(ulnaMesh);

      // Carpal Wrist Bones (Proximal & Distal Carpal Rows: Scaphoid, Lunate, Capitate, Hamate)
      for (let cr = 0; cr < 2; cr++) {
        const carpalRowGeo = trackGeo(
          new THREE.CapsuleGeometry(0.024 * m.torsoBulk, 0.064 * m.torsoBulk, 6, 8),
          0.005
        );
        carpalRowGeo.rotateZ(Math.PI / 2);
        const carpalRowMesh = new THREE.Mesh(carpalRowGeo, ivoryFangMat);
        carpalRowMesh.position.set(
          0,
          handY + 0.018 - cr * 0.022 * m.torsoBulk,
          handZ + (0.006 + cr * 0.016) * m.torsoBulk
        );
        elbowGroup.add(carpalRowMesh);
      }

      // 4 Distinct Metacarpal Hand Bones (Index, Middle, Ring, Little) + 3-Segment Finger Phalanges
      const fingerOffsets = [-0.042, -0.014, 0.014, 0.042];
      for (let f = 0; f < 4; f++) {
        const fx = fingerOffsets[f] * m.torsoBulk;
        const isMiddle = f === 1 || f === 2;
        const lenMult = isMiddle ? 1.08 : 0.94;

        const wristPt = new THREE.Vector3(fx * 0.72, handY + 0.002, handZ + 0.018);
        const knucklePt = new THREE.Vector3(
          fx,
          handY - 0.024 * m.torsoBulk,
          handZ + (0.068 + (isMiddle ? 0.006 : 0)) * m.torsoBulk
        );

        // Individual Metacarpal Bone Shaft across the back of the skeletal hand
        const metaCurve = new THREE.CatmullRomCurve3([wristPt, knucklePt]);
        const metaGeo = trackGeo(
          new THREE.TubeGeometry(metaCurve, 6, 0.013 * m.torsoBulk, 6, false),
          0.002
        );
        const metaMesh = new THREE.Mesh(metaGeo, ivoryFangMat);
        metaMesh.castShadow = true;
        elbowGroup.add(metaMesh);

        // Metacarpophalangeal (Knuckle) Condyle Bead
        const knuckleGeo = trackGeo(
          new THREE.SphereGeometry(0.016 * m.torsoBulk, 7, 6),
          0.002
        );
        const knuckleMesh = new THREE.Mesh(knuckleGeo, ivoryFangMat);
        knuckleMesh.position.copy(knucklePt);
        elbowGroup.add(knuckleMesh);

        if (isGrippingHandle) {
          // BLENDER-STYLE CLOSED SKELETAL FIST GRIPPING THE WEAPON / SHIELD SHAFT AT handSocket (0, handY - 0.035, handZ + 0.095)
          const pipPt = new THREE.Vector3(
            fx,
            handY - 0.018 * m.torsoBulk,
            handZ + (0.128 * lenMult) * m.torsoBulk
          );
          const dipPt = new THREE.Vector3(
            fx,
            handY - 0.064 * m.torsoBulk,
            handZ + (0.124 * lenMult) * m.torsoBulk
          );
          const tipPt = new THREE.Vector3(
            fx * 0.92,
            handY - 0.068 * m.torsoBulk,
            handZ + 0.072 * m.torsoBulk
          );

          const phalanxCurve = new THREE.CatmullRomCurve3([knucklePt, pipPt, dipPt, tipPt]);
          const phalanxGeo = trackGeo(
            new THREE.TubeGeometry(phalanxCurve, 10, 0.012 * m.torsoBulk, 6, false),
            0.002
          );
          const phalanxMesh = new THREE.Mesh(phalanxGeo, ivoryFangMat);
          phalanxMesh.castShadow = true;
          elbowGroup.add(phalanxMesh);

          const pipKnob = new THREE.Mesh(knuckleGeo, ivoryFangMat);
          pipKnob.position.copy(pipPt);
          elbowGroup.add(pipKnob);
        } else {
          // OPEN PREDATORY SKELETAL CLAW HAND: 3 Distinct Articulated Phalanges (Proximal, Middle, Distal Claw)
          const splayX = fx * 1.32;
          const pipPt = new THREE.Vector3(
            splayX,
            handY - 0.055 * lenMult * m.torsoBulk,
            handZ + (0.115 * lenMult) * m.torsoBulk
          );
          const dipPt = new THREE.Vector3(
            splayX * 1.08,
            handY - 0.095 * lenMult * m.torsoBulk,
            handZ + (0.145 * lenMult) * m.torsoBulk
          );
          const tipPt = new THREE.Vector3(
            splayX * 1.04,
            handY - 0.132 * lenMult * m.torsoBulk,
            handZ + (0.132 * lenMult) * m.torsoBulk
          );

          const phalanxCurve = new THREE.CatmullRomCurve3([knucklePt, pipPt, dipPt, tipPt]);
          const phalanxGeo = trackGeo(
            new THREE.TubeGeometry(phalanxCurve, 10, 0.012 * m.torsoBulk, 6, false),
            0.002
          );
          const phalanxMesh = new THREE.Mesh(phalanxGeo, ivoryFangMat);
          phalanxMesh.castShadow = true;
          elbowGroup.add(phalanxMesh);

          const pipKnob = new THREE.Mesh(knuckleGeo, ivoryFangMat);
          pipKnob.position.copy(pipPt);
          elbowGroup.add(pipKnob);
        }
      }

      // 5th Ray: Opposing Skeletal Thumb (1st Metacarpal + Proximal & Distal Thumb Phalanges)
      const thumbSign = side; // Medial/inner side of the hand
      const thumbBase = new THREE.Vector3(
        thumbSign * 0.042 * m.torsoBulk,
        handY - 0.002,
        handZ + 0.025 * m.torsoBulk
      );
      const thumbKnuckle = new THREE.Vector3(
        thumbSign * 0.062 * m.torsoBulk,
        handY - 0.028 * m.torsoBulk,
        handZ + 0.068 * m.torsoBulk
      );
      const thumbTip = isGrippingHandle
        ? new THREE.Vector3(
            -thumbSign * 0.012 * m.torsoBulk,
            handY - 0.052 * m.torsoBulk,
            handZ + 0.118 * m.torsoBulk
          )
        : new THREE.Vector3(
            thumbSign * 0.078 * m.torsoBulk,
            handY - 0.075 * m.torsoBulk,
            handZ + 0.112 * m.torsoBulk
          );
      const thumbCurve = new THREE.CatmullRomCurve3([thumbBase, thumbKnuckle, thumbTip]);
      const thumbGeo = trackGeo(
        new THREE.TubeGeometry(thumbCurve, 8, 0.013 * m.torsoBulk, 6, false),
        0.002
      );
      elbowGroup.add(new THREE.Mesh(thumbGeo, ivoryFangMat));
    } else {
      // LIVING HUMANOID / BEAST / CHAMPION ARM & SCULPTED HAND WITH ARTICULATED FINGER GRIP
      const deltoidGeo = trackGeo(
        new THREE.SphereGeometry(0.192 * m.torsoBulk, seg(14, 10), seg(10, 8)),
        0.022
      );
      deltoidGeo.scale(0.98, 1.24, 1.18);
      const deltoidMesh = new THREE.Mesh(deltoidGeo, primaryArmorMat);
      deltoidMesh.position.set(-side * 0.045, 0.035, 0);
      armGroup.add(deltoidMesh);

      const axillaBridgeCurve = new THREE.CatmullRomCurve3([
        new THREE.Vector3(-side * 0.16 * m.torsoBulk, 0.09, isQuadruped && !isCentaur ? 0.04 : 0.02),
        new THREE.Vector3(-side * 0.05 * m.torsoBulk, 0.02, 0.01),
        new THREE.Vector3(0, elbowY * 0.32, elbowZ * 0.3),
      ]);
      const axillaBridgeGeo = trackGeo(
        new THREE.TubeGeometry(
          axillaBridgeCurve,
          seg(12, 8),
          0.128 * m.torsoBulk,
          seg(10, 8),
          false
        ),
        0.016
      );
      const axillaBridgeMesh = new THREE.Mesh(
        axillaBridgeGeo,
        isQuadruped || isOrganicCreature ? primaryArmorMat : underbodyMat
      );
      armGroup.add(axillaBridgeMesh);

      const upperArmCurve = new THREE.CatmullRomCurve3([
        new THREE.Vector3(-side * 0.03, 0.08, 0),
        new THREE.Vector3(side * 0.01, elbowY * 0.5, elbowZ * 0.5),
        new THREE.Vector3(0, elbowY - 0.05, elbowZ),
      ]);
      const upperArmGeo = trackGeo(
        new THREE.TubeGeometry(
          upperArmCurve,
          seg(14, 10),
          0.138 * m.torsoBulk,
          seg(12, 10),
          false
        ),
        0.024
      );
      const upperArmMesh = new THREE.Mesh(
        upperArmGeo,
        isQuadruped ? primaryArmorMat : underbodyMat
      );
      upperArmMesh.castShadow = true;
      armGroup.add(upperArmMesh);

      if (muscleDef > 0.15) {
        const mScale = 0.78 + muscleDef * 0.34;
        const bicepGeo = trackGeo(
          new THREE.SphereGeometry(0.112 * m.torsoBulk * mScale, seg(12, 8), seg(10, 8)),
          0.024
        );
        bicepGeo.scale(0.84, 1.55, 1.08);
        const bicepMesh = new THREE.Mesh(
          bicepGeo,
          isQuadruped || isOrganicCreature ? primaryArmorMat : underbodyMat
        );
        bicepMesh.position.set(0, elbowY * 0.48, 0.055 * m.torsoBulk);
        bicepMesh.rotation.x = 0.12;
        armGroup.add(bicepMesh);

        const tricepGeo = trackGeo(
          new THREE.SphereGeometry(0.118 * m.torsoBulk * mScale, seg(12, 8), seg(10, 8)),
          0.024
        );
        tricepGeo.scale(0.86, 1.62, 1.12);
        const tricepMesh = new THREE.Mesh(
          tricepGeo,
          isQuadruped || isOrganicCreature ? primaryArmorMat : underbodyMat
        );
        tricepMesh.position.set(side * 0.025 * m.torsoBulk, elbowY * 0.38, -0.058 * m.torsoBulk);
        tricepMesh.rotation.x = -0.14;
        armGroup.add(tricepMesh);
      }

      const elbowSphereGeo = trackGeo(
        new THREE.CapsuleGeometry(0.128 * m.torsoBulk, 0.095 * m.torsoBulk, seg(10, 8), seg(14, 10)),
        0.015
      );
      elbowSphereGeo.scale(1.04, 1.12, 1.14);
      const elbowSphere = new THREE.Mesh(
        elbowSphereGeo,
        isQuadruped ? primaryArmorMat : secondaryTrimMat
      );
      elbowGroup.add(elbowSphere);

      const elbowBridgeCurve = new THREE.CatmullRomCurve3([
        new THREE.Vector3(0, 0.09, 0.038 * m.torsoBulk),
        new THREE.Vector3(0, 0.0, 0.058 * m.torsoBulk),
        new THREE.Vector3(0, -0.1, 0.042 * m.torsoBulk),
      ]);
      const elbowBridgeGeo = trackGeo(
        new THREE.TubeGeometry(elbowBridgeCurve, seg(10, 8), 0.062 * m.torsoBulk, seg(10, 8), false),
        0.012
      );
      const elbowBridgeMesh = new THREE.Mesh(
        elbowBridgeGeo,
        isOrganicCreature || isQuadruped ? primaryArmorMat : underbodyMat
      );
      elbowGroup.add(elbowBridgeMesh);

      const forearmCurve = new THREE.CatmullRomCurve3([
        new THREE.Vector3(0, 0.07, 0),
        new THREE.Vector3(0, -forearmLen * 0.48, handZ * 0.45),
        new THREE.Vector3(0, handY - 0.02, handZ),
      ]);
      const forearmGeo = trackGeo(
        new THREE.TubeGeometry(
          forearmCurve,
          seg(14, 10),
          0.12 * m.torsoBulk,
          seg(12, 10),
          false
        ),
        0.022
      );
      const forearmMesh = new THREE.Mesh(forearmGeo, primaryArmorMat);
      forearmMesh.castShadow = true;
      elbowGroup.add(forearmMesh);

      if (muscleDef > 0.15) {
        const mScale = 0.78 + muscleDef * 0.34;
        const brachioGeo = trackGeo(
          new THREE.SphereGeometry(0.098 * m.torsoBulk * mScale, seg(10, 8), seg(8, 6)),
          0.022
        );
        brachioGeo.scale(0.88, 1.58, 1.12);
        const brachioMesh = new THREE.Mesh(brachioGeo, primaryArmorMat);
        brachioMesh.position.set(side * 0.018 * m.torsoBulk, -forearmLen * 0.34, 0.028 * m.torsoBulk);
        elbowGroup.add(brachioMesh);
      }

      if (furDensity > 0.15 && hairStyle !== 'none') {
        const fScale = 0.74 + furDensity * 0.36;
        for (let t = 0; t < 3; t++) {
          const elbowTuftGeo = trackFurLock(0.055 * m.torsoBulk * fScale, (0.22 - t * 0.025) * fScale, 0.08);
          const elbowTuft = new THREE.Mesh(elbowTuftGeo, hairStrandMat);
          elbowTuft.position.set(
            side * 0.02 * m.torsoBulk,
            0.02 - forearmLen * (t * 0.25),
            -0.075 * m.torsoBulk
          );
          elbowTuft.rotation.x = -Math.PI * 0.66;
          elbowTuft.rotation.z = -side * 0.18;
          elbowGroup.add(elbowTuft);
        }
      }

      const wristCuffGeo = trackGeo(
        new THREE.CapsuleGeometry(0.105 * m.torsoBulk, 0.065 * m.torsoBulk, seg(8, 6), seg(12, 8)),
        0.012
      );
      const wristCuffMesh = new THREE.Mesh(wristCuffGeo, primaryArmorMat);
      wristCuffMesh.position.set(0, handY + 0.02, handZ + 0.01);
      wristCuffMesh.rotation.x = 0.35;
      elbowGroup.add(wristCuffMesh);

      const handGeo = trackGeo(
        new THREE.SphereGeometry(0.108 * m.torsoBulk, seg(12, 10), seg(10, 8))
      );
      handGeo.scale(0.95, 0.65, 1.15);
      const handMesh = new THREE.Mesh(handGeo, secondaryTrimMat);
      handMesh.position.set(0, handY - 0.012, handZ + 0.048);
      elbowGroup.add(handMesh);

      if (hasGraspingHands) {
        // 4 Sculpted 3D Gauntleted Fingers + Opposing Thumb wrapping tightly around the weapon/shield handle!
        const fingerOffsets = [-0.042, -0.014, 0.014, 0.042];
        for (let f = 0; f < 4; f++) {
          const fx = fingerOffsets[f] * m.torsoBulk;
          const knucklePt = new THREE.Vector3(fx, handY - 0.02 * m.torsoBulk, handZ + 0.068 * m.torsoBulk);
          const fingerCurve = isGrippingHandle
            ? new THREE.CatmullRomCurve3([
                knucklePt,
                new THREE.Vector3(fx, handY - 0.018 * m.torsoBulk, handZ + 0.132 * m.torsoBulk),
                new THREE.Vector3(fx, handY - 0.066 * m.torsoBulk, handZ + 0.125 * m.torsoBulk),
                new THREE.Vector3(fx * 0.9, handY - 0.068 * m.torsoBulk, handZ + 0.072 * m.torsoBulk),
              ])
            : new THREE.CatmullRomCurve3([
                knucklePt,
                new THREE.Vector3(fx * 1.15, handY - 0.055 * m.torsoBulk, handZ + 0.118 * m.torsoBulk),
                new THREE.Vector3(fx * 1.2, handY - 0.098 * m.torsoBulk, handZ + 0.138 * m.torsoBulk),
              ]);
          const fingerGeo = trackGeo(
            new THREE.TubeGeometry(fingerCurve, 8, 0.016 * m.torsoBulk, 6, false),
            0.003
          );
          elbowGroup.add(new THREE.Mesh(fingerGeo, primaryArmorMat));
        }

        // Opposing Thumb locking the grip across the weapon shaft
        const thumbCurve = new THREE.CatmullRomCurve3([
          new THREE.Vector3(side * 0.046 * m.torsoBulk, handY - 0.005, handZ + 0.032 * m.torsoBulk),
          new THREE.Vector3(side * 0.065 * m.torsoBulk, handY - 0.03 * m.torsoBulk, handZ + 0.072 * m.torsoBulk),
          isGrippingHandle
            ? new THREE.Vector3(-side * 0.01 * m.torsoBulk, handY - 0.052 * m.torsoBulk, handZ + 0.118 * m.torsoBulk)
            : new THREE.Vector3(side * 0.078 * m.torsoBulk, handY - 0.072 * m.torsoBulk, handZ + 0.108 * m.torsoBulk),
        ]);
        const thumbGeo = trackGeo(
          new THREE.TubeGeometry(thumbCurve, 7, 0.017 * m.torsoBulk, 6, false),
          0.003
        );
        elbowGroup.add(new THREE.Mesh(thumbGeo, primaryArmorMat));
      } else if (isQuadruped && !isCentaur) {
        for (let c = -1; c <= 1; c++) {
          const frontToeGeo = trackGeo(
            new THREE.SphereGeometry(0.052 * m.torsoBulk, seg(10, 8), seg(8, 6))
          );
          frontToeGeo.scale(0.8, 0.65, 1.4);
          const frontToe = new THREE.Mesh(frontToeGeo, primaryArmorMat);
          frontToe.position.set(c * 0.06 * m.torsoBulk, handY - 0.02, handZ + 0.16);
          elbowGroup.add(frontToe);
        }
      }
    }

    // Socket positioned at the exact center of the closed 3D palm/finger grip!
    const handSocket = new THREE.Object3D();
    handSocket.name = isLeft ? 'socket_shield_l' : 'socket_weapon_r';
    handSocket.position.set(0, handY - 0.038 * m.torsoBulk, handZ + 0.096 * m.torsoBulk);
    elbowGroup.add(handSocket);

    tagExplodablePart(armGroup, new THREE.Vector3(side * 0.65, 0.0, 0.25), {
      name: isQuadruped
        ? isLeft
          ? 'Extremidad Frontal Izquierda'
          : 'Extremidad Frontal Derecha'
        : isUndeadSkeleton
        ? isLeft
          ? 'Húmero, Radio-Cúbito y Mano Ósea Izquierda'
          : 'Húmero, Radio-Cúbito y Mano Ósea Derecha'
        : isLeft
        ? 'Brazo Articulado Izquierdo'
        : 'Brazo Articulado Derecho',
      category: 'Secondary Read',
      vertexCount: 420,
      triangleCount: 370,
      materialSummary: isUndeadSkeleton
        ? 'Húmero, radio y cúbito separados + carpo, 5 metacarpos y falanges con agarre 3D'
        : 'Bíceps, tríceps y braquiorradial esculpidos + mano con agarre 3D',
      socketName: handSocket.name,
    });
    explodableParts.push(armGroup);

    return { armGroup, elbowGroup, handSocket };
  };

  const {
    armGroup: leftArmGroup,
    elbowGroup: leftElbowGroup,
    handSocket: socketShieldL,
  } = buildArm(true);
  const {
    armGroup: rightArmGroup,
    elbowGroup: rightElbowGroup,
    handSocket: socketWeaponR,
  } = buildArm(false);

  // 8. WEAPON / NATURAL PAWS ON socket_weapon_r
  // ONLY enemies with grasping hands and a handheld weapon render weapons; all beasts/animals use natural claws/bite/breath!
  const weaponGroup = new THREE.Group();
  weaponGroup.name = `Weapon_${effectiveWeapon}`;
  weaponGroup.scale.setScalar(m.weaponScale);
  if (effectiveWeapon === 'bone_recurved_bow') {
    // Compensate for the downward (-Y) bind pose of the arm so when the right arm raises -90° (-Math.PI/2)
    // to aim forward, the bow stands vertically upright (World Y) and arrows shoot horizontally forward (World +Z)!
    weaponGroup.rotation.x = Math.PI / 2;
  }
  socketWeaponR.add(weaponGroup);

  const arrowProjectileMesh = new THREE.Group();
  arrowProjectileMesh.name = 'Arrow_Projectile_Mesh';
  arrowProjectileMesh.position.set(0, 0, 0.36);
  arrowProjectileMesh.scale.setScalar(effectiveWeapon === 'bone_recurved_bow' ? 1.0 : 0.0001);
  weaponGroup.add(arrowProjectileMesh);

  if (isNaturalUnarmed) {
    // Only add natural anatomical paw claws on non-skeleton beasts/animals (skeletons already have 5 sculpted phalanx claws!)
    if (!isUndeadSkeleton && config.helm !== 'tauren_bull_horns' && !isArachnid && !isBeholder) {
      for (let c = -1; c <= 1; c++) {
        const talonGeo = trackGeo(new THREE.ConeGeometry(0.028, 0.16, 5), 0.004);
        talonGeo.rotateX(Math.PI / 2.1);
        const talonR = new THREE.Mesh(talonGeo, ivoryFangMat);
        talonR.position.set(c * 0.055, -0.02, 0.09);
        weaponGroup.add(talonR);

        const talonL = talonR.clone();
        socketShieldL.add(talonL);
      }
    }
  } else if (effectiveWeapon === 'bone_recurved_bow') {
    // =========================================================================================
    // BLENDER-SCULPTED 3D RECURVED BONE & DARK-IRON COMPOSITE BOW + DORSAL QUIVER + 3D ARROWS
    // =========================================================================================
    const bowRiserGeo = trackGeo(
      new THREE.CylinderGeometry(0.034, 0.034, 0.34, seg(10, 8)),
      0.006
    );
    const bowRiserMesh = new THREE.Mesh(bowRiserGeo, secondaryTrimMat);
    weaponGroup.add(bowRiserMesh);

    // Upper & Lower Recurved Bone Limbs with Horned Siyah Tips
    for (const limbDir of [-1, 1]) {
      const limbCurve = new THREE.CatmullRomCurve3([
        new THREE.Vector3(0, limbDir * 0.14, 0.02),
        new THREE.Vector3(0, limbDir * 0.44, 0.18),
        new THREE.Vector3(0, limbDir * 0.68, 0.04),
        new THREE.Vector3(0, limbDir * 0.82, -0.16),
      ]);
      const limbGeo = trackGeo(
        new THREE.TubeGeometry(
          limbCurve,
          seg(14, 10),
          0.032,
          seg(8, 6),
          false
        ),
        0.006
      );
      const limbMesh = new THREE.Mesh(
        limbGeo,
        isUndeadSkeleton ? ivoryFangMat : primaryArmorMat
      );
      weaponGroup.add(limbMesh);

      // Decorative Metallic Limb Collar & Curved Bone Horn Tip
      const collarGeo = trackGeo(
        new THREE.CylinderGeometry(0.042, 0.042, 0.08, 8),
        0.004
      );
      const collarMesh = new THREE.Mesh(collarGeo, secondaryTrimMat);
      collarMesh.position.set(0, limbDir * 0.38, 0.14);
      collarMesh.rotation.x = limbDir * 0.35;
      weaponGroup.add(collarMesh);

      const tipSpikeGeo = trackGeo(
        new THREE.ConeGeometry(0.026, 0.16, 6),
        0.004
      );
      const tipSpike = new THREE.Mesh(tipSpikeGeo, emissiveRuneMat);
      tipSpike.position.set(0, limbDir * 0.86, -0.2);
      tipSpike.rotation.x = limbDir > 0 ? -0.45 : Math.PI + 0.45;
      weaponGroup.add(tipSpike);
    }

    // Dynamic 2-Segment Elastic Bowstring that pulls back into a V-shape to the archer's nose!
    const halfStringGeo = trackGeo(
      new THREE.CylinderGeometry(0.0075, 0.0075, 0.82, 6),
      0
    );
    const upperBowstringPivot = new THREE.Group();
    upperBowstringPivot.name = 'Bowstring_Upper_Pivot';
    upperBowstringPivot.position.set(0, 0.82, -0.16);
    const upperStringMesh = new THREE.Mesh(halfStringGeo, emissiveRuneMat);
    upperStringMesh.position.set(0, -0.41, 0);
    upperBowstringPivot.add(upperStringMesh);
    weaponGroup.add(upperBowstringPivot);

    const lowerBowstringPivot = new THREE.Group();
    lowerBowstringPivot.name = 'Bowstring_Lower_Pivot';
    lowerBowstringPivot.position.set(0, -0.82, -0.16);
    const lowerStringMesh = new THREE.Mesh(halfStringGeo, emissiveRuneMat);
    lowerStringMesh.position.set(0, 0.41, 0);
    lowerBowstringPivot.add(lowerStringMesh);
    weaponGroup.add(lowerBowstringPivot);

    weaponGroup.userData.bowstringPivots = {
      upper: upperBowstringPivot,
      lower: lowerBowstringPivot,
    };

    // 3D Animated Arrow Projectile (Shaft + Steel/Rune Broadhead + 3 Fletching Feathers + Flight Energy Trail)
    const arrowShaftGeo = trackGeo(
      new THREE.CylinderGeometry(0.015, 0.015, 1.12, 8),
      0.002
    );
    arrowShaftGeo.rotateX(Math.PI / 2);
    const arrowShaftMesh = new THREE.Mesh(arrowShaftGeo, ivoryFangMat);
    arrowProjectileMesh.add(arrowShaftMesh);

    const broadheadGeo = trackGeo(
      new THREE.ConeGeometry(0.052, 0.24, 6),
      0.003
    );
    broadheadGeo.rotateX(Math.PI / 2);
    broadheadGeo.scale(1.2, 0.65, 1.0);
    const broadheadMesh = new THREE.Mesh(broadheadGeo, emissiveRuneMat);
    broadheadMesh.position.set(0, 0, 0.62);
    arrowProjectileMesh.add(broadheadMesh);

    for (let v = 0; v < 3; v++) {
      const vaneGeo = trackGeo(new THREE.BoxGeometry(0.008, 0.095, 0.22), 0.002);
      const vaneMesh = new THREE.Mesh(vaneGeo, secondaryTrimMat);
      const vAng = (v / 3) * Math.PI * 2;
      vaneMesh.position.set(
        Math.sin(vAng) * 0.032,
        Math.cos(vAng) * 0.032,
        -0.44
      );
      vaneMesh.rotation.z = -vAng;
      arrowProjectileMesh.add(vaneMesh);
    }

    const arrowWakeGeo = trackGeo(
      new THREE.ConeGeometry(0.11, 0.92, 8, 1, true),
      0
    );
    arrowWakeGeo.rotateX(-Math.PI / 2);
    const arrowWakeMesh = new THREE.Mesh(arrowWakeGeo, slashTrailMat);
    arrowWakeMesh.name = 'Arrow_Flight_Wake';
    arrowWakeMesh.position.set(0, 0, 0.12);
    arrowProjectileMesh.add(arrowWakeMesh);

    // Sculpted 3D Dorsal Arrow Quiver with 6 Fletched Arrows mounted diagonally across the Archer's Back!
    const quiverGroup = new THREE.Group();
    quiverGroup.name = 'Dorsal_Arrow_Quiver';
    quiverGroup.position.set(
      0.16 * m.torsoBulk * m.shoulderWidth,
      torsoHeight * 0.56,
      -0.23 * m.torsoBulk
    );
    quiverGroup.rotation.set(0.16, 0, -0.34);
    torsoGroup.add(quiverGroup);

    const quiverTubeGeo = trackGeo(
      new THREE.CylinderGeometry(0.095 * m.torsoBulk, 0.075 * m.torsoBulk, 0.68, seg(12, 8)),
      0.008
    );
    const quiverTubeMesh = new THREE.Mesh(quiverTubeGeo, secondaryTrimMat);
    quiverGroup.add(quiverTubeMesh);

    const quiverRimGeo = trackGeo(
      new THREE.TorusGeometry(0.098 * m.torsoBulk, 0.018 * m.torsoBulk, 6, 12),
      0
    );
    quiverRimGeo.rotateX(Math.PI / 2);
    const quiverRimMesh = new THREE.Mesh(quiverRimGeo, primaryArmorMat);
    quiverRimMesh.position.y = 0.34;
    quiverGroup.add(quiverRimMesh);

    for (let qa = 0; qa < 6; qa++) {
      const qAng = (qa / 6) * Math.PI * 2;
      const qRad = 0.042 * m.torsoBulk;
      const qArrowGeo = trackGeo(
        new THREE.CylinderGeometry(0.011, 0.011, 0.38, 6),
        0
      );
      const qArrow = new THREE.Mesh(qArrowGeo, ivoryFangMat);
      qArrow.position.set(
        Math.cos(qAng) * qRad,
        0.44 + (qa % 2) * 0.04,
        Math.sin(qAng) * qRad
      );
      qArrow.rotation.z = (qa - 2.5) * 0.04;
      quiverGroup.add(qArrow);

      const qFeatherGeo = trackGeo(
        new THREE.ConeGeometry(0.028, 0.11, 4),
        0
      );
      const qFeather = new THREE.Mesh(qFeatherGeo, emissiveRuneMat);
      qFeather.position.set(
        Math.cos(qAng) * qRad,
        0.61 + (qa % 2) * 0.04,
        Math.sin(qAng) * qRad
      );
      quiverGroup.add(qFeather);
    }
  } else {
    const shaftLen =
      effectiveWeapon === 'arcane_staff' || effectiveWeapon === 'radiant_halberd' ? 1.85 : 1.15;
    const shaftTilt = Math.PI / 2.35;
    // Offset along the shaft axis so the handle passes DIRECTLY through (0, 0, 0) inside the closed 3D fingers & thumb!
    const gripOffset = shaftLen * 0.24;
    const shaftGeo = trackGeo(
      new THREE.CylinderGeometry(0.028, 0.034, shaftLen, seg(10, 6)),
      0.005
    );
    const shaftMesh = new THREE.Mesh(shaftGeo, underbodyMat);
    shaftMesh.position.set(
      0,
      Math.cos(shaftTilt) * gripOffset,
      Math.sin(shaftTilt) * gripOffset
    );
    shaftMesh.rotation.x = shaftTilt;
    weaponGroup.add(shaftMesh);

    // Sculpted Cord-Wrapped Hilt Grip Sleeve & Heavy Pommel right where the 3D hand fingers clamp around (0, 0, 0)!
    const gripSleeveGeo = trackGeo(
      new THREE.CylinderGeometry(0.036, 0.036, 0.24, seg(10, 6)),
      0.006
    );
    const gripSleeveMesh = new THREE.Mesh(gripSleeveGeo, secondaryTrimMat);
    gripSleeveMesh.position.set(0, -gripOffset, 0);
    shaftMesh.add(gripSleeveMesh);

    const pommelGeo = trackGeo(
      new THREE.OctahedronGeometry(0.068, 1),
      0.006
    );
    const pommelMesh = new THREE.Mesh(pommelGeo, secondaryTrimMat);
    pommelMesh.position.set(0, -shaftLen * 0.5 - 0.03, 0);
    shaftMesh.add(pommelMesh);

    const guardGeo = trackGeo(new THREE.SphereGeometry(0.14, 7, 5), 0.008);
    guardGeo.scale(1.2, 0.35, 0.55);
    const guardMesh = new THREE.Mesh(guardGeo, secondaryTrimMat);
    guardMesh.position.set(0, shaftLen * 0.45, 0);
    shaftMesh.add(guardMesh);

    if (effectiveWeapon === 'runic_greatsword' || effectiveWeapon === 'venom_scythe_claws') {
      const bladeShape = new THREE.Shape();
      bladeShape.moveTo(-0.085, 0);
      bladeShape.lineTo(0.085, 0);
      bladeShape.lineTo(0.055, 1.05);
      bladeShape.lineTo(0, 1.32);
      bladeShape.lineTo(-0.055, 1.05);
      bladeShape.closePath();

      const bladeGeo = trackGeo(
        new THREE.ExtrudeGeometry(bladeShape, {
          depth: 0.024,
          bevelEnabled: true,
          bevelThickness: 0.01,
          bevelSize: 0.012,
          bevelSegments: 1,
        }),
        0.004
      );
      bladeGeo.center();
      const bladeMesh = new THREE.Mesh(bladeGeo, bladeAlloyMat);
      bladeMesh.position.set(0, shaftLen * 0.5 + 0.52, 0);
      shaftMesh.add(bladeMesh);

      const runeStripGeo = trackGeo(new THREE.BoxGeometry(0.03, 0.85, 0.048), 0);
      const runeStripMesh = new THREE.Mesh(runeStripGeo, emissiveRuneMat);
      runeStripMesh.position.set(0, shaftLen * 0.5 + 0.38, 0);
      shaftMesh.add(runeStripMesh);
    } else if (effectiveWeapon === 'obsidian_warhammer') {
      const headGeo = trackGeo(new THREE.CylinderGeometry(0.22, 0.22, 0.54, 8), 0.018);
      headGeo.rotateZ(Math.PI / 2);
      const hammerHead = new THREE.Mesh(headGeo, primaryArmorMat);
      hammerHead.position.set(0, shaftLen * 0.5 + 0.14, 0);
      shaftMesh.add(hammerHead);
    } else if (effectiveWeapon === 'arcane_staff') {
      const finialGeo = trackGeo(
        new THREE.TorusGeometry(0.22, 0.035, 6, seg(16, 8)),
        0.008
      );
      const finialMesh = new THREE.Mesh(finialGeo, secondaryTrimMat);
      finialMesh.position.set(0, shaftLen * 0.5 + 0.2, 0);
      shaftMesh.add(finialMesh);

      const orbGeo = trackGeo(
        new THREE.OctahedronGeometry(0.12, density === 'low' ? 0 : 1),
        0
      );
      const orbMesh = new THREE.Mesh(orbGeo, crystalCoreMat);
      orbMesh.position.set(0, shaftLen * 0.5 + 0.2, 0);
      shaftMesh.add(orbMesh);
    } else {
      const axeShape = new THREE.Shape();
      axeShape.moveTo(0, -0.16);
      axeShape.quadraticCurveTo(0.42, -0.05, 0.48, 0.3);
      axeShape.quadraticCurveTo(0.25, 0.08, 0, 0.24);
      axeShape.closePath();

      const axeGeo = trackGeo(
        new THREE.ExtrudeGeometry(axeShape, {
          depth: 0.025,
          bevelEnabled: true,
          bevelThickness: 0.012,
          bevelSize: 0.012,
          bevelSegments: 1,
        }),
        0.005
      );
      const axeBladeR = new THREE.Mesh(axeGeo, bladeAlloyMat);
      axeBladeR.position.set(0.02, shaftLen * 0.45, -0.012);
      shaftMesh.add(axeBladeR);

      const axeBladeL = axeBladeR.clone();
      axeBladeL.scale.x = -1;
      axeBladeL.position.x = -0.02;
      shaftMesh.add(axeBladeL);
    }
  }

  // 8B. PROCEDURAL WEAPON SLASH TRAIL ARC (Used only for armed humanoid/champion weapons, never for animals)
  const slashTrailGeo = createWeaponSlashArcGeometry(isNaturalUnarmed, seg(22, 14));
  geometriesToDispose.push(slashTrailGeo);
  const slashTrailMesh = new THREE.Mesh(slashTrailGeo, slashTrailMat);
  slashTrailMesh.name = 'Weapon_Slash_Trail_Arc';
  slashTrailMesh.position.set(0, 0.08, 0.25);
  slashTrailMesh.scale.setScalar(0.0001); // Invisible at rest; animated only for weapon-wielding champions
  slashTrailMesh.visible = true;
  slashTrailMesh.frustumCulled = false;
  weaponGroup.add(slashTrailMesh);

  tagExplodablePart(weaponGroup, new THREE.Vector3(-0.65, 0.15, 0.75), {
    name: isNaturalUnarmed
      ? `Ataque Natural (${effectiveWeapon})`
      : `Armamento Principal (${effectiveWeapon})`,
    category: 'Weapon & Socket',
    vertexCount: isNaturalUnarmed ? 90 : 310,
    triangleCount: isNaturalUnarmed ? 60 : 250,
    materialSummary: isNaturalUnarmed
      ? 'Sin armas de mano — Ataque natural por garras, mordisco, embestida o aliento'
      : 'Empuñadura anclada dentro de los dedos de la mano (socket_weapon_r)',
    socketName: 'socket_weapon_r',
  });
  explodableParts.push(weaponGroup);

  // 9. OFF-HAND SHIELD / ORB / DAGGER ON socket_shield_l (Strictly restricted to creatures with grasping hands!)
  const offhandGroup = new THREE.Group();
  offhandGroup.name = `Offhand_${hasGraspingHands ? config.offhand : 'none'}`;
  socketShieldL.add(offhandGroup);

  if (hasGraspingHands && (config.offhand === 'tower_shield' || config.offhand === 'runic_buckler')) {
    const isTower = config.offhand === 'tower_shield';
    const shW = isTower ? 0.42 : 0.32;
    const shH = isTower ? 0.88 : 0.42;

    // Sculpted Inner Shield Handle Grip Bar passing directly through (0, 0, 0) inside the closed left fist!
    const shieldGripGeo = trackGeo(
      new THREE.CylinderGeometry(0.026, 0.026, 0.22, 8),
      0.004
    );
    shieldGripGeo.rotateX(Math.PI / 2);
    const shieldGripMesh = new THREE.Mesh(shieldGripGeo, secondaryTrimMat);
    offhandGroup.add(shieldGripMesh);

    const shieldShape = new THREE.Shape();
    shieldShape.moveTo(-shW, shH * 0.5);
    shieldShape.lineTo(shW, shH * 0.5);
    shieldShape.lineTo(shW * 0.85, -shH * 0.25);
    shieldShape.lineTo(0, -shH * 0.6);
    shieldShape.lineTo(-shW * 0.85, -shH * 0.25);
    shieldShape.closePath();

    const shieldGeo = trackGeo(
      new THREE.ExtrudeGeometry(shieldShape, {
        depth: 0.06,
        bevelEnabled: true,
        bevelThickness: 0.02,
        bevelSize: 0.02,
        bevelSegments: 1,
      }),
      0.008
    );
    const shieldMesh = new THREE.Mesh(shieldGeo, primaryArmorMat);
    shieldMesh.position.set(0.085, 0.02, 0.02);
    shieldMesh.rotation.y = 0.48;
    offhandGroup.add(shieldMesh);
  } else if (hasGraspingHands && config.offhand === 'phylactery_orb') {
    // Necromantic Censer Handle Rod gripped directly inside the left hand at (0, 0, 0) supporting the Phylactery Orb
    const censerRodGeo = trackGeo(
      new THREE.CylinderGeometry(0.024, 0.028, 0.34, 8),
      0.004
    );
    censerRodGeo.rotateX(Math.PI / 2.2);
    const censerRod = new THREE.Mesh(censerRodGeo, secondaryTrimMat);
    censerRod.position.set(0, 0.04, 0.06);
    offhandGroup.add(censerRod);

    const censerGeo = trackGeo(
      new THREE.IcosahedronGeometry(0.15, density === 'low' ? 0 : 1),
      0
    );
    const censerMesh = new THREE.Mesh(censerGeo, crystalCoreMat);
    censerMesh.position.set(0, 0.11, 0.22);
    offhandGroup.add(censerMesh);
  } else if (hasGraspingHands && config.offhand === 'dual_blade') {
    // Offhand Reaver Dagger with Hilt Handle gripped directly inside the closed left fist at (0, 0, 0)
    const daggerTilt = Math.PI / 2.25;
    const daggerHiltGeo = trackGeo(
      new THREE.CylinderGeometry(0.025, 0.028, 0.28, 8),
      0.004
    );
    const daggerHilt = new THREE.Mesh(daggerHiltGeo, secondaryTrimMat);
    daggerHilt.rotation.x = daggerTilt;
    offhandGroup.add(daggerHilt);

    const daggerGuardGeo = trackGeo(new THREE.BoxGeometry(0.18, 0.032, 0.055), 0.004);
    const daggerGuard = new THREE.Mesh(daggerGuardGeo, secondaryTrimMat);
    daggerGuard.position.set(0, 0.12, 0);
    daggerHilt.add(daggerGuard);

    const daggerGeo = trackGeo(new THREE.ConeGeometry(0.068, 0.72, seg(6, 4)), 0.005);
    daggerGeo.scale(1.0, 1.0, 0.36);
    const daggerMesh = new THREE.Mesh(daggerGeo, bladeAlloyMat);
    daggerMesh.position.set(0, 0.48, 0);
    daggerHilt.add(daggerMesh);
  }

  // 10. ARTICULATED WINGS & DORSAL STRUCTURES WITH SCULPTED SCAPULAR BACK SOCKETS
  const dorsalGroup = makeBone(`Dorsal_${config.dorsal}`);
  dorsalGroup.position.set(0, torsoHeight * 0.68, -0.18 * m.torsoBulk);
  dorsalGroup.scale.setScalar(m.dorsalScale);
  torsoGroup.add(dorsalGroup);

  if (config.dorsal !== 'none' && !isBeholder) {
    const dorsalSaddleGeo = trackGeo(
      new THREE.SphereGeometry(0.165 * m.torsoBulk, seg(12, 8), seg(10, 8)),
      0.016
    );
    dorsalSaddleGeo.scale(1.45, 0.92, 0.95);
    const dorsalSaddleMesh = new THREE.Mesh(
      dorsalSaddleGeo,
      isUndeadSkeleton ? ivoryFangMat : secondaryTrimMat
    );
    dorsalSaddleMesh.position.set(0, 0.04, 0.05);
    dorsalGroup.add(dorsalSaddleMesh);
  }

  const leftWingGroup = makeBone('Wing_L');
  const rightWingGroup = makeBone('Wing_R');
  dorsalGroup.add(leftWingGroup);
  dorsalGroup.add(rightWingGroup);

  if (config.dorsal === 'dragon_membrane_wings') {
    for (const side of [-1, 1]) {
      const targetWing = side === 1 ? leftWingGroup : rightWingGroup;

      const wingRootGeo = trackGeo(new THREE.SphereGeometry(0.16, seg(10, 8), seg(8, 6)), 0.014);
      wingRootGeo.scale(1.25, 1.05, 1.3);
      const wingRoot = new THREE.Mesh(wingRootGeo, secondaryTrimMat);
      wingRoot.position.set(side * 0.06, 0.02, 0.06);
      targetWing.add(wingRoot);

      const sparCurve = new THREE.CatmullRomCurve3([
        new THREE.Vector3(side * 0.04, 0, 0.04),
        new THREE.Vector3(side * 0.55, 0.48, -0.08),
        new THREE.Vector3(side * 1.25, 0.62, -0.16),
        new THREE.Vector3(side * 1.65, 0.15, -0.22),
      ]);
      const sparGeo = trackGeo(
        new THREE.TubeGeometry(sparCurve, seg(10, 6), 0.044, seg(6, 4), false),
        0.01
      );
      targetWing.add(new THREE.Mesh(sparGeo, primaryArmorMat));

      const webShape = new THREE.Shape();
      webShape.moveTo(0, 0);
      webShape.lineTo(side * 0.55, 0.46);
      webShape.lineTo(side * 1.25, 0.6);
      webShape.lineTo(side * 1.62, 0.14);
      webShape.quadraticCurveTo(side * 1.15, -0.15, side * 0.85, -0.45);
      webShape.quadraticCurveTo(side * 0.45, -0.2, 0, -0.35);
      webShape.closePath();

      const webGeo = trackGeo(
        new THREE.ExtrudeGeometry(webShape, { depth: 0.02, bevelEnabled: false }),
        0.01
      );
      const webMesh = new THREE.Mesh(webGeo, secondaryTrimMat);
      webMesh.position.z = -0.08;
      targetWing.add(webMesh);
    }
  } else if (config.dorsal === 'seraph_wings') {
    for (const side of [-1, 1]) {
      const targetWing = side === 1 ? leftWingGroup : rightWingGroup;
      const wingArchGeo = trackGeo(new THREE.SphereGeometry(0.12, 6, 5), 0.01);
      wingArchGeo.scale(1.8, 0.8, 0.8);
      const wingArch = new THREE.Mesh(wingArchGeo, secondaryTrimMat);
      wingArch.position.set(side * 0.15, 0.15, 0);
      targetWing.add(wingArch);

      for (let f = 0; f < 6; f++) {
        const featherGeo = trackGeo(new THREE.ConeGeometry(0.055, 0.85 - f * 0.09, 5), 0.008);
        featherGeo.scale(1.2, 1.0, 0.35);
        const featherMesh = new THREE.Mesh(
          featherGeo,
          f % 2 === 0 ? secondaryTrimMat : primaryArmorMat
        );
        featherMesh.position.set(side * (0.22 + f * 0.16), 0.22 - f * 0.05, -0.03 * f);
        featherMesh.rotation.z = Math.PI - side * (0.42 + f * 0.15);
        targetWing.add(featherMesh);
      }
    }
  } else if (config.dorsal === 'beholder_eye_tentacles') {
    for (let i = 0; i < 8; i++) {
      const ang = (i / 8) * Math.PI * 2;
      const rx = Math.cos(ang);
      const rz = Math.sin(ang);
      const stalkCurve = new THREE.CatmullRomCurve3([
        new THREE.Vector3(rx * 0.18, 0.05, rz * 0.18),
        new THREE.Vector3(rx * 0.65, 0.45 + (i % 2) * 0.18, rz * 0.65),
        new THREE.Vector3(rx * 0.88, 0.28 + (i % 3) * 0.15, rz * 0.88),
      ]);
      const stalkGeo = trackGeo(
        new THREE.TubeGeometry(stalkCurve, seg(8, 5), 0.04, seg(5, 4), false),
        0.01
      );
      dorsalGroup.add(new THREE.Mesh(stalkGeo, primaryArmorMat));

      const tipEye = new THREE.Mesh(
        trackGeo(new THREE.SphereGeometry(0.068, seg(8, 5), seg(6, 4)), 0),
        emissiveRuneMat
      );
      tipEye.position.set(rx * 0.88, 0.28 + (i % 3) * 0.15, rz * 0.88);
      dorsalGroup.add(tipEye);
    }
  } else if (config.dorsal === 'tattered_mantle') {
    const capeShape = new THREE.Shape();
    capeShape.moveTo(-0.42, 0.24);
    capeShape.lineTo(0.42, 0.24);
    capeShape.lineTo(0.58, -1.25);
    capeShape.lineTo(0, -1.32);
    capeShape.lineTo(-0.58, -1.25);
    capeShape.closePath();
    const capeGeo = trackGeo(
      new THREE.ExtrudeGeometry(capeShape, { depth: 0.03, bevelEnabled: false }),
      0.02,
      8.0
    );
    const capeMesh = new THREE.Mesh(capeGeo, underbodyMat);
    capeMesh.rotation.x = 0.14;
    dorsalGroup.add(capeMesh);
  } else if (config.dorsal === 'crystal_pylons' || config.dorsal === 'bio_chitin_spines') {
    for (let i = -2; i <= 2; i++) {
      if (i === 0) continue;
      const pylonGeo = trackGeo(new THREE.ConeGeometry(0.08, 0.55, 5), 0.012);
      const pylonMesh = new THREE.Mesh(
        pylonGeo,
        config.dorsal === 'crystal_pylons' ? crystalCoreMat : secondaryTrimMat
      );
      pylonMesh.position.set(i * 0.14, 0.18 + (2 - Math.abs(i)) * 0.1, -0.04);
      pylonMesh.rotation.z = -i * 0.25;
      pylonMesh.rotation.x = -0.35;
      dorsalGroup.add(pylonMesh);
    }
  } else if (config.dorsal === 'runic_halo_ring') {
    const haloGeo = trackGeo(new THREE.TorusGeometry(0.52, 0.035, 6, seg(20, 12)), 0);
    const haloMesh = new THREE.Mesh(haloGeo, emissiveRuneMat);
    haloMesh.position.set(0, 0.28, -0.04);
    dorsalGroup.add(haloMesh);
  }

  tagExplodablePart(dorsalGroup, new THREE.Vector3(0, 0.55, -0.85), {
    name: `Alas / Estructura Dorsal (${config.dorsal})`,
    category: 'Secondary Read',
    vertexCount: 380,
    triangleCount: 310,
    materialSummary: 'Unión escapular continua al torso dorsal',
  });
  explodableParts.push(dorsalGroup);

  // 11. CONTINUOUS 2-BONE ARTICULATED TAIL WITH CAUDAL BASE, BUSHY 3D FUR PLUME & WHIP TIP BONE
  const tailGroup = makeBone('Articulated_Tail');
  tailGroup.position.set(0, 0.06, isQuadruped || isArachnid ? -0.52 : -0.22 * m.torsoBulk);
  pelvisGroup.add(tailGroup);

  const tailTipBone = makeBone('Tail_Mid_Tip');
  tailTipBone.position.set(0, -0.12, -0.55 * m.tailLength);
  tailGroup.add(tailTipBone);

  if (m.tailLength > 0.08 && !isSerpentine) {
    const tLen = 1.15 * m.tailLength;
    const isScorpion =
      config.dorsal === 'scorpion_stinger_tail' ||
      isArachnid ||
      config.archetype === 'manticore_chimera' ||
      config.archetype === 'gale_wind_manticore' ||
      config.archetype === 'stitched_necromancy_chimera';

    const tailCurve = isScorpion
      ? new THREE.CatmullRomCurve3([
          new THREE.Vector3(0, 0.02, 0.14),
          new THREE.Vector3(0, 0.45 * tLen, -0.45 * tLen),
          new THREE.Vector3(0, 1.05 * tLen, -0.25 * tLen),
          new THREE.Vector3(0, 1.25 * tLen, 0.25 * tLen),
        ])
      : new THREE.CatmullRomCurve3([
          new THREE.Vector3(0, 0.02, 0.14),
          new THREE.Vector3(0, -0.14, -0.45 * tLen),
          new THREE.Vector3(0, -0.2, -0.95 * tLen),
          new THREE.Vector3(0, -0.06, -1.35 * tLen),
        ]);

    const midPt = tailCurve.getPoint(0.48);
    tailTipBone.position.copy(midPt);

    if (isUndeadSkeleton) {
      // BLENDER-SCULPTED 3D EXPOSED CAUDAL VERTEBRAL TAIL CHAIN FOR UNDEAD SKELETONS
      const caudalVertCount = 10;
      for (let cv = 0; cv < caudalVertCount; cv++) {
        const u = cv / (caudalVertCount - 1);
        const pt = tailCurve.getPoint(u);
        const tang = tailCurve.getTangent(u);
        const vScale = (1 - u * 0.65) * m.torsoBulk;

        const cBodyGeo = trackGeo(
          new THREE.CylinderGeometry(0.056 * vScale, 0.064 * vScale, 0.095 * tLen, seg(10, 8)),
          0.005
        );
        cBodyGeo.rotateX(Math.PI / 2);
        const cBody = new THREE.Mesh(cBodyGeo, ivoryFangMat);

        const cSpineGeo = trackGeo(
          new THREE.ConeGeometry(0.024 * vScale, 0.11 * vScale, 5),
          0.004
        );
        const cSpine = new THREE.Mesh(cSpineGeo, ivoryFangMat);

        if (u < 0.5) {
          cBody.position.copy(pt);
          cBody.lookAt(pt.clone().add(tang));
          tailGroup.add(cBody);

          cSpine.position.copy(pt).add(new THREE.Vector3(0, 0.065 * vScale, 0));
          tailGroup.add(cSpine);
        } else {
          const localPt = pt.clone().sub(midPt);
          cBody.position.copy(localPt);
          cBody.lookAt(localPt.clone().add(tang));
          tailTipBone.add(cBody);

          cSpine.position.copy(localPt).add(new THREE.Vector3(0, 0.065 * vScale, 0));
          tailTipBone.add(cSpine);
        }
      }

      const tipPt = tailCurve.getPoint(1);
      const boneSpurGeo = trackGeo(new THREE.ConeGeometry(0.085, 0.34, 6), 0.008);
      boneSpurGeo.rotateX(isScorpion ? Math.PI * 0.65 : -Math.PI / 2);
      const boneSpur = new THREE.Mesh(boneSpurGeo, ivoryFangMat);
      boneSpur.position.copy(tipPt).sub(midPt);
      tailTipBone.add(boneSpur);
    } else {
      // Blender-Sculpted Sacral-Caudal Tail Root Collar & Muscular Saddle (Seamlessly unites Pelvis_Chassis and Articulated_Tail)
      const tailRootGeo = trackGeo(
        new THREE.SphereGeometry(0.185 * m.torsoBulk, seg(12, 8), seg(10, 8)),
        0.016
      );
      tailRootGeo.scale(1.08, 0.96, 1.55);
      const tailRootMesh = new THREE.Mesh(tailRootGeo, primaryArmorMat);
      tailRootMesh.position.set(0, 0.02, 0.09);
      tailGroup.add(tailRootMesh);

      const sacralCollarGeo = trackGeo(
        new THREE.CylinderGeometry(
          0.13 * m.torsoBulk,
          0.22 * m.torsoBulk,
          0.28,
          seg(12, 8),
          2
        ),
        0.014
      );
      sacralCollarGeo.rotateX(Math.PI / 2 - 0.15);
      const sacralCollarMesh = new THREE.Mesh(sacralCollarGeo, primaryArmorMat);
      sacralCollarMesh.position.set(0, 0.02, 0.08);
      tailGroup.add(sacralCollarMesh);

      // Mid-Caudal Articulation Capsule at tailTipBone
      const midTailJointGeo = trackGeo(
        new THREE.SphereGeometry(0.125 * m.torsoBulk, seg(10, 8), seg(8, 6)),
        0.012
      );
      midTailJointGeo.scale(0.95, 0.95, 1.35);
      const midTailJointMesh = new THREE.Mesh(midTailJointGeo, primaryArmorMat);
      tailTipBone.add(midTailJointMesh);

      const isBushyTail =
        config.archetype === 'furious_keltir' ||
        config.archetype === 'rabid_grey_wolf' ||
        config.archetype === 'dire_wolf' ||
        furDensity > 0.35;

      const tailGeo = trackGeo(
        new THREE.TubeGeometry(
          tailCurve,
          seg(14, 8),
          (isBushyTail ? 0.165 : 0.11) * m.torsoBulk,
          seg(7, 4),
          false
        ),
        0.018
      );
      const tailMesh = new THREE.Mesh(tailGeo, primaryArmorMat);
      tailGroup.add(tailMesh);

      // 3D Bushy Hair Plume Strands along the tail on furry beasts
      if (furDensity > 0.15 && hairStyle !== 'none' && !isScorpion) {
        const fScale = 0.76 + furDensity * 0.38;
        for (let t = 0; t < 5; t++) {
          const u = 0.18 + t * 0.16;
          const pt = tailCurve.getPoint(u);
          const side = t % 2 === 0 ? 1 : -1;
          const plumeGeo = trackFurLock(0.085 * m.torsoBulk * fScale, 0.28 * fScale, 0.09);
          const plumeMesh = new THREE.Mesh(plumeGeo, hairStrandMat);
          if (u < 0.5) {
            plumeMesh.position.set(side * 0.05 * m.torsoBulk, pt.y, pt.z);
            plumeMesh.rotation.x = -Math.PI * 0.62;
            plumeMesh.rotation.z = -side * 0.32;
            tailGroup.add(plumeMesh);
          } else {
            plumeMesh.position.copy(pt).sub(midPt);
            plumeMesh.position.x += side * 0.05 * m.torsoBulk;
            plumeMesh.rotation.x = -Math.PI * 0.58;
            plumeMesh.rotation.z = -side * 0.28;
            tailTipBone.add(plumeMesh);
          }
        }
      }

      const tipPt = tailCurve.getPoint(1);
      const stingerGeo = trackGeo(
        new THREE.ConeGeometry(isBushyTail ? 0.18 : 0.13, isBushyTail ? 0.44 : 0.36, 6),
        0.01
      );
      stingerGeo.rotateX(isScorpion ? Math.PI * 0.65 : -Math.PI / 2);
      const stingerMesh = new THREE.Mesh(
        stingerGeo,
        isScorpion ? emissiveRuneMat : secondaryTrimMat
      );
      stingerMesh.position.copy(tipPt).sub(midPt);
      tailTipBone.add(stingerMesh);
    }

    tagExplodablePart(tailGroup, new THREE.Vector3(0, -0.1, -0.95), {
      name: isUndeadSkeleton
        ? 'Columna Caudal de Vértebras Óseas 3D'
        : isScorpion
        ? 'Aguijón Caudal Articulado'
        : 'Cola Orgánica con Penacho 3D',
      category: 'Secondary Read',
      vertexCount: 280,
      triangleCount: 230,
      materialSummary: isUndeadSkeleton
        ? 'Vértebras caudales individuales + espinas dorsales de marfil'
        : 'Malla continua deformada por 3 huesos + mechones de pelaje caudal',
    });
    explodableParts.push(tailGroup);
  }

  // 12. INSTANCED ORBITING RUNIC SHARDS
  let auraInstancedMesh: THREE.InstancedMesh | null = null;
  const shardData: BuiltEnemyRig['shardData'] = [];
  const rawShardCount = Math.max(0, Math.round(m.auraShardCount));
  const shardCount =
    density === 'low' ? Math.min(8, Math.ceil(rawShardCount * 0.5)) : rawShardCount;

  if (shardCount > 0) {
    const shardGeo = trackGeo(new THREE.OctahedronGeometry(0.055, 0), 0);
    shardGeo.scale(0.7, 1.8, 0.7);
    auraInstancedMesh = new THREE.InstancedMesh(shardGeo, emissiveRuneMat, shardCount);
    auraInstancedMesh.name = 'Aura_Runic_Shards';
    auraInstancedMesh.frustumCulled = false;

    const baseColor = new THREE.Color(matCfg.emissiveColor);
    for (let i = 0; i < shardCount; i++) {
      const angle = (i / shardCount) * Math.PI * 2;
      const radius = 0.85 + (i % 3) * 0.25;
      const yOffset = 0.35 + ((i * 7) % 10) * 0.18;
      const speed = 0.6 + (i % 5) * 0.18;
      const scale = 0.85 + ((i * 13) % 30) / 100;

      shardData.push({ angle, radius, yOffset, speed, scale });

      _dummyObj.position.set(Math.cos(angle) * radius, yOffset, Math.sin(angle) * radius);
      _dummyObj.rotation.set(0.2, angle, 0.15);
      _dummyObj.scale.setScalar(scale);
      _dummyObj.updateMatrix();
      auraInstancedMesh.setMatrixAt(i, _dummyObj.matrix);

      _shardColor.copy(baseColor).offsetHSL(((i % 5) - 2) * 0.03, 0, ((i % 3) - 1) * 0.08);
      auraInstancedMesh.setColorAt(i, _shardColor);
    }
    auraInstancedMesh.instanceMatrix.needsUpdate = true;
    if (auraInstancedMesh.instanceColor) auraInstancedMesh.instanceColor.needsUpdate = true;
    auraInstancedMesh.computeBoundingSphere();
    exportableGroup.add(auraInstancedMesh);
  }

  // 12B. BESPOKE 3D SPECIES ANATOMY FOR ELITE MONSTERS (Mammoth, Sabertooth, Porcupine, Rhino, Moose, Kitsune, Basilisk, Cobra, Chameleon, Tortoise, Pangolin, Ankylo, Angler, Salamander, 4-Armed Reaver, Jailer, Wheel Martyr, Myconid, Wendigo)
  {
    const arch = config.archetype;
    const hScale = 0.82 + m.torsoBulk * 0.18;
    const lThick = m.torsoBulk;

    // 1) Mammoth Trunk & Giant Curved Ivory Tusks (Living Woolly Mammoth & Ossuary Mammoth Behemoth)
    if (arch === 'elite_woolly_mammoth_calf' || arch === 'elite_mammoth_ossuary_behemoth') {
      for (const side of [-1, 1]) {
        const tuskCurve = new THREE.CatmullRomCurve3([
          new THREE.Vector3(side * 0.16 * hScale, -0.05 * hScale, 0.24 * hScale),
          new THREE.Vector3(side * 0.34 * hScale, -0.26 * hScale, 0.62 * hScale),
          new THREE.Vector3(side * 0.42 * hScale, -0.08 * hScale, 1.05 * hScale),
          new THREE.Vector3(side * 0.28 * hScale, 0.32 * hScale, 1.28 * hScale),
        ]);
        const tuskGeo = trackGeo(
          new THREE.TubeGeometry(tuskCurve, seg(16, 12), 0.068 * hScale, seg(10, 8), false)
        );
        headGroup.add(new THREE.Mesh(tuskGeo, ivoryFangMat));
      }
      if (!isUndeadSkeleton) {
        const trunkCurve = new THREE.CatmullRomCurve3([
          new THREE.Vector3(0, 0.04 * hScale, 0.28 * hScale),
          new THREE.Vector3(0, -0.22 * hScale, 0.56 * hScale),
          new THREE.Vector3(0, -0.58 * hScale, 0.62 * hScale),
          new THREE.Vector3(0, -0.78 * hScale, 0.44 * hScale),
          new THREE.Vector3(0, -0.68 * hScale, 0.28 * hScale),
        ]);
        const trunkGeo = trackGeo(
          new THREE.TubeGeometry(trunkCurve, seg(18, 12), 0.095 * hScale, seg(12, 8), false),
          0.015
        );
        headGroup.add(new THREE.Mesh(trunkGeo, primaryArmorMat));
      }
    }

    // 2) Oversized Curved Upper Saber Canines (Saber Prowler Leopard & Sabertooth Skull Stalker)
    if (arch === 'elite_saber_prowler_leopard' || arch === 'elite_sabertooth_skull_stalker') {
      for (const side of [-1, 1]) {
        const saberCurve = new THREE.CatmullRomCurve3([
          new THREE.Vector3(side * 0.13 * hScale, -0.02 * hScale, 0.34 * hScale),
          new THREE.Vector3(side * 0.14 * hScale, -0.24 * hScale, 0.42 * hScale),
          new THREE.Vector3(side * 0.12 * hScale, -0.46 * hScale, 0.36 * hScale),
        ]);
        const saberGeo = trackGeo(
          new THREE.TubeGeometry(saberCurve, seg(12, 8), 0.036 * hScale, seg(8, 6), false)
        );
        const saberTipGeo = trackGeo(
          new THREE.ConeGeometry(0.036 * hScale, 0.14 * hScale, seg(8, 6))
        );
        saberTipGeo.rotateX(Math.PI * 0.92);
        const saberTip = new THREE.Mesh(saberTipGeo, ivoryFangMat);
        saberTip.position.set(side * 0.12 * hScale, -0.51 * hScale, 0.34 * hScale);
        headGroup.add(new THREE.Mesh(saberGeo, ivoryFangMat), saberTip);
      }
    }

    // 3) Dorsal Quill Fan / Thorny Spines (Quillback Porcupine & Thorny Devil Moloch)
    if (arch === 'elite_quillback_porcupine' || arch === 'elite_thorny_devil_moloch') {
      const quillLen = arch === 'elite_quillback_porcupine' ? 0.62 : 0.34;
      const quillRad = arch === 'elite_quillback_porcupine' ? 0.028 : 0.042;
      for (let q = 0; q < 24; q++) {
        const row = Math.floor(q / 4);
        const col = (q % 4) - 1.5;
        const rFrac = row / 5;
        const qGeo = trackGeo(
          new THREE.ConeGeometry(quillRad * m.torsoBulk, quillLen * m.torsoBulk, seg(8, 6))
        );
        const qMesh = new THREE.Mesh(qGeo, ivoryFangMat);
        qMesh.position.set(
          col * 0.14 * m.torsoBulk * m.shoulderWidth,
          (0.28 + Math.sin(rFrac * Math.PI) * 0.14) * m.torsoBulk,
          -0.32 + rFrac * 0.78
        );
        qMesh.rotation.x = -0.68 - rFrac * 0.35;
        qMesh.rotation.z = -col * 0.32;
        torsoGroup.add(qMesh);
      }
    }

    // 4) Dual Nasal & Frontal Rhino Horns (Ironhorn Rhino Beast)
    if (arch === 'elite_ironhorn_rhino_beast') {
      const mainHornCurve = new THREE.CatmullRomCurve3([
        new THREE.Vector3(0, 0.08 * hScale, 0.36 * hScale),
        new THREE.Vector3(0, 0.28 * hScale, 0.58 * hScale),
        new THREE.Vector3(0, 0.56 * hScale, 0.64 * hScale),
      ]);
      const mainHornGeo = trackGeo(
        new THREE.TubeGeometry(mainHornCurve, seg(12, 8), 0.075 * hScale, seg(10, 8), false)
      );
      const rearHornGeo = trackGeo(
        new THREE.ConeGeometry(0.065 * hScale, 0.28 * hScale, seg(10, 8))
      );
      const rearHorn = new THREE.Mesh(rearHornGeo, ivoryFangMat);
      rearHorn.position.set(0, 0.22 * hScale, 0.22 * hScale);
      rearHorn.rotation.x = 0.25;
      headGroup.add(new THREE.Mesh(mainHornGeo, ivoryFangMat), rearHorn);
    }

    // 5) Palmate Moose Antlers & Wendigo Branching Antlers + Exposed Ribs
    if (arch === 'elite_great_antler_moose' || arch === 'elite_wendigo_antler_fiend') {
      for (const side of [-1, 1]) {
        const palmGeo = trackGeo(
          new THREE.BoxGeometry(0.46 * hScale, 0.28 * hScale, 0.06 * hScale),
          0.015
        );
        const palmMesh = new THREE.Mesh(palmGeo, ivoryFangMat);
        palmMesh.position.set(side * 0.36 * hScale, 0.34 * hScale, -0.04 * hScale);
        palmMesh.rotation.z = -side * 0.38;
        palmMesh.rotation.y = side * 0.24;
        headGroup.add(palmMesh);

        for (let t = 0; t < 4; t++) {
          const tineGeo = trackGeo(
            new THREE.ConeGeometry(0.032 * hScale, 0.32 * hScale, seg(8, 6))
          );
          const tineMesh = new THREE.Mesh(tineGeo, ivoryFangMat);
          tineMesh.position.set(
            side * (0.24 + t * 0.1) * hScale,
            (0.52 + (t % 2) * 0.08) * hScale,
            (-0.08 + t * 0.05) * hScale
          );
          tineMesh.rotation.z = -side * (0.2 + t * 0.12);
          headGroup.add(tineMesh);
        }
      }
      if (arch === 'elite_wendigo_antler_fiend') {
        for (const side of [-1, 1]) {
          for (let r = 0; r < 5; r++) {
            const wRibCurve = new THREE.CatmullRomCurve3([
              new THREE.Vector3(side * 0.05, (0.35 + r * 0.14) * torsoHeight, -0.14 * m.torsoBulk),
              new THREE.Vector3(
                side * 0.34 * m.torsoBulk,
                (0.33 + r * 0.14) * torsoHeight,
                0.08 * m.torsoBulk
              ),
              new THREE.Vector3(
                side * 0.1 * m.torsoBulk,
                (0.29 + r * 0.14) * torsoHeight,
                0.28 * m.torsoBulk
              ),
            ]);
            const wRibGeo = trackGeo(
              new THREE.TubeGeometry(wRibCurve, seg(10, 8), 0.028 * m.torsoBulk, seg(8, 6), false)
            );
            torsoGroup.add(new THREE.Mesh(wRibGeo, ivoryFangMat));
          }
        }
      }
    }

    // 6) Three-Tailed Kitsune Fox Lateral Tails Fan
    if (arch === 'elite_three_tailed_kitsune_fox') {
      for (const side of [-1, 1]) {
        const kTailCurve = new THREE.CatmullRomCurve3([
          new THREE.Vector3(side * 0.08, 0.02, -0.05),
          new THREE.Vector3(side * 0.36, 0.24, -0.45),
          new THREE.Vector3(side * 0.62, 0.56, -0.88),
          new THREE.Vector3(side * 0.74, 0.85, -1.12),
        ]);
        const kTailGeo = trackGeo(
          new THREE.TubeGeometry(kTailCurve, seg(16, 12), 0.14 * m.torsoBulk, seg(12, 8), false),
          0.02
        );
        tailGroup.add(new THREE.Mesh(kTailGeo, primaryArmorMat));
      }
    }

    // 7) Pleated Neck Frill Umbrella / Flared Cobra Hood (Frilled Venom Basilisk & Hooded King Cobra)
    if (arch === 'elite_frilled_venom_basilisk' || arch === 'elite_hooded_king_cobra') {
      const frillGeo = trackGeo(
        new THREE.CylinderGeometry(
          0.54 * hScale,
          0.18 * hScale,
          0.14 * hScale,
          seg(16, 12),
          1,
          true
        ),
        0.015
      );
      frillGeo.scale(1.25, 1.0, 0.55);
      const frillMesh = new THREE.Mesh(frillGeo, secondaryTrimMat);
      frillMesh.position.set(0, 0.12 * hScale, -0.06 * hScale);
      frillMesh.rotation.x = Math.PI * 0.42;
      headGroup.add(frillMesh);
    }

    // 8) Casqued Chameleon Tall Conical Casque, 3 Jackson Horns & Coiled Tail
    if (arch === 'elite_casqued_chameleon_stalker') {
      const casqueGeo = trackGeo(
        new THREE.ConeGeometry(0.24 * hScale, 0.52 * hScale, seg(10, 8)),
        0.015
      );
      const casqueMesh = new THREE.Mesh(casqueGeo, primaryArmorMat);
      casqueMesh.position.set(0, 0.38 * hScale, -0.14 * hScale);
      casqueMesh.rotation.x = -0.48;
      headGroup.add(casqueMesh);

      for (const hOff of [-0.11, 0, 0.11]) {
        const cHornGeo = trackGeo(
          new THREE.ConeGeometry(0.032 * hScale, 0.36 * hScale, seg(8, 6))
        );
        cHornGeo.rotateX(Math.PI * 0.46);
        const cHorn = new THREE.Mesh(cHornGeo, ivoryFangMat);
        cHorn.position.set(
          hOff * hScale,
          (hOff === 0 ? 0.04 : 0.12) * hScale,
          0.42 * hScale
        );
        headGroup.add(cHorn);
      }
    }

    // 9) Domed Carapace Shell with Pyramidal Scutes (Snapping Chelonian Tortoise & Armored Pangolin Roller)
    if (arch === 'elite_snapping_chelonian_tortoise' || arch === 'elite_armored_pangolin_roller') {
      const shellGeo = trackGeo(
        new THREE.SphereGeometry(
          0.56 * m.torsoBulk,
          seg(18, 12),
          seg(12, 8),
          0,
          Math.PI * 2,
          0,
          Math.PI * 0.58
        ),
        0.02
      );
      shellGeo.scale(1.08 * m.shoulderWidth, 0.82, 1.38);
      const shellMesh = new THREE.Mesh(shellGeo, secondaryTrimMat);
      shellMesh.position.set(0, 0.12 * m.torsoBulk, 0.12);
      torsoGroup.add(shellMesh);
    }

    // 10) Heavy Spiked Bony Tail Club (Clubtail Ankylo-Drake)
    if (arch === 'elite_clubtail_ankylo_drake') {
      const clubGeo = trackGeo(
        new THREE.SphereGeometry(0.24 * m.torsoBulk, seg(14, 10), seg(12, 8)),
        0.02
      );
      clubGeo.scale(1.35, 0.78, 1.15);
      const clubMesh = new THREE.Mesh(clubGeo, secondaryTrimMat);
      tailTipBone.add(clubMesh);
      for (const side of [-1, 1]) {
        const cSpikeGeo = trackGeo(
          new THREE.ConeGeometry(0.07 * m.torsoBulk, 0.26 * m.torsoBulk, seg(8, 6))
        );
        cSpikeGeo.rotateZ(-side * Math.PI * 0.5);
        const cSpike = new THREE.Mesh(cSpikeGeo, ivoryFangMat);
        cSpike.position.set(side * 0.28 * m.torsoBulk, 0, 0);
        tailTipBone.add(cSpike);
      }
    }

    // 11) Bioluminescent Angler Lure Stalk & Glowing Esca Lantern (Abyssal Angler Lurker)
    if (arch === 'elite_abyssal_angler_lurker') {
      const lureCurve = new THREE.CatmullRomCurve3([
        new THREE.Vector3(0, 0.22 * hScale, 0.12 * hScale),
        new THREE.Vector3(0, 0.55 * hScale, 0.38 * hScale),
        new THREE.Vector3(0, 0.42 * hScale, 0.68 * hScale),
      ]);
      const lureGeo = trackGeo(
        new THREE.TubeGeometry(lureCurve, seg(14, 10), 0.024 * hScale, seg(8, 6), false)
      );
      const escaGeo = trackGeo(
        new THREE.SphereGeometry(0.095 * hScale, seg(12, 8), seg(10, 8))
      );
      const escaMesh = new THREE.Mesh(escaGeo, emissiveRuneMat);
      escaMesh.position.set(0, 0.42 * hScale, 0.68 * hScale);
      headGroup.add(new THREE.Mesh(lureGeo, secondaryTrimMat), escaMesh);
    }

    // 12) External Feathery Amphibian Gill Stalks (Spitting Salamander Drake)
    if (arch === 'elite_spitting_salamander_drake') {
      for (const side of [-1, 1]) {
        for (let g = 0; g < 3; g++) {
          const gillGeo = trackGeo(
            new THREE.ConeGeometry(0.038 * hScale, 0.28 * hScale, seg(8, 6))
          );
          const gillMesh = new THREE.Mesh(gillGeo, emissiveRuneMat);
          gillMesh.position.set(
            side * 0.26 * hScale,
            (0.04 + g * 0.08) * hScale,
            -0.08 * hScale
          );
          gillMesh.rotation.z = -side * (0.85 + g * 0.18);
          headGroup.add(gillMesh);
        }
      }
    }

    // 13) Second Pair of Anatomical 3D Skeletal Arms (Four-Armed Bone Reaver: Humeral Head + Humerus + Epicondyles + Separated Radius & Ulna + Carpals + 5 Metacarpal & Phalangeal Fingers)
    if (arch === 'elite_four_armed_bone_reaver') {
      for (const side of [-1, 1]) {
        const subShoulderPos = new THREE.Vector3(
          side * 0.34 * m.torsoBulk * m.shoulderWidth,
          torsoHeight * 0.48,
          0.02
        );
        const subElbowPos = new THREE.Vector3(
          side * 0.56 * m.torsoBulk * m.shoulderWidth,
          torsoHeight * 0.28,
          0.14
        );
        const subWristPos = new THREE.Vector3(
          side * 0.48 * m.torsoBulk * m.shoulderWidth,
          torsoHeight * 0.12,
          0.42
        );

        // Glenoid Humeral Head & Distal Elbow Epicondyles
        const humHeadGeo = trackGeo(
          new THREE.SphereGeometry(0.068 * lThick, seg(10, 8), seg(8, 6)),
          0.006
        );
        const humHead = new THREE.Mesh(humHeadGeo, ivoryFangMat);
        humHead.position.copy(subShoulderPos);
        torsoGroup.add(humHead);

        // Humerus Shaft Curve
        const humCurve = new THREE.CatmullRomCurve3([subShoulderPos, subElbowPos]);
        const humGeo = trackGeo(
          new THREE.TubeGeometry(humCurve, seg(10, 8), 0.044 * lThick, seg(8, 6), false),
          0.005
        );
        torsoGroup.add(new THREE.Mesh(humGeo, ivoryFangMat));

        const elbowJointGeo = trackGeo(
          new THREE.SphereGeometry(0.058 * lThick, seg(10, 8), seg(8, 6)),
          0.005
        );
        elbowJointGeo.scale(1.3, 0.9, 1.0);
        const elbowJoint = new THREE.Mesh(elbowJointGeo, ivoryFangMat);
        elbowJoint.position.copy(subElbowPos);
        torsoGroup.add(elbowJoint);

        // Separated Radius & Ulna Forearm Bones with Interosseous Gap
        for (const boneOff of [-1, 1]) {
          const forearmMid = subElbowPos
            .clone()
            .lerp(subWristPos, 0.5)
            .add(new THREE.Vector3(side * boneOff * 0.028 * lThick, boneOff * 0.018, 0));
          const radUlnaCurve = new THREE.CatmullRomCurve3([
            subElbowPos.clone().add(new THREE.Vector3(side * boneOff * 0.02, 0, 0)),
            forearmMid,
            subWristPos.clone().add(new THREE.Vector3(side * boneOff * 0.022, 0, 0)),
          ]);
          const radUlnaGeo = trackGeo(
            new THREE.TubeGeometry(radUlnaCurve, seg(10, 8), 0.026 * lThick, seg(8, 6), false),
            0.004
          );
          torsoGroup.add(new THREE.Mesh(radUlnaGeo, ivoryFangMat));
        }

        // Carpal Wrist Cluster & 5 Articulated Metacarpal + Phalangeal Skeletal Claw Fingers
        const carpalGeo = trackGeo(
          new THREE.SphereGeometry(0.048 * lThick, seg(8, 6), seg(8, 6)),
          0.004
        );
        carpalGeo.scale(1.2, 0.65, 1.0);
        const carpalMesh = new THREE.Mesh(carpalGeo, ivoryFangMat);
        carpalMesh.position.copy(subWristPos);
        torsoGroup.add(carpalMesh);

        for (let f = 0; f < 5; f++) {
          const fFrac = (f - 2) * 0.026 * lThick;
          const fStart = subWristPos.clone().add(new THREE.Vector3(fFrac, -0.01, 0.03));
          const fMid = fStart.clone().add(new THREE.Vector3(fFrac * 0.6, -0.05, 0.08));
          const fTip = fMid.clone().add(new THREE.Vector3(0, -0.07, 0.05));
          const fingerCurve = new THREE.CatmullRomCurve3([fStart, fMid, fTip]);
          const fingerGeo = trackGeo(
            new THREE.TubeGeometry(fingerCurve, 6, 0.011 * lThick, 6, false),
            0.002
          );
          torsoGroup.add(new THREE.Mesh(fingerGeo, ivoryFangMat));
        }
      }
    }

    // 13B) Second Crowned Skeletal Skull on Left Shoulder (Twin-Skull Ettin Skeleton)
    if (arch === 'elite_twin_skull_ettin_skeleton') {
      const subSkullGroup = new THREE.Group();
      subSkullGroup.position.set(
        -0.28 * m.torsoBulk * m.shoulderWidth,
        torsoHeight + 0.14,
        0.08
      );
      subSkullGroup.rotation.set(0.06, -0.28, 0.16);
      torsoGroup.add(subSkullGroup);

      const subCraniumGeo = trackGeo(
        new THREE.SphereGeometry(0.18 * hScale, seg(14, 10), seg(12, 8)),
        0.01
      );
      subCraniumGeo.scale(0.92, 1.06, 1.12);
      subSkullGroup.add(new THREE.Mesh(subCraniumGeo, ivoryFangMat));

      const subJawGeo = trackGeo(
        new THREE.BoxGeometry(0.18 * hScale, 0.09 * hScale, 0.18 * hScale),
        0.006
      );
      const subJaw = new THREE.Mesh(subJawGeo, ivoryFangMat);
      subJaw.position.set(0, -0.14 * hScale, 0.08 * hScale);
      subSkullGroup.add(subJaw);

      for (const sSide of [-1, 1]) {
        const subEyeGeo = trackGeo(new THREE.SphereGeometry(0.032 * hScale, 8, 6));
        const subEye = new THREE.Mesh(subEyeGeo, emissiveRuneMat);
        subEye.position.set(sSide * 0.068 * hScale, -0.02 * hScale, 0.16 * hScale);
        subSkullGroup.add(subEye);
      }

      const subCrownGeo = trackGeo(
        new THREE.CylinderGeometry(0.16 * hScale, 0.15 * hScale, 0.09 * hScale, 12, 1, true),
        0.004
      );
      const subCrown = new THREE.Mesh(subCrownGeo, secondaryTrimMat);
      subCrown.position.set(0, 0.16 * hScale, 0.02 * hScale);
      subSkullGroup.add(subCrown);
    }

    // 14) Iron Prison Rib-Cage Bands & Dorsal Spiked Martyr Wheel
    if (arch === 'elite_caged_rib_jailer_skeleton') {
      for (let b = 0; b < 3; b++) {
        const cageRingGeo = trackGeo(
          new THREE.TorusGeometry(
            0.42 * m.torsoBulk * m.shoulderWidth,
            0.028 * m.torsoBulk,
            seg(8, 6),
            seg(20, 14)
          )
        );
        const cageRing = new THREE.Mesh(cageRingGeo, secondaryTrimMat);
        cageRing.position.set(0, (0.35 + b * 0.24) * torsoHeight, 0.04);
        cageRing.rotation.x = Math.PI * 0.5;
        torsoGroup.add(cageRing);
      }
    } else if (arch === 'elite_spiked_wheel_bone_martyr') {
      const wheelGeo = trackGeo(
        new THREE.TorusGeometry(0.68 * m.torsoBulk, 0.055 * m.torsoBulk, seg(10, 8), seg(24, 16))
      );
      const wheelMesh = new THREE.Mesh(wheelGeo, secondaryTrimMat);
      wheelMesh.position.set(0, torsoHeight * 0.72, -0.22 * m.torsoBulk);
      torsoGroup.add(wheelMesh);
      for (let sp = 0; sp < 8; sp++) {
        const ang = (sp / 8) * Math.PI * 2;
        const spokeGeo = trackGeo(
          new THREE.CylinderGeometry(0.022 * m.torsoBulk, 0.022 * m.torsoBulk, 1.36 * m.torsoBulk, 6)
        );
        const spoke = new THREE.Mesh(spokeGeo, ivoryFangMat);
        spoke.position.set(0, torsoHeight * 0.72, -0.22 * m.torsoBulk);
        spoke.rotation.z = ang;
        torsoGroup.add(spoke);
      }
    }

    // 15) Giant Bioluminescent Mushroom Cap & Spore Gills (Fungal Myconid Colossus)
    if (arch === 'elite_fungal_myconid_colossus') {
      const capGeo = trackGeo(
        new THREE.ConeGeometry(0.62 * hScale, 0.38 * hScale, seg(18, 12)),
        0.02
      );
      const capMesh = new THREE.Mesh(capGeo, secondaryTrimMat);
      capMesh.position.set(0, 0.36 * hScale, 0);
      headGroup.add(capMesh);
    }

    // 16) WORLD OF WARCRAFT ICONIC 3D ANATOMICAL & LEGENDARY SILHOUETTE FEATURES
    if (arch === 'wow_murloc_tidehunter') {
      // Bulbous Amphibious Murloc Eyes + Webbed Dorsal Fin Fan
      for (const side of [-1, 1]) {
        const bulbEyeGeo = trackGeo(
          new THREE.SphereGeometry(0.088 * hScale, seg(12, 8), seg(10, 8)),
          0.005
        );
        const bulbEye = new THREE.Mesh(bulbEyeGeo, emissiveRuneMat);
        bulbEye.position.set(side * 0.14 * hScale, 0.16 * hScale, 0.16 * hScale);
        headGroup.add(bulbEye);
      }
      for (let f = 0; f < 6; f++) {
        const u = f / 5;
        const finGeo = trackGeo(
          new THREE.ConeGeometry(0.045 * m.torsoBulk, (0.34 - u * 0.1) * m.dorsalScale, 6),
          0.006
        );
        const finMesh = new THREE.Mesh(finGeo, secondaryTrimMat);
        finMesh.position.set(0, (0.88 - u * 0.62) * torsoHeight, -0.22 * m.torsoBulk);
        finMesh.rotation.x = -0.82 - u * 0.25;
        torsoGroup.add(finMesh);
      }
    } else if (arch === 'wow_kobold_candle_geomancer') {
      // Iconic Tall Dripping Wax Candle & Glowing Flame on Kobold's Head ("You No Take Candle!")
      const candleGeo = trackGeo(
        new THREE.CylinderGeometry(0.065 * hScale, 0.082 * hScale, 0.34 * hScale, seg(12, 8)),
        0.008
      );
      const candleMesh = new THREE.Mesh(candleGeo, ivoryFangMat);
      candleMesh.position.set(0, 0.34 * hScale, 0.02 * hScale);
      headGroup.add(candleMesh);

      const flameGeo = trackGeo(
        new THREE.ConeGeometry(0.068 * hScale, 0.22 * hScale, seg(10, 8)),
        0.005
      );
      const flameMesh = new THREE.Mesh(flameGeo, emissiveRuneMat);
      flameMesh.position.set(0, 0.6 * hScale, 0.02 * hScale);
      headGroup.add(flameMesh);
    } else if (arch === 'wow_tauren_cairne_chieftain') {
      // Massive Carved Thunder Bluff Tauren Totem Trunk & Crossbeam Wing Totem on Back
      const totemTrunkGeo = trackGeo(
        new THREE.CylinderGeometry(
          0.16 * m.torsoBulk,
          0.19 * m.torsoBulk,
          1.45 * m.heightScale,
          seg(14, 10)
        ),
        0.015
      );
      const totemTrunk = new THREE.Mesh(totemTrunkGeo, secondaryTrimMat);
      totemTrunk.position.set(0, torsoHeight * 0.65, -0.34 * m.torsoBulk);
      torsoGroup.add(totemTrunk);

      const totemCrossGeo = trackGeo(
        new THREE.BoxGeometry(0.96 * m.shoulderWidth, 0.16 * m.torsoBulk, 0.16 * m.torsoBulk),
        0.01
      );
      const totemCross = new THREE.Mesh(totemCrossGeo, primaryArmorMat);
      totemCross.position.set(0, torsoHeight * 1.18, -0.34 * m.torsoBulk);
      torsoGroup.add(totemCross);
    } else if (arch === 'wow_orc_warsong_blademaster') {
      // Tall Warsong Clan Sashimono Battle Standard Pole & Fluttering Banner on Back
      const poleGeo = trackGeo(
        new THREE.CylinderGeometry(0.026, 0.032, 1.65 * m.heightScale, 8),
        0.004
      );
      const poleMesh = new THREE.Mesh(poleGeo, ivoryFangMat);
      poleMesh.position.set(0, torsoHeight * 0.92, -0.28 * m.torsoBulk);
      torsoGroup.add(poleMesh);

      const bannerGeo = trackGeo(
        new THREE.BoxGeometry(0.46 * m.dorsalScale, 0.78 * m.heightScale, 0.024),
        0.008
      );
      const bannerMesh = new THREE.Mesh(bannerGeo, secondaryTrimMat);
      bannerMesh.position.set(0.24 * m.dorsalScale, torsoHeight * 1.32, -0.28 * m.torsoBulk);
      torsoGroup.add(bannerMesh);
    } else if (arch === 'wow_illidan_betrayer_demonhunter') {
      // Glowing Fel-Green Demon Hunter Blindfold & Twin Curved Crescent Warglaives of Azzinoth
      const blindfoldGeo = trackGeo(
        new THREE.CylinderGeometry(
          0.245 * hScale,
          0.24 * hScale,
          0.11 * hScale,
          seg(16, 12),
          1,
          true
        ),
        0.004
      );
      const blindfoldMesh = new THREE.Mesh(blindfoldGeo, emissiveRuneMat);
      blindfoldMesh.position.set(0, 0.04 * hScale, 0.02 * hScale);
      headGroup.add(blindfoldMesh);

      for (const wSide of [-1, 1]) {
        const glaiveArcGeo = trackGeo(
          new THREE.TorusGeometry(
            0.58 * m.weaponScale,
            0.048 * m.weaponScale,
            seg(8, 6),
            seg(20, 14),
            Math.PI * 1.15
          ),
          0.006
        );
        glaiveArcGeo.scale(0.55, 1.25, 0.28);
        const glaiveBlade = new THREE.Mesh(glaiveArcGeo, emissiveRuneMat);
        glaiveBlade.rotation.y = Math.PI * 0.5;
        glaiveBlade.rotation.z = -Math.PI * 0.55;
        if (wSide === 1) {
          weaponGroup.add(glaiveBlade);
        } else {
          offhandGroup.add(glaiveBlade);
        }
      }
    } else if (arch === 'wow_arthas_lich_king') {
      // Iconic Tall Jagged Helm of Domination Spire Crown & Runic Brow Plate
      for (let sp = -2; sp <= 2; sp++) {
        const isCenter = sp === 0;
        const spireH = (isCenter ? 0.56 : 0.38 - Math.abs(sp) * 0.06) * hScale;
        const spireGeo = trackGeo(
          new THREE.ConeGeometry(0.042 * hScale, spireH, 6),
          0.004
        );
        const spireMesh = new THREE.Mesh(spireGeo, secondaryTrimMat);
        const ang = sp * 0.36;
        spireMesh.position.set(
          Math.sin(ang) * 0.21 * hScale,
          0.26 * hScale + spireH * 0.42,
          Math.cos(ang) * 0.21 * hScale
        );
        headGroup.add(spireMesh);
      }
    } else if (arch === 'wow_forsaken_royal_apothecary') {
      // Twin Glowing Blight Alchemy Cylinders Mounted on Skeletal Back
      for (const side of [-1, 1]) {
        const tankGeo = trackGeo(
          new THREE.CylinderGeometry(
            0.11 * m.torsoBulk,
            0.11 * m.torsoBulk,
            0.62 * m.heightScale,
            seg(12, 8)
          ),
          0.006
        );
        const tankMesh = new THREE.Mesh(tankGeo, emissiveRuneMat);
        tankMesh.position.set(
          side * 0.16 * m.torsoBulk,
          torsoHeight * 0.62,
          -0.24 * m.torsoBulk
        );
        torsoGroup.add(tankMesh);
      }
    } else if (
      arch === 'wow_shivarra_sixblade_priestess' ||
      arch === 'wow_lady_vashj_coilfang_matron'
    ) {
      // Multi-Armed Shivarra / Naga Matriarch Secondary & Tertiary Arms with Blades
      for (const side of [-1, 1]) {
        for (let pair = 0; pair < 2; pair++) {
          const yOff = torsoHeight * (0.66 - pair * 0.22);
          const armCurve = new THREE.CatmullRomCurve3([
            new THREE.Vector3(side * 0.28 * m.torsoBulk, yOff, 0),
            new THREE.Vector3(
              side * (0.62 + pair * 0.12) * m.torsoBulk,
              yOff - 0.12,
              0.16
            ),
            new THREE.Vector3(
              side * (0.78 + pair * 0.14) * m.torsoBulk,
              yOff + 0.08,
              0.42
            ),
          ]);
          const extraArmGeo = trackGeo(
            new THREE.TubeGeometry(armCurve, seg(12, 8), 0.058 * m.torsoBulk, seg(8, 6), false),
            0.008
          );
          torsoGroup.add(new THREE.Mesh(extraArmGeo, primaryArmorMat));

          const extraBladeGeo = trackGeo(
            new THREE.ConeGeometry(0.038 * m.weaponScale, 0.52 * m.weaponScale, 6),
            0.004
          );
          extraBladeGeo.rotateX(Math.PI * 0.35);
          const extraBlade = new THREE.Mesh(extraBladeGeo, secondaryTrimMat);
          extraBlade.position.set(
            side * (0.8 + pair * 0.14) * m.torsoBulk,
            yOff + 0.18,
            0.58
          );
          torsoGroup.add(extraBlade);
        }
      }
    } else if (
      arch === 'wow_goblin_steamwheedle_shredder' ||
      arch === 'wow_fel_reaver_construct'
    ) {
      // Twin Dorsal Furnace Exhaust Stacks & Circular Buzzsaw Blade
      for (const side of [-1, 1]) {
        const pipeGeo = trackGeo(
          new THREE.CylinderGeometry(
            0.068 * m.torsoBulk,
            0.055 * m.torsoBulk,
            0.68 * m.heightScale,
            10
          ),
          0.005
        );
        const pipeMesh = new THREE.Mesh(pipeGeo, secondaryTrimMat);
        pipeMesh.position.set(
          side * 0.24 * m.torsoBulk,
          torsoHeight * 0.88,
          -0.26 * m.torsoBulk
        );
        pipeMesh.rotation.x = -0.22;
        torsoGroup.add(pipeMesh);
      }
      const sawDiscGeo = trackGeo(
        new THREE.CylinderGeometry(0.38 * m.weaponScale, 0.38 * m.weaponScale, 0.035, 18),
        0.006
      );
      sawDiscGeo.rotateZ(Math.PI * 0.5);
      const sawDisc = new THREE.Mesh(sawDiscGeo, secondaryTrimMat);
      sawDisc.position.set(0, -0.18, 0.32);
      weaponGroup.add(sawDisc);
    } else if (arch === 'wow_kodo_ancient_warbeast') {
      // Twin Horde War Drums on Flanks + Massive Curved Kodo Nasal Horn
      for (const side of [-1, 1]) {
        const drumGeo = trackGeo(
          new THREE.CylinderGeometry(
            0.24 * m.torsoBulk,
            0.22 * m.torsoBulk,
            0.32 * m.torsoBulk,
            seg(14, 10)
          ),
          0.008
        );
        drumGeo.rotateZ(Math.PI * 0.5);
        const drumMesh = new THREE.Mesh(drumGeo, secondaryTrimMat);
        drumMesh.position.set(side * 0.42 * m.torsoBulk, 0.24, -0.12);
        pelvisGroup.add(drumMesh);
      }
      const kodoHornGeo = trackGeo(
        new THREE.ConeGeometry(0.095 * hScale, 0.52 * hScale, seg(10, 8)),
        0.008
      );
      kodoHornGeo.rotateX(Math.PI * 0.32);
      const kodoHorn = new THREE.Mesh(kodoHornGeo, ivoryFangMat);
      kodoHorn.position.set(0, 0.14 * hScale, (0.24 + m.snoutLength * 0.24) * hScale);
      headGroup.add(kodoHorn);
    } else if (arch === 'wow_faceless_kthir_general') {
      // Writhing N'raqi Facial Tentacles Hanging from the Lower Face
      for (let ten = -2; ten <= 2; ten++) {
        const tCurve = new THREE.CatmullRomCurve3([
          new THREE.Vector3(ten * 0.055 * hScale, -0.06 * hScale, 0.18 * hScale),
          new THREE.Vector3(ten * 0.085 * hScale, -0.26 * hScale, 0.28 * hScale),
          new THREE.Vector3(ten * 0.065 * hScale, -0.46 * hScale, 0.16 * hScale),
        ]);
        const tGeo = trackGeo(
          new THREE.TubeGeometry(tCurve, 10, 0.028 * hScale, 8, false),
          0.004
        );
        headGroup.add(new THREE.Mesh(tGeo, secondaryTrimMat));
      }
    }
  }

  // 13. UNIFIED CONTINUOUS SKIN (THREE.SkinnedMesh + THREE.Skeleton + Topology-Aware 4-Bone Weights)
  exportableGroup.updateMatrixWorld(true);

  const bones: THREE.Bone[] = [];
  const boneToIndex = new Map<THREE.Bone, number>();
  rootBone.traverse((obj) => {
    if ((obj as THREE.Bone).isBone) {
      const b = obj as THREE.Bone;
      boneToIndex.set(b, bones.length);
      bones.push(b);
    }
  });

  const boneWorldPos: THREE.Vector3[] = bones.map((b) => {
    b.userData.bindPos = b.position.clone();
    b.userData.bindQuat = b.quaternion.clone();
    b.userData.bindScale = b.scale.clone();
    return new THREE.Vector3().setFromMatrixPosition(b.matrixWorld);
  });

  // Build symmetrical anatomical joint adjacency map so parent & child bones share exact 50/50 weights at joints
  const neighborJoints: Array<Array<{ boneIdx: number; jointPos: THREE.Vector3; radius: number }>> =
    bones.map(() => []);

  const pelvisIdx = boneToIndex.get(pelvisGroup) ?? 0;
  const torsoIdx = boneToIndex.get(torsoGroup) ?? 0;
  const tailBaseIdx = boneToIndex.get(tailGroup) ?? 0;
  const tailTipIdx = boneToIndex.get(tailTipBone) ?? 0;

  for (let bIdx = 0; bIdx < bones.length; bIdx++) {
    const b = bones[bIdx];
    if (b === rootBone) continue;

    let anatomicalParent: THREE.Bone | null = null;
    if (b === leftLegGroup || b === rightLegGroup) {
      anatomicalParent = pelvisGroup;
    } else if (isQuadruped && !isCentaur && (b === leftArmGroup || b === rightArmGroup)) {
      anatomicalParent = torsoGroup;
    } else if (b.parent && (b.parent as THREE.Bone).isBone && b.parent !== rootBone) {
      anatomicalParent = b.parent as THREE.Bone;
    }

    if (anatomicalParent) {
      const pIdx = boneToIndex.get(anatomicalParent);
      if (pIdx !== undefined && pIdx !== bIdx) {
        const jPos = boneWorldPos[bIdx];
        const rad = Math.max(0.28, 0.42 * m.torsoBulk);
        neighborJoints[bIdx].push({ boneIdx: pIdx, jointPos: jPos, radius: rad });
        neighborJoints[pIdx].push({ boneIdx: bIdx, jointPos: jPos, radius: rad });
      }
    }

    // Also link Shoulder/Pauldron bones directly with Upper Arm bones for seamless shoulder-arm-torso deformation
    if (b === leftArmGroup || b === rightArmGroup) {
      const shBone = b === leftArmGroup ? leftShoulderGroup : rightShoulderGroup;
      const shIdx = boneToIndex.get(shBone);
      if (shIdx !== undefined && shIdx !== bIdx) {
        const jPos = boneWorldPos[bIdx];
        const rad = Math.max(0.26, 0.38 * m.torsoBulk);
        neighborJoints[bIdx].push({ boneIdx: shIdx, jointPos: jPos, radius: rad });
        neighborJoints[shIdx].push({ boneIdx: bIdx, jointPos: jPos, radius: rad });
      }
    }
  }

  // Collect all organic body meshes to weld into unified SkinnedMeshes (keep weapons, shields, crystal core & breath cone separate)
  const excludedAncestors = new Set<THREE.Object3D>([
    weaponGroup,
    offhandGroup,
    coreMesh,
    breathConeMesh,
  ]);
  const isExcludedMesh = (obj: THREE.Object3D): boolean => {
    let cur: THREE.Object3D | null = obj;
    while (cur) {
      if (excludedAncestors.has(cur)) return true;
      cur = cur.parent;
    }
    return false;
  };

  const findHomeBone = (obj: THREE.Object3D): THREE.Bone => {
    let cur: THREE.Object3D | null = obj.parent;
    while (cur) {
      if ((cur as THREE.Bone).isBone) return cur as THREE.Bone;
      cur = cur.parent;
    }
    return pelvisGroup;
  };

  const explodedSourceMeshes: THREE.Mesh[] = [];
  const matChunksMap = new Map<THREE.Material, THREE.BufferGeometry[]>();
  const _vPos = new THREE.Vector3();

  rootBone.traverse((obj) => {
    const mesh = obj as THREE.Mesh;
    if (!mesh.isMesh || !mesh.geometry || isExcludedMesh(mesh)) return;

    const homeBone = findHomeBone(mesh);
    const homeBoneIdx = boneToIndex.get(homeBone) ?? pelvisIdx;

    // Clone indexed geometry and preserve smooth analytical vertex normals!
    // Never call toNonIndexed() + computeVertexNormals(), which turns smooth surfaces into flat-faceted triangles.
    const chunkGeo = mesh.geometry.clone();
    if (!chunkGeo.getAttribute('normal')) {
      chunkGeo.computeVertexNormals();
    }
    chunkGeo.applyMatrix4(mesh.matrixWorld);

    const posAttr = chunkGeo.getAttribute('position');
    const vCount = posAttr.count;

    if (!chunkGeo.getAttribute('uv')) {
      chunkGeo.setAttribute('uv', new THREE.BufferAttribute(new Float32Array(vCount * 2), 2));
    }
    if (!chunkGeo.getAttribute('color')) {
      const fallbackColors = new Float32Array(vCount * 3);
      fallbackColors.fill(1.0);
      chunkGeo.setAttribute('color', new THREE.BufferAttribute(fallbackColors, 3));
    }

    const skinIndices = new Uint16Array(vCount * 4);
    const skinWeights = new Float32Array(vCount * 4);

    const isTailMesh = homeBone === tailGroup || homeBone === tailTipBone;
    const tailRootPos = boneWorldPos[tailBaseIdx];
    const tailSpan = Math.max(0.28, tailRootPos.distanceTo(boneWorldPos[tailTipIdx]) * 1.9);

    const nList = neighborJoints[homeBoneIdx];

    for (let v = 0; v < vCount; v++) {
      _vPos.set(posAttr.getX(v), posAttr.getY(v), posAttr.getZ(v));

      if (isTailMesh) {
        // Continuous 3-Bone Spine Spline Weighting along the Tail (Pelvis -> TailBase -> TailTip)
        const u = Math.min(1, Math.max(0, _vPos.distanceTo(tailRootPos) / tailSpan));
        const wPelvis = Math.max(0, 0.46 * (1 - u * 3.1));
        const wBase = Math.max(0.08, 1.0 - Math.abs(u - 0.34) * 1.6);
        const wTip = Math.max(0, (u - 0.2) * 1.4);
        const sum = wPelvis + wBase + wTip;

        skinIndices[v * 4] = tailBaseIdx;
        skinIndices[v * 4 + 1] = tailTipIdx;
        skinIndices[v * 4 + 2] = pelvisIdx;
        skinIndices[v * 4 + 3] = 0;

        skinWeights[v * 4] = wBase / sum;
        skinWeights[v * 4 + 1] = wTip / sum;
        skinWeights[v * 4 + 2] = wPelvis / sum;
        skinWeights[v * 4 + 3] = 0;
      } else {
        // Topology-Aware Multi-Bone Joint Blending (exact 50/50 convergence at joint centers!)
        const wHome = 0.5;
        let b1 = 0,
          w1 = 0;
        let b2 = 0,
          w2 = 0;
        let b3 = 0,
          w3 = 0;

        for (let k = 0; k < nList.length; k++) {
          const nj = nList[k];
          const dist = _vPos.distanceTo(nj.jointPos);
          const ratio = dist / nj.radius;
          if (ratio < 2.2) {
            const inf = 0.5 * Math.exp(-ratio * ratio * 2.2);
            if (inf > w1) {
              b3 = b2;
              w3 = w2;
              b2 = b1;
              w2 = w1;
              b1 = nj.boneIdx;
              w1 = inf;
            } else if (inf > w2) {
              b3 = b2;
              w3 = w2;
              b2 = nj.boneIdx;
              w2 = inf;
            } else if (inf > w3) {
              b3 = nj.boneIdx;
              w3 = inf;
            }
          }
        }

        // Also blend serpentine lower body smoothly between Pelvis_Chassis and Articulated_Tail
        if (isSerpentine && homeBone === pelvisGroup && _vPos.z < -0.15) {
          const serpInf = Math.min(0.75, Math.abs(_vPos.z + 0.15) * 0.85);
          b1 = tailBaseIdx;
          w1 = serpInf;
        }

        const totalW = wHome + w1 + w2 + w3;
        skinIndices[v * 4] = homeBoneIdx;
        skinIndices[v * 4 + 1] = b1;
        skinIndices[v * 4 + 2] = b2;
        skinIndices[v * 4 + 3] = b3;

        skinWeights[v * 4] = wHome / totalW;
        skinWeights[v * 4 + 1] = w1 / totalW;
        skinWeights[v * 4 + 2] = w2 / totalW;
        skinWeights[v * 4 + 3] = w3 / totalW;
      }
    }

    chunkGeo.setAttribute('skinIndex', new THREE.Uint16BufferAttribute(skinIndices, 4));
    chunkGeo.setAttribute('skinWeight', new THREE.Float32BufferAttribute(skinWeights, 4));

    const mat = mesh.material as THREE.Material;
    const list = matChunksMap.get(mat) || [];
    list.push(chunkGeo);
    matChunksMap.set(mat, list);

    // Hide rigid source mesh during normal SkinnedMesh view & .GLB export (only shown when Exploded View > 0.02)
    mesh.visible = false;
    mesh.userData.isExplodedSourceMesh = true;
    explodedSourceMeshes.push(mesh);
  });

  // Helper to concatenate indexed BufferGeometries with skinIndex & skinWeight while preserving smooth normals
  const mergeSkinnedChunks = (
    chunks: THREE.BufferGeometry[],
    skipSculptSmoothing = false
  ): THREE.BufferGeometry => {
    let totalVerts = 0;
    let totalIndices = 0;
    for (let i = 0; i < chunks.length; i++) {
      const c = chunks[i];
      const vCount = c.getAttribute('position').count;
      totalVerts += vCount;
      const idxAttr = c.getIndex();
      totalIndices += idxAttr ? idxAttr.count : vCount;
    }

    const mergedPos = new Float32Array(totalVerts * 3);
    const mergedNorm = new Float32Array(totalVerts * 3);
    const mergedUv = new Float32Array(totalVerts * 2);
    const mergedCol = new Float32Array(totalVerts * 3);
    const mergedSkinIdx = new Uint16Array(totalVerts * 4);
    const mergedSkinW = new Float32Array(totalVerts * 4);
    const mergedIndices =
      totalVerts > 65535 ? new Uint32Array(totalIndices) : new Uint16Array(totalIndices);

    let vOffset = 0;
    let iOffset = 0;
    for (let i = 0; i < chunks.length; i++) {
      const c = chunks[i];
      const count = c.getAttribute('position').count;
      mergedPos.set(c.getAttribute('position').array as Float32Array, vOffset * 3);
      mergedNorm.set(c.getAttribute('normal').array as Float32Array, vOffset * 3);
      mergedUv.set(c.getAttribute('uv').array as Float32Array, vOffset * 2);
      mergedCol.set(c.getAttribute('color').array as Float32Array, vOffset * 3);
      mergedSkinIdx.set(c.getAttribute('skinIndex').array as Uint16Array, vOffset * 4);
      mergedSkinW.set(c.getAttribute('skinWeight').array as Float32Array, vOffset * 4);

      const idxAttr = c.getIndex();
      if (idxAttr) {
        const idxArr = idxAttr.array;
        for (let k = 0; k < idxAttr.count; k++) {
          mergedIndices[iOffset + k] = idxArr[k] + vOffset;
        }
        iOffset += idxAttr.count;
      } else {
        for (let k = 0; k < count; k++) {
          mergedIndices[iOffset + k] = vOffset + k;
        }
        iOffset += count;
      }

      vOffset += count;
      c.dispose();
    }

    const out = new THREE.BufferGeometry();
    out.setIndex(new THREE.BufferAttribute(mergedIndices, 1));
    out.setAttribute('position', new THREE.BufferAttribute(mergedPos, 3));
    out.setAttribute('normal', new THREE.BufferAttribute(mergedNorm, 3));
    out.setAttribute('uv', new THREE.BufferAttribute(mergedUv, 2));
    out.setAttribute('color', new THREE.BufferAttribute(mergedCol, 3));
    out.setAttribute('skinIndex', new THREE.Uint16BufferAttribute(mergedSkinIdx, 4));
    out.setAttribute('skinWeight', new THREE.Float32BufferAttribute(mergedSkinW, 4));

    if (!skipSculptSmoothing) {
      // Apply Blender-style Taubin (\lambda-\mu) volume-preserving sculpt smoothing
      // and seam-welded vertex normal averaging so UV seams & anatomical junctions are seamless
      applyBlenderStyleSculptSmoothing(out, 2);
    }

    out.computeBoundingBox();
    out.computeBoundingSphere();
    return out;
  };

  /**
   * Simulates Blender's "Corrective Smooth" (Taubin \lambda-\mu volume-preserving filter)
   * + "Weighted / Seam-Welded Auto Smooth Normals" directly in the browser and .GLB export.
   * Welds coincident UV-seam vertices in a spatial hash so seams never split during smoothing.
   */
  function applyBlenderStyleSculptSmoothing(geo: THREE.BufferGeometry, iterations = 2): void {
    const posAttr = geo.getAttribute('position') as THREE.BufferAttribute;
    const normAttr = geo.getAttribute('normal') as THREE.BufferAttribute;
    const idxAttr = geo.getIndex();
    if (!posAttr || !idxAttr || posAttr.count === 0) return;

    const vCount = posAttr.count;
    const posArr = posAttr.array as Float32Array;
    const indices = idxAttr.array;

    // 1. Group spatially coincident vertices (e.g., along UV seams) into canonical clusters
    const quantScale = 400; // 2.5mm tolerance
    const clusterMap = new Map<string, number>();
    const vertToCluster = new Int32Array(vCount);
    const clusterVerts: number[][] = [];
    const clusterPos: number[] = []; // x, y, z per cluster

    for (let v = 0; v < vCount; v++) {
      const x = posArr[v * 3];
      const y = posArr[v * 3 + 1];
      const z = posArr[v * 3 + 2];
      const key = `${Math.round(x * quantScale)},${Math.round(y * quantScale)},${Math.round(z * quantScale)}`;
      let cIdx = clusterMap.get(key);
      if (cIdx === undefined) {
        cIdx = clusterVerts.length;
        clusterMap.set(key, cIdx);
        clusterVerts.push([v]);
        clusterPos.push(x, y, z);
      } else {
        clusterVerts[cIdx].push(v);
      }
      vertToCluster[v] = cIdx;
    }

    const cCount = clusterVerts.length;
    const neighbors: Set<number>[] = Array.from({ length: cCount }, () => new Set<number>());

    for (let i = 0; i < indices.length; i += 3) {
      const c0 = vertToCluster[indices[i]];
      const c1 = vertToCluster[indices[i + 1]];
      const c2 = vertToCluster[indices[i + 2]];
      if (c0 !== c1) {
        neighbors[c0].add(c1);
        neighbors[c1].add(c0);
      }
      if (c1 !== c2) {
        neighbors[c1].add(c2);
        neighbors[c2].add(c1);
      }
      if (c2 !== c0) {
        neighbors[c2].add(c0);
        neighbors[c0].add(c2);
      }
    }

    // 2. Taubin volume-preserving smoothing (\lambda = +0.32 shrink, \mu = -0.335 inflate)
    // Preserves sharp claw/horn tips (valence <= 2) while smoothing organic muscle/limb surfaces
    const tempPos = new Float32Array(cCount * 3);
    const curPos = new Float32Array(clusterPos);

    const runLaplacianStep = (factor: number) => {
      tempPos.set(curPos);
      for (let c = 0; c < cCount; c++) {
        const nSet = neighbors[c];
        const deg = nSet.size;
        // Skip isolated tips or low-valence spike tips so horns/claws remain sharp
        if (deg < 3) continue;

        let avgX = 0;
        let avgY = 0;
        let avgZ = 0;
        nSet.forEach((nIdx) => {
          avgX += curPos[nIdx * 3];
          avgY += curPos[nIdx * 3 + 1];
          avgZ += curPos[nIdx * 3 + 2];
        });
        avgX /= deg;
        avgY /= deg;
        avgZ /= deg;

        const cx = curPos[c * 3];
        const cy = curPos[c * 3 + 1];
        const cz = curPos[c * 3 + 2];

        // Protect ground contact feet (y < 0.03) from vertical drift
        const yWeight = cy < 0.035 ? 0.15 : 1.0;

        tempPos[c * 3] = cx + (avgX - cx) * factor;
        tempPos[c * 3 + 1] = cy + (avgY - cy) * factor * yWeight;
        tempPos[c * 3 + 2] = cz + (avgZ - cz) * factor;
      }
      curPos.set(tempPos);
    };

    for (let iter = 0; iter < iterations; iter++) {
      runLaplacianStep(0.32);   // Smooth pass (\lambda)
      runLaplacianStep(-0.334); // Volume-restoring inflate pass (\mu)
    }

    // Write smoothed positions back to all split UV vertices in each cluster
    for (let c = 0; c < cCount; c++) {
      const cx = curPos[c * 3];
      const cy = curPos[c * 3 + 1];
      const cz = curPos[c * 3 + 2];
      const verts = clusterVerts[c];
      for (let k = 0; k < verts.length; k++) {
        const v = verts[k];
        posArr[v * 3] = cx;
        posArr[v * 3 + 1] = cy;
        posArr[v * 3 + 2] = cz;
      }
    }
    posAttr.needsUpdate = true;

    // 3. Recompute face normals & average across coincident UV-seam clusters for crease-free shading
    geo.computeVertexNormals();
    if (normAttr) {
      const normArr = normAttr.array as Float32Array;
      for (let c = 0; c < cCount; c++) {
        const verts = clusterVerts[c];
        if (verts.length <= 1) continue;
        let nx = 0;
        let ny = 0;
        let nz = 0;
        for (let k = 0; k < verts.length; k++) {
          const v = verts[k];
          nx += normArr[v * 3];
          ny += normArr[v * 3 + 1];
          nz += normArr[v * 3 + 2];
        }
        const len = Math.hypot(nx, ny, nz);
        if (len > 1e-6) {
          nx /= len;
          ny /= len;
          nz /= len;
          for (let k = 0; k < verts.length; k++) {
            const v = verts[k];
            normArr[v * 3] = nx;
            normArr[v * 3 + 1] = ny;
            normArr[v * 3 + 2] = nz;
          }
        }
      }
      normAttr.needsUpdate = true;
    }
  }

  /**
   * Simulates Blender's "Displace Modifier (Texture Coordinates: UV)" on the welded skin mesh:
   * Evaluates `evaluateDragonScaleUVField(u, v)` at each seam-welded vertex cluster and displaces
   * vertices outward along their smooth normals on `isProtruding` scales and inward at scale crevices/recessed scales.
   */
  function applyBlenderUVScaleDisplacement(
    geo: THREE.BufferGeometry,
    textureRelief: number,
    torsoBulk: number
  ): void {
    const posAttr = geo.getAttribute('position') as THREE.BufferAttribute;
    const normAttr = geo.getAttribute('normal') as THREE.BufferAttribute;
    const uvAttr = geo.getAttribute('uv') as THREE.BufferAttribute;
    const colAttr = geo.getAttribute('color') as THREE.BufferAttribute;
    if (!posAttr || !normAttr || !uvAttr || posAttr.count === 0) return;

    const vCount = posAttr.count;
    const posArr = posAttr.array as Float32Array;
    const normArr = normAttr.array as Float32Array;
    const uvArr = uvAttr.array as Float32Array;
    const colArr = colAttr ? (colAttr.array as Float32Array) : null;

    // Group coincident UV-seam vertices into welded clusters so UV displacement never splits seams
    const quantScale = 400;
    const clusterMap = new Map<string, number>();
    const clusterVerts: number[][] = [];

    for (let v = 0; v < vCount; v++) {
      const x = posArr[v * 3];
      const y = posArr[v * 3 + 1];
      const z = posArr[v * 3 + 2];
      const key = `${Math.round(x * quantScale)},${Math.round(y * quantScale)},${Math.round(z * quantScale)}`;
      let cIdx = clusterMap.get(key);
      if (cIdx === undefined) {
        cIdx = clusterVerts.length;
        clusterMap.set(key, cIdx);
        clusterVerts.push([v]);
      } else {
        clusterVerts[cIdx].push(v);
      }
    }

    const dispAmp =
      0.0095 *
      Math.pow(Math.max(0.6, torsoBulk), 0.45) *
      (0.55 + 0.65 * Math.max(0.3, textureRelief));

    for (let c = 0; c < clusterVerts.length; c++) {
      const verts = clusterVerts[c];
      const v0 = verts[0];
      const cy = posArr[v0 * 3 + 1];
      // Protect ground-contact soles/claws from vertical drift
      if (cy < 0.085) continue;

      let avgDisp = 0;
      let avgShade = 0;
      let nx = 0;
      let ny = 0;
      let nz = 0;

      for (let k = 0; k < verts.length; k++) {
        const v = verts[k];
        const uu = uvArr[v * 2];
        const vv = uvArr[v * 2 + 1];
        const sample = evaluateDragonScaleUVField(uu, vv);
        // Smoothly taper displacement to 0 at primitive UV seams (u near 0/1) & polar caps (v near 0/1) so seam axes never form lines
        const uFade = Math.min(1, Math.min(uu, 1 - uu) * 14);
        const vFade = Math.min(1, Math.min(vv, 1 - vv) * 10);
        const seamWeight = Math.max(0, uFade * vFade);
        avgDisp += sample.displacement * seamWeight;
        avgShade += 1 + (sample.colorShade - 1) * seamWeight;
        nx += normArr[v * 3];
        ny += normArr[v * 3 + 1];
        nz += normArr[v * 3 + 2];
      }

      avgDisp /= verts.length;
      avgShade /= verts.length;
      const nLen = Math.hypot(nx, ny, nz);
      if (nLen < 1e-6) continue;
      nx /= nLen;
      ny /= nLen;
      nz /= nLen;

      const offset = avgDisp * dispAmp;
      for (let k = 0; k < verts.length; k++) {
        const v = verts[k];
        posArr[v * 3] += nx * offset;
        posArr[v * 3 + 1] += ny * offset;
        posArr[v * 3 + 2] += nz * offset;

        if (colArr) {
          colArr[v * 3] = Math.min(1.35, colArr[v * 3] * avgShade);
          colArr[v * 3 + 1] = Math.min(1.35, colArr[v * 3 + 1] * avgShade);
          colArr[v * 3 + 2] = Math.min(1.35, colArr[v * 3 + 2] * avgShade);
        }
      }
    }

    posAttr.needsUpdate = true;
    if (colAttr) colAttr.needsUpdate = true;

    // Recompute & seam-weld normals across displaced scale domes and crevices
    geo.computeVertexNormals();
    for (let c = 0; c < clusterVerts.length; c++) {
      const verts = clusterVerts[c];
      if (verts.length <= 1) continue;
      let nx = 0, ny = 0, nz = 0;
      for (let k = 0; k < verts.length; k++) {
        const v = verts[k];
        nx += normArr[v * 3];
        ny += normArr[v * 3 + 1];
        nz += normArr[v * 3 + 2];
      }
      const len = Math.hypot(nx, ny, nz);
      if (len > 1e-6) {
        nx /= len;
        ny /= len;
        nz /= len;
        for (let k = 0; k < verts.length; k++) {
          const v = verts[k];
          normArr[v * 3] = nx;
          normArr[v * 3 + 1] = ny;
          normArr[v * 3 + 2] = nz;
        }
      }
    }
    normAttr.needsUpdate = true;
    geo.computeBoundingBox();
    geo.computeBoundingSphere();
  }

  const sharedSkeleton = new THREE.Skeleton(bones);
  const skinnedMeshGroup = new THREE.Group();
  skinnedMeshGroup.name = 'Unified_SkinnedMesh_Group';
  exportableGroup.add(skinnedMeshGroup);

  const skinnedMeshes: THREE.SkinnedMesh[] = [];
  const resolvePartByPoint = (worldPoint: THREE.Vector3): PartInspectionInfo | undefined => {
    let bestInfo: PartInspectionInfo | undefined;
    let bestDistSq = Infinity;
    const tmpW = new THREE.Vector3();
    for (let i = 0; i < explodableParts.length; i++) {
      const p = explodableParts[i];
      const info = p.userData?.partInfo as PartInspectionInfo | undefined;
      if (!info) continue;
      p.getWorldPosition(tmpW);
      const dSq = tmpW.distanceToSquared(worldPoint);
      if (dSq < bestDistSq) {
        bestDistSq = dSq;
        bestInfo = info;
      }
    }
    return bestInfo;
  };

  const has3DDragonScaleRelief =
    !isBeholder &&
    !isUndeadSkeleton &&
    (isDragon ||
      isSerpentine ||
      config.raceCategory === 'dragons' ||
      getEffectiveTexturePattern(config) === 'scales');

  let matIdx = 0;
  let primaryMergedBodyGeo: THREE.BufferGeometry | null = null;
  let secondaryMergedBodyGeo: THREE.BufferGeometry | null = null;

  matChunksMap.forEach((chunks, mat) => {
    if (chunks.length === 0 || mat === hairStrandMat) return;
    const isCrispBoneOrToothMat =
      mat === ivoryFangMat ||
      mat.name === 'Mat_ToothEnamel' ||
      mat.name === 'Mat_SculptedCraniumBone' ||
      mat.name === 'Mat_SkullSocketVoid';
    const mergedGeo = mergeSkinnedChunks(chunks, isCrispBoneOrToothMat);
    geometriesToDispose.push(mergedGeo);

    if (mat === primaryArmorMat) {
      if (has3DDragonScaleRelief) {
        // Apply Blender-style UV-driven Displace Modifier before spawning UV-locked 3D scale hoods
        applyBlenderUVScaleDisplacement(mergedGeo, matCfg.textureRelief ?? 0.85, m.torsoBulk);
      }
      primaryMergedBodyGeo = mergedGeo;
    }
    if (mat === secondaryTrimMat) secondaryMergedBodyGeo = mergedGeo;

    const sMesh = new THREE.SkinnedMesh(mergedGeo, mat);
    sMesh.name = `Skinned_Unified_${mat.name || matIdx++}`;
    sMesh.castShadow = true;
    sMesh.receiveShadow = true;
    sMesh.frustumCulled = false;
    sMesh.layers.enable(1);
    sMesh.userData.resolvePartByPoint = resolvePartByPoint;

    sMesh.bind(sharedSkeleton, new THREE.Matrix4());
    skinnedMeshGroup.add(sMesh);
    skinnedMeshes.push(sMesh);
  });

  // 13B. PROCEDURAL 3D MULTI-STRAND HAIR & PELAGE COAT GENERATOR (Thousands of Curved 3D Hair Strands + Flowing Mane / Scalp Hair)
  if (hairStyle !== 'none' && furDensity > 0.05 && !isBeholder) {
    const hairChunks = matChunksMap.get(hairStrandMat) || [];
    const bodyGeosForHair: THREE.BufferGeometry[] = [];
    if (primaryMergedBodyGeo) bodyGeosForHair.push(primaryMergedBodyGeo);
    if (secondaryMergedBodyGeo) bodyGeosForHair.push(secondaryMergedBodyGeo);

    // Collect all head bones (supports 1, 2, or 3 heads for Cerberus / Chimera!)
    const allHeadBones: THREE.Bone[] = [headGroup];
    for (let eh = 0; eh < extraHeads.length; eh++) {
      extraHeads[eh].traverse((c) => {
        if ((c as THREE.Bone).isBone && c.name.startsWith('Head_Helm_Assembly')) {
          allHeadBones.push(c as THREE.Bone);
        }
      });
    }

    const strandCoatGeo = buildSkinned3DHairCoatGeometry({
      bodyGeos: bodyGeosForHair,
      hairStyle,
      furDensity,
      hairLength,
      lodMult,
      torsoBulk: m.torsoBulk,
      heightScale: m.heightScale,
      pelvisY,
      headBones: allHeadBones,
      neckIdx: boneToIndex.get(neckGroup) ?? torsoIdx,
      torsoIdx,
      pelvisIdx,
      tailBaseIdx,
      tailTipIdx,
      boneToIndex,
      boneWorldPos,
    });

    if (strandCoatGeo) {
      hairChunks.push(strandCoatGeo);
    }

    if (hairChunks.length > 0) {
      const mergedHairGeo = mergeSkinnedChunks(hairChunks, true);
      geometriesToDispose.push(mergedHairGeo);

      const hairSkinnedMesh = new THREE.SkinnedMesh(mergedHairGeo, hairStrandMat);
      hairSkinnedMesh.name = 'Skinned_3D_Hair_Strands_Coat';
      hairSkinnedMesh.castShadow = true;
      hairSkinnedMesh.receiveShadow = true;
      hairSkinnedMesh.frustumCulled = false;
      hairSkinnedMesh.layers.enable(1);
      hairSkinnedMesh.userData.resolvePartByPoint = resolvePartByPoint;

      hairSkinnedMesh.bind(sharedSkeleton, new THREE.Matrix4());
      skinnedMeshGroup.add(hairSkinnedMesh);
      skinnedMeshes.push(hairSkinnedMesh);
    }
  } else {
    const unusedHairChunks = matChunksMap.get(hairStrandMat);
    if (unusedHairChunks) {
      unusedHairChunks.forEach((c) => c.dispose());
    }
  }

  // 13C. PROCEDURAL 3D UV-LOCKED DRAGON OSTEODERM SCALE RELIEF (Matched 1:1 with Drawn Scale Texture)
  if (has3DDragonScaleRelief && primaryMergedBodyGeo) {
    const dragonScaleReliefGeo = buildSkinned3DDragonScaleReliefGeometry({
      bodyGeo: primaryMergedBodyGeo,
      lodMult,
      torsoBulk: m.torsoBulk,
      dorsalScale: m.dorsalScale,
      textureRelief: matCfg.textureRelief ?? 0.85,
      isDragon,
      tailBaseIdx,
      tailTipIdx,
      neckIdx: boneToIndex.get(neckGroup) ?? torsoIdx,
      boneWorldPos,
    });

    if (dragonScaleReliefGeo) {
      geometriesToDispose.push(dragonScaleReliefGeo);

      const scaleReliefSkinnedMesh = new THREE.SkinnedMesh(
        dragonScaleReliefGeo,
        primaryArmorMat
      );
      scaleReliefSkinnedMesh.name = 'Skinned_3D_Dragon_Scale_Relief';
      scaleReliefSkinnedMesh.castShadow = true;
      scaleReliefSkinnedMesh.receiveShadow = true;
      scaleReliefSkinnedMesh.frustumCulled = false;
      scaleReliefSkinnedMesh.layers.enable(1);
      scaleReliefSkinnedMesh.userData.resolvePartByPoint = resolvePartByPoint;

      scaleReliefSkinnedMesh.bind(sharedSkeleton, new THREE.Matrix4());
      skinnedMeshGroup.add(scaleReliefSkinnedMesh);
      skinnedMeshes.push(scaleReliefSkinnedMesh);
    }
  }

  // 14. HITBOXES & AGGRO RANGE VISUALIZER GROUP (Zero AxesHelper or SkeletonHelper lines bleeding through the model)
  const hitboxGroup = new THREE.Group();
  hitboxGroup.name = 'Tactical_Hitboxes_And_Sockets';
  rootGroup.add(hitboxGroup);

  const hitboxWireMat = trackMat(
    new THREE.MeshBasicMaterial({
      color: 0x10b981,
      wireframe: true,
      transparent: true,
      opacity: 0.45,
      depthWrite: false,
    })
  );
  const headHitboxMat = trackMat(
    new THREE.MeshBasicMaterial({
      color: 0xf43f5e,
      wireframe: true,
      transparent: true,
      opacity: 0.65,
      depthWrite: false,
    })
  );

  const torsoColGeo = trackGeo(
    new THREE.CylinderGeometry(
      config.stats.torsoHitboxRadius,
      config.stats.torsoHitboxRadius,
      config.stats.torsoHitboxHeight,
      12,
      1,
      true
    ),
    0
  );
  const torsoColMesh = new THREE.Mesh(torsoColGeo, hitboxWireMat);
  torsoColMesh.position.set(0, config.stats.torsoHitboxHeight * 0.5 + 0.25, 0);
  hitboxGroup.add(torsoColMesh);

  const headColGeo = trackGeo(new THREE.SphereGeometry(config.stats.headHitboxRadius, 10, 8), 0);
  const headColMesh = new THREE.Mesh(headColGeo, headHitboxMat);
  const headWorldPos = new THREE.Vector3();
  socketHead.getWorldPosition(headWorldPos);
  headColMesh.position.copy(headWorldPos);
  hitboxGroup.add(headColMesh);

  const rangeRingGeo = trackGeo(
    new THREE.RingGeometry(
      Math.max(0.5, config.stats.attackRange * 0.35 - 0.04),
      config.stats.attackRange * 0.35,
      36
    ),
    0
  );
  rangeRingGeo.rotateX(-Math.PI / 2);
  const rangeRingMat = trackMat(
    new THREE.MeshBasicMaterial({
      color: 0xf59e0b,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.65,
      depthWrite: false,
      polygonOffset: true,
      polygonOffsetFactor: -2,
      polygonOffsetUnits: -2,
    })
  );
  const rangeRingMesh = new THREE.Mesh(rangeRingGeo, rangeRingMat);
  rangeRingMesh.position.y = 0.03;
  hitboxGroup.add(rangeRingMesh);

  // 15. PROCEDURAL AURA PARTICLE EMITTER (Volcanic Smoke & Arcane Sparks emanating from Aura Anchor Points)
  const PARTICLE_COUNT = isSoftwareGl ? 72 : 160;
  const particleGeo = new THREE.BufferGeometry();
  geometriesToDispose.push(particleGeo);
  const particlePositions = new Float32Array(PARTICLE_COUNT * 3);
  const particleColors = new Float32Array(PARTICLE_COUNT * 3);
  particleGeo.setAttribute('position', new THREE.BufferAttribute(particlePositions, 3));
  particleGeo.setAttribute('color', new THREE.BufferAttribute(particleColors, 3));

  const smokeMat = trackMat(
    new THREE.PointsMaterial({
      name: 'Mat_Particles_VolcanicSmoke',
      map: textures.particleSmokeMap,
      size: 0.36 * m.heightScale,
      sizeAttenuation: true,
      transparent: true,
      opacity: 0.82,
      depthWrite: false,
      vertexColors: true,
      blending: THREE.NormalBlending,
    })
  );

  const sparkMat = trackMat(
    new THREE.PointsMaterial({
      name: 'Mat_Particles_ArcaneSparks',
      map: textures.particleSparkMap,
      size: 0.24 * m.heightScale,
      sizeAttenuation: true,
      transparent: true,
      opacity: 0.96,
      depthWrite: false,
      vertexColors: true,
      blending: THREE.AdditiveBlending,
    })
  );

  const auraParticlePoints = new THREE.Points(particleGeo, sparkMat);
  auraParticlePoints.name = 'Procedural_Aura_Particles';
  auraParticlePoints.frustumCulled = false;
  auraParticlePoints.visible = false;
  rootGroup.add(auraParticlePoints);

  const totalAnchors = shardData.length + 4; // Orbiting Aura Shards + 4 Anatomical Aura Sockets (Core, Head, WeaponR, ShieldL)
  const particlePool: BuiltEnemyRig['auraParticleSystem']['pool'] = [];
  for (let i = 0; i < PARTICLE_COUNT; i++) {
    const lifetime = 0.8 + (i % 11) * 0.11;
    particlePool.push({
      age: (i / PARTICLE_COUNT) * lifetime, // Staggered initial ages for continuous steady emission
      lifetime,
      anchorIndex: i % Math.max(1, totalAnchors),
      vx: ((i * 73) % 100 - 50) * 0.005,
      vy: 0.45 + ((i * 37) % 100) * 0.006,
      vz: ((i * 53) % 100 - 50) * 0.005,
      seed: i * 1.6180339,
    });
  }

  // 15B. REAL-TIME VOLUMETRIC MOUTH FIRE, FROST & BITE-SPARK PARTICLE STREAM
  const BREATH_PARTICLE_COUNT = isSoftwareGl ? 96 : 220;
  const breathParticleGeo = new THREE.BufferGeometry();
  geometriesToDispose.push(breathParticleGeo);
  const breathPositions = new Float32Array(BREATH_PARTICLE_COUNT * 3);
  const breathColors = new Float32Array(BREATH_PARTICLE_COUNT * 3);
  breathParticleGeo.setAttribute('position', new THREE.BufferAttribute(breathPositions, 3));
  breathParticleGeo.setAttribute('color', new THREE.BufferAttribute(breathColors, 3));

  const breathFireMat = trackMat(
    new THREE.PointsMaterial({
      name: 'Mat_MouthVolumetricFireParticles',
      map: textures.fireFlameSpriteMap,
      size: 0.48 * m.heightScale,
      sizeAttenuation: true,
      transparent: true,
      opacity: 0.96,
      depthWrite: false,
      vertexColors: true,
      blending: THREE.AdditiveBlending,
    })
  );

  const breathFrostMat = trackMat(
    new THREE.PointsMaterial({
      name: 'Mat_MouthVolumetricFrostParticles',
      map: textures.particleSmokeMap,
      size: 0.45 * m.heightScale,
      sizeAttenuation: true,
      transparent: true,
      opacity: 0.92,
      depthWrite: false,
      vertexColors: true,
      blending: THREE.AdditiveBlending,
    })
  );

  const breathParticlePoints = new THREE.Points(
    breathParticleGeo,
    effectiveWeapon === 'frost_breath_maw' ? breathFrostMat : breathFireMat
  );
  breathParticlePoints.name = 'Procedural_Mouth_Breath_Particles';
  breathParticlePoints.frustumCulled = false;
  breathParticlePoints.visible = false;
  rootGroup.add(breathParticlePoints);

  const breathParticlePool: BuiltEnemyRig['mouthBreathSystem']['pool'] = [];
  for (let i = 0; i < BREATH_PARTICLE_COUNT; i++) {
    const lifetime = 0.34 + (i % 9) * 0.048;
    breathParticlePool.push({
      age: (i / BREATH_PARTICLE_COUNT) * lifetime,
      lifetime,
      vx: 0,
      vy: 0,
      vz: 3.8,
      seed: i * 2.39996,
      isSpark: i % 6 === 0,
    });
  }

  // Compute exact vertex and triangle counts from visible exportable meshes (SkinnedMeshes + Weapons + Shards)
  let totalVertices = 0;
  let totalTriangles = 0;
  exportableGroup.traverse((obj) => {
    const mesh = obj as THREE.Mesh;
    if (mesh.isMesh && mesh.geometry && !mesh.userData.isExplodedSourceMesh) {
      const posAttr = mesh.geometry.getAttribute('position');
      if (posAttr) {
        const instanceMult = (mesh as THREE.InstancedMesh).isInstancedMesh
          ? (mesh as THREE.InstancedMesh).count
          : 1;
        totalVertices += posAttr.count * instanceMult;
        const index = mesh.geometry.getIndex();
        if (index) {
          totalTriangles += (index.count / 3) * instanceMult;
        } else {
          totalTriangles += (posAttr.count / 3) * instanceMult;
        }
      }
    }
  });

  const builtRig: BuiltEnemyRig = {
    rootGroup,
    exportableGroup,
    hitboxGroup,
    rootBone,
    skeleton: sharedSkeleton,
    skinnedMeshes,
    skinnedMeshGroup,
    explodedSourceMeshes,
    animations: [],
    materials: {
      primaryArmorMat,
      secondaryTrimMat,
      underbodyMat,
      hairStrandMat,
      emissiveRuneMat,
      crystalCoreMat,
      slashTrailMat,
    },
    parts: {
      pelvisGroup,
      torsoGroup,
      neckGroup,
      headGroup,
      jawGroup,
      extraHeads,
      leftShoulderGroup,
      rightShoulderGroup,
      leftArmGroup,
      rightArmGroup,
      leftElbowGroup,
      rightElbowGroup,
      leftLegGroup,
      rightLegGroup,
      leftKneeGroup,
      rightKneeGroup,
      leftWingGroup,
      rightWingGroup,
      tailGroup,
      tailTipBone,
      dorsalGroup,
      weaponGroup,
      offhandGroup,
      coreMesh,
      breathConeMesh,
      slashTrailMesh,
      arrowProjectileMesh,
      auraInstancedMesh,
    },
    sockets: {
      head: socketHead,
      weaponR: socketWeaponR,
      shieldL: socketShieldL,
      core: socketCore,
    },
    explodableParts,
    shardData,
    auraParticleSystem: {
      points: auraParticlePoints,
      smokeMat,
      sparkMat,
      activeMode: 'none',
      pool: particlePool,
    },
    mouthBreathSystem: {
      points: breathParticlePoints,
      fireMat: breathFireMat,
      frostMat: breathFrostMat,
      breathLight,
      fireBreathMap: textures.fireBreathMap,
      frostBreathMap: textures.frostBreathMap,
      pool: breathParticlePool,
    },
    metrics: {
      totalVertices: Math.round(totalVertices),
      totalTriangles: Math.round(totalTriangles),
      partCount: bones.length,
      boneCount: bones.length,
      clipCount: 6,
    },
    dispose: () => {
      geometriesToDispose.forEach((g) => g.dispose());
      materialsToDispose.forEach((mItem) => mItem.dispose());
    },
  };

  // Bake 6 GLTF 2.0 Compliant Skeletal AnimationClips (Idle, Walk, Attack, JumpBite, Hit, Death)
  builtRig.animations = bakeEnemyAnimationClips(builtRig);
  exportableGroup.animations = builtRig.animations;
  exportableGroup.userData.skeletalRig = {
    unifiedSkinnedMeshes: skinnedMeshes.length,
    boneCount: bones.length,
    embeddedAnimations: builtRig.animations.map((a) => a.name),
  };

  return builtRig;
}

/**
 * Resets all skeleton bones on the rig back to their clean bind pose.
 */
export function resetRigToBindPose(rig: BuiltEnemyRig) {
  for (let i = 0; i < rig.skeleton.bones.length; i++) {
    const b = rig.skeleton.bones[i];
    if (b.userData.bindPos) b.position.copy(b.userData.bindPos);
    if (b.userData.bindQuat) b.quaternion.copy(b.userData.bindQuat);
    if (b.userData.bindScale) b.scale.copy(b.userData.bindScale);
  }
  rig.parts.breathConeMesh.visible = true;
  rig.parts.breathConeMesh.scale.setScalar(0.0001);
  rig.parts.slashTrailMesh.visible = true;
  rig.parts.slashTrailMesh.scale.setScalar(0.0001);
  rig.parts.slashTrailMesh.rotation.set(0, 0, 0);
  const isBow = rig.exportableGroup.userData.effectiveWeapon === 'bone_recurved_bow';
  rig.parts.arrowProjectileMesh.position.set(0, 0, 0.36);
  rig.parts.arrowProjectileMesh.scale.setScalar(isBow ? 1.0 : 0.0001);
  const bowPivots = rig.parts.weaponGroup.userData.bowstringPivots as
    | { upper: THREE.Group; lower: THREE.Group }
    | undefined;
  if (bowPivots) {
    bowPivots.upper.rotation.set(0, 0, 0);
    bowPivots.upper.scale.set(1, 1, 1);
    bowPivots.lower.rotation.set(0, 0, 0);
    bowPivots.lower.scale.set(1, 1, 1);
  }
  rig.mouthBreathSystem.breathLight.intensity = 0;
  rig.mouthBreathSystem.points.visible = false;
  rig.skinnedMeshGroup.visible = true;
  for (let i = 0; i < rig.explodedSourceMeshes.length; i++) {
    rig.explodedSourceMeshes[i].visible = false;
  }
  rig.exportableGroup.updateMatrixWorld(true);
}

/**
 * Bakes 6 standard THREE.AnimationClip tracks ('Idle', 'Walk', 'Attack', 'JumpBite', 'Hit', 'Death')
 * using QuaternionKeyframeTrack and VectorKeyframeTrack so GLTFExporter embeds them directly inside .GLB
 * (including the Weapon Slash Trail Arc and Elemental Breath Cone tracks!).
 */
function bakeEnemyAnimationClips(rig: BuiltEnemyRig): THREE.AnimationClip[] {
  const specs: Array<{
    name: string;
    state: AnimationState;
    duration: number;
    steps: number;
    loopSeamless: boolean;
  }> = [
    { name: 'Idle', state: 'idle', duration: 2.73, steps: 24, loopSeamless: true },
    { name: 'Walk', state: 'walk', duration: 1.18, steps: 24, loopSeamless: true },
    { name: 'Attack', state: 'attack', duration: 2.513, steps: 26, loopSeamless: true },
    { name: 'JumpBite', state: 'jump_bite', duration: 1.85, steps: 28, loopSeamless: true },
    { name: 'Hit', state: 'hit', duration: 0.75, steps: 16, loopSeamless: false },
    { name: 'Death', state: 'death', duration: 1.5, steps: 22, loopSeamless: false },
  ];

  const bones = rig.skeleton.bones;
  const clips: THREE.AnimationClip[] = [];

  for (const spec of specs) {
    resetRigToBindPose(rig);

    // Pre-warm looping animations so damped channels settle into steady periodic motion
    if (spec.loopSeamless) {
      for (let w = 0; w < 8; w++) {
        updateEnemyAnimation(rig, spec.state, (w / 8) * spec.duration, 0.25, 1.0, 0);
      }
    }

    const numFrames = spec.steps + 1;
    const times = new Float32Array(numFrames);
    const quatBuffers = bones.map(() => new Float32Array(numFrames * 4));
    const rootBonePosBuffer = new Float32Array(numFrames * 3);
    const pelvisPosBuffer = new Float32Array(numFrames * 3);
    const torsoScaleBuffer = new Float32Array(numFrames * 3);
    const slashScaleBuffer = new Float32Array(numFrames * 3);
    const slashQuatBuffer = new Float32Array(numFrames * 4);
    const breathScaleBuffer = new Float32Array(numFrames * 3);
    const breathQuatBuffer = new Float32Array(numFrames * 4);
    const arrowPosBuffer = new Float32Array(numFrames * 3);
    const arrowScaleBuffer = new Float32Array(numFrames * 3);

    for (let s = 0; s <= spec.steps; s++) {
      const t = (s / spec.steps) * spec.duration;
      times[s] = Number(t.toFixed(4));

      updateEnemyAnimation(rig, spec.state, t, 0.35, 1.0, 0);

      for (let bIdx = 0; bIdx < bones.length; bIdx++) {
        const b = bones[bIdx];
        const qBuf = quatBuffers[bIdx];
        if (spec.loopSeamless && s === spec.steps) {
          qBuf[s * 4] = qBuf[0];
          qBuf[s * 4 + 1] = qBuf[1];
          qBuf[s * 4 + 2] = qBuf[2];
          qBuf[s * 4 + 3] = qBuf[3];
        } else {
          qBuf[s * 4] = b.quaternion.x;
          qBuf[s * 4 + 1] = b.quaternion.y;
          qBuf[s * 4 + 2] = b.quaternion.z;
          qBuf[s * 4 + 3] = b.quaternion.w;
        }
      }

      if (spec.loopSeamless && s === spec.steps) {
        rootBonePosBuffer[s * 3] = rootBonePosBuffer[0];
        rootBonePosBuffer[s * 3 + 1] = rootBonePosBuffer[1];
        rootBonePosBuffer[s * 3 + 2] = rootBonePosBuffer[2];

        pelvisPosBuffer[s * 3] = pelvisPosBuffer[0];
        pelvisPosBuffer[s * 3 + 1] = pelvisPosBuffer[1];
        pelvisPosBuffer[s * 3 + 2] = pelvisPosBuffer[2];

        torsoScaleBuffer[s * 3] = torsoScaleBuffer[0];
        torsoScaleBuffer[s * 3 + 1] = torsoScaleBuffer[1];
        torsoScaleBuffer[s * 3 + 2] = torsoScaleBuffer[2];

        slashScaleBuffer[s * 3] = slashScaleBuffer[0];
        slashScaleBuffer[s * 3 + 1] = slashScaleBuffer[1];
        slashScaleBuffer[s * 3 + 2] = slashScaleBuffer[2];

        slashQuatBuffer[s * 4] = slashQuatBuffer[0];
        slashQuatBuffer[s * 4 + 1] = slashQuatBuffer[1];
        slashQuatBuffer[s * 4 + 2] = slashQuatBuffer[2];
        slashQuatBuffer[s * 4 + 3] = slashQuatBuffer[3];

        breathScaleBuffer[s * 3] = breathScaleBuffer[0];
        breathScaleBuffer[s * 3 + 1] = breathScaleBuffer[1];
        breathScaleBuffer[s * 3 + 2] = breathScaleBuffer[2];

        breathQuatBuffer[s * 4] = breathQuatBuffer[0];
        breathQuatBuffer[s * 4 + 1] = breathQuatBuffer[1];
        breathQuatBuffer[s * 4 + 2] = breathQuatBuffer[2];
        breathQuatBuffer[s * 4 + 3] = breathQuatBuffer[3];

        arrowPosBuffer[s * 3] = arrowPosBuffer[0];
        arrowPosBuffer[s * 3 + 1] = arrowPosBuffer[1];
        arrowPosBuffer[s * 3 + 2] = arrowPosBuffer[2];

        arrowScaleBuffer[s * 3] = arrowScaleBuffer[0];
        arrowScaleBuffer[s * 3 + 1] = arrowScaleBuffer[1];
        arrowScaleBuffer[s * 3 + 2] = arrowScaleBuffer[2];
      } else {
        rootBonePosBuffer[s * 3] = rig.rootBone.position.x;
        rootBonePosBuffer[s * 3 + 1] = rig.rootBone.position.y;
        rootBonePosBuffer[s * 3 + 2] = rig.rootBone.position.z;

        pelvisPosBuffer[s * 3] = rig.parts.pelvisGroup.position.x;
        pelvisPosBuffer[s * 3 + 1] = rig.parts.pelvisGroup.position.y;
        pelvisPosBuffer[s * 3 + 2] = rig.parts.pelvisGroup.position.z;

        torsoScaleBuffer[s * 3] = rig.parts.torsoGroup.scale.x;
        torsoScaleBuffer[s * 3 + 1] = rig.parts.torsoGroup.scale.y;
        torsoScaleBuffer[s * 3 + 2] = rig.parts.torsoGroup.scale.z;

        slashScaleBuffer[s * 3] = rig.parts.slashTrailMesh.scale.x;
        slashScaleBuffer[s * 3 + 1] = rig.parts.slashTrailMesh.scale.y;
        slashScaleBuffer[s * 3 + 2] = rig.parts.slashTrailMesh.scale.z;

        slashQuatBuffer[s * 4] = rig.parts.slashTrailMesh.quaternion.x;
        slashQuatBuffer[s * 4 + 1] = rig.parts.slashTrailMesh.quaternion.y;
        slashQuatBuffer[s * 4 + 2] = rig.parts.slashTrailMesh.quaternion.z;
        slashQuatBuffer[s * 4 + 3] = rig.parts.slashTrailMesh.quaternion.w;

        breathScaleBuffer[s * 3] = rig.parts.breathConeMesh.scale.x;
        breathScaleBuffer[s * 3 + 1] = rig.parts.breathConeMesh.scale.y;
        breathScaleBuffer[s * 3 + 2] = rig.parts.breathConeMesh.scale.z;

        breathQuatBuffer[s * 4] = rig.parts.breathConeMesh.quaternion.x;
        breathQuatBuffer[s * 4 + 1] = rig.parts.breathConeMesh.quaternion.y;
        breathQuatBuffer[s * 4 + 2] = rig.parts.breathConeMesh.quaternion.z;
        breathQuatBuffer[s * 4 + 3] = rig.parts.breathConeMesh.quaternion.w;

        arrowPosBuffer[s * 3] = rig.parts.arrowProjectileMesh.position.x;
        arrowPosBuffer[s * 3 + 1] = rig.parts.arrowProjectileMesh.position.y;
        arrowPosBuffer[s * 3 + 2] = rig.parts.arrowProjectileMesh.position.z;

        arrowScaleBuffer[s * 3] = rig.parts.arrowProjectileMesh.scale.x;
        arrowScaleBuffer[s * 3 + 1] = rig.parts.arrowProjectileMesh.scale.y;
        arrowScaleBuffer[s * 3 + 2] = rig.parts.arrowProjectileMesh.scale.z;
      }
    }

    const tracks: THREE.KeyframeTrack[] = [];
    for (let bIdx = 0; bIdx < bones.length; bIdx++) {
      const b = bones[bIdx];
      if (b === rig.rootBone) continue;
      tracks.push(
        new THREE.QuaternionKeyframeTrack(`${b.name}.quaternion`, times, quatBuffers[bIdx])
      );
    }

    tracks.push(
      new THREE.VectorKeyframeTrack(
        `${rig.rootBone.name}.position`,
        times,
        rootBonePosBuffer
      )
    );
    tracks.push(
      new THREE.VectorKeyframeTrack(
        `${rig.parts.pelvisGroup.name}.position`,
        times,
        pelvisPosBuffer
      )
    );
    tracks.push(
      new THREE.VectorKeyframeTrack(
        `${rig.parts.torsoGroup.name}.scale`,
        times,
        torsoScaleBuffer
      )
    );
    tracks.push(
      new THREE.VectorKeyframeTrack(
        `${rig.parts.slashTrailMesh.name}.scale`,
        times,
        slashScaleBuffer
      )
    );
    tracks.push(
      new THREE.QuaternionKeyframeTrack(
        `${rig.parts.slashTrailMesh.name}.quaternion`,
        times,
        slashQuatBuffer
      )
    );
    tracks.push(
      new THREE.VectorKeyframeTrack(
        `${rig.parts.breathConeMesh.name}.scale`,
        times,
        breathScaleBuffer
      )
    );
    tracks.push(
      new THREE.QuaternionKeyframeTrack(
        `${rig.parts.breathConeMesh.name}.quaternion`,
        times,
        breathQuatBuffer
      )
    );
    tracks.push(
      new THREE.VectorKeyframeTrack(
        `${rig.parts.arrowProjectileMesh.name}.position`,
        times,
        arrowPosBuffer
      )
    );
    tracks.push(
      new THREE.VectorKeyframeTrack(
        `${rig.parts.arrowProjectileMesh.name}.scale`,
        times,
        arrowScaleBuffer
      )
    );

    const clip = new THREE.AnimationClip(spec.name, spec.duration, tracks);
    clip.optimize();
    clips.push(clip);
  }

  resetRigToBindPose(rig);
  return clips;
}

export function updateEnemyAnimation(
  rig: BuiltEnemyRig,
  animState: AnimationState,
  elapsed: number,
  delta: number,
  speedMultiplier: number,
  explodedAmount: number,
  auraParticleMode: AuraParticleEffectId = 'none'
) {
  const t = elapsed * speedMultiplier;
  const {
    pelvisGroup,
    torsoGroup,
    neckGroup,
    headGroup,
    jawGroup,
    extraHeads,
    leftShoulderGroup,
    rightShoulderGroup,
    leftArmGroup,
    rightArmGroup,
    leftElbowGroup,
    rightElbowGroup,
    leftLegGroup,
    rightLegGroup,
    leftKneeGroup,
    rightKneeGroup,
    leftWingGroup,
    rightWingGroup,
    tailGroup,
    tailTipBone,
    dorsalGroup,
    coreMesh,
    breathConeMesh,
    slashTrailMesh,
    arrowProjectileMesh,
    auraInstancedMesh,
  } = rig.parts;

  const bodyPlan = rig.exportableGroup.userData.bodyPlan as string;
  const isFloating = bodyPlan === 'floating_magical';
  const isDragon = bodyPlan === 'dragon_wyvern';
  const isSerpentine = bodyPlan === 'serpentine_naga';
  const isQuadruped = bodyPlan === 'quadruped_beast' || bodyPlan === 'dragon_wyvern';
  const isAnimal = Boolean(rig.exportableGroup.userData.isAnimal);
  const effectiveWeapon =
    (rig.exportableGroup.userData.effectiveWeapon as string) || 'feral_claws_fangs';
  const isBowWeapon = effectiveWeapon === 'bone_recurved_bow';
  const isHandheldWeaponMonster =
    !isAnimal &&
    effectiveWeapon !== 'feral_claws_fangs' &&
    effectiveWeapon !== 'leap_fang_bite' &&
    effectiveWeapon !== 'ram_charge' &&
    effectiveWeapon !== 'dragon_breath_maw' &&
    effectiveWeapon !== 'frost_breath_maw';
  const elementAffinity =
    (rig.exportableGroup.userData.rpgStats?.element as string) || 'Salvaje / Físico';
  const isFireAffinity =
    effectiveWeapon === 'dragon_breath_maw' ||
    elementAffinity === 'Fuego Dracónico' ||
    elementAffinity === 'Fuego Solar';
  const isFrostAffinity =
    effectiveWeapon === 'frost_breath_maw' || elementAffinity === 'Escarcha';

  // Default resting nocked state for 3D Arrow Projectile when not actively shooting
  if (animState !== 'attack' && animState !== 'jump_bite') {
    arrowProjectileMesh.position.set(0, 0.02, 0.18);
    arrowProjectileMesh.scale.setScalar(isBowWeapon ? 1.0 : 0.0001);
  } else if (!isBowWeapon) {
    arrowProjectileMesh.scale.setScalar(0.0001);
  }

  // Toggle between Unified Continuous SkinnedMesh (normal view) and Discrete Sub-Meshes (Exploded View > 0.02)
  const isExploded = explodedAmount > 0.02;
  if (rig.skinnedMeshGroup.visible === isExploded) {
    rig.skinnedMeshGroup.visible = !isExploded;
    for (let i = 0; i < rig.explodedSourceMeshes.length; i++) {
      rig.explodedSourceMeshes[i].visible = isExploded;
    }
  }

  // 1. Smoothly interpolate Exploded View separations for non-pelvis parts
  for (let i = 0; i < rig.explodableParts.length; i++) {
    const part = rig.explodableParts[i];
    if (part === pelvisGroup) continue; // pelvisGroup position is updated deterministically with _pelvisAnimOffset below
    const basePos = part.userData.basePos as THREE.Vector3;
    const explodeVec = part.userData.explodeVec as THREE.Vector3;
    if (basePos && explodeVec) {
      _targetPos.copy(basePos).addScaledVector(explodeVec, explodedAmount);
      part.position.x = THREE.MathUtils.damp(part.position.x, _targetPos.x, 8, delta);
      part.position.y = THREE.MathUtils.damp(part.position.y, _targetPos.y, 8, delta);
      part.position.z = THREE.MathUtils.damp(part.position.z, _targetPos.z, 8, delta);
    }
  }

  // Reset deterministic pelvic offset for this frame (ZERO frame-to-frame += accumulation!)
  _pelvisAnimOffset.set(0, 0, 0);
  if (isFloating) {
    _pelvisAnimOffset.y = Math.sin(t * 2.3) * 0.12;
  }

  // Track intensity of mouth fire/frost/bite particle emission this frame
  let mouthBreathIntensity = 0;
  let mouthParticleMode: 'fire' | 'frost' | 'bite_sparks' = isFrostAffinity
    ? 'frost'
    : isFireAffinity
    ? 'fire'
    : 'bite_sparks';

  // 2. Rotate Runic Core Crystal & Update Instanced Aura Shards
  coreMesh.rotation.y = t * 1.4;
  coreMesh.rotation.z = Math.sin(t * 2.0) * 0.25;

  if (auraInstancedMesh && rig.shardData.length > 0) {
    for (let i = 0; i < rig.shardData.length; i++) {
      const s = rig.shardData[i];
      const curAngle = s.angle + t * s.speed;
      const r = s.radius + explodedAmount * 0.45;
      const y = s.yOffset + Math.sin(t * 2.2 + i) * 0.12;
      _dummyObj.position.set(Math.cos(curAngle) * r, y, Math.sin(curAngle) * r);
      _dummyObj.rotation.set(t * 0.8 + i, curAngle, 0.25);
      _dummyObj.scale.setScalar(s.scale);
      _dummyObj.updateMatrix();
      auraInstancedMesh.setMatrixAt(i, _dummyObj.matrix);
    }
    auraInstancedMesh.instanceMatrix.needsUpdate = true;
  }

  // Continuous Organic 2-Bone S-Wave Tail Sway, Ribcage Breathing & Wing Motion
  const tailFreq =
    animState === 'walk'
      ? 5.33
      : animState === 'jump_bite'
      ? 6.78
      : animState === 'attack'
      ? 5.0
      : 2.3;
  const tailWave = Math.sin(t * tailFreq);
  const tailTipWave = Math.sin(t * tailFreq - 0.95);
  tailGroup.rotation.y = THREE.MathUtils.damp(tailGroup.rotation.y, tailWave * 0.26, 10, delta);
  tailTipBone.rotation.y = THREE.MathUtils.damp(tailTipBone.rotation.y, tailTipWave * 0.36, 10, delta);
  tailTipBone.rotation.x = THREE.MathUtils.damp(
    tailTipBone.rotation.x,
    Math.cos(t * tailFreq - 0.6) * 0.13,
    10,
    delta
  );

  for (let i = 0; i < extraHeads.length; i++) {
    const eh = extraHeads[i];
    const phase = i === 0 ? 1.4 : -1.4;
    eh.rotation.x = Math.sin(t * 2.3 + phase) * 0.12;
  }

  // Aerodynamic 2-phase wing kinematics (strong downstroke, feathered recovery)
  const wingFreq = animState === 'jump_bite' ? 6.78 : animState === 'attack' ? 5.0 : 2.3;
  const wingFlap = Math.sin(t * wingFreq);
  const wingFold = Math.cos(t * wingFreq) * 0.16;
  leftWingGroup.rotation.y = THREE.MathUtils.damp(leftWingGroup.rotation.y, wingFlap * 0.38, 10, delta);
  rightWingGroup.rotation.y = THREE.MathUtils.damp(rightWingGroup.rotation.y, -wingFlap * 0.38, 10, delta);
  leftWingGroup.rotation.z = THREE.MathUtils.damp(leftWingGroup.rotation.z, wingFold, 10, delta);
  rightWingGroup.rotation.z = THREE.MathUtils.damp(rightWingGroup.rotation.z, -wingFold, 10, delta);

  const dampLambda = 12;

  // Reset secondary rotation channels when not actively overridden
  if (animState !== 'attack' && animState !== 'jump_bite') {
    rig.rootBone.position.x = THREE.MathUtils.damp(rig.rootBone.position.x, 0, dampLambda, delta);
    rig.rootBone.position.y = THREE.MathUtils.damp(rig.rootBone.position.y, 0, dampLambda, delta);
    rig.rootBone.position.z = THREE.MathUtils.damp(rig.rootBone.position.z, 0, dampLambda, delta);
    headGroup.rotation.z = THREE.MathUtils.damp(headGroup.rotation.z, 0, dampLambda, delta);
  }
  if (animState !== 'walk') {
    pelvisGroup.rotation.y = THREE.MathUtils.damp(pelvisGroup.rotation.y, 0, dampLambda, delta);
    pelvisGroup.rotation.z = THREE.MathUtils.damp(pelvisGroup.rotation.z, 0, dampLambda, delta);
    torsoGroup.rotation.y = THREE.MathUtils.damp(torsoGroup.rotation.y, 0, dampLambda, delta);
    torsoGroup.rotation.z = THREE.MathUtils.damp(torsoGroup.rotation.z, 0, dampLambda, delta);
  }
  if (animState !== 'hit' && animState !== 'jump_bite' && !(animState === 'attack' && isBowWeapon)) {
    rightArmGroup.rotation.z = THREE.MathUtils.damp(rightArmGroup.rotation.z, 0, dampLambda, delta);
    leftArmGroup.rotation.z = THREE.MathUtils.damp(leftArmGroup.rotation.z, 0, dampLambda, delta);
    rightArmGroup.rotation.y = THREE.MathUtils.damp(rightArmGroup.rotation.y, 0, dampLambda, delta);
    leftArmGroup.rotation.y = THREE.MathUtils.damp(leftArmGroup.rotation.y, 0, dampLambda, delta);
  }

  const setBowstringDraw = (pullBackZ: number) => {
    const bowPivots = rig.parts.weaponGroup.userData.bowstringPivots as
      | { upper: THREE.Group; lower: THREE.Group }
      | undefined;
    if (!bowPivots) return;
    const halfLen = 0.82;
    const stretchedLen = Math.hypot(halfLen, pullBackZ);
    const angle = Math.atan2(pullBackZ, halfLen);
    bowPivots.upper.rotation.x = angle;
    bowPivots.upper.scale.y = stretchedLen / halfLen;
    bowPivots.lower.rotation.x = -angle;
    bowPivots.lower.scale.y = stretchedLen / halfLen;
  };
  if (isBowWeapon && animState !== 'attack' && animState !== 'jump_bite') {
    setBowstringDraw(0);
    arrowProjectileMesh.position.set(0, 0, 0.36);
    arrowProjectileMesh.scale.setScalar(1.0);
  }

  // Helper for:
  // 1) Monstruos con Arma Cuerpo a Cuerpo -> "Salto y Golpeo Brutal con Arma" (Heroic Leap + Overhead Weapon Slam + 3D Slash Trail)
  // 2) Arqueros con Arco -> "Salto Acrobático, Tensión a la Nariz y Disparo Horizontal al Frente"
  // 3) Bestias y Animales -> "Salto con Mordida Depredadora" (Predatory Leap + Lethal Fang Bite)
  const applyPredatoryJumpBite = (cycleDuration: number) => {
    const u = ((t / cycleDuration) % 1.0 + 1.0) % 1.0;

    if (isBowWeapon) {
      // =========================================================================================
      // CASE A: ARCHER ACROBATIC JUMP-SHOT (Tensión a la Altura de la Nariz + Disparo Horizontal +Z)
      // =========================================================================================
      breathConeMesh.scale.setScalar(0.0001);
      if (u < 0.22) {
        const p = u / 0.22;
        const coil = Math.sin(p * Math.PI * 0.5);
        _pelvisAnimOffset.y = -0.22 * coil;
        _pelvisAnimOffset.z = -0.18 * coil;
        rig.rootBone.position.z = THREE.MathUtils.damp(rig.rootBone.position.z, -0.14 * coil, 18, delta);
        rig.rootBone.position.y = THREE.MathUtils.damp(rig.rootBone.position.y, 0, 18, delta);

        pelvisGroup.rotation.x = THREE.MathUtils.damp(pelvisGroup.rotation.x, 0, 18, delta);
        torsoGroup.rotation.x = THREE.MathUtils.damp(torsoGroup.rotation.x, 0, 18, delta);
        torsoGroup.rotation.y = THREE.MathUtils.damp(torsoGroup.rotation.y, -0.22 * coil, 18, delta);
        headGroup.rotation.x = THREE.MathUtils.damp(headGroup.rotation.x, 0, 18, delta);
        headGroup.rotation.y = THREE.MathUtils.damp(headGroup.rotation.y, 0.22 * coil, 18, delta);

        leftLegGroup.rotation.x = THREE.MathUtils.damp(leftLegGroup.rotation.x, -0.56 * coil, 18, delta);
        rightLegGroup.rotation.x = THREE.MathUtils.damp(rightLegGroup.rotation.x, -0.56 * coil, 18, delta);
        leftKneeGroup.rotation.x = THREE.MathUtils.damp(leftKneeGroup.rotation.x, 0.25 + 0.65 * coil, 18, delta);
        rightKneeGroup.rotation.x = THREE.MathUtils.damp(rightKneeGroup.rotation.x, 0.25 + 0.65 * coil, 18, delta);

        // Raise right bow arm forward to nose height
        rightArmGroup.rotation.x = THREE.MathUtils.damp(rightArmGroup.rotation.x, -1.52 * coil, 20, delta);
        rightArmGroup.rotation.y = THREE.MathUtils.damp(rightArmGroup.rotation.y, 0.22 * coil, 20, delta);
        rightArmGroup.rotation.z = THREE.MathUtils.damp(rightArmGroup.rotation.z, 0, 20, delta);
        rightElbowGroup.rotation.x = THREE.MathUtils.damp(rightElbowGroup.rotation.x, -0.0508 * coil, 20, delta);

        // Left hand reaches forward to grab the bowstring
        leftArmGroup.rotation.x = THREE.MathUtils.damp(leftArmGroup.rotation.x, -1.5 * coil, 20, delta);
        leftArmGroup.rotation.y = THREE.MathUtils.damp(leftArmGroup.rotation.y, -0.36 * coil, 20, delta);
        leftArmGroup.rotation.z = THREE.MathUtils.damp(leftArmGroup.rotation.z, 0.06 * coil, 20, delta);
        leftElbowGroup.rotation.x = THREE.MathUtils.damp(leftElbowGroup.rotation.x, -0.24 * coil, 20, delta);

        slashTrailMesh.scale.setScalar(0.0001);
        setBowstringDraw(0);
        arrowProjectileMesh.position.set(0, 0, 0.36);
        arrowProjectileMesh.scale.setScalar(1.0);
      } else if (u < 0.52) {
        // High Acrobatic Leap while pulling the bowstring tension straight back to the NOSE!
        const airU = (u - 0.22) / 0.3;
        const arcHeight = Math.sin(airU * Math.PI) * 0.72;
        _pelvisAnimOffset.y = arcHeight;
        _pelvisAnimOffset.z = THREE.MathUtils.lerp(-0.18, 0.12, airU);
        rig.rootBone.position.y = THREE.MathUtils.damp(rig.rootBone.position.y, arcHeight * 0.35, 22, delta);

        // Keep spine and pelvis level (0 pitch) so the bow aims 100% horizontally forward along +Z!
        pelvisGroup.rotation.x = THREE.MathUtils.damp(pelvisGroup.rotation.x, 0, 24, delta);
        torsoGroup.rotation.x = THREE.MathUtils.damp(torsoGroup.rotation.x, 0, 24, delta);
        torsoGroup.rotation.y = THREE.MathUtils.damp(torsoGroup.rotation.y, -0.22, 22, delta);
        neckGroup.rotation.x = THREE.MathUtils.damp(neckGroup.rotation.x, 0, 22, delta);
        headGroup.rotation.x = THREE.MathUtils.damp(headGroup.rotation.x, 0, 22, delta);
        headGroup.rotation.y = THREE.MathUtils.damp(headGroup.rotation.y, 0.22, 22, delta);

        // Right arm extended straight ahead at nose height (-1.52 + -0.0508 = -Math.PI/2 => 0° world pitch!)
        rightShoulderGroup.rotation.z = THREE.MathUtils.damp(rightShoulderGroup.rotation.z, 0, 24, delta);
        rightArmGroup.rotation.x = THREE.MathUtils.damp(rightArmGroup.rotation.x, -1.52, 26, delta);
        rightArmGroup.rotation.y = THREE.MathUtils.damp(rightArmGroup.rotation.y, 0.22, 26, delta);
        rightArmGroup.rotation.z = THREE.MathUtils.damp(rightArmGroup.rotation.z, 0, 26, delta);
        rightElbowGroup.rotation.x = THREE.MathUtils.damp(rightElbowGroup.rotation.x, -0.0508, 26, delta);

        // Left hand pulls bowstring and arrow nock straight back to the height of the nose!
        leftArmGroup.rotation.x = THREE.MathUtils.damp(
          leftArmGroup.rotation.x,
          THREE.MathUtils.lerp(-1.5, -1.56, airU),
          24,
          delta
        );
        leftArmGroup.rotation.y = THREE.MathUtils.damp(
          leftArmGroup.rotation.y,
          THREE.MathUtils.lerp(-0.36, 0.18, airU),
          24,
          delta
        );
        leftArmGroup.rotation.z = THREE.MathUtils.damp(
          leftArmGroup.rotation.z,
          THREE.MathUtils.lerp(0.06, 0.26, airU),
          24,
          delta
        );
        leftElbowGroup.rotation.x = THREE.MathUtils.damp(
          leftElbowGroup.rotation.x,
          THREE.MathUtils.lerp(-0.24, -1.96, airU),
          24,
          delta
        );

        leftLegGroup.rotation.x = THREE.MathUtils.damp(leftLegGroup.rotation.x, -0.48, 20, delta);
        rightLegGroup.rotation.x = THREE.MathUtils.damp(rightLegGroup.rotation.x, 0.52, 20, delta);
        leftKneeGroup.rotation.x = THREE.MathUtils.damp(leftKneeGroup.rotation.x, 0.86, 20, delta);
        rightKneeGroup.rotation.x = THREE.MathUtils.damp(rightKneeGroup.rotation.x, 0.44, 20, delta);

        const pullZ = airU * 0.56;
        setBowstringDraw(pullZ);
        slashTrailMesh.scale.setScalar(0.0001);
        arrowProjectileMesh.position.set(0, 0, 0.36 - pullZ);
        arrowProjectileMesh.scale.setScalar(1.05);
      } else if (u < 0.84) {
        // Release Bowstring from Nose Height & Shoot 3D Arrow Horizontally Straight Forward (+Z)!
        const shotU = (u - 0.52) / 0.32;
        const landAbsorb = Math.sin(shotU * Math.PI);
        _pelvisAnimOffset.y = -0.1 * landAbsorb;
        rig.rootBone.position.y = THREE.MathUtils.damp(rig.rootBone.position.y, 0, 20, delta);

        pelvisGroup.rotation.x = THREE.MathUtils.damp(pelvisGroup.rotation.x, 0, 26, delta);
        torsoGroup.rotation.x = THREE.MathUtils.damp(torsoGroup.rotation.x, 0, 26, delta);
        torsoGroup.rotation.y = THREE.MathUtils.damp(torsoGroup.rotation.y, -0.22, 24, delta);
        neckGroup.rotation.x = THREE.MathUtils.damp(neckGroup.rotation.x, 0, 24, delta);
        headGroup.rotation.x = THREE.MathUtils.damp(headGroup.rotation.x, 0, 24, delta);
        headGroup.rotation.y = THREE.MathUtils.damp(headGroup.rotation.y, 0.22, 24, delta);

        rightShoulderGroup.rotation.z = THREE.MathUtils.damp(rightShoulderGroup.rotation.z, 0, 26, delta);
        rightArmGroup.rotation.x = THREE.MathUtils.damp(rightArmGroup.rotation.x, -1.52, 28, delta);
        rightArmGroup.rotation.y = THREE.MathUtils.damp(rightArmGroup.rotation.y, 0.22, 28, delta);
        rightArmGroup.rotation.z = THREE.MathUtils.damp(rightArmGroup.rotation.z, 0, 28, delta);
        rightElbowGroup.rotation.x = THREE.MathUtils.damp(rightElbowGroup.rotation.x, -0.0508, 28, delta);

        leftArmGroup.rotation.x = THREE.MathUtils.damp(leftArmGroup.rotation.x, -1.56, 24, delta);
        leftArmGroup.rotation.y = THREE.MathUtils.damp(leftArmGroup.rotation.y, 0.24, 24, delta);
        leftArmGroup.rotation.z = THREE.MathUtils.damp(leftArmGroup.rotation.z, 0.28, 24, delta);
        leftElbowGroup.rotation.x = THREE.MathUtils.damp(leftElbowGroup.rotation.x, -2.02, 24, delta);

        leftLegGroup.rotation.x = THREE.MathUtils.damp(leftLegGroup.rotation.x, -0.26, 18, delta);
        rightLegGroup.rotation.x = THREE.MathUtils.damp(rightLegGroup.rotation.x, 0.22, 18, delta);
        leftKneeGroup.rotation.x = THREE.MathUtils.damp(leftKneeGroup.rotation.x, 0.48, 18, delta);
        rightKneeGroup.rotation.x = THREE.MathUtils.damp(rightKneeGroup.rotation.x, 0.42, 18, delta);

        // Bowstring snaps taut and arrow flies 100% horizontally forward along +Z (up to 11.2m)!
        const stringSnap = Math.max(0, Math.sin(shotU * Math.PI * 5) * Math.exp(-shotU * 7) * 0.05);
        setBowstringDraw(stringSnap);
        arrowProjectileMesh.position.set(0, 0, 0.36 + shotU * 11.2);
        arrowProjectileMesh.scale.set(1.25, 1.25, 1.65);

        if (shotU < 0.28) {
          const flash = (1.0 - shotU / 0.28) * 0.95;
          slashTrailMesh.scale.set(0.85, flash, flash);
          slashTrailMesh.rotation.set(0, 0, 0);
        } else {
          slashTrailMesh.scale.setScalar(0.0001);
        }
      } else {
        const recU = (u - 0.84) / 0.16;
        _pelvisAnimOffset.y = 0;
        rig.rootBone.position.z = THREE.MathUtils.damp(rig.rootBone.position.z, 0, 16, delta);
        rig.rootBone.position.y = THREE.MathUtils.damp(rig.rootBone.position.y, 0, 16, delta);
        pelvisGroup.rotation.x = THREE.MathUtils.damp(pelvisGroup.rotation.x, 0, 16, delta);
        torsoGroup.rotation.x = THREE.MathUtils.damp(torsoGroup.rotation.x, 0, 16, delta);
        torsoGroup.rotation.y = THREE.MathUtils.damp(torsoGroup.rotation.y, 0, 16, delta);
        rightArmGroup.rotation.x = THREE.MathUtils.damp(rightArmGroup.rotation.x, -0.35, 16, delta);
        rightArmGroup.rotation.y = THREE.MathUtils.damp(rightArmGroup.rotation.y, 0, 16, delta);
        leftArmGroup.rotation.x = THREE.MathUtils.damp(leftArmGroup.rotation.x, -0.35, 16, delta);
        leftArmGroup.rotation.y = THREE.MathUtils.damp(leftArmGroup.rotation.y, 0, 16, delta);
        slashTrailMesh.scale.setScalar(0.0001);
        setBowstringDraw(0);
        arrowProjectileMesh.position.set(0, 0, 0.36);
        arrowProjectileMesh.scale.setScalar(recU);
      }
      return;
    }

    if (isHandheldWeaponMonster) {
      // =========================================================================================
      // CASE B: ARMED MONSTER HEROIC LEAP + CRUSHING OVERHEAD WEAPON SLAM ("Salto y Golpeo")
      // =========================================================================================
      breathConeMesh.scale.setScalar(0.0001);
      if (u < 0.24) {
        // PHASE 1: Deep Warrior Crouch & Two-Stage Weapon Windup (0.00 -> 0.24)
        const p = u / 0.24;
        const coil = Math.sin(p * Math.PI * 0.5);
        _pelvisAnimOffset.y = -0.24 * coil;
        _pelvisAnimOffset.z = -0.22 * coil;
        rig.rootBone.position.z = THREE.MathUtils.damp(rig.rootBone.position.z, -0.16 * coil, 18, delta);
        rig.rootBone.position.y = THREE.MathUtils.damp(rig.rootBone.position.y, 0, 18, delta);

        pelvisGroup.rotation.x = THREE.MathUtils.damp(pelvisGroup.rotation.x, 0.2 * coil, 18, delta);
        torsoGroup.rotation.x = THREE.MathUtils.damp(torsoGroup.rotation.x, 0.26 * coil, 18, delta);
        torsoGroup.rotation.y = THREE.MathUtils.damp(torsoGroup.rotation.y, -0.38 * coil, 18, delta);
        neckGroup.rotation.x = THREE.MathUtils.damp(neckGroup.rotation.x, -0.22 * coil, 18, delta);
        headGroup.rotation.x = THREE.MathUtils.damp(headGroup.rotation.x, -0.18 * coil, 18, delta);
        jawGroup.rotation.x = THREE.MathUtils.damp(jawGroup.rotation.x, 0.24 * coil, 18, delta);

        // Deep leg compression before explosive takeoff
        leftLegGroup.rotation.x = THREE.MathUtils.damp(leftLegGroup.rotation.x, -0.62 * coil, 18, delta);
        rightLegGroup.rotation.x = THREE.MathUtils.damp(rightLegGroup.rotation.x, -0.58 * coil, 18, delta);
        leftKneeGroup.rotation.x = THREE.MathUtils.damp(leftKneeGroup.rotation.x, 0.26 + 0.68 * coil, 18, delta);
        rightKneeGroup.rotation.x = THREE.MathUtils.damp(rightKneeGroup.rotation.x, 0.26 + 0.64 * coil, 18, delta);

        // Cock weapon back behind the right shoulder ready for the aerial leap-slam
        rightArmGroup.rotation.x = THREE.MathUtils.damp(rightArmGroup.rotation.x, -1.55 * coil, 18, delta);
        rightArmGroup.rotation.z = THREE.MathUtils.damp(rightArmGroup.rotation.z, -0.22 * coil, 18, delta);
        rightElbowGroup.rotation.x = THREE.MathUtils.damp(rightElbowGroup.rotation.x, -0.68 * coil, 18, delta);

        leftArmGroup.rotation.x = THREE.MathUtils.damp(leftArmGroup.rotation.x, -0.58 * coil, 18, delta);
        leftArmGroup.rotation.z = THREE.MathUtils.damp(leftArmGroup.rotation.z, 0.18 * coil, 18, delta);
        leftElbowGroup.rotation.x = THREE.MathUtils.damp(leftElbowGroup.rotation.x, -0.48 * coil, 18, delta);

        slashTrailMesh.scale.setScalar(0.0001);
      } else if (u < 0.56) {
        // PHASE 2: Explosive Heroic Leap & Raising Weapon High Overhead in Mid-Air (0.24 -> 0.56)
        const airU = (u - 0.24) / 0.32;
        const arcHeight = Math.sin(airU * Math.PI) * 0.82;
        const forwardDist = THREE.MathUtils.lerp(-0.2, 0.88, airU);

        _pelvisAnimOffset.y = arcHeight;
        _pelvisAnimOffset.z = forwardDist * 0.45;
        rig.rootBone.position.z = THREE.MathUtils.damp(rig.rootBone.position.z, forwardDist * 0.55, 22, delta);
        rig.rootBone.position.y = THREE.MathUtils.damp(rig.rootBone.position.y, arcHeight * 0.32, 22, delta);

        const pitch = THREE.MathUtils.lerp(-0.28, 0.32, airU);
        pelvisGroup.rotation.x = THREE.MathUtils.damp(pelvisGroup.rotation.x, pitch * 0.7, 22, delta);
        torsoGroup.rotation.x = THREE.MathUtils.damp(torsoGroup.rotation.x, pitch, 22, delta);
        torsoGroup.rotation.y = THREE.MathUtils.damp(
          torsoGroup.rotation.y,
          THREE.MathUtils.lerp(-0.38, 0.12, airU),
          22,
          delta
        );
        jawGroup.rotation.x = THREE.MathUtils.damp(jawGroup.rotation.x, 0.68, 22, delta);

        // Dynamic tucked warrior jump pose in mid-air
        leftLegGroup.rotation.x = THREE.MathUtils.damp(leftLegGroup.rotation.x, -0.54, 22, delta);
        rightLegGroup.rotation.x = THREE.MathUtils.damp(rightLegGroup.rotation.x, 0.48, 22, delta);
        leftKneeGroup.rotation.x = THREE.MathUtils.damp(leftKneeGroup.rotation.x, 0.88, 22, delta);
        rightKneeGroup.rotation.x = THREE.MathUtils.damp(rightKneeGroup.rotation.x, 0.46, 22, delta);

        // Raise weapon high above the head at the apex, then initiate the downward cleave!
        const targetWeaponArmX =
          airU < 0.72
            ? THREE.MathUtils.lerp(-1.65, -2.42, airU / 0.72)
            : THREE.MathUtils.lerp(-2.42, -0.45, (airU - 0.72) / 0.28);
        rightArmGroup.rotation.x = THREE.MathUtils.damp(rightArmGroup.rotation.x, targetWeaponArmX, 24, delta);
        rightArmGroup.rotation.z = THREE.MathUtils.damp(rightArmGroup.rotation.z, -0.14, 22, delta);
        rightElbowGroup.rotation.x = THREE.MathUtils.damp(rightElbowGroup.rotation.x, -0.28, 22, delta);

        leftArmGroup.rotation.x = THREE.MathUtils.damp(leftArmGroup.rotation.x, -1.15, 20, delta);
        leftArmGroup.rotation.z = THREE.MathUtils.damp(leftArmGroup.rotation.z, 0.28, 20, delta);
        leftElbowGroup.rotation.x = THREE.MathUtils.damp(leftElbowGroup.rotation.x, -0.35, 20, delta);

        leftWingGroup.rotation.y = THREE.MathUtils.damp(leftWingGroup.rotation.y, 0.72, 18, delta);
        rightWingGroup.rotation.y = THREE.MathUtils.damp(rightWingGroup.rotation.y, -0.72, 18, delta);

        if (airU > 0.7) {
          const preSlash = (airU - 0.7) / 0.3;
          slashTrailMesh.scale.set(1.15, preSlash * 1.25, preSlash * 1.35);
          slashTrailMesh.rotation.set(-0.45 + preSlash * 0.85, 0.1, -0.12);
        } else {
          slashTrailMesh.scale.setScalar(0.0001);
        }
      } else if (u < 0.78) {
        // PHASE 3: Crushing Overhead Weapon Slam on Landing Impact + Massive 3D Slash Arc (0.56 -> 0.78)
        const slamU = (u - 0.56) / 0.22;
        const shockAbsorb = Math.sin(slamU * Math.PI);

        _pelvisAnimOffset.y = -0.18 * shockAbsorb;
        _pelvisAnimOffset.z = 0.38 * (1 - slamU * 0.25);
        rig.rootBone.position.z = THREE.MathUtils.damp(
          rig.rootBone.position.z,
          0.46 * (1 - slamU * 0.25),
          22,
          delta
        );
        rig.rootBone.position.y = THREE.MathUtils.damp(rig.rootBone.position.y, 0, 22, delta);

        pelvisGroup.rotation.x = THREE.MathUtils.damp(pelvisGroup.rotation.x, 0.22, 22, delta);
        torsoGroup.rotation.x = THREE.MathUtils.damp(torsoGroup.rotation.x, 0.36, 22, delta);
        torsoGroup.rotation.y = THREE.MathUtils.damp(torsoGroup.rotation.y, 0.38, 22, delta);
        headGroup.rotation.x = THREE.MathUtils.damp(headGroup.rotation.x, 0.16, 22, delta);
        jawGroup.rotation.x = THREE.MathUtils.damp(jawGroup.rotation.x, 0.45, 22, delta);

        // Weapon arm slams all the way down through the target!
        rightArmGroup.rotation.x = THREE.MathUtils.damp(rightArmGroup.rotation.x, 0.58, 26, delta);
        rightArmGroup.rotation.z = THREE.MathUtils.damp(rightArmGroup.rotation.z, 0.08, 22, delta);
        rightElbowGroup.rotation.x = THREE.MathUtils.damp(rightElbowGroup.rotation.x, -0.14, 24, delta);

        leftArmGroup.rotation.x = THREE.MathUtils.damp(leftArmGroup.rotation.x, -0.35, 22, delta);
        leftArmGroup.rotation.z = THREE.MathUtils.damp(leftArmGroup.rotation.z, 0.12, 20, delta);
        leftElbowGroup.rotation.x = THREE.MathUtils.damp(leftElbowGroup.rotation.x, -0.52, 22, delta);

        leftLegGroup.rotation.x = THREE.MathUtils.damp(leftLegGroup.rotation.x, -0.38, 20, delta);
        rightLegGroup.rotation.x = THREE.MathUtils.damp(rightLegGroup.rotation.x, 0.34, 20, delta);
        leftKneeGroup.rotation.x = THREE.MathUtils.damp(leftKneeGroup.rotation.x, 0.62, 20, delta);
        rightKneeGroup.rotation.x = THREE.MathUtils.damp(rightKneeGroup.rotation.x, 0.48, 20, delta);

        // Massive 3D Weapon Slash Trail Arc + Impact Shockwave Sparks!
        const sMag = Math.max(0.0001, (1.0 - slamU * 0.48) * 1.55);
        slashTrailMesh.scale.set(1.25, sMag, sMag * 1.32);
        slashTrailMesh.rotation.set(0.45 + slamU * 0.55, 0.14, -0.18);
        mouthBreathIntensity = (1.0 - slamU * 0.5) * 0.75;
      } else {
        // PHASE 4: Agile Warrior Push-Off & Recovery Back to Stance (0.78 -> 1.00)
        const recU = (u - 0.78) / 0.22;
        const easeBack = 1.0 - Math.sin(recU * Math.PI * 0.5);
        const hopBackArc = Math.sin(recU * Math.PI) * 0.1;

        _pelvisAnimOffset.y = hopBackArc;
        _pelvisAnimOffset.z = 0.28 * easeBack;
        rig.rootBone.position.z = THREE.MathUtils.damp(rig.rootBone.position.z, 0.34 * easeBack, 18, delta);
        rig.rootBone.position.y = THREE.MathUtils.damp(rig.rootBone.position.y, 0, 18, delta);

        pelvisGroup.rotation.x = THREE.MathUtils.damp(pelvisGroup.rotation.x, 0, 16, delta);
        torsoGroup.rotation.x = THREE.MathUtils.damp(torsoGroup.rotation.x, 0, 16, delta);
        torsoGroup.rotation.y = THREE.MathUtils.damp(torsoGroup.rotation.y, 0, 16, delta);
        neckGroup.rotation.x = THREE.MathUtils.damp(neckGroup.rotation.x, 0, 16, delta);
        headGroup.rotation.x = THREE.MathUtils.damp(headGroup.rotation.x, 0, 16, delta);
        jawGroup.rotation.x = THREE.MathUtils.damp(jawGroup.rotation.x, 0.08, 16, delta);

        rightArmGroup.rotation.x = THREE.MathUtils.damp(rightArmGroup.rotation.x, -0.18, 16, delta);
        leftArmGroup.rotation.x = THREE.MathUtils.damp(leftArmGroup.rotation.x, -0.18, 16, delta);
        rightArmGroup.rotation.z = THREE.MathUtils.damp(rightArmGroup.rotation.z, 0, 16, delta);
        leftArmGroup.rotation.z = THREE.MathUtils.damp(leftArmGroup.rotation.z, 0, 16, delta);
        rightElbowGroup.rotation.x = THREE.MathUtils.damp(rightElbowGroup.rotation.x, -0.32, 16, delta);
        leftElbowGroup.rotation.x = THREE.MathUtils.damp(leftElbowGroup.rotation.x, -0.32, 16, delta);

        leftLegGroup.rotation.x = THREE.MathUtils.damp(leftLegGroup.rotation.x, -0.06, 16, delta);
        rightLegGroup.rotation.x = THREE.MathUtils.damp(rightLegGroup.rotation.x, 0.06, 16, delta);
        leftKneeGroup.rotation.x = THREE.MathUtils.damp(leftKneeGroup.rotation.x, 0.12, 16, delta);
        rightKneeGroup.rotation.x = THREE.MathUtils.damp(rightKneeGroup.rotation.x, 0.12, 16, delta);

        slashTrailMesh.scale.setScalar(0.0001);
      }
      return;
    }

    // =========================================================================================
    // CASE C: UNARMED BEAST / DRAGON PREDATORY LEAP + FANG BITE ("Salto con Mordida")
    // =========================================================================================
    slashTrailMesh.scale.setScalar(0.0001);

    if (u < 0.24) {
      // PHASE 1: Predatory Stalk Crouch & Coiled Haunches (0.00 -> 0.24)
      const p = u / 0.24;
      const coil = Math.sin(p * Math.PI * 0.5);
      _pelvisAnimOffset.y = -0.22 * coil;
      _pelvisAnimOffset.z = -0.2 * coil;
      rig.rootBone.position.z = THREE.MathUtils.damp(rig.rootBone.position.z, -0.14 * coil, 18, delta);
      rig.rootBone.position.y = THREE.MathUtils.damp(rig.rootBone.position.y, 0, 18, delta);

      pelvisGroup.rotation.x = THREE.MathUtils.damp(pelvisGroup.rotation.x, 0.18 * coil, 18, delta);
      torsoGroup.rotation.x = THREE.MathUtils.damp(torsoGroup.rotation.x, 0.24 * coil, 18, delta);
      neckGroup.rotation.x = THREE.MathUtils.damp(neckGroup.rotation.x, -0.28 * coil, 18, delta);
      headGroup.rotation.x = THREE.MathUtils.damp(headGroup.rotation.x, -0.18 * coil, 18, delta);
      headGroup.rotation.z = THREE.MathUtils.damp(headGroup.rotation.z, 0, 18, delta);
      jawGroup.rotation.x = THREE.MathUtils.damp(jawGroup.rotation.x, 0.18 * coil, 18, delta);

      // Deep hind-leg compression & coiled forelegs ready to spring
      leftLegGroup.rotation.x = THREE.MathUtils.damp(leftLegGroup.rotation.x, -0.58 * coil, 18, delta);
      rightLegGroup.rotation.x = THREE.MathUtils.damp(rightLegGroup.rotation.x, -0.58 * coil, 18, delta);
      leftKneeGroup.rotation.x = THREE.MathUtils.damp(leftKneeGroup.rotation.x, 0.24 + 0.62 * coil, 18, delta);
      rightKneeGroup.rotation.x = THREE.MathUtils.damp(rightKneeGroup.rotation.x, 0.24 + 0.62 * coil, 18, delta);

      leftArmGroup.rotation.x = THREE.MathUtils.damp(leftArmGroup.rotation.x, 0.44 * coil, 18, delta);
      rightArmGroup.rotation.x = THREE.MathUtils.damp(rightArmGroup.rotation.x, 0.44 * coil, 18, delta);
      leftElbowGroup.rotation.x = THREE.MathUtils.damp(leftElbowGroup.rotation.x, -0.26 - 0.48 * coil, 18, delta);
      rightElbowGroup.rotation.x = THREE.MathUtils.damp(rightElbowGroup.rotation.x, -0.26 - 0.48 * coil, 18, delta);

      tailGroup.rotation.x = THREE.MathUtils.damp(tailGroup.rotation.x, -0.25 * coil, 16, delta);
      breathConeMesh.scale.setScalar(0.0001);
    } else if (u < 0.56) {
      // PHASE 2: Explosive Ballistic Leap & Maximum Jaw Gape in Mid-Air (0.24 -> 0.56)
      const airU = (u - 0.24) / 0.32;
      const arcHeight = Math.sin(airU * Math.PI) * 0.64;
      const forwardDist = THREE.MathUtils.lerp(-0.2, 0.82, airU);

      _pelvisAnimOffset.y = arcHeight;
      _pelvisAnimOffset.z = forwardDist * 0.45;
      rig.rootBone.position.z = THREE.MathUtils.damp(rig.rootBone.position.z, forwardDist * 0.55, 22, delta);
      rig.rootBone.position.y = THREE.MathUtils.damp(rig.rootBone.position.y, arcHeight * 0.25, 22, delta);

      // Pitch body upward on takeoff (-0.44 rad), level at apex, pitch downward onto prey (+0.34 rad)
      const pitch = THREE.MathUtils.lerp(-0.44, 0.34, airU);
      pelvisGroup.rotation.x = THREE.MathUtils.damp(pelvisGroup.rotation.x, pitch, 22, delta);
      torsoGroup.rotation.x = THREE.MathUtils.damp(torsoGroup.rotation.x, pitch * 0.5, 22, delta);

      // Extend neck toward target and open jaws wide for the lethal bite
      neckGroup.rotation.x = THREE.MathUtils.damp(neckGroup.rotation.x, 0.34, 22, delta);
      headGroup.rotation.x = THREE.MathUtils.damp(headGroup.rotation.x, -0.16 + airU * 0.28, 22, delta);
      jawGroup.rotation.x = THREE.MathUtils.damp(jawGroup.rotation.x, 0.96, 24, delta);

      // Hind legs fully extended backward from the explosive push-off
      leftLegGroup.rotation.x = THREE.MathUtils.damp(leftLegGroup.rotation.x, 0.74, 22, delta);
      rightLegGroup.rotation.x = THREE.MathUtils.damp(rightLegGroup.rotation.x, 0.74, 22, delta);
      leftKneeGroup.rotation.x = THREE.MathUtils.damp(leftKneeGroup.rotation.x, 0.08, 22, delta);
      rightKneeGroup.rotation.x = THREE.MathUtils.damp(rightKneeGroup.rotation.x, 0.08, 22, delta);

      // Forelegs / Claws reaching forward and spread wide to seize the prey
      leftArmGroup.rotation.x = THREE.MathUtils.damp(leftArmGroup.rotation.x, -1.12, 22, delta);
      rightArmGroup.rotation.x = THREE.MathUtils.damp(rightArmGroup.rotation.x, -1.12, 22, delta);
      leftArmGroup.rotation.z = THREE.MathUtils.damp(leftArmGroup.rotation.z, 0.24, 20, delta);
      rightArmGroup.rotation.z = THREE.MathUtils.damp(rightArmGroup.rotation.z, -0.24, 20, delta);
      leftElbowGroup.rotation.x = THREE.MathUtils.damp(leftElbowGroup.rotation.x, -0.22, 22, delta);
      rightElbowGroup.rotation.x = THREE.MathUtils.damp(rightElbowGroup.rotation.x, -0.22, 22, delta);

      // Wings flare wide at apex of the pounce
      leftWingGroup.rotation.y = THREE.MathUtils.damp(leftWingGroup.rotation.y, 0.68, 18, delta);
      rightWingGroup.rotation.y = THREE.MathUtils.damp(rightWingGroup.rotation.y, -0.68, 18, delta);
      tailGroup.rotation.x = THREE.MathUtils.damp(tailGroup.rotation.x, 0.48, 18, delta);

      if (airU > 0.75) {
        const preSnap = (airU - 0.75) / 0.25;
        breathConeMesh.scale.setScalar(Math.max(0.0001, preSnap * 0.65));
        mouthBreathIntensity = preSnap * 0.75;
      } else {
        breathConeMesh.scale.setScalar(0.0001);
      }
    } else if (u < 0.76) {
      // PHASE 3: Violent Fang Bite Clamp, Savage Head-Tear & Forepaw Pin on Impact (0.56 -> 0.76)
      const biteU = (u - 0.56) / 0.2;
      const shockAbsorb = Math.sin(biteU * Math.PI);

      _pelvisAnimOffset.y = -0.14 * shockAbsorb;
      _pelvisAnimOffset.z = 0.36 * (1 - biteU * 0.25);
      rig.rootBone.position.z = THREE.MathUtils.damp(
        rig.rootBone.position.z,
        0.44 * (1 - biteU * 0.25),
        22,
        delta
      );
      rig.rootBone.position.y = THREE.MathUtils.damp(rig.rootBone.position.y, 0, 22, delta);

      pelvisGroup.rotation.x = THREE.MathUtils.damp(pelvisGroup.rotation.x, 0.14, 22, delta);
      torsoGroup.rotation.x = THREE.MathUtils.damp(torsoGroup.rotation.x, 0.18, 22, delta);

      // Jaws SLAM SHUT violently and head shakes/twists ferociously to tear prey!
      const tearShake = Math.sin(biteU * Math.PI * 4.0);
      jawGroup.rotation.x = THREE.MathUtils.damp(jawGroup.rotation.x, 0.02, 32, delta);
      neckGroup.rotation.x = THREE.MathUtils.damp(neckGroup.rotation.x, 0.38, 22, delta);
      neckGroup.rotation.y = THREE.MathUtils.damp(neckGroup.rotation.y, tearShake * 0.22, 24, delta);
      headGroup.rotation.x = THREE.MathUtils.damp(headGroup.rotation.x, 0.18, 22, delta);
      headGroup.rotation.z = THREE.MathUtils.damp(headGroup.rotation.z, tearShake * 0.34, 26, delta);
      headGroup.rotation.y = THREE.MathUtils.damp(
        headGroup.rotation.y,
        Math.cos(biteU * Math.PI * 4.0) * 0.22,
        24,
        delta
      );

      // Forepaws slam down pinning the target while elbows & knees absorb the landing impact
      leftArmGroup.rotation.x = THREE.MathUtils.damp(leftArmGroup.rotation.x, -0.42, 22, delta);
      rightArmGroup.rotation.x = THREE.MathUtils.damp(rightArmGroup.rotation.x, -0.42, 22, delta);
      leftArmGroup.rotation.z = THREE.MathUtils.damp(leftArmGroup.rotation.z, 0.06, 20, delta);
      rightArmGroup.rotation.z = THREE.MathUtils.damp(rightArmGroup.rotation.z, -0.06, 20, delta);
      leftElbowGroup.rotation.x = THREE.MathUtils.damp(leftElbowGroup.rotation.x, -0.54, 22, delta);
      rightElbowGroup.rotation.x = THREE.MathUtils.damp(rightElbowGroup.rotation.x, -0.54, 22, delta);

      leftLegGroup.rotation.x = THREE.MathUtils.damp(leftLegGroup.rotation.x, -0.28, 20, delta);
      rightLegGroup.rotation.x = THREE.MathUtils.damp(rightLegGroup.rotation.x, -0.28, 20, delta);
      leftKneeGroup.rotation.x = THREE.MathUtils.damp(leftKneeGroup.rotation.x, 0.56, 20, delta);
      rightKneeGroup.rotation.x = THREE.MathUtils.damp(rightKneeGroup.rotation.x, 0.56, 20, delta);

      // Bite impact shockwave & fiery/spark burst at the jaws
      const fxScale = Math.max(0.0001, (1.0 - biteU * 0.55) * 1.28);
      breathConeMesh.scale.set(
        fxScale,
        fxScale,
        fxScale * (effectiveWeapon === 'dragon_breath_maw' || effectiveWeapon === 'frost_breath_maw' ? 0.85 : 1.05)
      );
      breathConeMesh.rotation.z = tearShake * 0.25;
      mouthBreathIntensity = 1.0 - biteU * 0.4;
    } else {
      // PHASE 4: Agile Predatory Push-Off & Recovery Back to Stance (0.76 -> 1.00)
      const recU = (u - 0.76) / 0.24;
      const easeBack = 1.0 - Math.sin(recU * Math.PI * 0.5);
      const hopBackArc = Math.sin(recU * Math.PI) * 0.12;

      _pelvisAnimOffset.y = hopBackArc;
      _pelvisAnimOffset.z = 0.27 * easeBack;
      rig.rootBone.position.z = THREE.MathUtils.damp(rig.rootBone.position.z, 0.33 * easeBack, 18, delta);
      rig.rootBone.position.y = THREE.MathUtils.damp(rig.rootBone.position.y, 0, 18, delta);

      pelvisGroup.rotation.x = THREE.MathUtils.damp(pelvisGroup.rotation.x, 0, 16, delta);
      torsoGroup.rotation.x = THREE.MathUtils.damp(torsoGroup.rotation.x, 0, 16, delta);
      neckGroup.rotation.x = THREE.MathUtils.damp(neckGroup.rotation.x, 0, 16, delta);
      neckGroup.rotation.y = THREE.MathUtils.damp(neckGroup.rotation.y, 0, 16, delta);
      headGroup.rotation.x = THREE.MathUtils.damp(headGroup.rotation.x, 0, 16, delta);
      headGroup.rotation.y = THREE.MathUtils.damp(headGroup.rotation.y, 0, 16, delta);
      headGroup.rotation.z = THREE.MathUtils.damp(headGroup.rotation.z, 0, 16, delta);
      jawGroup.rotation.x = THREE.MathUtils.damp(jawGroup.rotation.x, 0.08, 16, delta);

      leftArmGroup.rotation.x = THREE.MathUtils.damp(leftArmGroup.rotation.x, isQuadruped ? 0.08 : -0.18, 16, delta);
      rightArmGroup.rotation.x = THREE.MathUtils.damp(rightArmGroup.rotation.x, isQuadruped ? 0.08 : -0.18, 16, delta);
      leftArmGroup.rotation.z = THREE.MathUtils.damp(leftArmGroup.rotation.z, 0, 16, delta);
      rightArmGroup.rotation.z = THREE.MathUtils.damp(rightArmGroup.rotation.z, 0, 16, delta);
      leftElbowGroup.rotation.x = THREE.MathUtils.damp(leftElbowGroup.rotation.x, isQuadruped ? -0.18 : -0.32, 16, delta);
      rightElbowGroup.rotation.x = THREE.MathUtils.damp(rightElbowGroup.rotation.x, isQuadruped ? -0.18 : -0.32, 16, delta);

      leftLegGroup.rotation.x = THREE.MathUtils.damp(leftLegGroup.rotation.x, isQuadruped ? -0.12 : -0.06, 16, delta);
      rightLegGroup.rotation.x = THREE.MathUtils.damp(rightLegGroup.rotation.x, isQuadruped ? -0.12 : 0.06, 16, delta);
      leftKneeGroup.rotation.x = THREE.MathUtils.damp(leftKneeGroup.rotation.x, isQuadruped ? 0.24 : 0.12, 16, delta);
      rightKneeGroup.rotation.x = THREE.MathUtils.damp(rightKneeGroup.rotation.x, isQuadruped ? 0.24 : 0.12, 16, delta);

      breathConeMesh.scale.setScalar(0.0001);
    }
  };

  if (animState === 'idle') {
    const breath = Math.sin(t * 2.3);
    const weightShift = Math.sin(t * 2.3);
    const alertLook = Math.cos(t * 2.3);
    breathConeMesh.scale.setScalar(0.0001);
    slashTrailMesh.scale.setScalar(0.0001);

    // Subtle grounded lateral weight shift between stance limbs (zero vertical hopping!)
    if (!isFloating) {
      _pelvisAnimOffset.x = weightShift * 0.018;
      _pelvisAnimOffset.y = breath * 0.008;
    }

    // Organic ribcage breathing expansion
    const breathScale = 1.0 + breath * 0.026;
    torsoGroup.scale.set(breathScale, 1.0 + breath * 0.016, breathScale);
    torsoGroup.rotation.x = THREE.MathUtils.damp(torsoGroup.rotation.x, breath * 0.025, dampLambda, delta);

    jawGroup.rotation.x = THREE.MathUtils.damp(jawGroup.rotation.x, 0.06 + Math.max(0, breath * 0.11), dampLambda, delta);
    neckGroup.rotation.y = THREE.MathUtils.damp(neckGroup.rotation.y, weightShift * 0.12, dampLambda, delta);
    neckGroup.rotation.x = THREE.MathUtils.damp(neckGroup.rotation.x, breath * 0.035, dampLambda, delta);
    headGroup.rotation.y = THREE.MathUtils.damp(headGroup.rotation.y, weightShift * 0.11 + alertLook * 0.06, dampLambda, delta);
    headGroup.rotation.x = THREE.MathUtils.damp(headGroup.rotation.x, -breath * 0.045, dampLambda, delta);

    leftShoulderGroup.rotation.z = THREE.MathUtils.damp(leftShoulderGroup.rotation.z, breath * 0.045, dampLambda, delta);
    rightShoulderGroup.rotation.z = THREE.MathUtils.damp(rightShoulderGroup.rotation.z, -breath * 0.045, dampLambda, delta);

    // Natural bent elbows & knees in grounded idle stance
    rightArmGroup.rotation.x = THREE.MathUtils.damp(rightArmGroup.rotation.x, isQuadruped ? 0.08 : -0.18 + breath * 0.05, dampLambda, delta);
    leftArmGroup.rotation.x = THREE.MathUtils.damp(leftArmGroup.rotation.x, isQuadruped ? 0.08 : -0.22 - breath * 0.04, dampLambda, delta);
    leftElbowGroup.rotation.x = THREE.MathUtils.damp(leftElbowGroup.rotation.x, isQuadruped ? -0.18 : -0.32 - breath * 0.05, dampLambda, delta);
    rightElbowGroup.rotation.x = THREE.MathUtils.damp(rightElbowGroup.rotation.x, isQuadruped ? -0.18 : -0.32 + breath * 0.05, dampLambda, delta);

    leftLegGroup.rotation.x = THREE.MathUtils.damp(leftLegGroup.rotation.x, isQuadruped ? -0.12 : -0.08, dampLambda, delta);
    rightLegGroup.rotation.x = THREE.MathUtils.damp(rightLegGroup.rotation.x, isQuadruped ? -0.12 : 0.06, dampLambda, delta);
    leftKneeGroup.rotation.x = THREE.MathUtils.damp(leftKneeGroup.rotation.x, isQuadruped ? 0.24 : 0.14, dampLambda, delta);
    rightKneeGroup.rotation.x = THREE.MathUtils.damp(rightKneeGroup.rotation.x, isQuadruped ? 0.24 : 0.08, dampLambda, delta);

    pelvisGroup.rotation.x = THREE.MathUtils.damp(pelvisGroup.rotation.x, 0, dampLambda, delta);
    tailGroup.rotation.x = THREE.MathUtils.damp(tailGroup.rotation.x, Math.cos(t * 2.3) * 0.08, dampLambda, delta);
  } else if (animState === 'walk') {
    breathConeMesh.scale.setScalar(0.0001);
    slashTrailMesh.scale.setScalar(0.0001);
    // Period = 2 * PI / 5.3247 = 1.1800s (matches Walk clip duration for 100% seamless loop)
    const phase = t * 5.3247;

    if (isQuadruped) {
      // BIOMECHANICAL 4-BEAT QUADRUPED PROWL / TROT GAIT (Zero hopping!)
      // Diagonal-lateral phase offsets: Hind-Left (0), Fore-Left (+1.25), Hind-Right (PI), Fore-Right (PI + 1.25)
      const pHL = phase;
      const pHR = phase + Math.PI;
      const pFL = phase + 1.25;
      const pFR = phase + Math.PI + 1.25;

      const sinHL = Math.sin(pHL);
      const sinHR = Math.sin(pHR);
      const sinFL = Math.sin(pFL);
      const sinFR = Math.sin(pFR);

      // Grounded hip height: lowers slightly when legs reach full extension, never hops upward!
      _pelvisAnimOffset.y = Math.cos(phase * 2.0) * 0.015 - 0.012;
      _pelvisAnimOffset.x = Math.cos(phase) * 0.022;
      _pelvisAnimOffset.z = Math.sin(phase * 2.0) * 0.014;

      // Pelvic & Scapular weight transfer (hip roll, pelvic yaw, counter-rolling chest)
      pelvisGroup.rotation.x = THREE.MathUtils.damp(pelvisGroup.rotation.x, Math.sin(phase * 2.0) * 0.028, dampLambda, delta);
      pelvisGroup.rotation.z = THREE.MathUtils.damp(pelvisGroup.rotation.z, Math.cos(pHL) * 0.048, dampLambda, delta);
      pelvisGroup.rotation.y = THREE.MathUtils.damp(pelvisGroup.rotation.y, sinHL * 0.055, dampLambda, delta);

      torsoGroup.rotation.x = THREE.MathUtils.damp(torsoGroup.rotation.x, 0.04 - Math.sin(phase * 2.0) * 0.024, dampLambda, delta);
      torsoGroup.rotation.z = THREE.MathUtils.damp(torsoGroup.rotation.z, -Math.cos(pFL) * 0.042, dampLambda, delta);
      torsoGroup.rotation.y = THREE.MathUtils.damp(torsoGroup.rotation.y, -sinFL * 0.048, dampLambda, delta);

      // Shoulder blades slide up/down as each foreleg bears weight in stance phase
      leftShoulderGroup.rotation.z = THREE.MathUtils.damp(leftShoulderGroup.rotation.z, -sinFL * 0.08, dampLambda, delta);
      rightShoulderGroup.rotation.z = THREE.MathUtils.damp(rightShoulderGroup.rotation.z, sinFR * 0.08, dampLambda, delta);

      // Gyroscopic predator head stabilization: neck absorbs torso sway so head stays focused on target
      neckGroup.rotation.x = THREE.MathUtils.damp(neckGroup.rotation.x, 0.08 + Math.sin(phase * 2.0) * 0.032, dampLambda, delta);
      neckGroup.rotation.y = THREE.MathUtils.damp(neckGroup.rotation.y, sinFL * 0.045, dampLambda, delta);
      headGroup.rotation.x = THREE.MathUtils.damp(headGroup.rotation.x, -0.05 - Math.sin(phase * 2.0) * 0.032, dampLambda, delta);
      headGroup.rotation.y = THREE.MathUtils.damp(headGroup.rotation.y, -sinFL * 0.035, dampLambda, delta);
      jawGroup.rotation.x = THREE.MathUtils.damp(jawGroup.rotation.x, 0.12 + Math.sin(phase * 2.0) * 0.03, dampLambda, delta);

      // Hind legs: swing forward (-rotX) with hock/knee flexion during swing (cos > 0), firm extension on push-off
      const swingLiftHL = Math.max(0, Math.cos(pHL));
      const swingLiftHR = Math.max(0, Math.cos(pHR));
      leftLegGroup.rotation.x = THREE.MathUtils.damp(leftLegGroup.rotation.x, -0.08 - sinHL * 0.44, dampLambda, delta);
      rightLegGroup.rotation.x = THREE.MathUtils.damp(rightLegGroup.rotation.x, -0.08 - sinHR * 0.44, dampLambda, delta);
      leftKneeGroup.rotation.x = THREE.MathUtils.damp(
        leftKneeGroup.rotation.x,
        0.18 + swingLiftHL * 0.48 + Math.max(0, -sinHL) * 0.08,
        dampLambda,
        delta
      );
      rightKneeGroup.rotation.x = THREE.MathUtils.damp(
        rightKneeGroup.rotation.x,
        0.18 + swingLiftHR * 0.48 + Math.max(0, -sinHR) * 0.08,
        dampLambda,
        delta
      );

      // Forelegs: reach forward (-rotX) with carpal/elbow bend during swing (cos > 0), planted straight during stance
      const swingLiftFL = Math.max(0, Math.cos(pFL));
      const swingLiftFR = Math.max(0, Math.cos(pFR));
      leftArmGroup.rotation.x = THREE.MathUtils.damp(leftArmGroup.rotation.x, 0.06 - sinFL * 0.42, dampLambda, delta);
      rightArmGroup.rotation.x = THREE.MathUtils.damp(rightArmGroup.rotation.x, 0.06 - sinFR * 0.42, dampLambda, delta);
      leftElbowGroup.rotation.x = THREE.MathUtils.damp(
        leftElbowGroup.rotation.x,
        -0.14 - swingLiftFL * 0.46,
        dampLambda,
        delta
      );
      rightElbowGroup.rotation.x = THREE.MathUtils.damp(
        rightElbowGroup.rotation.x,
        -0.14 - swingLiftFR * 0.46,
        dampLambda,
        delta
      );
    } else if (isSerpentine) {
      // SERPENTINE S-WAVE LATERAL SLITHER (Grounded undulation, zero vertical bounce)
      const slither = Math.sin(phase);
      _pelvisAnimOffset.x = slither * 0.085;
      _pelvisAnimOffset.y = -0.01;
      pelvisGroup.rotation.y = THREE.MathUtils.damp(pelvisGroup.rotation.y, Math.cos(phase) * 0.18, dampLambda, delta);
      torsoGroup.rotation.y = THREE.MathUtils.damp(torsoGroup.rotation.y, -Math.cos(phase) * 0.14, dampLambda, delta);
      torsoGroup.rotation.x = THREE.MathUtils.damp(torsoGroup.rotation.x, 0.08, dampLambda, delta);
      neckGroup.rotation.y = THREE.MathUtils.damp(neckGroup.rotation.y, Math.cos(phase - 0.8) * 0.12, dampLambda, delta);
      headGroup.rotation.y = THREE.MathUtils.damp(headGroup.rotation.y, -Math.cos(phase - 0.8) * 0.1, dampLambda, delta);
      rightArmGroup.rotation.x = THREE.MathUtils.damp(rightArmGroup.rotation.x, slither * 0.25 - 0.15, dampLambda, delta);
      leftArmGroup.rotation.x = THREE.MathUtils.damp(leftArmGroup.rotation.x, -slither * 0.25 - 0.15, dampLambda, delta);
    } else {
      // BIOMECHANICAL BIPED / TITAN HEAVY STRIDE (Contrapposto hip-drop, heel-to-toe knee bend, torso counter-twist)
      const stride = Math.sin(phase);
      const liftL = Math.max(0, Math.cos(phase));
      const liftR = Math.max(0, -Math.cos(phase));

      if (!isFloating) {
        // Hips drop slightly when legs are apart at peak stride, never hopping above standing height
        _pelvisAnimOffset.y = -Math.abs(stride) * 0.024 + Math.cos(phase * 2.0) * 0.012;
        _pelvisAnimOffset.x = -stride * 0.026;
      }

      pelvisGroup.rotation.x = THREE.MathUtils.damp(pelvisGroup.rotation.x, 0.04, dampLambda, delta);
      pelvisGroup.rotation.z = THREE.MathUtils.damp(pelvisGroup.rotation.z, stride * 0.052, dampLambda, delta);
      pelvisGroup.rotation.y = THREE.MathUtils.damp(pelvisGroup.rotation.y, -stride * 0.085, dampLambda, delta);

      // Upper torso leans slightly forward and counter-rotates against the pelvis
      torsoGroup.rotation.x = THREE.MathUtils.damp(torsoGroup.rotation.x, 0.07 + Math.abs(stride) * 0.025, dampLambda, delta);
      torsoGroup.rotation.y = THREE.MathUtils.damp(torsoGroup.rotation.y, stride * 0.095, dampLambda, delta);
      torsoGroup.rotation.z = THREE.MathUtils.damp(torsoGroup.rotation.z, -stride * 0.038, dampLambda, delta);

      headGroup.rotation.y = THREE.MathUtils.damp(headGroup.rotation.y, -stride * 0.06, dampLambda, delta);
      jawGroup.rotation.x = THREE.MathUtils.damp(jawGroup.rotation.x, 0.12, dampLambda, delta);

      // Legs: forward swing (-rotX) bends the knee cleanly to clear the ground; stance leg extends firmly
      leftLegGroup.rotation.x = THREE.MathUtils.damp(leftLegGroup.rotation.x, -stride * 0.46, dampLambda, delta);
      rightLegGroup.rotation.x = THREE.MathUtils.damp(rightLegGroup.rotation.x, stride * 0.46, dampLambda, delta);
      leftKneeGroup.rotation.x = THREE.MathUtils.damp(
        leftKneeGroup.rotation.x,
        0.1 + liftL * 0.56 + Math.max(0, -stride) * 0.08,
        dampLambda,
        delta
      );
      rightKneeGroup.rotation.x = THREE.MathUtils.damp(
        rightKneeGroup.rotation.x,
        0.1 + liftR * 0.56 + Math.max(0, stride) * 0.08,
        dampLambda,
        delta
      );

      // Arms swing in counter-phase to legs with natural elbow lag
      rightArmGroup.rotation.x = THREE.MathUtils.damp(rightArmGroup.rotation.x, -stride * 0.42 - 0.12, dampLambda, delta);
      leftArmGroup.rotation.x = THREE.MathUtils.damp(leftArmGroup.rotation.x, stride * 0.42 - 0.12, dampLambda, delta);
      rightElbowGroup.rotation.x = THREE.MathUtils.damp(
        rightElbowGroup.rotation.x,
        -0.28 - Math.max(0, stride) * 0.36,
        dampLambda,
        delta
      );
      leftElbowGroup.rotation.x = THREE.MathUtils.damp(
        leftElbowGroup.rotation.x,
        -0.28 - Math.max(0, -stride) * 0.36,
        dampLambda,
        delta
      );
    }

    dorsalGroup.rotation.x = THREE.MathUtils.damp(
      dorsalGroup.rotation.x,
      0.12 + Math.abs(Math.sin(phase)) * 0.08,
      dampLambda,
      delta
    );
  } else if (animState === 'jump_bite') {
    // DEDICATED PREDATORY POUNCE + FANG BITE SKILL ("Salto con Mordida")
    applyPredatoryJumpBite(1.85);
  } else if (animState === 'attack') {
    // If weapon is explicitly set to 'leap_fang_bite', 'attack' also executes the full Jump-Bite pounce!
    if (effectiveWeapon === 'leap_fang_bite') {
      applyPredatoryJumpBite(2.513);
    } else {
      const cycle = (t * 2.5) % (Math.PI * 2);
      const swing = Math.sin(cycle);
      const lunge = Math.max(0, Math.sin(cycle - 0.4));
      const windup = Math.max(0, -swing);

      if (
        isAnimal ||
        effectiveWeapon === 'ram_charge' ||
        effectiveWeapon === 'dragon_breath_maw' ||
        effectiveWeapon === 'frost_breath_maw'
      ) {
        // ANIMALS NEVER SWING HANDHELD WEAPONS: Suppress paw slash trail completely
        slashTrailMesh.scale.setScalar(0.0001);

        if (effectiveWeapon === 'ram_charge') {
          // 1. EMBESTIDA BRUTAL / CORNADA (Grounded accelerating stampede + lowered horns + upward gore lift)
          const goreSnap = Math.max(0, Math.sin(cycle - 0.85));
          const chargeZ = -0.26 * windup + 0.76 * lunge;

          _pelvisAnimOffset.y = -0.08 * windup + 0.04 * goreSnap;
          _pelvisAnimOffset.z = chargeZ * 0.35;
          rig.rootBone.position.z = THREE.MathUtils.damp(rig.rootBone.position.z, chargeZ * 0.65, 18, delta);
          rig.rootBone.position.y = THREE.MathUtils.damp(rig.rootBone.position.y, 0, 18, delta);

          pelvisGroup.rotation.x = THREE.MathUtils.damp(
            pelvisGroup.rotation.x,
            0.14 * windup - 0.16 * goreSnap,
            18,
            delta
          );
          torsoGroup.rotation.x = THREE.MathUtils.damp(
            torsoGroup.rotation.x,
            0.22 * (windup + lunge * 0.5) - 0.24 * goreSnap,
            18,
            delta
          );

          // Lower neck & head during windup/charge to aim horns/tusks, then snap upward on gore impact
          neckGroup.rotation.x = THREE.MathUtils.damp(
            neckGroup.rotation.x,
            0.44 * (windup + lunge * 0.6) - goreSnap * 0.88,
            18,
            delta
          );
          headGroup.rotation.x = THREE.MathUtils.damp(
            headGroup.rotation.x,
            0.56 * (windup + lunge * 0.5) - goreSnap * 1.08,
            18,
            delta
          );
          jawGroup.rotation.x = THREE.MathUtils.damp(jawGroup.rotation.x, 0.12 + goreSnap * 0.48, 16, delta);

          // Kinetic Ram Shockwave Cone at snout/horns
          const ramScale = lunge > 0.22 ? Math.max(0.0001, lunge * 1.15) : 0.0001;
          breathConeMesh.scale.set(ramScale, ramScale, ramScale);
          breathConeMesh.rotation.z = t * 4.0;

          // Galloping/driving leg stride during the charge
          const drivePhase = cycle * 2.0;
          leftArmGroup.rotation.x = THREE.MathUtils.damp(
            leftArmGroup.rotation.x,
            0.28 * windup - 0.54 * lunge + Math.sin(drivePhase) * 0.18 * lunge,
            16,
            delta
          );
          rightArmGroup.rotation.x = THREE.MathUtils.damp(
            rightArmGroup.rotation.x,
            0.28 * windup - 0.54 * lunge - Math.sin(drivePhase) * 0.18 * lunge,
            16,
            delta
          );
          leftElbowGroup.rotation.x = THREE.MathUtils.damp(leftElbowGroup.rotation.x, -0.28, 14, delta);
          rightElbowGroup.rotation.x = THREE.MathUtils.damp(rightElbowGroup.rotation.x, -0.28, 14, delta);

          leftLegGroup.rotation.x = THREE.MathUtils.damp(
            leftLegGroup.rotation.x,
            -0.34 * windup + 0.62 * lunge - Math.sin(drivePhase) * 0.16 * lunge,
            16,
            delta
          );
          rightLegGroup.rotation.x = THREE.MathUtils.damp(
            rightLegGroup.rotation.x,
            -0.34 * windup + 0.62 * lunge + Math.sin(drivePhase) * 0.16 * lunge,
            16,
            delta
          );
          leftKneeGroup.rotation.x = THREE.MathUtils.damp(leftKneeGroup.rotation.x, 0.26 + lunge * 0.24, 14, delta);
          rightKneeGroup.rotation.x = THREE.MathUtils.damp(rightKneeGroup.rotation.x, 0.26 + lunge * 0.24, 14, delta);
          tailGroup.rotation.x = THREE.MathUtils.damp(tailGroup.rotation.x, lunge * 0.55, 14, delta);
        } else if (
          effectiveWeapon === 'dragon_breath_maw' ||
          effectiveWeapon === 'frost_breath_maw'
        ) {
          // 2. FUEGO REAL O SKILL DE HIELO POR LA BOCA (Grounded brace + deep chest inhale + sweeping volumetric fire stream)
          const isFrost = effectiveWeapon === 'frost_breath_maw';
          _pelvisAnimOffset.y = -0.05 * lunge;
          _pelvisAnimOffset.z = -0.1 * windup + 0.14 * lunge;
          rig.rootBone.position.z = THREE.MathUtils.damp(
            rig.rootBone.position.z,
            -0.08 * windup + 0.12 * lunge,
            14,
            delta
          );
          rig.rootBone.position.y = THREE.MathUtils.damp(rig.rootBone.position.y, 0, 14, delta);

          // Deep chest inhalation before blasting fire/ice from the mouth
          const inhale = 1.0 + windup * 0.09 + lunge * 0.045;
          torsoGroup.scale.set(inhale, 1.0 + lunge * 0.035, inhale);
          torsoGroup.rotation.x = THREE.MathUtils.damp(
            torsoGroup.rotation.x,
            -0.14 * windup + 0.16 * lunge,
            16,
            delta
          );

          neckGroup.rotation.x = THREE.MathUtils.damp(
            neckGroup.rotation.x,
            -0.28 * windup + 0.34 * lunge,
            16,
            delta
          );
          headGroup.rotation.x = THREE.MathUtils.damp(
            headGroup.rotation.x,
            -0.24 * windup - 0.12 * lunge,
            16,
            delta
          );
          // Natural sweeping fire arc across the battlefield
          const sweepYaw = lunge > 0.08 ? Math.sin(t * 4.2) * 0.22 * lunge : 0;
          neckGroup.rotation.y = THREE.MathUtils.damp(neckGroup.rotation.y, sweepYaw * 0.45, 14, delta);
          headGroup.rotation.y = THREE.MathUtils.damp(headGroup.rotation.y, sweepYaw, 14, delta);
          jawGroup.rotation.x = THREE.MathUtils.damp(jawGroup.rotation.x, 0.1 + lunge * 0.84, 18, delta);

          for (let i = 0; i < extraHeads.length; i++) {
            extraHeads[i].rotation.x = -0.18 * windup + 0.28 * lunge;
          }

          // Volumetric Plume organic turbulence & UV scrolling (NO mechanical drill spinning!)
          if (lunge > 0.06) {
            const bScale = Math.max(0.0001, lunge * 1.18);
            const flickerX = 1.0 + Math.sin(t * 31.0) * 0.09 + Math.cos(t * 47.0) * 0.05;
            const flickerY = 1.0 + Math.cos(t * 29.0) * 0.09 + Math.sin(t * 43.0) * 0.05;
            const stretchZ = (isFrost ? 1.38 : 1.35) * (0.94 + Math.sin(t * 23.0) * 0.08);
            breathConeMesh.scale.set(bScale * flickerX, bScale * flickerY, bScale * stretchZ);
            // Subtle organic flame wave oscillation instead of 360-degree spinning cone
            breathConeMesh.rotation.z = Math.sin(t * 14.0) * 0.16;

            // Counter-oscillate inner flame layers for 3D volumetric depth
            for (let c = 0; c < breathConeMesh.children.length; c++) {
              const child = breathConeMesh.children[c];
              if (child.name === 'Breath_SubLayer_Mid') {
                child.rotation.z = -Math.sin(t * 19.0 + 1.2) * 0.22;
              } else if (child.name === 'Breath_SubLayer_Core') {
                child.scale.set(
                  0.92 + Math.sin(t * 37.0) * 0.12,
                  0.92 + Math.cos(t * 41.0) * 0.12,
                  1.0
                );
              }
            }

            mouthBreathIntensity = Math.min(1.0, lunge * 1.25);
            mouthParticleMode = isFrost ? 'frost' : 'fire';
          } else {
            breathConeMesh.scale.setScalar(0.0001);
          }

          // Grounded braced stance on all legs while firing from the mouth
          leftArmGroup.rotation.x = THREE.MathUtils.damp(leftArmGroup.rotation.x, -0.18 * lunge, 14, delta);
          rightArmGroup.rotation.x = THREE.MathUtils.damp(rightArmGroup.rotation.x, -0.18 * lunge, 14, delta);
          leftElbowGroup.rotation.x = THREE.MathUtils.damp(leftElbowGroup.rotation.x, -0.24, 14, delta);
          rightElbowGroup.rotation.x = THREE.MathUtils.damp(rightElbowGroup.rotation.x, -0.24, 14, delta);

          leftLegGroup.rotation.x = THREE.MathUtils.damp(leftLegGroup.rotation.x, 0.18 * lunge, 14, delta);
          rightLegGroup.rotation.x = THREE.MathUtils.damp(rightLegGroup.rotation.x, 0.18 * lunge, 14, delta);
          leftKneeGroup.rotation.x = THREE.MathUtils.damp(leftKneeGroup.rotation.x, 0.26, 14, delta);
          rightKneeGroup.rotation.x = THREE.MathUtils.damp(rightKneeGroup.rotation.x, 0.26, 14, delta);
          tailGroup.rotation.x = THREE.MathUtils.damp(tailGroup.rotation.x, lunge * 0.42, 14, delta);
        } else {
          // 3. MORDISCO SALVAJE CON FAUCES (Grounded predatory lunge + jaws open wide + violent jaw snap & tear)
          const jawOpenWave = Math.max(0, Math.sin(cycle + 0.35));
          const biteSnap = Math.max(0, Math.sin(cycle - 0.72));
          const lungeZ = -0.16 * windup + 0.56 * lunge;

          _pelvisAnimOffset.y = -0.09 * windup + 0.04 * lunge;
          _pelvisAnimOffset.z = lungeZ * 0.35;
          rig.rootBone.position.z = THREE.MathUtils.damp(rig.rootBone.position.z, lungeZ * 0.65, 18, delta);
          rig.rootBone.position.y = THREE.MathUtils.damp(rig.rootBone.position.y, 0, 18, delta);

          pelvisGroup.rotation.x = THREE.MathUtils.damp(pelvisGroup.rotation.x, 0.12 * windup - 0.08 * lunge, 18, delta);
          torsoGroup.rotation.x = THREE.MathUtils.damp(torsoGroup.rotation.x, 0.16 * windup + 0.12 * lunge, 18, delta);

          neckGroup.rotation.x = THREE.MathUtils.damp(
            neckGroup.rotation.x,
            -0.2 * windup + 0.42 * lunge,
            18,
            delta
          );
          headGroup.rotation.x = THREE.MathUtils.damp(
            headGroup.rotation.x,
            -0.16 * windup + 0.16 * lunge - biteSnap * 0.28,
            18,
            delta
          );
          // Predatory head twist during the bite tear
          headGroup.rotation.z = THREE.MathUtils.damp(
            headGroup.rotation.z,
            Math.sin(cycle * 2.0) * 0.26 * lunge,
            18,
            delta
          );

          // Open jaws wide on windup/approach, then slam shut violently on biteSnap!
          const targetJaw = Math.max(0.02, jawOpenWave * 0.88 - biteSnap * 0.92);
          jawGroup.rotation.x = THREE.MathUtils.damp(jawGroup.rotation.x, targetJaw, 22, delta);

          // Flash 3D Fang Bite Shockwave at the mouth when the jaws snap shut
          const biteFxScale = biteSnap > 0.12 ? Math.max(0.0001, biteSnap * 1.18) : 0.0001;
          breathConeMesh.scale.set(biteFxScale, biteFxScale, biteFxScale);
          breathConeMesh.rotation.z = 0;
          if (biteSnap > 0.2) {
            mouthBreathIntensity = biteSnap * 0.85;
          }

          // Grounded predatory lunge on forelegs & hindlegs
          leftArmGroup.rotation.x = THREE.MathUtils.damp(
            leftArmGroup.rotation.x,
            0.22 * windup - 0.46 * lunge,
            16,
            delta
          );
          rightArmGroup.rotation.x = THREE.MathUtils.damp(
            rightArmGroup.rotation.x,
            0.22 * windup - 0.46 * lunge,
            16,
            delta
          );
          leftElbowGroup.rotation.x = THREE.MathUtils.damp(leftElbowGroup.rotation.x, -0.28, 14, delta);
          rightElbowGroup.rotation.x = THREE.MathUtils.damp(rightElbowGroup.rotation.x, -0.28, 14, delta);

          leftLegGroup.rotation.x = THREE.MathUtils.damp(
            leftLegGroup.rotation.x,
            -0.26 * windup + 0.5 * lunge,
            16,
            delta
          );
          rightLegGroup.rotation.x = THREE.MathUtils.damp(
            rightLegGroup.rotation.x,
            -0.26 * windup + 0.5 * lunge,
            16,
            delta
          );
          leftKneeGroup.rotation.x = THREE.MathUtils.damp(leftKneeGroup.rotation.x, 0.24 + lunge * 0.24, 14, delta);
          rightKneeGroup.rotation.x = THREE.MathUtils.damp(rightKneeGroup.rotation.x, 0.24 + lunge * 0.24, 14, delta);
          tailGroup.rotation.x = THREE.MathUtils.damp(tailGroup.rotation.x, lunge * 0.54, 14, delta);
        }
      } else if (isBowWeapon) {
        // NATURAL ARCHER BIOMECHANICS: RAISE BOW AT NOSE LEVEL, DRAW STRING TO THE NOSE, AND SHOOT HORIZONTALLY FORWARD (+Z)
        breathConeMesh.scale.setScalar(0.0001);
        const bowU = ((t / 1.2565) % 1.0 + 1.0) % 1.0;

        if (bowU < 0.44) {
          // PHASE 1: Aim Bow Horizontally Forward & Draw Bowstring + Arrow Nock Back to the Height of the Nose (0.00 -> 0.44)
          const drawP = bowU / 0.44;
          _pelvisAnimOffset.y = -0.03 * drawP;
          _pelvisAnimOffset.z = -0.05 * drawP;
          rig.rootBone.position.z = THREE.MathUtils.damp(rig.rootBone.position.z, -0.05 * drawP, 18, delta);

          // Keep pelvis & torso pitch at 0 so the bow and arrow remain 100% horizontal (zero upward tilt!)
          pelvisGroup.rotation.x = THREE.MathUtils.damp(pelvisGroup.rotation.x, 0, 22, delta);
          torsoGroup.rotation.x = THREE.MathUtils.damp(torsoGroup.rotation.x, 0, 22, delta);
          torsoGroup.rotation.y = THREE.MathUtils.damp(torsoGroup.rotation.y, -0.22, 20, delta);
          neckGroup.rotation.x = THREE.MathUtils.damp(neckGroup.rotation.x, 0, 20, delta);
          headGroup.rotation.x = THREE.MathUtils.damp(headGroup.rotation.x, 0, 20, delta);
          headGroup.rotation.y = THREE.MathUtils.damp(headGroup.rotation.y, 0.22, 20, delta);
          jawGroup.rotation.x = THREE.MathUtils.damp(jawGroup.rotation.x, 0.08 * drawP, 18, delta);

          // Right arm holds the bow vertically upright directly in front at nose height:
          // rightArmGroup.rotation.x (-1.52) + rightElbowGroup.rotation.x (-0.0508) = -Math.PI/2 => 0° world pitch!
          rightShoulderGroup.rotation.z = THREE.MathUtils.damp(rightShoulderGroup.rotation.z, 0, 22, delta);
          rightArmGroup.rotation.x = THREE.MathUtils.damp(rightArmGroup.rotation.x, -1.52, 24, delta);
          rightArmGroup.rotation.y = THREE.MathUtils.damp(rightArmGroup.rotation.y, 0.22, 24, delta);
          rightArmGroup.rotation.z = THREE.MathUtils.damp(rightArmGroup.rotation.z, 0, 24, delta);
          rightElbowGroup.rotation.x = THREE.MathUtils.damp(rightElbowGroup.rotation.x, -0.0508, 24, delta);

          // Left arm reaches to the front bowstring (drawP = 0) and pulls tension straight back to the NOSE (drawP = 1)!
          leftArmGroup.rotation.x = THREE.MathUtils.damp(
            leftArmGroup.rotation.x,
            THREE.MathUtils.lerp(-1.5, -1.56, drawP),
            22,
            delta
          );
          leftArmGroup.rotation.y = THREE.MathUtils.damp(
            leftArmGroup.rotation.y,
            THREE.MathUtils.lerp(-0.36, 0.18, drawP),
            22,
            delta
          );
          leftArmGroup.rotation.z = THREE.MathUtils.damp(
            leftArmGroup.rotation.z,
            THREE.MathUtils.lerp(0.06, 0.26, drawP),
            22,
            delta
          );
          leftElbowGroup.rotation.x = THREE.MathUtils.damp(
            leftElbowGroup.rotation.x,
            THREE.MathUtils.lerp(-0.24, -1.96, drawP),
            22,
            delta
          );

          leftLegGroup.rotation.x = THREE.MathUtils.damp(leftLegGroup.rotation.x, -0.16 * drawP, 14, delta);
          rightLegGroup.rotation.x = THREE.MathUtils.damp(rightLegGroup.rotation.x, 0.18 * drawP, 14, delta);

          const pullZ = drawP * 0.56;
          setBowstringDraw(pullZ);
          slashTrailMesh.scale.setScalar(0.0001);
          arrowProjectileMesh.position.set(0, 0, 0.36 - pullZ);
          arrowProjectileMesh.scale.setScalar(1.05);
        } else if (bowU < 0.86) {
          // PHASE 2: Release String from the Nose & Shoot 3D Arrow Horizontally Straight Forward Along +Z (0.44 -> 0.86)
          const flyP = (bowU - 0.44) / 0.42;
          _pelvisAnimOffset.y = -0.02;
          rig.rootBone.position.z = THREE.MathUtils.damp(rig.rootBone.position.z, -0.06, 16, delta);

          pelvisGroup.rotation.x = THREE.MathUtils.damp(pelvisGroup.rotation.x, 0, 24, delta);
          torsoGroup.rotation.x = THREE.MathUtils.damp(torsoGroup.rotation.x, 0, 24, delta);
          torsoGroup.rotation.y = THREE.MathUtils.damp(torsoGroup.rotation.y, -0.22, 22, delta);
          neckGroup.rotation.x = THREE.MathUtils.damp(neckGroup.rotation.x, 0, 22, delta);
          headGroup.rotation.x = THREE.MathUtils.damp(headGroup.rotation.x, 0, 22, delta);
          headGroup.rotation.y = THREE.MathUtils.damp(headGroup.rotation.y, 0.22, 22, delta);
          jawGroup.rotation.x = THREE.MathUtils.damp(jawGroup.rotation.x, 0.24 * (1 - flyP), 18, delta);

          rightShoulderGroup.rotation.z = THREE.MathUtils.damp(rightShoulderGroup.rotation.z, 0, 24, delta);
          rightArmGroup.rotation.x = THREE.MathUtils.damp(rightArmGroup.rotation.x, -1.52, 26, delta);
          rightArmGroup.rotation.y = THREE.MathUtils.damp(rightArmGroup.rotation.y, 0.22, 26, delta);
          rightArmGroup.rotation.z = THREE.MathUtils.damp(rightArmGroup.rotation.z, 0, 26, delta);
          rightElbowGroup.rotation.x = THREE.MathUtils.damp(rightElbowGroup.rotation.x, -0.0508, 26, delta);

          // Left hand anchors cleanly at the nose/cheek with crisp release follow-through
          leftArmGroup.rotation.x = THREE.MathUtils.damp(leftArmGroup.rotation.x, -1.56, 24, delta);
          leftArmGroup.rotation.y = THREE.MathUtils.damp(leftArmGroup.rotation.y, 0.24, 24, delta);
          leftArmGroup.rotation.z = THREE.MathUtils.damp(leftArmGroup.rotation.z, 0.28, 24, delta);
          leftElbowGroup.rotation.x = THREE.MathUtils.damp(leftElbowGroup.rotation.x, -2.02, 24, delta);

          // Bowstring snaps taut and 3D Arrow shoots 100% horizontally forward along +Z (zero upward climb!)
          const stringSnap = Math.max(0, Math.sin(flyP * Math.PI * 5) * Math.exp(-flyP * 7) * 0.05);
          setBowstringDraw(stringSnap);
          arrowProjectileMesh.position.set(0, 0, 0.36 + flyP * 11.4);
          arrowProjectileMesh.scale.set(1.25, 1.25, 1.6);

          if (flyP < 0.24) {
            const flash = (1.0 - flyP / 0.24) * 0.85;
            slashTrailMesh.scale.set(0.8, flash, flash);
            slashTrailMesh.rotation.set(0, 0, 0);
          } else {
            slashTrailMesh.scale.setScalar(0.0001);
          }
        } else {
          // PHASE 3: Draw Fresh Arrow from Dorsal Quiver & Nock on Bowstring (0.86 -> 1.00)
          const relP = (bowU - 0.86) / 0.14;
          pelvisGroup.rotation.x = THREE.MathUtils.damp(pelvisGroup.rotation.x, 0, 18, delta);
          torsoGroup.rotation.x = THREE.MathUtils.damp(torsoGroup.rotation.x, 0, 18, delta);
          torsoGroup.rotation.y = THREE.MathUtils.damp(torsoGroup.rotation.y, -0.18, 16, delta);

          rightArmGroup.rotation.x = THREE.MathUtils.damp(rightArmGroup.rotation.x, -1.42, 18, delta);
          rightArmGroup.rotation.y = THREE.MathUtils.damp(rightArmGroup.rotation.y, 0.18, 18, delta);
          rightElbowGroup.rotation.x = THREE.MathUtils.damp(rightElbowGroup.rotation.x, -0.15, 18, delta);

          leftArmGroup.rotation.x = THREE.MathUtils.damp(
            leftArmGroup.rotation.x,
            THREE.MathUtils.lerp(-1.72, -1.5, relP),
            20,
            delta
          );
          leftArmGroup.rotation.y = THREE.MathUtils.damp(
            leftArmGroup.rotation.y,
            THREE.MathUtils.lerp(0.15, -0.36, relP),
            20,
            delta
          );
          leftArmGroup.rotation.z = THREE.MathUtils.damp(leftArmGroup.rotation.z, 0.08, 18, delta);
          leftElbowGroup.rotation.x = THREE.MathUtils.damp(
            leftElbowGroup.rotation.x,
            THREE.MathUtils.lerp(-1.25, -0.24, relP),
            20,
            delta
          );

          setBowstringDraw(0);
          slashTrailMesh.scale.setScalar(0.0001);
          arrowProjectileMesh.position.set(0, 0, 0.36);
          arrowProjectileMesh.scale.setScalar(relP);
        }
      } else {
        // HUMANOID / CHAMPION WEAPON STRIKE (Grounded weight transfer, torso twist, follow-through)
        _pelvisAnimOffset.y = -0.06 * lunge;
        _pelvisAnimOffset.z = -0.08 * windup + 0.22 * lunge;
        rig.rootBone.position.z = THREE.MathUtils.damp(rig.rootBone.position.z, 0.16 * lunge, 14, delta);
        jawGroup.rotation.x = THREE.MathUtils.damp(jawGroup.rotation.x, lunge * 0.52, 16, delta);
        breathConeMesh.scale.setScalar(0.0001);

        // Torso windup twist (-Y) and explosive strike follow-through (+Y, +X)
        torsoGroup.rotation.y = THREE.MathUtils.damp(
          torsoGroup.rotation.y,
          -0.28 * windup + 0.36 * lunge,
          16,
          delta
        );
        torsoGroup.rotation.x = THREE.MathUtils.damp(
          torsoGroup.rotation.x,
          -0.1 * windup + 0.22 * lunge,
          16,
          delta
        );

        const slashPower = Math.max(0, Math.sin(cycle - 0.25));
        if (slashPower > 0.05) {
          const sMag = Math.min(1.35, slashPower * 1.35);
          slashTrailMesh.scale.set(1.0, sMag, sMag * 1.15);
          slashTrailMesh.rotation.set(-0.35 + slashPower * 1.15, 0.08 * swing, -0.18 * swing);
        } else {
          slashTrailMesh.scale.setScalar(0.0001);
        }

        tailGroup.rotation.x = THREE.MathUtils.damp(tailGroup.rotation.x, lunge * 0.6, 14, delta);

        rightArmGroup.rotation.x = THREE.MathUtils.damp(
          rightArmGroup.rotation.x,
          swing > 0 ? -1.45 + lunge * 2.25 : -0.35 - windup * 0.55,
          16,
          delta
        );
        rightElbowGroup.rotation.x = THREE.MathUtils.damp(
          rightElbowGroup.rotation.x,
          swing > 0 ? -0.65 + lunge * 0.55 : -0.45,
          16,
          delta
        );
        leftArmGroup.rotation.x = THREE.MathUtils.damp(leftArmGroup.rotation.x, -0.52 + swing * 0.32, 12, delta);
        leftElbowGroup.rotation.x = THREE.MathUtils.damp(leftElbowGroup.rotation.x, -0.42, 12, delta);

        leftLegGroup.rotation.x = THREE.MathUtils.damp(leftLegGroup.rotation.x, -lunge * 0.34, 12, delta);
        rightLegGroup.rotation.x = THREE.MathUtils.damp(rightLegGroup.rotation.x, lunge * 0.34, 12, delta);
        leftKneeGroup.rotation.x = THREE.MathUtils.damp(leftKneeGroup.rotation.x, 0.24 + lunge * 0.34, 12, delta);
        rightKneeGroup.rotation.x = THREE.MathUtils.damp(leftKneeGroup.rotation.x, 0.22 + lunge * 0.18, 12, delta);
      }
    }
  } else if (animState === 'hit') {
    breathConeMesh.scale.setScalar(0.0001);
    slashTrailMesh.scale.setScalar(0.0001);
    // Kinetic recoil wave: sharp knockback followed by muscular brace & shudder
    const stagger = Math.sin(t * 10.0) * 0.18;
    _pelvisAnimOffset.z = -0.14;
    _pelvisAnimOffset.y = -0.08;
    pelvisGroup.rotation.x = THREE.MathUtils.damp(pelvisGroup.rotation.x, -0.16, 16, delta);
    torsoGroup.rotation.x = THREE.MathUtils.damp(torsoGroup.rotation.x, -0.24 + stagger * 0.4, 16, delta);
    torsoGroup.rotation.y = THREE.MathUtils.damp(torsoGroup.rotation.y, stagger * 0.5, 16, delta);
    jawGroup.rotation.x = THREE.MathUtils.damp(jawGroup.rotation.x, 0.48, 16, delta);
    headGroup.rotation.x = THREE.MathUtils.damp(headGroup.rotation.x, -0.38 + stagger, 16, delta);
    rightArmGroup.rotation.z = THREE.MathUtils.damp(rightArmGroup.rotation.z, -0.45, 14, delta);
    leftArmGroup.rotation.z = THREE.MathUtils.damp(leftArmGroup.rotation.z, 0.45, 14, delta);
    leftLegGroup.rotation.x = THREE.MathUtils.damp(leftLegGroup.rotation.x, -0.22, 14, delta);
    rightLegGroup.rotation.x = THREE.MathUtils.damp(rightLegGroup.rotation.x, 0.26, 14, delta);
    leftKneeGroup.rotation.x = THREE.MathUtils.damp(leftKneeGroup.rotation.x, 0.48, 14, delta);
    rightKneeGroup.rotation.x = THREE.MathUtils.damp(rightKneeGroup.rotation.x, 0.42, 14, delta);
  } else if (animState === 'death') {
    breathConeMesh.scale.setScalar(0.0001);
    slashTrailMesh.scale.setScalar(0.0001);
    // Grounded collapse: knees buckle and pelvis drops onto the floor plane
    _pelvisAnimOffset.y = -0.42;
    _pelvisAnimOffset.z = -0.12;
    pelvisGroup.rotation.x = THREE.MathUtils.damp(pelvisGroup.rotation.x, 1.02, 5, delta);
    torsoGroup.rotation.x = THREE.MathUtils.damp(torsoGroup.rotation.x, 0.24, 5, delta);
    headGroup.rotation.x = THREE.MathUtils.damp(headGroup.rotation.x, 0.42, 5, delta);
    jawGroup.rotation.x = THREE.MathUtils.damp(jawGroup.rotation.x, 0.35, 5, delta);
    rightArmGroup.rotation.x = THREE.MathUtils.damp(rightArmGroup.rotation.x, -1.15, 5, delta);
    leftArmGroup.rotation.x = THREE.MathUtils.damp(leftArmGroup.rotation.x, -1.05, 5, delta);
    leftKneeGroup.rotation.x = THREE.MathUtils.damp(leftKneeGroup.rotation.x, 0.78, 5, delta);
    rightKneeGroup.rotation.x = THREE.MathUtils.damp(rightKneeGroup.rotation.x, 0.78, 5, delta);
  }

  // Apply deterministic pelvic position (basePos + explodeVec * explodedAmount + _pelvisAnimOffset)
  if (pelvisGroup.userData.basePos) {
    const bp = pelvisGroup.userData.basePos as THREE.Vector3;
    const ex = (pelvisGroup.userData.explodeVec as THREE.Vector3) || _zeroVec;
    const tx = bp.x + ex.x * explodedAmount + _pelvisAnimOffset.x;
    const ty = bp.y + ex.y * explodedAmount + _pelvisAnimOffset.y;
    const tz = bp.z + ex.z * explodedAmount + _pelvisAnimOffset.z;
    pelvisGroup.position.x = THREE.MathUtils.damp(pelvisGroup.position.x, tx, 16, delta);
    pelvisGroup.position.y = THREE.MathUtils.damp(pelvisGroup.position.y, ty, 16, delta);
    pelvisGroup.position.z = THREE.MathUtils.damp(pelvisGroup.position.z, tz, 16, delta);
  }

  // Ensure world transforms of animated bones & mouth emitter are up to date
  rig.exportableGroup.updateMatrixWorld(true);

  // 2B. Update Real-Time Volumetric Mouth Fire / Frost / Bite-Spark Stream & Dynamic Light
  const mSys = rig.mouthBreathSystem;
  const isBakingClip = delta >= 0.2;
  if (!isBakingClip) {
    // Scroll volumetric flame & frost textures rapidly outward along the plume
    mSys.fireBreathMap.offset.y = (mSys.fireBreathMap.offset.y - delta * 3.6) % 1.0;
    mSys.fireBreathMap.offset.x = Math.sin(t * 5.5) * 0.08;
    mSys.frostBreathMap.offset.y = (mSys.frostBreathMap.offset.y - delta * 3.1) % 1.0;
    mSys.frostBreathMap.offset.x = Math.cos(t * 4.8) * 0.06;

    if (mouthBreathIntensity > 0.02) {
      // Flickering dynamic mouth point light
      if (mouthParticleMode === 'frost') {
        mSys.breathLight.color.setHex(0x38bdf8);
        mSys.breathLight.intensity =
          mouthBreathIntensity * (3.2 + Math.sin(t * 34.0) * 0.7);
      } else if (mouthParticleMode === 'fire') {
        mSys.breathLight.color.setHex(0xff6a00);
        mSys.breathLight.intensity =
          mouthBreathIntensity *
          (4.2 + Math.sin(t * 39.0) * 1.15 + Math.cos(t * 61.0) * 0.75);
      } else {
        mSys.breathLight.color.setHex(0xef4444);
        mSys.breathLight.intensity = mouthBreathIntensity * 2.4;
      }
    } else {
      mSys.breathLight.intensity = THREE.MathUtils.damp(
        mSys.breathLight.intensity,
        0,
        18,
        delta
      );
    }

    // Check if any particles are still alive or if mouth is actively emitting
    const bPosAttr = mSys.points.geometry.getAttribute('position') as THREE.BufferAttribute;
    const bColAttr = mSys.points.geometry.getAttribute('color') as THREE.BufferAttribute;
    const bDt = Math.max(0.004, Math.min(delta, 0.06)) * Math.max(0.4, speedMultiplier);

    if (mouthBreathIntensity > 0.02 || mSys.points.visible) {
      mSys.points.material =
        mouthParticleMode === 'frost' ? mSys.frostMat : mSys.fireMat;

      // Resolve mouth origin and forward direction vector in rig.rootGroup local space
      breathConeMesh.getWorldPosition(_mouthWorldPos);
      _mouthLocalPos.copy(_mouthWorldPos);
      rig.rootGroup.worldToLocal(_mouthLocalPos);

      headGroup.getWorldQuaternion(_headWorldQuat);
      rig.rootGroup.getWorldQuaternion(_rootWorldQuat);
      _rootWorldQuat.invert();
      _headWorldQuat.premultiply(_rootWorldQuat);
      _mouthForwardDir.set(0, -0.06, 1).applyQuaternion(_headWorldQuat).normalize();

      let activeCount = 0;
      for (let i = 0; i < mSys.pool.length; i++) {
        const p = mSys.pool[i];
        p.age += bDt;

        if (p.age >= p.lifetime) {
          if (mouthBreathIntensity > 0.02) {
            p.age = 0;
            p.lifetime =
              mouthParticleMode === 'bite_sparks'
                ? 0.22 + (i % 6) * 0.035
                : 0.34 + (i % 9) * 0.048;

            const spreadAngle = t * 13.7 + p.seed * 6.28318;
            const throatR = 0.045 + (i % 4) * 0.015;
            const ox = Math.cos(spreadAngle) * throatR;
            const oy = Math.sin(spreadAngle) * throatR;

            bPosAttr.setXYZ(
              i,
              _mouthLocalPos.x + ox,
              _mouthLocalPos.y + oy,
              _mouthLocalPos.z
            );

            const jetSpeed =
              mouthParticleMode === 'bite_sparks'
                ? 2.2 + (i % 5) * 0.45
                : (3.8 + (i % 7) * 0.36) * (0.75 + mouthBreathIntensity * 0.35);
            const coneSpread =
              mouthParticleMode === 'bite_sparks' ? 1.35 : p.isSpark ? 0.85 : 0.52;

            p.vx = _mouthForwardDir.x * jetSpeed + Math.cos(spreadAngle) * coneSpread;
            p.vy = _mouthForwardDir.y * jetSpeed + Math.sin(spreadAngle) * coneSpread;
            p.vz = _mouthForwardDir.z * jetSpeed + Math.sin(spreadAngle * 1.7) * 0.35;
          } else {
            // Hide expired particle when breath has stopped
            bColAttr.setXYZ(i, 0, 0, 0);
            continue;
          }
        }

        activeCount++;
        const ratio = Math.min(1, p.age / p.lifetime);
        let px = bPosAttr.getX(i);
        let py = bPosAttr.getY(i);
        let pz = bPosAttr.getZ(i);

        // 3D turbulent curl & thermal buoyancy (fire rises, frost sinks slightly)
        const turbFreq = 11.0;
        const turbAmp = (0.35 + ratio * 0.85) * (p.isSpark ? 1.4 : 1.0);
        const curlX = Math.sin(t * turbFreq + p.seed + pz * 3.2) * turbAmp;
        const curlY =
          Math.cos(t * (turbFreq * 0.9) + p.seed * 1.7 + px * 3.2) * (turbAmp * 0.65) +
          (mouthParticleMode === 'frost' ? -0.25 : 0.72 * ratio);
        const curlZ = Math.cos(t * turbFreq + p.seed * 2.3 + py * 3.2) * (turbAmp * 0.5);

        const drag = 1.0 - ratio * 0.32;
        px += (p.vx * drag + curlX) * bDt;
        py += (p.vy * drag + curlY) * bDt;
        pz += (p.vz * drag + curlZ) * bDt;
        bPosAttr.setXYZ(i, px, py, pz);

        // Blackbody temperature color curve
        const fade = Math.pow(Math.max(0, 1.0 - Math.pow(ratio, 1.35)), 1.25);
        if (mouthParticleMode === 'frost') {
          if (ratio < 0.25) {
            _particleColorOut.setRGB(0.95, 0.99, 1.0);
          } else if (ratio < 0.65) {
            _particleColorOut.setRGB(0.35, 0.82, 1.0);
          } else {
            _particleColorOut.setRGB(0.08, 0.42, 0.85);
          }
          _particleColorOut.multiplyScalar(fade * 1.15);
        } else if (mouthParticleMode === 'bite_sparks') {
          if (ratio < 0.3) {
            _particleColorOut.setRGB(1.0, 0.92, 0.75);
          } else {
            _particleColorOut.setRGB(0.95, 0.18, 0.12);
          }
          _particleColorOut.multiplyScalar(fade * 1.2);
        } else {
          // Real Fire: White-Hot Core -> Bright Yellow-Gold -> Roaring Orange -> Deep Crimson -> Smoldering Smoke
          if (p.isSpark) {
            _particleColorOut.setRGB(1.0, 0.85 - ratio * 0.45, 0.25 * (1 - ratio));
            _particleColorOut.multiplyScalar(fade * 1.45);
          } else if (ratio < 0.16) {
            const u = ratio / 0.16;
            _particleColorOut.setRGB(1.0, 1.0 - u * 0.06, 0.92 - u * 0.38);
            _particleColorOut.multiplyScalar(1.35);
          } else if (ratio < 0.46) {
            const u = (ratio - 0.16) / 0.3;
            _particleColorOut.setRGB(1.0, 0.94 - u * 0.48, 0.54 - u * 0.48);
            _particleColorOut.multiplyScalar(1.25);
          } else if (ratio < 0.76) {
            const u = (ratio - 0.46) / 0.3;
            _particleColorOut.setRGB(1.0 - u * 0.22, 0.46 - u * 0.36, 0.06 * (1 - u));
            _particleColorOut.multiplyScalar(fade * 1.15);
          } else {
            const u = (ratio - 0.76) / 0.24;
            _particleColorOut.setRGB(
              (0.78 - u * 0.62) * fade,
              (0.1 - u * 0.08) * fade,
              0.02 * fade
            );
          }
        }

        bColAttr.setXYZ(i, _particleColorOut.r, _particleColorOut.g, _particleColorOut.b);
      }

      mSys.points.visible = activeCount > 0;
      bPosAttr.needsUpdate = true;
      bColAttr.needsUpdate = true;
    }
  }

  // 3. Update Procedural Aura Particle Emitter (Volcanic Smoke or Arcane Sparks from Aura Anchor Points)
  const pSys = rig.auraParticleSystem;
  if (auraParticleMode === 'none') {
    if (pSys.points.visible) {
      pSys.points.visible = false;
      pSys.activeMode = 'none';
    }
    return;
  }

  const posAttr = pSys.points.geometry.getAttribute('position') as THREE.BufferAttribute;
  const colAttr = pSys.points.geometry.getAttribute('color') as THREE.BufferAttribute;
  const shardCount = rig.shardData.length;
  const socketList = [
    rig.sockets.core,
    rig.sockets.head,
    rig.sockets.weaponR,
    rig.sockets.shieldL,
  ];
  const totalAnchors = shardCount + socketList.length;

  // Resolve exact local-space coordinates of the i-th Aura Anchor Point (orbiting Aura Shards + 4 Aura Sockets)
  const sampleAuraAnchor = (anchorIdx: number, out: THREE.Vector3) => {
    const safeIdx = ((anchorIdx % totalAnchors) + totalAnchors) % totalAnchors;
    if (safeIdx < shardCount) {
      const s = rig.shardData[safeIdx];
      const curAngle = s.angle + t * s.speed;
      const r = s.radius + explodedAmount * 0.45;
      const y = s.yOffset + Math.sin(t * 2.2 + safeIdx) * 0.12;
      out.set(Math.cos(curAngle) * r, y, Math.sin(curAngle) * r);
    } else {
      const sock = socketList[safeIdx - shardCount] || rig.sockets.core;
      sock.getWorldPosition(out);
      rig.rootGroup.worldToLocal(out);
    }
  };

  const isVolcanic = auraParticleMode === 'volcanic_smoke';
  if (!pSys.points.visible || pSys.activeMode !== auraParticleMode) {
    pSys.points.visible = true;
    pSys.points.material = isVolcanic ? pSys.smokeMat : pSys.sparkMat;
    pSys.activeMode = auraParticleMode;

    // Seed all particles immediately around the enemy's live aura anchor points when switching modes
    for (let i = 0; i < pSys.pool.length; i++) {
      const p = pSys.pool[i];
      p.anchorIndex = i % totalAnchors;
      p.lifetime = isVolcanic ? 1.25 + (i % 9) * 0.14 : 0.72 + (i % 7) * 0.11;
      p.age = (i / pSys.pool.length) * p.lifetime;
      sampleAuraAnchor(p.anchorIndex, _anchorLocalPos);
      const frac = p.age / p.lifetime;
      posAttr.setXYZ(
        i,
        _anchorLocalPos.x + Math.sin(p.seed + i) * 0.08,
        _anchorLocalPos.y + frac * (isVolcanic ? 0.65 : 0.42),
        _anchorLocalPos.z + Math.cos(p.seed + i) * 0.08
      );
    }
  }

  if (isVolcanic) {
    // Volcanic Smoke: glowing magma-ember birth -> billowing dark volcanic ash plume -> smooth fade
    _particleBaseColor.set('#ff5511').lerp(rig.materials.emissiveRuneMat.color, 0.35);
    _particleAccentColor.set('#261d1e');
  } else {
    // Arcane Sparks: white-hot aether core -> enemy's emissive rune energy color -> crackling twinkle
    _particleBaseColor.copy(rig.materials.emissiveRuneMat.color);
    _particleAccentColor.set('#ffffff').lerp(_particleBaseColor, 0.25);
  }

  const dt = Math.max(0.004, Math.min(delta, 0.08)) * Math.max(0.35, speedMultiplier);

  for (let i = 0; i < pSys.pool.length; i++) {
    const p = pSys.pool[i];
    p.age += dt;

    if (p.age >= p.lifetime) {
      p.age = 0;
      p.anchorIndex = (p.anchorIndex + 1 + (i % 3)) % totalAnchors;
      p.lifetime = isVolcanic
        ? 1.2 + ((i * 19) % 10) * 0.13
        : 0.68 + ((i * 23) % 10) * 0.095;

      sampleAuraAnchor(p.anchorIndex, _anchorLocalPos);
      const jitterR = isVolcanic ? 0.09 : 0.055;
      const ang = t * 3.7 + p.seed * 6.283;
      posAttr.setXYZ(
        i,
        _anchorLocalPos.x + Math.cos(ang) * jitterR,
        _anchorLocalPos.y + (Math.sin(p.seed * 3.1 + t) * 0.05),
        _anchorLocalPos.z + Math.sin(ang) * jitterR
      );

      if (isVolcanic) {
        p.vx = Math.cos(ang) * 0.16;
        p.vy = 0.52 + ((i * 31) % 10) * 0.045;
        p.vz = Math.sin(ang) * 0.16;
      } else {
        p.vx = Math.cos(ang) * 0.42;
        p.vy = 0.48 + ((i * 41) % 10) * 0.065;
        p.vz = Math.sin(ang) * 0.42;
      }
    }

    const lifeRatio = Math.min(1, p.age / p.lifetime);
    let px = posAttr.getX(i);
    let py = posAttr.getY(i);
    let pz = posAttr.getZ(i);

    if (isVolcanic) {
      // Billowing turbulent curl drift upward for volcanic smoke
      const curlX = Math.sin(t * 2.4 + p.seed + py * 2.1) * 0.28;
      const curlZ = Math.cos(t * 2.2 + p.seed * 1.3 + py * 2.1) * 0.28;
      px += (p.vx + curlX) * dt;
      py += p.vy * (1.0 - lifeRatio * 0.25) * dt;
      pz += (p.vz + curlZ) * dt;

      // Color curve: hot volcanic ember at anchor (0..0.28) -> dark smoldering smoke (0.28..0.75) -> fade out
      if (lifeRatio < 0.28) {
        const u = lifeRatio / 0.28;
        _particleColorOut.copy(_particleBaseColor).lerp(_particleAccentColor, u * 0.78);
      } else {
        const u = (lifeRatio - 0.28) / 0.72;
        const fade = Math.max(0, 1.0 - u * 0.88);
        _particleColorOut.copy(_particleAccentColor).multiplyScalar(fade);
      }
    } else {
      // Spiraling arcane sparks bursting and orbiting from the aura anchors
      const swirlAngle = t * 6.5 + p.seed * 4.0;
      const swirlX = -Math.sin(swirlAngle) * 0.36;
      const swirlZ = Math.cos(swirlAngle) * 0.36;
      px += (p.vx * (1.0 - lifeRatio * 0.5) + swirlX) * dt;
      py += (p.vy + Math.sin(t * 9.0 + p.seed) * 0.22) * dt;
      pz += (p.vz * (1.0 - lifeRatio * 0.5) + swirlZ) * dt;

      const twinkle = 0.62 + 0.38 * Math.sin(t * 20.0 + p.seed * 13.0);
      const fade = Math.sin((1.0 - lifeRatio) * Math.PI * 0.5) * twinkle;
      if (lifeRatio < 0.32) {
        _particleColorOut.copy(_particleAccentColor).lerp(_particleBaseColor, lifeRatio / 0.32);
      } else {
        _particleColorOut.copy(_particleBaseColor);
      }
      _particleColorOut.multiplyScalar(Math.max(0, fade * 1.25));
    }

    posAttr.setXYZ(i, px, py, pz);
    colAttr.setXYZ(i, _particleColorOut.r, _particleColorOut.g, _particleColorOut.b);
  }

  posAttr.needsUpdate = true;
  colAttr.needsUpdate = true;
}
