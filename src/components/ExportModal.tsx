import React, { useMemo, useState } from 'react';
import {
  Check,
  Code2,
  Copy,
  Download,
  FileCode,
  FileJson,
  Globe,
  Package,
  Sparkles,
  Upload,
  X,
} from 'lucide-react';
import { EnemyConfig, calculateThreatScore } from '../types/enemy';
import {
  downloadTextFile,
  generateBlenderPythonScript,
  generateThreeJsModuleCode,
} from '../three/exporters';

interface ExportModalProps {
  isOpen: boolean;
  onClose: () => void;
  config: EnemyConfig;
  metrics: { totalVertices: number; totalTriangles: number; partCount: number };
  onExportGLB: (binary: boolean) => Promise<void>;
  onExportHTML: () => Promise<void>;
  onImportConfig: (cfg: EnemyConfig) => void;
  exportStatusMessage: string | null;
}

export const ExportModal: React.FC<ExportModalProps> = ({
  isOpen,
  onClose,
  config,
  metrics,
  onExportGLB,
  onExportHTML,
  onImportConfig,
  exportStatusMessage,
}) => {
  const [activeFormat, setActiveFormat] = useState<
    'comparison' | 'blender_py' | 'three_ts' | 'rpg_json'
  >('comparison');
  const [copied, setCopied] = useState(false);
  const [isExportingGlb, setIsExportingGlb] = useState(false);
  const [isExportingHtml, setIsExportingHtml] = useState(false);
  const [importError, setImportError] = useState<string | null>(null);

  const generatedTsCode = useMemo(() => generateThreeJsModuleCode(config), [config]);
  const generatedBlenderPy = useMemo(() => generateBlenderPythonScript(config), [config]);
  const generatedJson = useMemo(() => JSON.stringify(config, null, 2), [config]);
  const threatScore = useMemo(() => calculateThreatScore(config.stats), [config.stats]);

  if (!isOpen) return null;

  const handleCopy = (text: string) => {
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 1800);
  };

  const handleGlbTrigger = async (binary: boolean) => {
    setIsExportingGlb(true);
    try {
      await onExportGLB(binary);
    } finally {
      setIsExportingGlb(false);
    }
  };

  const handleHtmlTrigger = async () => {
    setIsExportingHtml(true);
    try {
      await onExportHTML();
    } finally {
      setIsExportingHtml(false);
    }
  };

  const handleJsonUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      try {
        const parsed = JSON.parse(String(reader.result));
        if (parsed && parsed.archetype && parsed.morphology && parsed.materials && parsed.stats) {
          onImportConfig(parsed as EnemyConfig);
          setImportError(null);
          onClose();
        } else {
          setImportError('El archivo JSON no tiene el esquema de EnemyConfig válido.');
        }
      } catch {
        setImportError('No se pudo parsear el archivo JSON.');
      }
    };
    reader.readAsText(file);
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/75 p-4"
      role="dialog"
      aria-modal="true"
      aria-label="Centro de Exportación GLB, HTML, Blender y Three.js"
    >
      <div className="w-full max-w-5xl rounded-xl border border-slate-800 bg-[#0e1118] text-slate-100 shadow-2xl overflow-hidden flex flex-col max-h-[88vh]">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-800/80">
          <div>
            <h2 className="font-display text-lg font-bold tracking-wide text-slate-100">
              Pipeline de Exportación: .GLB Binario · Visor .HTML · Script Blender (bpy) · Three.js
            </h2>
            <p className="text-xs text-slate-400 mt-0.5 font-mono-tabular">
              {config.name} · SkinnedMesh Suavizada (Taubin Sculpt) + {metrics.partCount} huesos · 6 Clips (incl. Salto Mordida) + Fuego Volumétrico ·{' '}
              {metrics.totalVertices.toLocaleString()} vértices · {metrics.totalTriangles.toLocaleString()} tris · Amenaza {threatScore}
            </p>
          </div>
          <button
            onClick={onClose}
            className="p-2 text-slate-400 hover:text-slate-100 hover:bg-slate-800/70 rounded-lg transition-colors cursor-pointer"
            aria-label="Cerrar ventana de exportación"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Format Switcher Bar */}
        <div className="flex flex-wrap items-center justify-between gap-3 px-6 py-3 bg-[#131722] border-b border-slate-800/80">
          <div className="flex flex-wrap items-center gap-1.5 p-1 bg-[#0b0d12] rounded-lg border border-slate-800/80">
            <button
              onClick={() => setActiveFormat('comparison')}
              className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                activeFormat === 'comparison'
                  ? 'bg-amber-500 text-slate-950 font-semibold'
                  : 'text-slate-300 hover:text-white'
              }`}
            >
              1. Descarga Directa (.GLB + .HTML + Blender)
            </button>
            <button
              onClick={() => setActiveFormat('blender_py')}
              className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                activeFormat === 'blender_py'
                  ? 'bg-amber-500 text-slate-950 font-semibold'
                  : 'text-slate-300 hover:text-white'
              }`}
            >
              2. Script Blender 4.x (.PY / bpy)
            </button>
            <button
              onClick={() => setActiveFormat('three_ts')}
              className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                activeFormat === 'three_ts'
                  ? 'bg-amber-500 text-slate-950 font-semibold'
                  : 'text-slate-300 hover:text-white'
              }`}
            >
              3. Código Three.js (.TS)
            </button>
            <button
              onClick={() => setActiveFormat('rpg_json')}
              className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                activeFormat === 'rpg_json'
                  ? 'bg-amber-500 text-slate-950 font-semibold'
                  : 'text-slate-300 hover:text-white'
              }`}
            >
              4. Blueprint JSON + Importar
            </button>
          </div>

          {exportStatusMessage && (
            <span className="text-xs text-amber-400 font-mono-tabular truncate">
              {exportStatusMessage}
            </span>
          )}
        </div>

        {/* Body */}
        <div className="p-6 overflow-y-auto space-y-6 flex-1">
          {activeFormat === 'comparison' && (
            <div className="space-y-5">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 items-stretch">
                {/* Pipeline 1: GLB Binary */}
                <div className="rounded-xl border border-slate-800 bg-[#131722] p-4 flex flex-col justify-between">
                  <div className="space-y-2.5">
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-semibold text-amber-400 tracking-wide">
                        OPCIÓN A · ARCHIVO .GLB / .GLTF
                      </span>
                      <Package className="w-4 h-4 text-amber-400" />
                    </div>
                    <h3 className="text-sm font-bold text-slate-100">
                      Binario .GLB con Suavizado Taubin Sculpt
                    </h3>
                    <p className="text-xs text-slate-300 leading-relaxed">
                      Empaqueta la `SkinnedMesh` unificada con suavizado Taubin, {metrics.partCount} huesos, **6 animaciones esqueléticas** (`Idle`, `Walk`, `Attack`, `JumpBite`, `Hit`, `Death`), **Colores PBR + Colores de Vértice (`COLOR_0`)** y **Texturas PNG incrustadas** (Albedo, Normal Map 3D y Roughness).
                    </p>
                    <ul className="text-xs text-slate-400 space-y-1 pt-1">
                      <li>· Incluye texturas de escamas, pelaje, acero Damasco, fuego/hielo y mapas de normales.</li>
                      <li>· Las pistas `Attack` y `JumpBite` activan la estela de corte, salto con mordida y aliento elemental.</li>
                      <li>· Compatible directamente con Blender, Unity, Godot, Unreal y `GLTFLoader`.</li>
                    </ul>
                  </div>

                  <div className="pt-4 mt-4 border-t border-slate-800/80 flex flex-wrap items-center gap-2">
                    <button
                      onClick={() => handleGlbTrigger(true)}
                      disabled={isExportingGlb}
                      className="flex-1 inline-flex items-center justify-center gap-1.5 px-3 py-2.5 text-xs font-semibold bg-amber-500 hover:bg-amber-400 text-slate-950 rounded-lg transition-colors whitespace-nowrap shrink-0 cursor-pointer"
                    >
                      <Download className="w-4 h-4" />
                      {isExportingGlb ? 'Empaquetando...' : `Descargar .GLB`}
                    </button>
                    <button
                      onClick={() => handleGlbTrigger(false)}
                      disabled={isExportingGlb}
                      className="inline-flex items-center justify-center gap-1 px-2.5 py-2.5 text-xs font-medium bg-slate-800 hover:bg-slate-700 text-slate-200 rounded-lg transition-colors whitespace-nowrap shrink-0 cursor-pointer"
                    >
                      <FileJson className="w-3.5 h-3.5" />
                      .GLTF
                    </button>
                  </div>
                </div>

                {/* Pipeline 2: Standalone Single-File HTML */}
                <div className="rounded-xl border border-emerald-500/40 bg-[#131722] p-4 flex flex-col justify-between">
                  <div className="space-y-2.5">
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-semibold text-emerald-400 tracking-wide">
                        OPCIÓN B · VISOR WEB .HTML AUTÓNOMO
                      </span>
                      <Globe className="w-4 h-4 text-emerald-400" />
                    </div>
                    <h3 className="text-sm font-bold text-slate-100">
                      Archivo .HTML Todo-en-Uno (Base64 .GLB + VFX)
                    </h3>
                    <p className="text-xs text-slate-300 leading-relaxed">
                      Genera un único archivo `.html` autocontenido que lleva el binario `.GLB` completo incrustado en Base64 junto con el motor de partículas de **Humo Volcánico / Chispas Arcanas** y la **Estela de Ataque**.
                    </p>
                    <ul className="text-xs text-slate-400 space-y-1 pt-1">
                      <li>· Ábrelo con doble clic en cualquier navegador o incrústalo con `&lt;iframe&gt;` en tu web.</li>
                      <li>· Incluye HUD interactivo para alternar las 6 animaciones (incl. Salto Mordida) y las partículas.</li>
                      <li>· Texturas PBR (Albedo, Normal Map y Roughness) y materiales completos sin servidores externos.</li>
                    </ul>
                  </div>

                  <div className="pt-4 mt-4 border-t border-slate-800/80 flex flex-wrap items-center gap-2">
                    <button
                      onClick={handleHtmlTrigger}
                      disabled={isExportingHtml}
                      className="w-full inline-flex items-center justify-center gap-1.5 px-3 py-2.5 text-xs font-semibold bg-emerald-400 hover:bg-emerald-300 text-slate-950 rounded-lg transition-colors whitespace-nowrap shrink-0 cursor-pointer"
                    >
                      <Globe className="w-4 h-4" />
                      {isExportingHtml
                        ? 'Generando HTML...'
                        : `Descargar ${config.id}_viewer.html`}
                    </button>
                  </div>
                </div>

                {/* Pipeline 3: Blender Python (bpy) + Three.js TS */}
                <div className="rounded-xl border border-orange-500/40 bg-[#131722] p-4 flex flex-col justify-between">
                  <div className="space-y-2.5">
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-semibold text-orange-400 tracking-wide">
                        OPCIÓN C · AUTOMATIZACIÓN BLENDER (.PY)
                      </span>
                      <Sparkles className="w-4 h-4 text-orange-400" />
                    </div>
                    <h3 className="text-sm font-bold text-slate-100">
                      Script Blender 3.6 / 4.x (`bpy`) + Subsurf & SSS
                    </h3>
                    <p className="text-xs text-slate-300 leading-relaxed">
                      ¿Quieres llevar el `.GLB` a Blender para calidad cinematográfica? Este script `bpy` aplica automáticamente **Weld + Corrective Smooth + Subdivision Catmull-Clark** y nodos **Subsurface Scattering (SSS) + Curvatura**.
                    </p>
                    <ul className="text-xs text-slate-400 space-y-1 pt-1">
                      <li>· Mantiene intacto el `Armature` de {metrics.partCount} huesos y las 6 acciones de animación.</li>
                      <li>· Añade desgaste de bordes por geometría (`Pointiness`) y relieve 3D.</li>
                    </ul>
                  </div>

                  <div className="pt-4 mt-4 border-t border-slate-800/80 flex flex-wrap items-center gap-2">
                    <button
                      onClick={() =>
                        downloadTextFile(
                          generatedBlenderPy,
                          `enhance_${config.id}_blender.py`,
                          'text/x-python'
                        )
                      }
                      className="flex-1 inline-flex items-center justify-center gap-1.5 px-3 py-2.5 text-xs font-semibold bg-orange-500 hover:bg-orange-400 text-slate-950 rounded-lg transition-colors whitespace-nowrap shrink-0 cursor-pointer"
                    >
                      <FileCode className="w-4 h-4" />
                      Descargar Script .PY
                    </button>
                    <button
                      onClick={() => setActiveFormat('blender_py')}
                      className="inline-flex items-center justify-center gap-1 px-2.5 py-2.5 text-xs font-medium bg-slate-800 hover:bg-slate-700 text-slate-200 rounded-lg transition-colors whitespace-nowrap shrink-0 cursor-pointer"
                    >
                      Ver Script
                    </button>
                  </div>
                </div>
              </div>

              {/* Quick Integration Recipe Box */}
              <div className="rounded-xl border border-slate-800/90 bg-[#0b0d12] p-4 space-y-2">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <span className="text-xs font-semibold text-slate-200">
                    Flujo Dual: Suavizado Taubin en Tiempo Real + Post-Procesado Automático en Blender
                  </span>
                  <span className="text-xs text-slate-400 font-mono-tabular">
                    Pivote de Suelo: Y = 0.00m · Escala: 1 unidad = 1 metro
                  </span>
                </div>
                <p className="text-xs text-slate-400 leading-relaxed">
                  1) Directamente en el navegador, el motor aplica un filtro volumétrico estilo Blender (<code className="text-amber-300">Taubin λ-μ Sculpt Smoothing</code> + soldadura de normales en costuras UV) sobre la <code className="text-emerald-300">SkinnedMesh</code> antes de exportar a <code className="text-amber-300">.glb</code> y <code className="text-emerald-300">.html</code>.{' '}
                  2) Si abres el <code className="text-amber-300">.glb</code> en Blender y ejecutas <code className="text-orange-300">enhance_{config.id}_blender.py</code>, Blender añade modificadores <code className="text-orange-300">CORRECTIVE_SMOOTH</code>, subdivisión <code className="text-orange-300">SUBSURF</code> y shaders Cycles/EEVEE con <code className="text-orange-300">Subsurface Scattering</code>.
                </p>
              </div>
            </div>
          )}

          {activeFormat === 'blender_py' && (
            <div className="space-y-3">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <span className="text-xs text-slate-400">
                  Script Python (<code className="text-orange-300">bpy</code>) para Blender 3.6 / 4.x — añade Corrective Smooth, Subsurf Catmull-Clark y nodos PBR con Subsurface Scattering + Pointiness:
                </span>
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => handleCopy(generatedBlenderPy)}
                    className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium bg-slate-800 hover:bg-slate-700 text-slate-100 rounded-lg transition-colors cursor-pointer"
                  >
                    {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
                    {copied ? 'Copiado' : 'Copiar Script bpy'}
                  </button>
                  <button
                    onClick={() =>
                      downloadTextFile(
                        generatedBlenderPy,
                        `enhance_${config.id}_blender.py`,
                        'text/x-python'
                      )
                    }
                    className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold bg-orange-500 hover:bg-orange-400 text-slate-950 rounded-lg transition-colors cursor-pointer"
                  >
                    <Download className="w-3.5 h-3.5" />
                    Descargar enhance_{config.id}_blender.py
                  </button>
                </div>
              </div>
              <pre className="p-4 rounded-xl bg-[#080a0e] border border-slate-800 text-xs font-mono-tabular text-slate-300 overflow-x-auto max-h-[48vh] leading-relaxed select-text">
                {generatedBlenderPy}
              </pre>
            </div>
          )}

          {activeFormat === 'three_ts' && (
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-xs text-slate-400">
                  Módulo autónomo TypeScript con geometrías procedurales, sockets y loop de animación:
                </span>
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => handleCopy(generatedTsCode)}
                    className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium bg-slate-800 hover:bg-slate-700 text-slate-100 rounded-lg transition-colors cursor-pointer"
                  >
                    {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
                    {copied ? 'Copiado' : 'Copiar Código TS'}
                  </button>
                  <button
                    onClick={() =>
                      downloadTextFile(generatedTsCode, `create_${config.id}.ts`, 'text/typescript')
                    }
                    className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold bg-amber-500 hover:bg-amber-400 text-slate-950 rounded-lg transition-colors cursor-pointer"
                  >
                    <Download className="w-3.5 h-3.5" />
                    Descargar .TS
                  </button>
                </div>
              </div>
              <pre className="p-4 rounded-xl bg-[#080a0e] border border-slate-800 text-xs font-mono-tabular text-slate-300 overflow-x-auto max-h-[48vh] leading-relaxed select-text">
                {generatedTsCode}
              </pre>
            </div>
          )}

          {activeFormat === 'rpg_json' && (
            <div className="space-y-4">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <span className="text-xs text-slate-400">
                  Guarda o carga el blueprint JSON completo del enemigo (morfología, materiales PBR y estadísticas RPG):
                </span>
                <div className="flex items-center gap-2">
                  <label className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium bg-slate-800 hover:bg-slate-700 text-slate-100 rounded-lg transition-colors cursor-pointer">
                    <Upload className="w-3.5 h-3.5" />
                    Importar JSON
                    <input
                      type="file"
                      accept=".json"
                      onChange={handleJsonUpload}
                      className="hidden"
                    />
                  </label>
                  <button
                    onClick={() => handleCopy(generatedJson)}
                    className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium bg-slate-800 hover:bg-slate-700 text-slate-100 rounded-lg transition-colors cursor-pointer"
                  >
                    {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
                    {copied ? 'Copiado' : 'Copiar JSON'}
                  </button>
                  <button
                    onClick={() =>
                      downloadTextFile(
                        generatedJson,
                        `${config.id}_blueprint.json`,
                        'application/json'
                      )
                    }
                    className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold bg-amber-500 hover:bg-amber-400 text-slate-950 rounded-lg transition-colors cursor-pointer"
                  >
                    <Download className="w-3.5 h-3.5" />
                    Descargar .JSON
                  </button>
                </div>
              </div>
              {importError && (
                <div className="p-3 rounded-lg bg-rose-950/60 border border-rose-800 text-xs text-rose-200">
                  {importError}
                </div>
              )}
              <pre className="p-4 rounded-xl bg-[#080a0e] border border-slate-800 text-xs font-mono-tabular text-slate-300 overflow-x-auto max-h-[45vh] leading-relaxed select-text">
                {generatedJson}
              </pre>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
