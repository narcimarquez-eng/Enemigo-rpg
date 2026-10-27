import React, { useMemo, useState } from 'react';
import {
  Check,
  Code2,
  Copy,
  Download,
  FileCode,
  FileJson,
  Package,
  Upload,
  X,
} from 'lucide-react';
import { EnemyConfig, calculateThreatScore } from '../types/enemy';
import { downloadTextFile, generateThreeJsModuleCode } from '../three/exporters';

interface ExportModalProps {
  isOpen: boolean;
  onClose: () => void;
  config: EnemyConfig;
  metrics: { totalVertices: number; totalTriangles: number; partCount: number };
  onExportGLB: (binary: boolean) => Promise<void>;
  onImportConfig: (cfg: EnemyConfig) => void;
  exportStatusMessage: string | null;
}

export const ExportModal: React.FC<ExportModalProps> = ({
  isOpen,
  onClose,
  config,
  metrics,
  onExportGLB,
  onImportConfig,
  exportStatusMessage,
}) => {
  const [activeFormat, setActiveFormat] = useState<'comparison' | 'three_ts' | 'rpg_json'>(
    'comparison'
  );
  const [copied, setCopied] = useState(false);
  const [isExportingGlb, setIsExportingGlb] = useState(false);
  const [importError, setImportError] = useState<string | null>(null);

  const generatedTsCode = useMemo(() => generateThreeJsModuleCode(config), [config]);
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
      aria-label="Centro de Exportación GLB y Three.js"
    >
      <div className="w-full max-w-4xl rounded-xl border border-slate-800 bg-[#0e1118] text-slate-100 shadow-2xl overflow-hidden flex flex-col max-h-[88vh]">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-800/80">
          <div>
            <h2 className="font-display text-lg font-bold tracking-wide text-slate-100">
              Pipeline de Integración 3D: .GLB Binario vs. Módulo Three.js Nativo
            </h2>
            <p className="text-xs text-slate-400 mt-0.5 font-mono-tabular">
              {config.name} · {metrics.partCount} sub-mallas · {metrics.totalVertices.toLocaleString()} vértices ·{' '}
              {metrics.totalTriangles.toLocaleString()} triángulos · Índice de Amenaza {threatScore}
            </p>
          </div>
          <button
            onClick={onClose}
            className="p-2 text-slate-400 hover:text-slate-100 hover:bg-slate-800/70 rounded-lg transition-colors"
            aria-label="Cerrar ventana de exportación"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Format Switcher Bar */}
        <div className="flex items-center justify-between gap-4 px-6 py-3 bg-[#131722] border-b border-slate-800/80">
          <div className="flex items-center gap-1.5 p-1 bg-[#0b0d12] rounded-lg border border-slate-800/80">
            <button
              onClick={() => setActiveFormat('comparison')}
              className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors whitespace-nowrap shrink-0 ${
                activeFormat === 'comparison'
                  ? 'bg-amber-500 text-slate-950 font-semibold'
                  : 'text-slate-300 hover:text-white'
              }`}
            >
              1. Guía y Descarga Directa (.GLB + Three.js)
            </button>
            <button
              onClick={() => setActiveFormat('three_ts')}
              className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors whitespace-nowrap shrink-0 ${
                activeFormat === 'three_ts'
                  ? 'bg-amber-500 text-slate-950 font-semibold'
                  : 'text-slate-300 hover:text-white'
              }`}
            >
              2. Código Three.js / TypeScript
            </button>
            <button
              onClick={() => setActiveFormat('rpg_json')}
              className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors whitespace-nowrap shrink-0 ${
                activeFormat === 'rpg_json'
                  ? 'bg-amber-500 text-slate-950 font-semibold'
                  : 'text-slate-300 hover:text-white'
              }`}
            >
              3. Blueprint JSON + Importar
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
            <div className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-5 items-stretch">
                {/* Pipeline 1: GLB Binary */}
                <div className="rounded-xl border border-slate-800 bg-[#131722] p-5 flex flex-col justify-between">
                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-semibold text-amber-400 tracking-wide">
                        OPCIÓN A · ESTÁNDAR DE MOTOR 3D
                      </span>
                      <Package className="w-4 h-4 text-amber-400" />
                    </div>
                    <h3 className="text-base font-bold text-slate-100">
                      Exportación Binaria .GLB (GLTF 2.0)
                    </h3>
                    <p className="text-xs text-slate-300 leading-relaxed">
                      Empaqueta la jerarquía 3D completa, materiales PBR (`MeshPhysicalMaterial`),
                      nodos de anclaje (`socket_weapon_r`, `socket_shield_l`, `socket_head`,
                      `socket_core`) y los atributos RPG dentro de `scene.userData` en un único
                      archivo binario compacto.
                    </p>
                    <ul className="text-xs text-slate-400 space-y-1.5 pt-1">
                      <li>· Ideal para cargar con `GLTFLoader` en Three.js, React Three Fiber, Godot o Unity.</li>
                      <li>· Permite previsualizar o retocar la malla en Blender conservando los pivotes en Y=0.</li>
                      <li>· Incluye todas las sub-partes separadas por nodo para animar o desmembrar en combate.</li>
                    </ul>
                  </div>

                  <div className="pt-5 mt-5 border-t border-slate-800/80 flex flex-wrap items-center gap-2.5">
                    <button
                      onClick={() => handleGlbTrigger(true)}
                      disabled={isExportingGlb}
                      className="flex-1 inline-flex items-center justify-center gap-2 px-4 py-2.5 text-xs font-semibold bg-amber-500 hover:bg-amber-400 text-slate-950 rounded-lg transition-colors whitespace-nowrap shrink-0 cursor-pointer"
                    >
                      <Download className="w-4 h-4" />
                      {isExportingGlb ? 'Empaquetando .GLB...' : `Descargar ${config.id}.glb`}
                    </button>
                    <button
                      onClick={() => handleGlbTrigger(false)}
                      disabled={isExportingGlb}
                      className="inline-flex items-center justify-center gap-1.5 px-3 py-2.5 text-xs font-medium bg-slate-800 hover:bg-slate-700 text-slate-200 rounded-lg transition-colors whitespace-nowrap shrink-0 cursor-pointer"
                    >
                      <FileJson className="w-3.5 h-3.5" />
                      .GLTF (JSON)
                    </button>
                  </div>
                </div>

                {/* Pipeline 2: Direct Three.js Procedural Code */}
                <div className="rounded-xl border border-slate-800 bg-[#131722] p-5 flex flex-col justify-between">
                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-semibold text-sky-400 tracking-wide">
                        OPCIÓN B · CERO ARCHIVOS EXTERNOS
                      </span>
                      <Code2 className="w-4 h-4 text-sky-400" />
                    </div>
                    <h3 className="text-base font-bold text-slate-100">
                      Módulo Nativo Three.js / TypeScript (.TS)
                    </h3>
                    <p className="text-xs text-slate-300 leading-relaxed">
                      Genera una función constructora TypeScript (`createEnemy()`) que instancia
                      las geometrías (`LatheGeometry`, `ExtrudeGeometry`, `TubeGeometry`),
                      materiales PBR y el controlador cinemático (`update(state, elapsed, delta)`)
                      en tiempo real.
                    </p>
                    <ul className="text-xs text-slate-400 space-y-1.5 pt-1">
                      <li>· Cero peticiones de red ni errores 404: pesa menos de 6 KB en tu bundle.</li>
                      <li>· Permite mutar colores, escala y estadísticas dinámicamente según el nivel de mazmorra.</li>
                      <li>· Incluye animaciones procedurales listas (`idle`, `walk`, `attack`, `hit`, `death`).</li>
                    </ul>
                  </div>

                  <div className="pt-5 mt-5 border-t border-slate-800/80 flex flex-wrap items-center gap-2.5">
                    <button
                      onClick={() =>
                        downloadTextFile(
                          generatedTsCode,
                          `create_${config.id}.ts`,
                          'text/typescript'
                        )
                      }
                      className="flex-1 inline-flex items-center justify-center gap-2 px-4 py-2.5 text-xs font-semibold bg-sky-500 hover:bg-sky-400 text-slate-950 rounded-lg transition-colors whitespace-nowrap shrink-0 cursor-pointer"
                    >
                      <FileCode className="w-4 h-4" />
                      Descargar create_{config.id}.ts
                    </button>
                    <button
                      onClick={() => setActiveFormat('three_ts')}
                      className="inline-flex items-center justify-center gap-1.5 px-3 py-2.5 text-xs font-medium bg-slate-800 hover:bg-slate-700 text-slate-200 rounded-lg transition-colors whitespace-nowrap shrink-0 cursor-pointer"
                    >
                      Ver Código
                    </button>
                  </div>
                </div>
              </div>

              {/* Quick Integration Recipe Box */}
              <div className="rounded-xl border border-slate-800/90 bg-[#0b0d12] p-4 space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-semibold text-slate-200">
                     Nodos y Sockets Embebidos en Ambos Formatos
                  </span>
                  <span className="text-xs text-slate-400 font-mono-tabular">
                    Pivote de Suelo: Y = 0.00m · Escala: 1 unidad = 1 metro
                  </span>
                </div>
                <p className="text-xs text-slate-400 leading-relaxed">
                  Al importar el archivo <code className="text-amber-300">.glb</code> con{' '}
                  <code className="text-slate-300">GLTFLoader</code> o ejecutar el módulo{' '}
                  <code className="text-sky-300">createEnemy()</code>, puedes acoplar partículas,
                  barras de vida flotantes o nuevas armas buscando los nodos{' '}
                  <code className="text-emerald-300">enemy.getObjectByName('socket_weapon_r')</code>,{' '}
                  <code className="text-emerald-300">'socket_shield_l'</code>,{' '}
                  <code className="text-emerald-300">'socket_head'</code> y{' '}
                  <code className="text-emerald-300">'socket_core'</code>, y leer sus estadísticas
                  directamente desde <code className="text-amber-300">enemy.userData.rpgStats</code>.
                </p>
              </div>
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
