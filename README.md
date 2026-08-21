# Run

cd src && uv run python server.py

# Parlor Vision — Windows + Ollama Edition

Conversaciones de voz y visión con IA 100% local en tu máquina. Adaptado de macOS a Windows con Ollama, Whisper y Kokoro TTS. Optimizado para español con soporte para compartir pantalla.

> **Estado actual:** Prototipo funcional. Conversaciones básicas operativas con latencia moderada y soporte multi-idioma.

## ![texto del vínculo](https://miro.medium.com/1*psOrNqpYas3Nym3PyeyCVA.png)

## 🎯 Qué Logramos

### ✅ Migración Completa Mac → Windows

| Componente    | Original (macOS)            | Windows Edition              |
| ------------- | --------------------------- | ---------------------------- |
| **Motor LLM** | `litert-lm` (Apple GPU)     | **Ollama** (multiplataforma) |
| **Modelo**    | `gemma-4-E2B` (.litertlm)   | **`gemma4:e2b`** (Ollama)    |
| **TTS**       | `mlx-audio` (Apple Silicon) | **`kokoro-onnx`** (CPU)      |
| **Visión**    | Nativa via LiteRT           | **Via Ollama API**           |

### ✅ Reconocimiento de Voz (STT)

- **`faster-whisper`** transcribe audio en ~1 segundo
- **Detección automática de idioma** (español, inglés, italiano, etc.)
- Funciona en CPU sin GPU CUDA

### 🌎 Optimizado para Español

- **Respuesta prioritaria en español** — El sistema detecta automáticamente cuando el usuario habla en español y fuerza respuestas **exclusivas** en ese idioma.
- **System prompt dinámico reforzado** — Instrucciones explícitas que previenen respuestas en inglés: _"EL USUARIO TE ESTÁ HABLANDO EN ESPAÑOL. DEBES RESPONDER EXCLUSIVAMENTE EN ESPAÑOL."_
- **Whisper configurado para español** — Transcripción forzada con `language="es"` y `initial_prompt` en español para mayor precisión.
- **TTS con voz nativa hispana** — `ef_dora` (Kokoro, español) y `es-MX-DaliaNeural` (Edge-TTS fallback, voz mexicana profesional).
- **Historial con contexto lingüístico** — Mantiene los últimos 20 intercambios para consistencia en idioma y tema.
- **Soporte multi-idioma extendido** — También funciona en inglés, italiano, francés y más (detección automática).

### ✅ Visión (Cámara + Pantalla)

- Envío de imágenes a Ollama en cada turno
- **Compartir pantalla** — Muestra tu pantalla completa en vez de la cámara
- Descripción contextual de lo que ve la cámara o pantalla
- Combinación audio + imagen para respuestas ricas

### ✅ Text-to-Speech Streaming

- **Kokoro-ONNX** genera voz en streaming (frase por frase)
- Primera frase suena antes de que termine la respuesta completa
- Fallback a Edge-TTS si Kokoro falla

### ✅ Interfaz Completa

- Cámara en vivo con efectos de glow dinámicos
- Visualización de ondas de audio
- Transcripción en tiempo real
- Detección automática de voz (VAD) — manos libres
- Barge-in: puedes interrumpir la AI hablando

### ✅ Latencia Optimizada

| Etapa         | Primera vez | Subsiguientes |
| ------------- | ----------- | ------------- |
| Whisper (STT) | ~1.2s       | ~1.0s         |
| LLM (Ollama)  | ~15-20s     | ~2-4s         |
| TTS (Kokoro)  | ~3-5s       | ~3-4s         |
| **Total**     | **~20-30s** | **~6-10s**    |

---

## 🎯 Casos de Uso y Potencial

Esta base de **visión local + análisis inteligente + alerta por voz** se adapta a entornos reales donde **los datos nunca salen de la red**:

### Uso Personal y Social

| Caso                     | Ejemplo                                      |
| ------------------------ | -------------------------------------------- |
| **Asistente Personal**   | "¿Qué tengo pendiente hoy?"                  |
| **Práctica de Idiomas**  | Conversación fluida en español/inglés        |
| **Accesibilidad**        | Interacción por voz para discapacidad visual |
| **Educación**            | Tutor conversacional local                   |
| **Monitor de Seguridad** | Alerta cuando detecta personas o cambios     |

### Uso Industrial y Comercial

| Caso                         | Ejemplo                                     |
| ---------------------------- | ------------------------------------------- |
| **Control de Calidad**       | Detecta defectos en línea de producción     |
| **Mantenimiento Industrial** | Lee indicadores y detecta fallas en equipos |
| **Supervisor de EPP**        | Verifica uso de equipo de protección        |
| **Análisis de Pantalla**     | Debugging de código consultando a la IA     |

---

## 🏗️ Arquitectura Actual

```
Browser (mic + camera/screen)
    │
    │  WebSocket (audio WAV base64 + JPEG base64)
    ▼
FastAPI server (Python)
    │
    ├── Whisper (faster-whisper, CPU)
    │   └── Transcribe audio → texto
    │
    ├── Ollama API (localhost:11434)
    │   └── Gemma 4:2b con GPU (RTX 4050)
    │       └── Genera respuesta de texto
    │
    └── Kokoro TTS (ONNX Runtime, CPU)
        └── Convierte texto → audio
    │
    │  WebSocket (audio chunks PCM base64)
    ▼
Browser (playback + transcript)
```

---

## 🎨 Diseño y Flujo de Interacción

La interfaz usa un estilo **dark-mode moderno** con cámara en vivo, waveform animado, transcripción tipo chat e indicadores de estado con glow dinámico (verde → listening, ámbar → processing, índigo → speaking).

### Flujo de Conversación

```
1. Usuario habla → VAD detecta inicio de voz
2. VAD captura audio → Whisper transcribe (~1s)
3. Imagen de cámara/pantalla capturada → Envío via WebSocket
4. Ollama genera respuesta → Texto + imágenes (~2-20s)
5. TTS convierte texto → audio en streaming
6. Primera frase suena antes de completar respuesta
7. Usuario puede interrumpir hablando (barge-in)
8. Historial actualizado → Persiste entre sesiones
```

---

## 🚀 Inicio Rápido

### 1. Prerrequisitos

- **Python 3.12** (exacto)
- **Ollama** instalado con modelo `gemma4:e2b`
- **Windows 10/11** con NVIDIA RTX (recomendado)
- **~8 GB RAM** libre

### 2. Verificar Ollama

```bash
ollama list
# Debes ver: gemma4:e2b
```

### 3. Crear Entorno

```bash
conda create -n parlor python=3.12 -y
conda activate parlor
```

### 4. Instalar Dependencias

```bash
cd C:\Users\EdwinQuintero\Documents\Anaconda 3\parlor\parlor_2\src
pip install -e .
```

### 5. Ejecutar

```bash
python server.py
```

### 6. Abrir en Navegador

**http://localhost:8000**

Permite cámara y micrófono, ¡y habla!

---

## ⚙️ Configuración

| Variable          | Default                  | Descripción             |
| ----------------- | ------------------------ | ----------------------- |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | URL del servidor Ollama |
| `OLLAMA_MODEL`    | `gemma4:e2b`             | Modelo Ollama a usar    |
| `PORT`            | `8000`                   | Puerto del servidor     |

### Ejemplo personalizado:

```bash
set OLLAMA_MODEL=gemma3:4b
set PORT=8080
python server.py
```

---

## ⚠️ Retos Pendientes

### 🔴 Críticos

#### 1. Latencia de Primera Respuesta (15-20s)

**Problema:** La primera interacción tarda ~20 segundos, las subsiguientes ~3s.

**Causa:**

- Carga inicial del modelo en memoria de GPU
- Primera inferencia requiere warm-up de Ollama

**Posibles soluciones:**

- [ ] **Precarga del modelo:** Enviar petición dummy al iniciar servidor
- [ ] **Modelo más ligero:** Probar `gemma3:4b` o `llama3.2:3b`
- [ ] **Mantener modelo activo:** Evitar que Ollama descargue el modelo

#### 2. Consistencia de Idioma

**Problema:** A veces responde en inglés aunque le hables en español.

**Causa:**

- Gemma 4 tiende al inglés por defecto
- Detección heurística no cubre todos los casos

**Estado actual:**

- ✅ Detecta español con palabras clave
- ✅ Agrega instrucción explícita al system prompt
- ❌ No funciona 100% consistente

**Posibles soluciones:**

- [ ] **Prompt más agresivo:** `"RESPONDE EXCLUSIVAMENTE EN ESPAÑOL. No uses inglés."`
- [ ] **Modelo con mejor soporte español:** `qwen2.5:7b` o `llama3.2:3b`
- [ ] **Traducción post-LLM:** Si detecta inglés, traducir respuesta

#### 3. Whisper en CPU (no GPU)

**Problema:** `faster-whisper` no funciona con GPU por falta de `cublas64_12.dll`.

**Causa:** CUDA Toolkit no instalado o configuración incompleta.

**Posibles soluciones:**

- [ ] **Instalar CUDA Toolkit 12.x:** https://developer.nvidia.com/cuda-downloads
- [ ] **Usar `whisper.cpp`:** Alternativa con mejor soporte Windows GPU
- [ ] **Mantener CPU:** ~1s es aceptable para muchos casos

---

### 🟡 Importantes

#### 4. Modelo Óptimo para Windows

**Evaluación de alternativas:**

| Modelo            | Tamaño | GPU VRAM | Velocidad  | Multi-idioma | Visión |
| ----------------- | ------ | -------- | ---------- | ------------ | ------ |
| `gemma4:e2b`      | 7.2 GB | 6 GB     | ⚡⚡⚡     | ⚠️ Parcial   | ✅ Sí  |
| `gemma3:4b`       | 3.3 GB | 4 GB     | ⚡⚡⚡⚡   | ✅ Bueno     | ✅ Sí  |
| `qwen2.5:7b`      | 4.7 GB | 6 GB     | ⚡⚡⚡     | ✅ Excelente | ❌ No  |
| `llama3.2-vision` | 3 GB   | 4 GB     | ⚡⚡⚡⚡   | ✅ Bueno     | ✅ Sí  |
| `llama3.2:3b`     | 2 GB   | 2 GB     | ⚡⚡⚡⚡⚡ | ⚠️ Parcial   | ❌ No  |

**Recomendación a probar:**

- **`llama3.2-vision`** — Balance velocidad/visión/multi-idioma
- **`gemma3:4b`** — Más rápido que gemma4 con buena capacidad

#### 5. TTS Solo Voz Femenina

**Problema:** Kokoro usa `af_heart` (voz femenina) por defecto.

**Posibles soluciones:**

- [ ] **Agregar selección de voces:** Kokoro tiene 10+ voces disponibles
- [ ] **Integrar Edge-TTS:** Voces masculinas/femeninas de Azure
- [ ] **Voz configurable:** Parámetro en UI o `.env`

#### 6. Transcripción de Audio Cortada

**Problema:** Whisper a veces transcribe frases incompletas.

**Causa:** VAD del navegador corta audio demasiado pronto.

**Posibles soluciones:**

- [ ] **Ajustar VAD:** `redemptionMs`, `minSpeechMs` en frontend
- [ ] **Whisper con context:** Usar audio previo como contexto
- [ ] **Punctuation restoration:** Modelo que añade puntuación

---

### 🟢 Deseables

#### 7. Mejoras de UX

- [ ] **Indicador de idioma:** Mostrar "🇪🇸 Español" o "🇬🇧 English" en UI
- [ ] **Selector de voz:** Dropdown para elegir voz del TTS
- [ ] **Configuración de cámara:** Resolución, flip horizontal
- [ ] **Historial persistente:** Guardar conversaciones entre sesiones
- [ ] **Modo oscuro/claro:** Toggle de tema visual

#### 8. Funcionalidades Avanzadas

- [ ] **Memoria a largo plazo:** Recordar conversaciones previas
- [ ] **RAG con documentos:** Cargar PDFs/TXT para consultas
- [ ] **Múltiples cámaras:** Soporte para cámara frontal + trasera
- [ ] **Modo offline total:** Sin dependencia de internet
- [ ] **Grabación de conversaciones:** Exportar audio + texto

#### 9. Optimización de Rendimiento

- [ ] **GPU para Whisper:** Instalar CUDA correctamente
- [ ] **Streaming de LLM:** Token por token en vez de respuesta completa
- [ ] **Cache de respuestas:** Respuestas frecuentes en cache
- [ ] **Compresión de imágenes:** JPEG con menor calidad para velocidad
- [ ] **Batch processing:** Múltiples frases en un solo request

---

## 📊 Benchmarks

Los números de latencia están en la tabla "Latencia Optimizada" más arriba. Para medir en tu entorno:

```bash
# Iniciar servidor
python server.py

# En otra terminal — benchmark end-to-end
python benchmarks/bench.py

# Benchmark solo TTS
python benchmarks/benchmark_tts.py
```

---

## 📁 Estructura del Proyecto

```
parlor_2/
├── src/
│   ├── server.py              # FastAPI + Ollama + Whisper + TTS
│   ├── tts.py                 # Kokoro-ONNX + Edge-TTS fallback
│   ├── index.html             # Frontend (VAD, cámara, audio)
│   ├── pyproject.toml         # Dependencias
│   └── benchmarks/
│       ├── bench.py           # Benchmark WebSocket
│       └── benchmark_tts.py   # Benchmark TTS
├── models/
│   └── whisper/               # Whisper descargado automáticamente
├── .env.example               # Variables de entorno
├── .gitignore
└── README.md
```

---

## 🔧 Troubleshooting

### Ollama no conecta

```bash
ollama list
ollama serve  # Si no está corriendo
```

### Whisper falla

```bash
# Verificar instalación
python -c "from faster_whisper import WhisperModel; print('OK')"

# Reinstalar si hay errores
pip install --force-reinstall faster-whisper
```

### TTS no funciona

```bash
# Kokoro falla, instalar fallback
pip install edge-tts

# El servidor usará edge-tts automáticamente
```

### GPU no se usa

```bash
# Verificar que Ollama usa GPU
ollama run gemma4:e2b "test"

# Revisar Task Manager → Performance → GPU
```

---

## 🙏 Agradecimientos

- **Original:** [Parlor](https://github.com/fikrikarim/parlor) por Fikri Karim
- **Ollama:** https://ollama.com — Serving local models
- **Gemma 4:** https://ai.google.dev/gemma — Google DeepMind
- **Whisper:** https://github.com/openai/whisper — OpenAI
- **faster-whisper:** https://github.com/SYSTRAN/faster-whisper — SYSTRAN
- **Kokoro:** https://huggingface.co/hexgrad/Kokoro-82M — Hexgrad
- **Silero VAD:** https://github.com/snakers4/silero-vad

---

## 📄 Licencia

[Apache 2.0](LICENSE) — Mismo proyecto original

---

## 📝 Historial de Cambios

### v0.3.0 — Windows + Ollama + Whisper

- ✅ Migración completa de macOS a Windows
- ✅ Integración con Ollama para LLM
- ✅ Transcripción de voz con `faster-whisper`
- ✅ Detección automática de idioma
- ✅ Respuestas multi-idioma (español/inglés)
- ✅ Visión funcional con cámara en vivo
- ✅ TTS con Kokoro-ONNX
- ✅ Latencia optimizada (~7s en respuestas subsequentes)

### v0.2.0 — Windows con llama-cpp-python

- ⚠️ Intento inicial con `llama-cpp-python` (abandonado)
- ⚠️ Problemas de compilación y compatibilidad

### v0.1.0 — Adaptación inicial

- 📋 Port básico de macOS a Windows
- ❌ Sin STT, sin multi-idioma, sin visión funcional
