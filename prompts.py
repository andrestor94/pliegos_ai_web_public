# -*- coding: utf-8 -*-
"""
prompts.py
Centraliza textos de prompts para análisis de pliegos y chat.

Nota:
- Estos prompts son genéricos; si querés reglas dinámicas de citas (multi-anexo vs. único),
  usá la función auxiliar `reglas_citas(varios_anexos: bool)` que está al final.
- utils.py hoy trae estas constantes: PROMPT_PARAMETRIZADO, PROMPT_ANALITICO, NO_RENGLONES_RULE
"""

# --- Regla puntual que a veces queremos reforzar en los modelos ---
NO_RENGLONES_RULE = (
    "Para el campo 'Número de renglón' en la Ficha, escribir exactamente: "
    "'Total de renglones: <cantidad>; ver Sección 9 para el detalle completo'. "
    "Nunca uses 'N' como placeholder ni inventes cantidades."
)

# --- Prompt principal estilo 'Ficha estandarizada' (Andrés) ---
PROMPT_PARAMETRIZADO = r"""
# (Instrucciones internas: NO imprimir este encabezado ni estas reglas en la salida)

Objetivo
- Generar un informe de análisis de licitación en Argentina (ámbitos nacional, provincial o municipal), exhaustivo y sin invenciones.
- La salida DEBE comenzar con el título EXACTO: "Ficha estandarizada del procedimiento" (sin numeración, NO escribir "0)").
- Dentro de esa Ficha, incluir los campos con estos rótulos exactos (cada uno con su valor):
  • N° de proceso
  • Nombre de proceso
  • Objeto de la contratación
  • Procedimiento de selección
  • Tipo de cotización
  • Tipo de adjudicación
  • Cantidad de ofertas permitidas
  • Estado
  • Plazo de mantenimiento de la oferta
  • Número de renglón (escribir: "Total de renglones: <cantidad>; ver Sección 9 para el detalle completo"; JAMÁS usar 'N')
  • Objeto del gasto
  • Código del item (si corresponde a nivel renglón, dejar referencia a Sección 9)
  • Descripción   (si corresponde a nivel renglón, dejar referencia a Sección 9)
  • Cantidad      (si corresponde a nivel renglón, dejar referencia a Sección 9)
  • Inicio y final de consultas
  • Fecha y hora del acto de apertura
  • Monto
  • Moneda
  • Duración del contrato
- Si algo NO figura en los archivos, escribir "NO ESPECIFICADO" y no inventar ni inferir.
- Cada línea con dato crítico debe terminar con su cita de fuente, según las Reglas de Citas.

{REGLAS_CITAS}

Estilo
- Encabezados y listas claras; sin meta-texto ("parte X de Y", "revise el resto", etc.).
- Deduplicar, fusionar y no repetir información.
- Mantener terminología del pliego. Usar 2 decimales si el pliego lo exige para precios.
- No mencionar nombres de portales/sistemas salvo que figuren explícitamente en los documentos.

Estructura de salida EXACTA (usar estos títulos tal cual)
Ficha estandarizada del procedimiento (campos estandarizados)
1) Resumen ejecutivo (<=200 palabras)
2) Datos clave del llamado
3) Alcance contractual y vigencias
4) Entregas y logística
5) Presentación y contenido de la oferta
6) Evaluación, empate y mejora de oferta
7) Garantías
8) Muestras, envases, etiquetado y caducidad (si aplica)
9) Renglones y planilla de cotización
10) Checklist operativo
11) Fechas y plazos críticos
12) Observaciones finales

Cobertura obligatoria por sección (según aplique)
- 2) Datos clave: Organismo, Expediente/N° proceso, Tipo/Modalidad/Etapa, Objeto, Rubro, Lugar/área; contactos/portales (mails/URLs) si figuran.
- 3) Alcance/vigencias: mantenimiento de oferta y prórroga; perfeccionamiento; ampliaciones/topes; duración/término del contrato.
- 4) Entregas: lugar/horarios; forma (única/parcelada); plazos; flete/descarga.
- 5) Presentación: sobre/caja, duplicado, firma, rotulado; documentación fiscal/registral; costo/valor del pliego si existe.
- 6) Evaluación: cuadro comparativo; tipo de cambio; criterios cuali/cuantitativos; empates; mejora de precio.
- 7) Garantías: umbrales por UC si aplica; % mantenimiento y % cumplimiento con plazos/condiciones; contragarantías.
- 8) Muestras/envases/etiquetado/caducidad: ANMAT/BPM; cadena de frío; rotulados; vigencia mínima.
- 9) Renglones/planilla: incluir TODOS los renglones (si existe planilla). Por renglón: Cantidad, Código (si hay), Descripción y especificaciones técnicas relevantes en 1 línea. Si hay demasiados, mantener listado completo aunque la descripción se acote.
- 10) Checklist: acciones para el oferente.
- 11) Fechas críticas: presentación, apertura, mantenimiento, entregas, consultas, etc.
- 12) Observaciones finales: alertas y condicionantes.

Guía de sinónimos/normalización (Argentina)
- "Número de proceso" ~ "Expediente", "N° de procedimiento", "N° de trámite", "EX-...", "IF-...".
- "Nombre de proceso" ~ "Denominación del procedimiento", "Título del llamado".
- "Objeto de la contratación" ~ "Objeto", "Adquisición/Contratación de", "Finalidad".
- "Procedimiento de selección" ~ "Tipo de procedimiento", "Modalidad", "Clase del llamado" (Licitación Pública/Privada, Contratación Directa, Compra Menor, Subasta, etc.).
- "Tipo de cotización" ~ "Forma de cotización", "Modo de cotizar", "Planilla de precios", "Ítem por ítem", "Global/Total", "Por renglón/lote".
- "Tipo de adjudicación" ~ "Criterio de adjudicación", "Adjudicación por renglón/lote/total".
- "Cantidad de ofertas permitidas" ~ "Número de propuestas por oferente", "Ofertas alternativas/adicionales".
- "Estado" ~ "Situación del trámite" (vigente, abierto, cerrado, desierto, fracasado, adjudicado).
- "Plazo de mantenimiento de la oferta" ~ "Validez de la oferta".
- "Número de renglón" ~ "Renglón", "Ítem (número)".
- "Objeto del gasto" ~ "Partida presupuestaria", "Clasificador/Objeto del gasto", "Estructura programática".
- "Código del ítem" ~ "Código interno", "Código catálogo", "SKU".
- "Descripción" ~ "Descripción del ítem", "Especificaciones técnicas".
- "Cantidad" ~ "Cantidad solicitada/Requerida".
- "Inicio y final de consultas" ~ "Plazo de consultas/aclaraciones", "Recepción de consultas", "Preguntas y respuestas".
- "Fecha y hora del acto de apertura" ~ "Apertura", "Acto de apertura de ofertas".
- "Monto" ~ "Presupuesto oficial/referencial", "Monto estimado", "Crédito disponible".
- "Moneda" ~ "Moneda de cotización" (ARS, USD, etc.), "Tipo de cambio".
- "Duración del contrato" ~ "Plazo contractual", "Vigencia", "Por el término de".
- "Presentación de ofertas" ~ "Acto de presentación", "Límite de recepción".
- "Garantía de mantenimiento" ~ "Garantía de oferta".
- "Garantía de cumplimiento" ~ "Garantía contractual".
- "Planilla de cotización" ~ "Formulario de oferta", "Cuadro comparativo", "Planilla de precios".
- "Tipo de cambio BNA" ~ "Banco Nación vendedor del día anterior".

Regla especial:
- {NO_RENGLONES_RULE}
"""

# --- Prompt alternativo "analítico" (estructura 2.1–2.16) ---
PROMPT_ANALITICO = r"""
# (Instrucciones internas: NO imprimir este encabezado ni estas reglas en la salida)
Reglas clave:
- Cero invenciones; si falta o es ambiguo: escribir "NO ESPECIFICADO" y explicarlo en la misma sección.
- Cada dato crítico debe terminar con su fuente entre paréntesis, según las Reglas de Citas.
- Cobertura completa (oferta -> ejecución), con normativa citada.
- Deduplicar, fusionar, no repetir; un único informe integrado.
- Prohibido meta texto tipo "parte X de Y" o "revise el resto".
- No imprimir etiquetas internas como [PÁGINA N].
- No usar los títulos literales "Informe Completo" ni "Informe Original".

Formato de salida:
1) RESUMEN DE PLIEGO (<=200 palabras)
2) INFORME DETALLADO CON TRAZABILIDAD
   2.1 Identificación del llamado
   2.2 Calendario y lugares
   2.3 Contactos y portales (listar TODOS los e-mails y URLs detectados)
   2.4 Alcance y plazo contractual
   2.5 Tipología / modalidad (citar norma/artículos)
   2.6 Mantenimiento de oferta y prórroga
   2.7 Garantías (umbral UC, %, plazos, formas)
   2.8 Presentación de ofertas (soporte, firmas, docs obligatorias) e incluir costo/valor del pliego y mecanismo de adquisición/pago
   2.9 Apertura, evaluación y adjudicación (tipo de cambio BNA, comisión, criterios, preferencias)
   2.10 Subsanación (qué sí/no)
   2.11 Perfeccionamiento y modificaciones
   2.12 Entrega, lugares y plazos
   2.13 Planilla de cotización y renglones (enumerar TODOS los renglones; por renglón incluir cantidades, UM, descripción y especificaciones técnicas relevantes)
   2.14 Muestras
   2.15 Normativa aplicable (todas las leyes/decretos/resoluciones/disposiciones citadas, con número/año y fuente)
   2.16 Catálogo de artículos citados (Art. N — síntesis 1–2 líneas; una línea por artículo; con cita)

Estilo:
- Títulos con mayúsculas iniciales, listas claras, tablas simples. Sin "#".
- Aplicar la guía de sinónimos y conservar la terminología encontrada.

{REGLAS_CITAS}
"""

# ========== Ayudantes opcionales (por si querés formatear dinámicamente) ==========

def reglas_citas(varios_anexos: bool) -> str:
    """
    Devuelve el bloque de reglas de citas recomendado para insertar en los prompts,
    según se trate de un documento único o un multi-anexo.
    """
    if varios_anexos:
        return (
            "Reglas de Citas:\n"
            "- Documento MULTI-ANEXO: al final de cada línea con dato, usar (Anexo X, p. N).\n"
            "- Deducir N tomando la etiqueta [PÁGINA N] más cercana dentro del texto del ANEXO correspondiente.\n"
            "- Si no hay paginación: (Fuente: documento provisto)."
        )
    else:
        return (
            "Reglas de Citas:\n"
            "- Documento ÚNICO: al final de cada línea con dato, usar (p. N) a partir de la etiqueta [PÁGINA N] más cercana.\n"
            "- Si no hay paginación: (Fuente: documento provisto)."
        )
