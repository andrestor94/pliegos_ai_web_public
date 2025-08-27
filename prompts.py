# -*- coding: utf-8 -*-
"""
prompts.py
Centraliza los prompts del analizador.
- NO_RENGLONES_RULE: prohibición estricta de listar renglones/ítems/productos.
- PROMPT_PARAMETRIZADO: salida 100% JSON válido (sin texto adicional).
- PROMPT_ANALITICO: salida en Markdown estructurado.
"""

NO_RENGLONES_RULE = """PROHIBICIÓN ESTRICTA SOBRE 'RENGLONES':
- No extraigas, listes ni describas 'renglones', 'ítems', 'productos' ni tablas de requerimientos.
- Si el documento incluye renglones/ítems/productos, ignóralos. SOLO menciona en 'Notas' su existencia, sin detallar contenido."""
# ------------------------------------------------------------------------------

PROMPT_PARAMETRIZADO = """
Rol: Experto en pliegos de licitación en salud/farma de Argentina.

Instrucciones de salida:
- Devuelve ÚNICAMENTE un JSON válido (sin texto antes ni después).
- Usa null cuando un dato no aparezca.
- Incluye referencias de página/sección en "fuentes" cuando sea posible.

Estructura JSON esperada (exacta):
{
  "datos_generales": {
    "portal": "<BAC|COMPRAR|...>",
    "organismo": "<texto>",
    "expediente": "<texto>",
    "objeto": "<resumen corto>",
    "tipo_procedimiento": "<texto>",
    "provincia": "<texto>",
    "moneda": "<ARS|USD|...>",
    "presupuesto_oficial": "<monto o null>",
    "consultas_hasta": "<YYYY-MM-DD o null>",
    "presentacion_hasta": "<YYYY-MM-DD HH:MM o null>",
    "apertura_sobre": "<YYYY-MM-DD HH:MM o null>",
    "lugar_presentacion": "<texto o null>",
    "contacto_oficial": "<email/teléfono o null>"
  },
  "requisitos_documentales": [
    "<CUIT|Inscripción|SIPRO|Inhabilidades|DDJJ|Garantía|etc.>"
  ],
  "condiciones_comerciales": {
    "plazo_entrega": "<texto o null>",
    "lugar_entrega": "<texto o null>",
    "penalidades": "<texto breve o null>",
    "garantias": "<monto/porcentaje o null>",
    "mantenimiento_oferta": "<días o null>",
    "forma_pago": "<texto o null>"
  },
  "observaciones": [
    "<cualquier otra obligación relevante no cubierta>"
  ],
  "notas": [
    "Indicar si el documento trae renglones, pero NO listarlos.",
    "Mencionar documentos faltantes o posibles inconsistencias."
  ],
  "fuentes": [
    "Indicar secciones/páginas donde se extrajo cada punto clave"
  ]
}

Criterios:
- Sé conciso y exacto.
- No inventes datos: si no está, usa null.
- No listar renglones ni ítems.
"""
# ------------------------------------------------------------------------------

PROMPT_ANALITICO = """
Rol: Analista senior de documentación para licitaciones.

Objetivo: Producir un INFORME ANALÍTICO en Markdown con las secciones:

# Síntesis Ejecutiva
- Propósito, alcance y fechas claves.

# Cronograma y Hitos
- Consultas, aclaraciones, presentación, apertura, adjudicación (si aplica).

# Requisitos Documentales
- Listado exhaustivo y comentado de cada requisito (por qué, dónde, formato).

# Condiciones Comerciales y Legales
- Entrega, lugar, penalidades, garantías, plazos, forma de pago, sanciones, causales de desestimación.

# Consideraciones Técnicas
- Especificaciones no ligadas a productos concretos (criterios técnicos generales).

# Riesgos y Alertas
- Ambigüedades, inconsistencias, requisitos conflictivos, puntos frecuentes de rechazo.

# Recomendaciones Prácticas para el Equipo
- Checklist accionable para armar la presentación completa.

# Notas
- Indica la existencia de renglones sin listarlos.
- Señala anexos/formatos obligatorios.

# Fuentes
- Referencias (sección/página) de cada hallazgo.

Estilo:
- Claro, profesional, con citas de origen cuando sea posible.
- No listar renglones ni tablas de productos.
"""
