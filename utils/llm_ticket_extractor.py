# utils/llm_ticket_extractor.py
import json
import os
import re
from typing import Dict, Optional, List
import ollama


class LLMTicketExtractor:
    """
    Extractor de tickets v3 — Correcciones acumuladas:
    - num_predict aumentado a 800 para modelos 7b+
    - stop tokens para evitar texto extra tras el JSON
    - Extracción de JSON con balance de llaves (evita cortes)
    - Fusión defensiva: no rompe si el LLM falla
    - Regex expandido: cubre VALOR TOTAL, BASE IMPONIBLE IVA, etc.
    - IVA extraído con su propio patrón
    - Corrección de decimales perdidos por OCR (ej: 196 → 1.96)
    - Extractor de establecimiento inteligente (busca RAZÓN SOCIAL primero)
    - Fallback de fecha mejorado
    - Campo llm_ok para facilitar debug
    """

    def __init__(self, model: str = "qwen2.5:7b"):
        self.model = model
        ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        os.environ["OLLAMA_HOST"] = ollama_host
        print(f"LLMTicketExtractor v3 con modelo: {model}")

    # ------------------------------------------------------------------
    # Preprocesamiento de texto OCR
    # ------------------------------------------------------------------

    def _preprocess_text(self, text: str) -> str:
        """Limpieza inicial del texto OCR"""
        # Eliminar paréntesis/corchetes pegados a números
        text = re.sub(r'[\(\[\{]\s*(\d)', r'\1', text)
        # Normalizar etiquetas comunes con dos puntos
        text = re.sub(r'(?i)total\s*[:\.]', 'TOTAL:', text)
        text = re.sub(r'(?i)subtotal\s*[:\.]', 'SUBTOTAL:', text)
        # Arreglos de URL mal formateadas por OCR
        text = text.replace("https:l", "https://").replace("fmodelo com", "fmodelo.com")
        return text

    # ------------------------------------------------------------------
    # Prompt del sistema
    # ------------------------------------------------------------------

    def _create_system_prompt(self) -> str:
        return (
            "Eres un analizador de tickets de caja guatemaltecos (sistema FEL).\n"
            "Extrae datos y responde SOLO con JSON válido, sin texto extra.\n\n"
            "REGLAS:\n"
            '1. "nit": NIT del EMISOR (establecimiento). '
            "Está en el encabezado, ANTES de la sección FACTURA. "
            "Formato: dígitos con guión al final (ej: 3535591-3). "
            "IGNORA el NIT del certificador y del comprador.\n"
            '2. "serie": Código alfanumérico corto después de \"serie:\". '
            "Ej: E9628AD9. Solo el primer segmento antes de guión.\n"
            '3. "dte": Número después de \"Numero de DTE:\" o \"DTE:\". '
            "Solo dígitos, 7-12 caracteres. Ej: 49498949.\n"
            '4. "establecimiento": Solo primeras 2-3 líneas. Máx 40 chars.\n'
            '5. "fecha": Formato DD/MM/YYYY.\n'
            '6. "total": Float. Total final.\n'
            '7. "subtotal": Float. Antes de impuestos. Si no existe: 0.00.\n'
            '8. "iva": Float. Si no existe: 0.00.\n'
            '9. "propina": Float. Si no existe: 0.00.\n'
            '10. "factura_url": URL o null.\n\n'
            "RESPUESTA (JSON en una sola línea):\n"
            '{"establecimiento":null,"fecha":null,"subtotal":0.00,'
            '"iva":0.00,"total":0.00,"propina":0.00,"factura_url":null,'
            '"nit":null,"serie":null,"dte":null}'
        )

    # ------------------------------------------------------------------
    # Llamada al LLM
    # ------------------------------------------------------------------

    def _call_llm(self, ocr_text: str) -> str:
        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._create_system_prompt()},
                    {
                        "role": "user",
                        "content": (
                            "Extrae los datos de este ticket. "
                            "Responde ÚNICAMENTE con el JSON, nada más.\n\n"
                            f"[TICKET]\n{ocr_text}\n[/TICKET]"
                        ),
                    },
                ],
                options={
                    "temperature": 0.0,
                    "num_predict": 800,          # era 400, insuficiente para 7b
                    "stop": ["\n\n", "```"],     # corta si el modelo agrega texto extra
                },
            )
            return response["message"]["content"]
        except Exception as e:
            raise Exception(f"Error Ollama: {e}")

    # ------------------------------------------------------------------
    # Limpieza y parseo de la respuesta JSON
    # ------------------------------------------------------------------

    def _clean_json_response(self, response: str) -> str:
        """
        Extrae el bloque JSON de la respuesta del LLM.
        Usa balance de llaves para evitar cortes parciales.
        """
        response = re.sub(r"```json\s*", "", response)
        response = re.sub(r"```\s*", "", response)

        start = response.find("{")
        if start == -1:
            return ""

        depth = 0
        for i, ch in enumerate(response[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return response[start : i + 1]

        return response[start:].strip()

    # ------------------------------------------------------------------
    # Conversión de strings a float
    # ------------------------------------------------------------------

    def _parse_money(self, value) -> float:
        """Convierte string a float manejando formatos diversos"""
        if isinstance(value, (int, float)):
            return float(value)
        if not value:
            return 0.0

        s_val = re.sub(r"[^\d.,]", "", str(value).strip())
        if not s_val:
            return 0.0

        try:
            # "1,500.00" o "12,045.82" → coma de miles, punto decimal
            if "," in s_val and "." in s_val:
                s_val = s_val.replace(",", "")
            elif "," in s_val:
                parts = s_val.split(",")
                if len(parts[-1]) == 3:
                    # "1,500" → miles
                    s_val = s_val.replace(",", "")
                else:
                    # "12,50" → formato europeo
                    s_val = s_val.replace(",", ".")
            return float(s_val)
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # Corrección de decimales perdidos por OCR
    # ------------------------------------------------------------------

    def _fix_missing_decimal(self, value: float, reference: float) -> float:
        """
        Corrige valores donde el OCR perdió el punto decimal.
        Ejemplo: 1.96 leído como 196 cuando el total es 15.00.
        Intenta insertar el decimal en distintas posiciones y elige
        el candidato más grande que siga siendo <= reference.
        """
        if reference <= 0 or value <= reference:
            return value  # Parece correcto, no tocar

        s = str(int(value))
        candidates = []
        for i in range(1, len(s)):
            try:
                candidate = float(s[:i] + "." + s[i:])
                if candidate <= reference:
                    candidates.append(candidate)
            except Exception:
                continue

        return max(candidates) if candidates else value

    # ------------------------------------------------------------------
    # Búsqueda de valores numéricos mediante Regex
    # ------------------------------------------------------------------

    def _find_values_via_regex(self, text: str) -> Dict:
        candidates = {
            "total": 0.0,
            "subtotal": 0.0,
            "iva": 0.0,
            "url": None,
            "nit": None,
            "serie": None,
            "dte": None
        }

        _num = r"\$?\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{1,2})?"

        total_pattern = re.compile(
            rf"(?i)(VALOR\s+TOTAL|TOTAL\s+A\s+PAGAR|IMPORTE\s+TOTAL|TOTAL:?)"
            rf"(?:[^a-zA-Z0-9\n]{{0,25}})\s*({_num})"
        )
        subtotal_pattern = re.compile(
            rf"(?i)(BASE\s+IMPONIBLE(?:\s+IVA)?|SUBTOTAL:?|SUB\s+TOTAL)"
            rf"(?:[^a-zA-Z0-9\n]{{0,25}})\s*({_num})"
        )
        iva_pattern = re.compile(
            rf"(?i)(?:^|\s)(IVA|TAX|IMPUESTO|GRAV)(?:[^a-zA-Z0-9\n]{{0,25}})\s*({_num})"
        )

        found_totals: List[float] = []
        found_subtotals: List[float] = []
        found_iva: List[float] = []

        def _safe_add(matches, bucket: List[float]):
            for _, mount_str in matches:
                val = self._parse_money(mount_str)
                if val <= 0:
                    continue
                if val > 50_000 and "," not in mount_str and "." not in mount_str:
                    continue
                bucket.append(val)

        _safe_add(total_pattern.findall(text), found_totals)
        _safe_add(subtotal_pattern.findall(text), found_subtotals)
        _safe_add(iva_pattern.findall(text), found_iva)

        if found_totals:
            candidates["total"] = max(found_totals)

        if found_subtotals:
            candidates["subtotal"] = max(found_subtotals)
            if candidates["total"] > 0 and candidates["subtotal"] > candidates["total"] * 1.05:
                valid_subs = [x for x in found_subtotals if x <= candidates["total"]]
                candidates["subtotal"] = max(valid_subs) if valid_subs else 0.0

        if found_iva:
            candidates["iva"] = max(found_iva)

        # Fallback total: si sigue en 0, tomar el mayor monto decimal del texto
        if candidates["total"] == 0.0:
            # Excluir números entre paréntesis (suelen ser precios unitarios)
            text_clean = re.sub(r'\([^)]*\)', '', text)
            all_amounts = re.findall(r'\b(\d{1,4}[.,]\d{2})\b', text_clean)
            parsed_amounts = [
                self._parse_money(a) for a in all_amounts
                if 0 < self._parse_money(a) <= 10_000
            ]
            if parsed_amounts:
                candidates["total"] = max(parsed_amounts)

        # URL
        text_nospaces = text.replace(" ", "")
        url_match = re.search(
            r"(https?://[\w\.-]+(?:/[\w\.-]*)*)", text_nospaces, re.IGNORECASE
        )
        if url_match:
            candidates["url"] = url_match.group(1)

        # ── NIT ──────────────────────────────────────────────────────────────
        # Tomar el NIT del encabezado (antes de la sección FACTURA/DATOS DEL)
        pre_factura = re.split(r'(?i)(FACTURA|DATOS DEL)', text)[0]
        nit_match = re.search(
            r'(?i)NIT\s*[:\.-]?\s*([0-9]{1,10}-?[0-9Kk])',
            pre_factura
        )
        if nit_match:
            candidates["nit"] = nit_match.group(1).strip()
        else:
            # Fallback: OCR puede confundir N con H o K
            nit_fallback = re.search(
                r'(?i)[HKN][I1][T]\s*[:\.-]?\s*([0-9]{1,10}-?[0-9Kk])',
                text
            )
            if nit_fallback:
                candidates["nit"] = nit_fallback.group(1).strip()

        # ── SERIE ─────────────────────────────────────────────────────────────
        serie_match = re.search(
            r'(?i)serie\s*[:\.-]?\s*([A-Za-z0-9]{6,20})',
            text
        )
        if serie_match:
            candidates["serie"] = serie_match.group(1).upper().strip()

        # ── DTE ───────────────────────────────────────────────────────────────
        # Normalizar: el OCR puede partir "DTE:\n:49498949" en dos líneas
        text_normalized = re.sub(
            r'(?i)(D[T1I][E3]\s*[:\.-]*)\s*\n\s*[:\.-]*\s*([0-9]{6,12})',
            r'\1\2',
            text
        )
        dte_match = re.search(
            r'(?i)(?:numero\s+de\s+)?D[T1I][E3]\s*[:\.-]*\s*([0-9]{6,12})',
            text_normalized
        )
        if dte_match:
            candidates["dte"] = dte_match.group(1).strip()
        else:
            dte_context = re.search(
                r'(?i)(?:autorizaci[oó]n|folio|numero)\s*[:\.-]*\s*([0-9]{7,12})',
                text_normalized
            )
            if dte_context:
                candidates["dte"] = dte_context.group(1).strip()

        return candidates

    # ------------------------------------------------------------------
    # Extracción inteligente del nombre del establecimiento
    # ------------------------------------------------------------------

    def _extract_establecimiento(self, text: str, llm_data: dict) -> str:
        """
        Prioridades:
        1. LLM (si devolvió algo razonable)
        2. Búsqueda de 'RAZÓN SOCIAL:' en el texto OCR
        3. Primera línea corta que no parezca metadato
        """
        llm_val = str(llm_data.get("establecimiento", ""))
        if llm_val and llm_val.lower() != "null" and len(llm_val) <= 50:
            return llm_val

        # Buscar etiqueta explícita de razón social (común en México/LATAM)
        match = re.search(
            r"(?i)RAZ[ÓO]N\s+SOCIAL\s*[:\-]?\s*"
            r"([A-ZÁÉÍÓÚÑ][A-Za-záéíóúñ\s\.\,&]{2,40})",
            text,
        )
        if match:
            return match.group(1).strip()

        # Fallback: primera línea razonable que no parezca metadato
        skip_patterns = re.compile(
            r"(?i)(r\.u\.c|factura|n[uú]m|referencia|fecha|\d{6,}|www\.|http)"
        )
        lines = [l.strip() for l in text.split("\n") if 3 < len(l.strip()) <= 50]
        for line in lines:
            if not skip_patterns.search(line):
                return line

        return lines[0] if lines else ""

    # ------------------------------------------------------------------
    # Punto de entrada principal
    # ------------------------------------------------------------------

    def extract_ticket_fields(self, ocr_text: str) -> Dict:
        if not ocr_text:
            return {"success": False, "error": "Texto vacío"}

        clean_text = self._preprocess_text(ocr_text)

        # 1. Llamada al LLM (falla de forma silenciosa, no rompe el flujo)
        llm_data: Dict = {}
        llm_ok = False
        try:
            llm_response = self._call_llm(clean_text)
            cleaned = self._clean_json_response(llm_response)
            if cleaned:
                llm_data = json.loads(cleaned)
                llm_ok = True
        except Exception as e:
            print(f"[WARN] LLM falló o devolvió JSON inválido: {e}")

        # 2. Regex como fuente numérica confiable
        regex_data = self._find_values_via_regex(clean_text)

        # 3. Fusión defensiva: Regex tiene prioridad en números
        data: Dict = {}

        data["total"] = (
            regex_data["total"]
            if regex_data["total"] > 0
            else self._parse_money(llm_data.get("total", 0))
        )
        data["subtotal"] = (
            regex_data["subtotal"]
            if regex_data["subtotal"] > 0
            else self._parse_money(llm_data.get("subtotal", 0))
        )
        data["iva"] = (
            regex_data["iva"]
            if regex_data["iva"] > 0
            else self._parse_money(llm_data.get("iva", 0))
        )
        data["propina"] = self._parse_money(llm_data.get("propina", 0))

        # 4. Corrección de decimales perdidos por OCR
        #    Ej: IVA "196" cuando el total es 15.00 → corrige a 1.96
        if data["total"] > 0:
            data["iva"]      = self._fix_missing_decimal(data["iva"],      data["total"])
            data["subtotal"] = self._fix_missing_decimal(data["subtotal"], data["total"])
            data["propina"]  = self._fix_missing_decimal(data["propina"],  data["total"])

        # 5. URL: Regex tiene prioridad
        data["factura_url"] = regex_data.get("url") or llm_data.get("factura_url")

        # 6. Establecimiento: lógica multicapa
        data["establecimiento"] = self._extract_establecimiento(clean_text, llm_data)

        # 7. Fecha: LLM primero, regex como fallback
        fecha = str(llm_data.get("fecha", ""))
        if not fecha or fecha.lower() == "null" or len(fecha) > 20:
            date_match = re.search(
                r"(\d{1,2}[/-](?:[a-zA-Z]{3}|\d{1,2})[/-]\d{2,4})", clean_text
            )
            fecha = date_match.group(1) if date_match else ""
        data["fecha"] = fecha

        # 8. Campos específicos de Guatemala (FEL)
        data["nit"] = regex_data.get("nit") or llm_data.get("nit")
        data["serie"] = regex_data.get("serie") or llm_data.get("serie")
        data["dte"] = regex_data.get("dte") or llm_data.get("dte")

        # Metadatos de respuesta
        data["success"]   = True
        data["llm_ok"]    = llm_ok   # útil para debug: indica si el LLM respondió bien
        data["llm_model"] = self.model

        return data


# ------------------------------------------------------------------
# Función helper (API pública del módulo)
# ------------------------------------------------------------------

def extract_with_ollama(ocr_text: str, model: str = "qwen2.5:7b") -> Dict:
    extractor = LLMTicketExtractor(model=model)
    return extractor.extract_ticket_fields(ocr_text)