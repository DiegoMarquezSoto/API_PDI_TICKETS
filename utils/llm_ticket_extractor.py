# utils/llm_ticket_extractor.py
import json
import os
import re
from typing import Dict, Optional, List
import ollama

class LLMTicketExtractor:
    """
    Extractor Final: Soporta enteros (totales cerrados) y decimales.
    Filtra códigos alfanuméricos para no confundirlos con precios.
    """
    
    def __init__(self, model: str = "qwen2.5:0.5b"):
        self.model = model
        ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        os.environ["OLLAMA_HOST"] = ollama_host
        print(f"LLMTicketExtractor (Soporte Enteros + Filtro Códigos) con modelo: {model}")
    
    def _preprocess_text(self, text: str) -> str:
        """Limpieza inicial del texto OCR"""
        # 1. Eliminar paréntesis/corchetes pegados a números
        text = re.sub(r'[\(\[\{]\s*(\d)', r'\1', text)
        
        # 2. Normalizar etiquetas comunes
        text = re.sub(r'(?i)total\s*[:\.]', 'TOTAL:', text)
        text = re.sub(r'(?i)subtotal\s*[:\.]', 'SUBTOTAL:', text)
        
        # 3. Arreglos de URL y dominio
        text = text.replace("https:l", "https://").replace("fmodelo com", "fmodelo.com")
        
        return text

    def _create_prompt(self, ocr_text: str) -> str:
        return f"""Extrae datos del ticket en JSON.
        
TEXTO:
{ocr_text}

REGLAS:
1. establecimiento: Nombre tienda (primera línea).
2. fecha: DD/MM/YYYY.
3. importes: Busca TOTAL y SUBTOTAL.
4. url: Busca enlaces web.
5. propina: Opcional, si no hay, 0.00.
6. iva: puede aparecer como "grav" "iva" "impuesto" o "tax".

Responde SOLO JSON:
{{
  "establecimiento": "string",
  "fecha": "string",
  "subtotal": 0.00,
  "iva": 0.00,
  "total": 0.00,
  "propina": 0.00,
  "factura_url": "string"
}}"""
    
    def _call_llm(self, prompt: str) -> str:
        try:
            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.1, "num_predict": 600}
            )
            return response['message']['content']
        except Exception as e:
            raise Exception(f"Error Ollama: {e}")
    
    def _clean_json_response(self, response: str) -> str:
        try:
            response = re.sub(r'```json\s*', '', response)
            response = re.sub(r'```\s*', '', response)
            start = response.find('{')
            end = response.rfind('}') + 1
            if start != -1 and end != 0:
                return response[start:end]
            return response.strip()
        except:
            return response

    def _parse_money(self, value) -> float:
        """Convierte string a float manejando formatos diversos"""
        if isinstance(value, (int, float)): return float(value)
        if not value: return 0.0
        
        s_val = str(value).strip()
        # Eliminar símbolos de moneda y espacios
        s_val = re.sub(r'[^\d.,]', '', s_val)
        
        if not s_val: return 0.0
        
        try:
            # Caso "1,500.00" o "12,045.82" (Coma miles, Punto decimal)
            if ',' in s_val and '.' in s_val:
                s_val = s_val.replace(',', '')
            # Caso "1,500" (Entero con coma de miles)
            elif ',' in s_val:
                # Si los últimos dígitos son 3 (ej: 1,000), asumimos miles
                parts = s_val.split(',')
                if len(parts[-1]) == 3:
                    s_val = s_val.replace(',', '')
                else:
                    # Caso raro 12,50 (formato europeo/sudamericano) -> 12.50
                    s_val = s_val.replace(',', '.')
            
            return float(s_val)
        except:
            return 0.0

    def _find_values_via_regex(self, text: str) -> Dict:
        """
        Búsqueda de valores numéricos que soporta ENTEROS pero evita CÓDIGOS.
        """
        candidates = {"total": 0.0, "subtotal": 0.0, "url": None}
        
        # --- LÓGICA DE PRECIO ---
        # Explicación del Regex mejorado:
        # (?i)(TOTAL|SUBTOTAL)  -> Busca la palabra clave
        # [^a-zA-Z0-9\n]{0,20}  -> Permite hasta 20 caracteres intermedios (espacios, puntos, $, 'GRAV AL')
        #                          PERO PROHÍBE LETRAS Y NÚMEROS PEGADOS. Esto evita 'SUBTOTAL H563036'.
        # (\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{1,2})?) -> Captura el número. 
        #                          Soporta: 12,045.82 | 500.00 | 500 | 1,200
        
        price_pattern = re.compile(r'(?i)(TOTAL|SUBTOTAL)(?:[^a-zA-Z0-9\n]{0,25})\s*(\$?\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{1,2})?)')
        
        matches = price_pattern.findall(text)
        
        found_totals = []
        found_subtotals = []

        for label, mount_str in matches:
            val = self._parse_money(mount_str)
            if val <= 0: continue
            
            # Filtro de seguridad: Si es un entero gigante sin formato, suele ser código de barras
            # Ej: 563036 se convierte en 563036.0. Si no tenía coma, es sospechoso si es > 50000.
            if val > 50000 and ',' not in mount_str and '.' not in mount_str:
                continue

            if "TOTAL" in label.upper() and "SUB" not in label.upper():
                found_totals.append(val)
            elif "SUBTOTAL" in label.upper():
                found_subtotals.append(val)

        # Selección inteligente
        if found_totals:
            # El total suele ser el monto mayor encontrado asociado a la etiqueta TOTAL
            candidates["total"] = max(found_totals)
        
        if found_subtotals:
            # El subtotal suele ser el mayor encontrado (para evitar desgloces pequeños)
            candidates["subtotal"] = max(found_subtotals)
            # Corrección lógica: Subtotal no puede ser mayor al Total (con margen de error)
            if candidates["total"] > 0 and candidates["subtotal"] > candidates["total"] * 1.05:
                # Si el subtotal es absurdo, probablemente agarramos un código mal filtrado
                # Intentamos buscar uno menor
                valid_subs = [x for x in found_subtotals if x <= candidates["total"]]
                if valid_subs:
                    candidates["subtotal"] = max(valid_subs)

        # --- LÓGICA DE URL ---
        text_nospaces = text.replace(" ", "")
        url_match = re.search(r'(https?://[\w\.-]+(?:/[\w\.-]*)*)', text_nospaces, re.IGNORECASE)
        if url_match:
            candidates["url"] = url_match.group(1)

        return candidates

    def extract_ticket_fields(self, ocr_text: str) -> Dict:
        if not ocr_text: return {"success": False, "error": "Texto vacío"}
        
        clean_text = self._preprocess_text(ocr_text)
        
        # 1. LLM para estructura básica
        try:
            llm_response = self._call_llm(self._create_prompt(clean_text))
            data = json.loads(self._clean_json_response(llm_response))
        except Exception:
            data = {"success": False}

        # 2. Regex Especializada (Enteros + Decimales)
        regex_data = self._find_values_via_regex(clean_text)

        # 3. Fusión de datos (Prioridad a Regex en números)
        if regex_data["total"] > 0:
            data["total"] = regex_data["total"]
        else:
            data["total"] = self._parse_money(data.get("total"))

        if regex_data["subtotal"] > 0:
            data["subtotal"] = regex_data["subtotal"]
        else:
            data["subtotal"] = self._parse_money(data.get("subtotal"))
            
        if regex_data["url"]:
            data["factura_url"] = regex_data["url"]
        
        # Recuperar establecimiento si falta
        if not data.get("establecimiento") or data.get("establecimiento") == "null":
            lines = [l.strip() for l in clean_text.split('\n') if len(l.strip()) > 3]
            if lines: data["establecimiento"] = lines[0]

        data["success"] = True
        data["llm_model"] = self.model
        data.setdefault("iva", 0.0)
        data.setdefault("propina", 0.0)
        
        return data

# Función helper (MANTENIDA)
def extract_with_ollama(ocr_text: str, model: str = "qwen2.5:0.5b") -> Dict:
    extractor = LLMTicketExtractor(model=model)
    return extractor.extract_ticket_fields(ocr_text)