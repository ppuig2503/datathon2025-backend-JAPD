from fastapi import APIRouter, HTTPException
from dotenv import load_dotenv
from openai import OpenAI
import os
from app.types.openAiTypes import (
    AnswerInput, 
    AnswerResponse, 
    LocalExplanationInput, 
    LocalExplanationResponse,
    GlobalExplanationInput,
    GlobalExplanationResponse,
    PDPResponse
)
from data import data

router = APIRouter(
    prefix="/ai",
    tags=["OpenAI"]
)

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


@router.post("/answer", response_model=AnswerResponse)
def answer(input_data: AnswerInput):
    """
    Answer endpoint using OpenAI API.
    
    Args:
        input_data: AnswerInput object with the question
        
    Returns:
        AnswerResponse with the answer from OpenAI
    """
    try:
        # Get explainability data from data.py
        local_data = data.get_local_data()
        global_data = data.get_global_data()
        
        # Build context from available explainability data
        context_parts = []
        
        # Add local explanation context if available
        if local_data["prediction"] is not None and local_data["explanation"] is not None:
            prediction_label = "GANADA" if local_data["prediction"] == 1 else "PERDIDA"
            probability_percent = round(local_data["probability"] * 100, 2)
            
            context_parts.append(f"PREDICCIÓN LOCAL:\n- Resultado: {prediction_label}\n- Probabilidad de ganar: {probability_percent}%")
            
            # Top 5 local features
            top_features = list(local_data["explanation"].items())[:5]
            features_text = "\n".join([f"  · {feature}: {contribution:.4f}" for feature, contribution in top_features])
            context_parts.append(f"- Top features que influyeron en esta predicción:\n{features_text}")
        
        # Add global explanation context if available
        if global_data["shap_global"] is not None:
            top_global = list(global_data["shap_global"].items())[:5]
            global_text = "\n".join([f"  · {feature}: {value:.4f}" for feature, value in top_global])
            context_parts.append(f"\nIMPORTANCIAS GLOBALES DEL MODELO:\n- Top features más importantes en general:\n{global_text}")
        
        # Add PDP context if available
        if global_data["pdp_data"] is not None:
            pdp_data = global_data["pdp_data"]
            feature_name = pdp_data.get('feature_type', 'N/A')
            context_parts.append(f"\nPARTIAL DEPENDENCE:\n- Feature analizada: {feature_name}")
            
            if pdp_data.get('grids') and pdp_data.get('pdp_values'):
                grids = pdp_data['grids']
                values = pdp_data['pdp_values']
                min_val, max_val = min(values), max(values)
                context_parts.append(f"- Rango de valores de la feature: {min(grids):.2f} a {max(grids):.2f}")
                context_parts.append(f"- Probabilidad predicha varía de {min_val:.4f} a {max_val:.4f}")
                
                # Determine trend
                if values[-1] > values[0] + 0.05:
                    context_parts.append("- Tendencia: A mayor valor de esta feature, MAYOR probabilidad de ganar")
                elif values[-1] < values[0] - 0.05:
                    context_parts.append("- Tendencia: A mayor valor de esta feature, MENOR probabilidad de ganar")
                else:
                    context_parts.append("- Tendencia: Esta feature tiene impacto moderado/constante en la probabilidad")
        
        # Combine all context
        context = "\n\n".join(context_parts) if context_parts else "No hay datos de explicabilidad disponibles aún."
        
        # Add context to system message
        system_message = f"""Eres un asistente conversacional especializado en explicar modelos de Machine Learning de clasificación binaria a usuarios de negocio no técnicos. 

Contexto del caso de uso:
- Empresa: Schneider Electric.
- Problema: predecir si una oportunidad comercial (venta) será GANADA (1) o PERDIDA (0).
- Modelo: clasificador binario que devuelve una probabilidad de ganar la oportunidad.
- En la pantalla: el usuario estará viendo la predicción (GANADA/PERDIDA) y la probabilidad asociada, junto con explicaciones de por qué el modelo tomó esa decisión, junto con un gráfico formado por las features más importantes y su impacto.

DATOS DE EXPLICABILIDAD DISPONIBLES:
{context}

Tu objetivo:
1. Responder preguntas del usuario sobre la predicción y el modelo usando los datos de explicabilidad proporcionados arriba.
2. Traducir los resultados técnicos (probabilidades, importancias, SHAP, LIME, PDP, etc.) a un lenguaje de negocio claro y entendible.
3. Enlazar siempre las explicaciones con las features de negocio.
4. Recordar que el modelo es PROBABILÍSTICO: habla de probabilidades, no de certezas.

Cómo debes responder:
- Idioma: Por defecto español. Si el usuario habla en otro idioma, respóndele en ese idioma.
- Estilo: Claro, profesional y cercano. Sin jerga técnica innecesaria.
- Usa SIEMPRE los datos específicos proporcionados en el contexto arriba. NO inventes números.
- Si te preguntan algo que no está en los datos disponibles, dilo claramente.
- Respuestas concisas: 2 párrafos máximo por respuesta.

Ejemplos de preguntas que podrías recibir:
- "¿Por qué esta oportunidad se clasificó como ganada/perdida?"
- "¿Qué puedo hacer para mejorar la probabilidad de ganar?"
- "¿Qué variables son más importantes?"
- "¿Por qué esta feature tiene tanto peso?"
- "¿Es confiable esta predicción?"

Recuerda: Usa SOLO los datos del contexto proporcionado arriba. Sé específico, claro y útil."""
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": input_data.question}
            ]
        )
        
        answer = response.choices[0].message.content
        
        return AnswerResponse(answer=answer)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {str(e)}")


@router.post("/text/local", response_model=LocalExplanationResponse)
def generate_local_explanation_text():
    """
    Generate natural language explanation for local LIME results.
    Uses data stored in data.py from the /explain endpoint.
        
    Returns:
        LocalExplanationResponse with the explanation text from OpenAI
    """
    try:
        # Get data from data.py
        local_data = data.get_local_data()
        
        # Validate that data exists
        if local_data["prediction"] is None or local_data["explanation"] is None:
            raise HTTPException(status_code=400, detail="No local explainability data available. Please call /explain endpoint first.")
        
        # Prepare the explanation data for the prompt
        prediction_label = "GANADA" if local_data["prediction"] == 1 else "PERDIDA"
        probability_percent = round(local_data["probability"] * 100, 2)
        
        # Format the feature explanations
        features_text = "\n".join(
            [f"- {feature}: {contribution:.4f}" for feature, contribution in local_data["explanation"].items()]
        )
        
        # Create the user prompt with the explanation data
        user_prompt = f"""Genera un texto explicativo ESPECÍFICO sobre los resultados de explainability LOCAL (LIME/SHAP local) para esta oportunidad de venta concreta:

PREDICCIÓN: {prediction_label}
PROBABILIDAD DE GANAR: {probability_percent}%

CONTRIBUCIONES LOCALES DE LAS FEATURES (LIME/SHAP):
{features_text}

Este texto se mostrará en una página de resultados. Debe ser:
- ESPECÍFICO sobre ESTA oportunidad en particular
- Explicar POR QUÉ el modelo predijo este resultado para ESTE caso
- Identificar las entre dos i cuatro features que MÁS influyeron en la predicción de esta oportunidad
- Focus on abnormal
- Traducir las contribuciones numéricas a lenguaje de negocio
- Incluir qué factores empujaron hacia GANADA y cuáles hacia PERDIDA
- 3 líneas máximo
- Lenguaje claro para usuarios de negocio

NO hagas generalizaciones sobre el modelo. Céntrate solo en explicar ESTA predicción específica."""
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Eres un experto en explicar resultados de EXPLAINABILITY LOCAL (LIME y SHAP local) de modelos de ML a usuarios de negocio. Contexto: Modelo de clasificación binaria de Schneider Electric que predice si una oportunidad comercial será GANADA (1) o PERDIDA (0). Tu tarea ESPECÍFICA: Explicar POR QUÉ el modelo ha hecho una predicción concreta para UNA oportunidad específica, basándote en las contribuciones LIME/SHAP locales que se te proporcionan. IMPORTANTE: - Céntrate SOLO en esta oportunidad en particular, NO hagas generalizaciones del modelo - Identifica qué features específicas empujaron la predicción hacia GANADA vs PERDIDA - Usa los valores numéricos exactos de las contribuciones que se te dan - Traduce los nombres técnicos de features a lenguaje de negocio - Explica el impacto de cada feature importante en términos comerciales - Sé específico y concreto, evita frases genéricas - Responde en español, claro y profesional - Estructura: resumen breve + explicación detallada de top 3-5 features + implicación práctica"},
                {"role": "user", "content": user_prompt}
            ]
        )
        
        text = response.choices[0].message.content
        
        return LocalExplanationResponse(text=text)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {str(e)}")


@router.post("/text/global", response_model=GlobalExplanationResponse)
def generate_global_explanation_text():
    """
    Generate natural language explanation for global explainability results (PDP, SHAP global, feature importance).
    Uses data stored in data.py.
    
    Returns:
        GlobalExplanationResponse with the explanation text from OpenAI
    """
    try:
        # Get data from data.py
        global_data = data.get_global_data()
        
        # Validate that at least SHAP or PDP data exists
        if global_data["shap_global"] is None and global_data["pdp_data"] is None:
            raise HTTPException(status_code=400, detail="No global explainability data available. Please call /explain_shap or /explain_pdp endpoints first.")
        
        # Format SHAP global if provided
        shap_text = ""
        if global_data["shap_global"]:
            shap_text = "IMPORTANCIAS GLOBALES SHAP (contribución promedio de cada feature):\n" + "\n".join(
                [f"- {feature}: {value:.4f}" for feature, value in global_data["shap_global"].items()]
            )
            if global_data["shap_base_value"] is not None:
                shap_text += f"\n\nValor base del modelo: {global_data['shap_base_value']:.4f}"
        
        # Format PDP data if provided
        pdp_text = ""
        if global_data["pdp_data"]:
            pdp_data = global_data["pdp_data"]
            pdp_text = "\n\nPARTIAL DEPENDENCE PLOT (PDP):\n"
            pdp_text += f"Feature analizada: {pdp_data.get('feature_type', 'N/A')}\n"
            if pdp_data.get('grids') and pdp_data.get('pdp_values'):
                # Summarize PDP trend
                grids = pdp_data['grids']
                values = pdp_data['pdp_values']
                pdp_text += f"Rango de valores: {min(grids):.2f} a {max(grids):.2f}\n"
                pdp_text += f"Probabilidad predicha varía de {min(values):.4f} a {max(values):.4f}\n"
                # Determine trend
                if values[-1] > values[0]:
                    pdp_text += "Tendencia: A mayor valor de esta feature, MAYOR probabilidad de ganar\n"
                elif values[-1] < values[0]:
                    pdp_text += "Tendencia: A mayor valor de esta feature, MENOR probabilidad de ganar\n"
                else:
                    pdp_text += "Tendencia: Esta feature tiene poco impacto en la probabilidad\n"
        
        # Create the user prompt with the global explanation data
        user_prompt = f"""Genera un texto explicativo ESPECÍFICO sobre los resultados de explainability GLOBAL del modelo de predicción de oportunidades de venta:

{shap_text}{pdp_text}

Este texto se mostrará en una página de resultados. Debe ser:
- ESPECÍFICO sobre el comportamiento GENERAL del modelo
- Explicar QUÉ variables son más importantes en el modelo (según SHAP)
- Describir CÓMO cada variable importante influye en las predicciones en general
- Si hay información de PDP, explicar las tendencias observadas (ej: "a mayor X, mayor probabilidad de ganar")
- Identificar entre dos i cuatro razones principales por las que el modelo toma sus decisiones
- Focus on abnormal
- Traducir a lenguaje de negocio con ejemplos prácticos
- 3 líneas máximo
- Lenguaje claro para usuarios de negocio

NO hables de casos individuales. Céntrate en patrones GLOBALES y tendencias del modelo en general."""
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Eres un experto en explicar resultados de EXPLAINABILITY GLOBAL (Feature Importance, SHAP global, Partial Dependence Plots) de modelos de ML a usuarios de negocio. Contexto: Modelo de clasificación binaria de Schneider Electric que predice si una oportunidad comercial será GANADA (1) o PERDIDA (0). Tu tarea ESPECÍFICA: Explicar QUÉ variables son más importantes en el modelo en general y CÓMO influyen en las predicciones, basándote en métricas globales. IMPORTANTE: - Céntrate en PATRONES GLOBALES del modelo, NO en casos individuales - Identifica las features más importantes según feature importance / SHAP global - Explica el efecto general de cada feature (si aumenta o disminuye prob. de ganar) - Si hay info de PDP, describe las tendencias observadas - Traduce los nombres técnicos de features a lenguaje de negocio - Usa ejemplos prácticos orientados a ventas - Sé específico con los datos numéricos que se te proporcionan - Responde en español, claro y profesional - Estructura: resumen breve + top 5-7 features con su influencia + implicaciones para el negocio"},
                {"role": "user", "content": user_prompt}
            ]
        )
        
        text = response.choices[0].message.content
        
        return GlobalExplanationResponse(text=text)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {str(e)}")


@router.post("/pdp_sentence", response_model=PDPResponse)
def generate_pdp_sentence():
    """
    Generate a one-sentence explanation of the PDP distribution for a given feature.
    Uses PDP data stored in data.py from the /explain_pdp endpoint.
    
    Returns:
        AnswerResponse with a concise sentence explaining the PDP pattern
    """
    try:
        # Get PDP data from data.py
        global_data = data.get_global_data()
        
        # Validate that PDP data exists
        if global_data["pdp_data"] is None:
            raise HTTPException(status_code=400, detail="No PDP data available. Please call /explain_pdp endpoint first.")
        
        pdp_data = global_data["pdp_data"]
        feature_name = pdp_data.get('feature_type', 'N/A')
        grids = pdp_data.get('grids', [])
        pdp_values = pdp_data.get('pdp_values', [])
        
        if not grids or not pdp_values:
            raise HTTPException(status_code=400, detail="PDP data is incomplete.")
        
        # Create context for the AI
        min_grid, max_grid = min(grids), max(grids)
        min_prob, max_prob = min(pdp_values), max(pdp_values)
        start_prob, end_prob = pdp_values[0], pdp_values[-1]
        
        # Calculate trend
        prob_change = end_prob - start_prob
        prob_range = max_prob - min_prob
        
        user_prompt = f"""Analiza esta gráfica de Partial Dependence Plot (PDP) y resume en UNA SOLA FRASE cómo está distribuida y por qué es importante:

FEATURE ANALIZADA: {feature_name}

DATOS DE LA GRÁFICA:
- Eje X (valores de la feature): de {min_grid:.2f} a {max_grid:.2f}
- Eje Y (probabilidad de ganar): de {min_prob:.4f} a {max_prob:.4f}
- Probabilidad al inicio (valor mínimo de la feature): {start_prob:.4f}
- Probabilidad al final (valor máximo de la feature): {end_prob:.4f}
- Cambio total de probabilidad: {prob_change:.4f}
- Rango de variación: {prob_range:.4f}

INSTRUCCIONES:
- Genera UNA SOLA FRASE (máximo 25 palabras)
- Describe la tendencia principal (crece, decrece, se mantiene estable, tiene forma de U, etc.)
- Explica qué significa eso para el negocio en términos simples
- NO uses términos técnicos como "PDP", "eje X", "eje Y"
- Habla de la feature por su nombre y de la "probabilidad de ganar la oportunidad"
- Sé directo y específico

EJEMPLO DE RESPUESTA:
"A mayor [nombre_feature], mayor probabilidad de ganar, especialmente por encima de [valor_clave]."
"La probabilidad se mantiene estable independientemente del valor de [nombre_feature]."
"Valores bajos de [nombre_feature] reducen drásticamente las posibilidades de ganar."
"""
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Eres un experto en interpretar gráficas de Partial Dependence Plot (PDP) y traducir patrones técnicos a insights de negocio concisos. Debes generar UNA SOLA FRASE clara y directa que explique la distribución de los puntos en la gráfica y su implicación práctica para el negocio."},
                {"role": "user", "content": user_prompt}
            ]
        )
        
        sentence = response.choices[0].message.content
        
        return PDPResponse(text=sentence)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {str(e)}")
