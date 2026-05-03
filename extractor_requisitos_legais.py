import os
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser

# 1. Esquema de Extracción Legal
class AnalisisContrato(BaseModel):
    tipo_contrato: str = Field(
        ..., 
        description="Categoría jurídica del documento.",
        enum=["Arrendamiento", "Prestación de Servicios", "Confidencialidad (NDA)", "Laboral", "Mercantil"]
    )
    fecha_vencimiento: str = Field(
        ..., 
        description="Fecha de finalización o prórroga del contrato. Si no hay, indicar 'Indefinido'."
    )
    clausula_penalizacion: bool = Field(
        ..., 
        description="¿Existe alguna sanción económica o penalización por incumplimiento?"
    )
    monto_penalidad: str = Field(
        ..., 
        description="Detalle de la multa o método de cálculo si existe penalización. Si no, 'N/A'."
    )
    resumen_juridico: str = Field(
        ..., 
        description="Resumen técnico-legal del objeto del contrato (máximo 12 palabras)."
    )

# 2. Configuración de IA
llm = ChatOllama(model="mistral", temperature=0)
parser = JsonOutputParser(pydantic_object=AnalisisContrato)

# 3. Prompt de Analista Legal (Paralegal AI)
tagging_prompt = ChatPromptTemplate.from_template(
    """
    Eres un asistente legal especializado en auditoría de contratos (Due Diligence). 
    Analiza el fragmento de contrato proporcionado y extrae los requisitos solicitados con absoluta precisión.
    Si una fecha o monto no es explícito pero se puede inferir, indícalo claramente.
    
    {format_instructions}
    
    Fragmento del Contrato:
    {input}
    
    Respuesta (JSON):
    """
)

# 4. Matriz de Contratos (Ejemplos variados)
contratos_data = [
    "Este contrato de prestación de servicios de consultoría informática tendrá una duración de doce meses, comenzando el 1 de junio de 2024 y finalizando el 31 de mayo de 2025. En caso de desistimiento unilateral por parte del Consultor antes de la fecha de término, este deberá abonar a la Empresa una indemnización fija de 5.000 euros por daños y perjuicios.",
    "El presente acuerdo de confidencialidad (NDA) vincula a ambas partes desde la firma del mismo de forma indefinida, mientras dure la relación comercial. No se establecen penalizaciones económicas directas, pero el incumplimiento dará derecho a reclamar por la vía civil los daños que se demuestren en sede judicial.",
    "Contrato de arrendamiento de local comercial situado en Calle Mayor 5. El plazo de vigencia se establece en 5 años, con vencimiento el 15 de marzo de 2029. Si el arrendatario abandona el local antes del segundo año, perderá la fianza depositada equivalente a dos mensualidades de 1.200 euros cada una.",
    "Acuerdo mercantil de distribución de productos de software. Las partes acuerdan que el contrato se renovará anualmente a menos que una parte notifique lo contrario. Si el distribuidor vende productos de la competencia, se aplicará una penalización del 20% sobre las ventas brutas del último trimestre."
]

contratos_data = [
    # 1. Contrato de Arrendamiento denso
    "CONTRATO DE ARRENDAMIENTO DE BIEN INMUEBLE. " + 
    "En la ciudad de Madrid, a 4 de mayo de 2026. " +
    "Reunidos de una parte el Arrendador y de otra el Arrendatario. " +
    "Las partes se reconocen capacidad legal suficiente. " +
    "EXPOSICIÓN DE MOTIVOS: I. Que el arrendador es propietario de la finca. " +
    "II. Que el arrendatario desea alquilar el espacio para uso comercial. " +
    "CLÁUSULAS: 1. OBJETO. El objeto es el local situado en Calle Génova. " +
    "2. DURACIÓN. El presente contrato tendrá una vigencia obligatoria de 5 años. " +
    "Se establece como FECHA DE VENCIMIENTO FINAL el 4 de mayo de 2031. " * 20 + # Simulando extensión
    "15. PENALIZACIONES. En caso de impago de una sola mensualidad, el Arrendatario deberá " +
    "abonar un recargo del 10% sobre la renta mensual de 2.500 euros. " +
    "Asimismo, si se abandona el local antes del tercer año, la penalización será de 3 mensualidades completas. " +
    "El presente documento consta de 50 páginas anexas sobre el estado de la finca...",

    # 2. NDA de Confidencialidad (Propiedad Intelectual)
    "ACUERDO DE NO DIVULGACIÓN Y CONFIDENCIALIDAD (NDA). " +
    "Este acuerdo se celebra entre TechCorp y el Consultor Externo. " +
    "CONSIDERANDOS: Ambas partes desean intercambiar información sensible. " +
    "DEFINICIONES: Se entiende por información confidencial todo código fuente y algoritmos. " +
    "OBLIGACIONES: El receptor no podrá revelar los datos a terceros. " * 30 + 
    "VIGENCIA: El compromiso de confidencialidad será de carácter INDEFINIDO. " +
    "Incluso tras la finalización de los servicios prestados. " +
    "SANCIONES: La violación de este acuerdo conlleva una sanción automática de 50.000 euros " +
    "independientemente de las acciones legales adicionales por daños y perjuicios. " +
    "No se admitirán excepciones salvo requerimiento judicial expreso...",

    # 3. Contrato de Prestación de Servicios (Software)
    "CONTRATO MERCANTIL DE DESARROLLO DE SOFTWARE. " +
    "Las partes acuerdan el desarrollo de una plataforma de IA. " +
    "METODOLOGÍA: Se trabajará bajo metodología Agile-Scrum. " +
    "HITOS DE ENTREGA: Fase 1 en junio, Fase 2 en agosto. " * 25 +
    "PLAZO: El proyecto finalizará con la entrega del código final el 30 de noviembre de 2027. " +
    "FECHA DE VENCIMIENTO: 30-11-2027. " +
    "PENALIDAD POR RETRASO: Cada día de retraso imputable al desarrollador " +
    "supondrá un descuento de 200 euros sobre el pago final del hito. " +
    "Si el retraso supera los 30 días, la empresa podrá rescindir el contrato sin pago alguno...",

    # 4. Contrato Laboral de Alta Dirección
    "CONTRATO LABORAL ESPECIAL DE ALTA DIRECCIÓN. " +
    "EMPLEADOR: Global Logistics SL. EMPLEADO: Director de Operaciones. " +
    "FUNCIONES: Gestión de la red logística europea y supervisión de 500 empleados. " +
    "SALARIO: Se fija una retribución anual de 120.000 euros brutos. " * 15 +
    "Pacto de no competencia post-contractual de 2 años de duración. " +
    "VENCIMIENTO: El contrato es de carácter INDEFINIDO, sujeto a revisión de objetivos anual. " +
    "CLÁUSULA DE RESCISIÓN: Si el directivo incumple el preaviso de 6 meses, " +
    "deberá pagar una penalización equivalente a medio año de sueldo bruto...",

    # 5. Contrato de Suministro Industrial
    "CONTRATO DE SUMINISTRO DE GAS NATURAL. " +
    "Proveedor energético y Planta Industrial de Aceros. " +
    "VOLUMEN: El proveedor garantiza un flujo constante de 50.000 m3 mensuales. " +
    "PRECIO: Indexado al mercado diario con un margen fijo del 2%. " * 40 +
    "DURACIÓN: El contrato expira el 01 de enero de 2030. " +
    "PENALIZACIONES POR EXCESO: Si la planta supera el consumo contratado, " +
    "pagará un sobrecoste del 50% sobre el precio de mercado por cada m3 excedido. " +
    "Fuerza mayor incluye guerras y desastres naturales documentados...",

    # 6. Contrato de Mantenimiento de Hardware
    "CONTRATO DE MANTENIMIENTO PREVENTIVO Y CORRECTIVO. " +
    "Objeto: Servidores de datos situados en CPD principal. " +
    "SLA: El tiempo de respuesta debe ser inferior a 4 horas. " +
    "COBERTURA: Incluye piezas, mano de obra y desplazamientos. " * 20 +
    "FECHA DE VENCIMIENTO: 15 de agosto de 2028. " +
    "INCUMPLIMIENTO DE SLA: Si el técnico llega tarde, se aplicará una penalización " +
    "de 500 euros por cada hora de demora respecto al SLA acordado de 4h...",

    # 7. Contrato de Franquicia (Retail)
    "CONTRATO DE FRANQUICIA PARA CADENA DE RESTAURACIÓN. " +
    "El franquiciador cede el uso de la marca y el Know-How. " +
    "CÁNONES: Entrada de 30.000 euros y royalties mensuales del 5%. " +
    "ESTÁNDARES DE CALIDAD: El local debe seguir el manual de identidad corporativa. " * 30 +
    "VIGENCIA: El acuerdo tiene una duración de 10 años, venciendo el 20 de octubre de 2036. " +
    "PENALIZACIÓN POR COMPETENCIA: Si el franquiciado abre un negocio similar " +
    "en un radio de 5km, pagará 100.000 euros al franquiciador de forma inmediata...",

    # 8. Acuerdo de Nivel de Servicio (Cloud)
    "SERVICE LEVEL AGREEMENT (SLA) - SERVICIOS NUBE. " +
    "DISPONIBILIDAD: Se garantiza un Uptime del 99.99% anual. " +
    "EXCLUSIONES: Ventanas de mantenimiento programadas de madrugada. " +
    "SOPORTE: 24/7 vía ticket y teléfono para incidencias críticas. " * 25 +
    "VENCIMIENTO: Contrato renovable automáticamente cada 12 meses. Fecha final: 12-12-2029. " +
    "CRÉDITOS DE SERVICIO: Por cada 0.1% de caída por debajo del 99.99%, " +
    "se devolverá el 10% de la cuota mensual como compensación o penalización...",

    # 9. Contrato de Compraventa de Activos
    "CONTRATO DE COMPRAVENTA DE MAQUINARIA INDUSTRIAL. " +
    "Vendedor y Comprador acuerdan la transferencia de 5 prensas hidráulicas. " +
    "PRECIO TOTAL: 1.500.000 euros pagaderos en tres plazos. " +
    "GARANTÍA: El vendedor garantiza el funcionamiento por 24 meses. " * 15 +
    "FECHA DE ENTREGA Y VENCIMIENTO DE PAGOS: 05 de mayo de 2027. " +
    "CLÁUSULA PENAL: El retraso en el pago de cualquiera de los plazos " +
    "devengará un interés de demora del 15% anual y una multa fija de 10.000 euros...",

    # 10. Contrato de Colaboración de Influencer
    "CONTRATO DE CAMPAÑA DE MARKETING Y COLABORACIÓN. " +
    "Marca de Moda y Creador de Contenido. " +
    "CONTENIDO: 3 posts en Instagram y 5 Stories mencionando la marca. " +
    "DERECHOS DE IMAGEN: La marca puede usar las fotos por 6 meses. " * 20 +
    "VENCIMIENTO: La campaña finaliza el 31 de diciembre de 2026. " +
    "PENALIZACIÓN POR MALA IMAGEN: Si el influencer se ve envuelto en polémicas, " +
    "deberá devolver el 100% del pago y pagar una multa por daños reputacionales de 20.000€."
]

# 5. Pipeline de Ejecución
prompt_final = tagging_prompt.partial(format_instructions=parser.get_format_instructions())
tagging_chain = prompt_final | llm | parser

# 6. Procesamiento y Salida
resultados_legales = []

print(f"--- Iniciando Auditoría Legal de {len(contratos_data)} documentos ---\n")

for i, contrato in enumerate(contratos_data, 1):
    try:
        resultado = tagging_chain.invoke({"input": contrato})
        resultados_legales.append(resultado)
        
        print(f"📄 DOCUMENTO #{i}:")
        print(f"   - Tipo: {resultado.get('tipo_contrato')}")
        print(f"   - Vencimiento: {resultado.get('fecha_vencimiento')}")
        print(f"   - ¿Penalización?: {'SÍ' if resultado.get('clausula_penalizacion') else 'NO'}")
        print(f"   - Detalle Multa: {resultado.get('monto_penalidad')}")
        print(f"   - Objeto: {resultado.get('resumen_juridico')}")
        print("-" * 50)
        
    except Exception as e:
        print(f"❌ Error analizando documento #{i}: {e}")