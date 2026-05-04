import os
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser

# 1. Esquema de Triaje Médico
class TriajeMedico(BaseModel):
    urgencia: str = Field(
        ..., 
        description="Nivel de prioridad médica basado en la gravedad.",
        enum=["Urgente", "Consulta", "Seguimiento"]
    )
    especialidad: str = Field(
        ..., 
        description="Especialidad médica recomendada para el caso.",
        enum=["Cardiología", "Pediatría", "Dermatología", "Neurología", "Medicina General", "Traumatología"]
    )
    sintomas_clave: list = Field(
        ..., 
        description="Lista de síntomas principales detectados en el relato."
    )
    resumen_paciente: str = Field(
        ..., 
        description="Resumen clínico breve de máximo 12 palabras."
    )

# 2. Configuración de IA
llm = ChatOllama(model="mistral", temperature=0)
parser = JsonOutputParser(pydantic_object=TriajeMedico)

# 3. Prompt de Asistente de Telemedicina
tagging_prompt = ChatPromptTemplate.from_template(
    """
    Eres un asistente virtual de triaje médico de alta precisión. 
    Tu objetivo es analizar la descripción del paciente para categorizar la urgencia y derivar al especialista correcto.
    
    REGLA CRÍTICA: Si el paciente menciona dolor de pecho, dificultad para respirar o pérdida de conciencia, marca SIEMPRE como 'Urgente'.

    {format_instructions}
    
    Descripción del Paciente:
    {input}
    
    Respuesta (JSON):
    """
)

# 4. Matriz de Casos de Telemedicina
pacientes_data = [
    "Siento una presión muy fuerte en el pecho que se extiende al brazo izquierdo y me cuesta respirar desde hace 10 minutos.",
    "Mi hijo de 3 años tiene una erupción cutánea roja en las mejillas desde esta mañana, no tiene fiebre y está animado.",
    "Tengo un dolor de cabeza punzante en el lado derecho que me provoca náuseas y sensibilidad a la luz; me suele pasar una vez al mes.",
    "Hola, quería revisar los resultados de mi analítica de colesterol que me hice la semana pasada para ver si la medicación funciona.",
    "Me doblé el tobillo jugando al fútbol ayer. Está muy hinchado y no puedo apoyar el pie en el suelo del dolor.",
    "Desde hace dos horas mi padre balbucea al hablar y tiene un lado de la cara como caído, no sabemos qué le pasa."
]
pacientes_data = [
    # 1. Caso Urgente: Infarto/Cardiología
    "Llevo unos 20 minutos sintiendo un dolor opresivo muy fuerte justo en el centro del pecho, como si tuviera un peso encima que no me deja expandir los pulmones. "
    "El dolor ha empezado a irradiarse hacia la mandíbula y noto un hormigueo extraño en el brazo izquierdo. "
    "Estoy sudando frío, me siento muy mareado y me cuesta horrores mantener una respiración normal. "
    "Tengo antecedentes de hipertensión en mi familia y estoy bastante asustado por la intensidad de la presión.",

    # 2. Caso Consulta: Dermatología/Pediatría
    "Mi hija de apenas 4 años se despertó hoy con unas manchitas rojas circulares repartidas por todo el tronco y la espalda. "
    "Dice que le pican un poco, pero no tiene fiebre, está comiendo perfectamente y su nivel de energía es el de siempre. "
    "Ayer estuvimos en el campo y no sé si podrá ser una reacción alérgica a alguna planta o alguna picadura de insecto. "
    "Quisiera que un especialista le echara un vistazo para descartar algo contagioso antes de mandarla al colegio mañana.",

    # 3. Caso Urgente: Ictus/Neurología
    "Estoy escribiendo esto porque mi marido, de repente, ha dejado de poder mover el brazo derecho mientras estábamos desayunando. "
    "Cuando intenta hablar me doy cuenta de que balbucea y no se le entiende nada de lo que dice, parece que tiene la boca torcida. "
    "Intento que sonría y solo se le eleva un lado de la cara, el otro permanece caído y sin expresión. "
    "No se ha caído ni se ha dado ningún golpe, ha sido algo totalmente súbito y no sabemos cómo reaccionar.",

    # 4. Caso Seguimiento: Medicina General/Endocrinología
    "Hola, soy paciente crónico y me gustaría programar una cita para revisar los últimos resultados de mi analítica de control. "
    "He estado siguiendo la dieta y tomando la medicación para la glucosa tal como acordamos en la última visita de hace tres meses. "
    "Me noto bien de energía, aunque he perdido un par de kilos y quería saber si eso es normal dentro del tratamiento actual. "
    "No tengo ninguna molestia aguda, simplemente es para ajustar la dosis de la receta si fuera necesario.",

    # 5. Caso Consulta: Traumatología
    "Ayer tarde, mientras jugaba un partido de pádel, hice un giro brusco con la rodilla derecha y escuché una especie de chasquido seco. "
    "Al principio pude seguir caminando, pero hoy me he levantado con la rodilla muy inflamada y me duele mucho al intentar subir escaleras. "
    "He estado aplicando hielo y manteniendo la pierna en alto, pero noto una inestabilidad extraña al apoyar el peso. "
    "Necesito que un traumatólogo valore si puede haber alguna afectación en los ligamentos o en el menisco.",

    # 6. Caso Urgente: Dificultad respiratoria/Neumología
    "Mi abuela tiene EPOC y desde hace un par de horas dice que no le entra el aire, está usando el inhalador de rescate pero no mejora. "
    "Tiene los labios con un tono azulado y está haciendo un ruido sibilante muy fuerte al intentar inhalar. "
    "Está muy inquieta, no puede terminar las frases porque se queda sin aliento y tiene las pulsaciones muy aceleradas. "
    "Normalmente suele estabilizarse rápido, pero hoy la saturación de oxígeno que marca el pulsioxímetro está bajando de 88.",

    # 7. Caso Consulta: Neurología (Migraña)
    "Tengo una migraña muy intensa que empezó anoche y no remite con el paracetamol habitual de 1 gramo. "
    "Siento como latidos dentro de la cabeza, sobre todo detrás del ojo izquierdo, y me molesta muchísimo cualquier tipo de luz o ruido. "
    "He tenido episodios parecidos en el pasado, pero este parece más duradero de lo normal y me dan ganas de vomitar constantemente. "
    "Quería consultar si existe algún tratamiento preventivo más fuerte porque estos ataques me están invalidando para trabajar.",

    # 8. Caso Seguimiento: Dermatología
    "Quisiera una cita de seguimiento para comprobar cómo está evolucionando la mancha que me quitaron de la espalda hace dos semanas. "
    "La cicatriz parece estar cerrando bien, no hay signos de infección ni pus, pero noto la zona un poco tirante y algo enrojecida. "
    "También quería aprovechar para que el doctor revise un lunar nuevo que me ha salido en el brazo y que tiene bordes irregulares. "
    "Tengo los resultados de la biopsia previa y me gustaría que me los explicaran con calma.",

    # 9. Caso Consulta: Pediatría/Infecciosas
    "Mi hijo de 6 años lleva tres días con fiebre alta, llegando casi a los 39 grados, que solo baja un poco con el ibuprofeno. "
    "Se queja mucho de que le duele la garganta al tragar y he visto que tiene como unas placas blancas al fondo de la boca. "
    "No tiene tos ni moqueo, pero está muy decaído, no quiere comer nada sólido y se queja de dolor de tripa. "
    "Sospecho que puede ser una amigdalitis bacteriana y me gustaría saber si necesita empezar con antibióticos.",

    # 10. Caso Urgente: Traumatología (Fractura abierta/Grave)
    "Acabo de sufrir una caída de unos dos metros desde una escalera mientras trabajaba y creo que me he roto el brazo. "
    "Hay una deformidad evidente cerca del codo, me duele muchísimo y creo que se ve un poco de hueso a través de la herida. "
    "Estoy perdiendo algo de sangre y empiezo a sentir un hormigueo frío en los dedos de la mano, apenas puedo moverlos. "
    "Me siento muy flojo, como si me fuera a desmayar, y necesito asistencia médica lo más rápido posible."
]

# 5. Pipeline
prompt_final = tagging_prompt.partial(format_instructions=parser.get_format_instructions())
tagging_chain = prompt_final | llm | parser

# 6. Ejecución del Triaje
resultados_medicos = []

print("--- Procesando Triaje de Telemedicina ---\n")

for i, caso in enumerate(pacientes_data, 1):
    try:
        resultado = tagging_chain.invoke({"input": caso})
        resultados_medicos.append(resultado)
        
        print(f"🏥 PACIENTE #{i}:")
        print(f"   - Prioridad: {resultado.get('urgencia')}")
        print(f"   - Derivación: {resultado.get('especialidad')}")
        print(f"   - Síntomas:  {', '.join(resultado.get('sintomas_clave', []))}")
        print(f"   - Resumen:   {resultado.get('resumen_paciente')}")
        print("-" * 50)
        
    except Exception as e:
        print(f"❌ Error en el procesamiento del paciente #{i}: {e}")