import os
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser

# 1. Esquema de Triaje para Recursos Humanos
class ClasificacionCV(BaseModel):
    nivel_experiencia: str = Field(
        ..., 
        description="Nivel basado en años y responsabilidades.",
        enum=["Trainee", "Junior", "Middle", "Senior", "Lead"]
    )
    tecnologia_principal: str = Field(
        ..., 
        description="La stack o lenguaje más fuerte del candidato.",
        enum=["Python", "JavaScript", ".NET", "Java", "C++", "Go"]
    )
    disponibilidad: str = Field(
        ..., 
        description="Situación laboral actual.",
        enum=["Inmediata", "15 días", "1 mes", "No especificada"]
    )
    salario_esperado: str = Field(
        ..., 
        description="Pretensiones económicas si se mencionan, si no, 'No especificado'."
    )
    resumen_perfil: str = Field(
        ..., 
        description="Resumen profesional de máximo 10 palabras."
    )

# 2. Configuración de IA
llm = ChatOllama(model="mistral", temperature=0)
parser = JsonOutputParser(pydantic_object=ClasificacionCV)

# 3. Prompt de Reclutador Técnico
tagging_prompt = ChatPromptTemplate.from_template(
    """
    Eres un experto Technical Recruiter. Tu tarea es analizar el resumen del currículum proporcionado
    y clasificar al candidato con precisión quirúrgica para el sistema de triaje de la empresa.
    
    {format_instructions}
    
    Resumen del CV:
    {input}
    
    Respuesta (JSON):
    """
)

# 4. Matriz de Candidatos (Strings extensos)
candidatos_cv = [
    "Desarrollador con 8 años de trayectoria liderando equipos en entornos distribuidos. Experto en arquitectura de microservicios con Python y Django. He gestionado presupuestos de infraestructura cloud en AWS. Actualmente busco nuevos retos por finalización de proyecto. Mi incorporación podría ser de carácter inmediato. Mi último salario base estaba en el rango de los 65k anuales. Me apasiona el código limpio y la mentoría de perfiles más jóvenes. He trabajado con bases de datos SQL y NoSQL. Poseo certificación oficial de soluciones arquitectónicas. Mi enfoque siempre está en la escalabilidad y la seguridad del dato.",
    "Graduado reciente en Ingeniería Informática con muchísimas ganas de aprender. Durante la carrera hice prácticas de 6 meses desarrollando aplicaciones web sencillas con Java y Spring Boot. Conozco los fundamentos de Git y metodologías ágiles. Estoy disponible para empezar mañana mismo si fuera necesario. No tengo una pretensión salarial cerrada, busco mi primera oportunidad real. He realizado varios proyectos personales subidos a mi GitHub. Me considero una persona proactiva y con capacidad de trabajo en equipo. Busco un entorno donde pueda crecer técnicamente bajo supervisión senior.",
    "Especialista en desarrollo backend con .NET Core y C# durante los últimos 4 años. He trabajado principalmente en el sector bancario optimizando procesos de transacciones. Tengo un nivel de inglés C1 certificado. Actualmente trabajo pero escucho ofertas, requeriría un preaviso de 1 mes. Aspiro a un salario mínimo de 45.000 euros. Poseo conocimientos avanzados en Azure y contenedores Docker. Me interesa el desarrollo basado en pruebas (TDD). Soy una persona metódica y enfocada a resultados. He participado en la migración de sistemas legacy a arquitecturas modernas.",
    "Fullstack senior especializado en ecosistema JavaScript, principalmente Node.js y React. 10 años de experiencia total. Disponibilidad de 15 días.",
    "Ingeniera de software con 2 años de experiencia usando Python para análisis de datos y backend. Disponibilidad inmediata. 30k.",
    "Arquitecto Java con más de 12 años de experiencia. Experto en Spring Cloud y Kubernetes. Disponibilidad 1 mes. 75k.",
    "Junior .NET developer, 1 año de experiencia en consultoría. Busco cambio por crecimiento. Disponibilidad inmediata.",
    "Lead Engineer en Go y C++. 15 años de experiencia. Experto en sistemas de baja latencia. Disponibilidad inmediata.",
    "Candidato Middle Java. 5 años de experiencia. Conocimientos de Angular. Preaviso de 15 días. 40k.",
    "Estudiante de último año buscando convenio de prácticas. Conocimientos básicos de Python y algoritmos. Inmediata."
]

# 5. Pipeline
prompt_final = tagging_prompt.partial(format_instructions=parser.get_format_instructions())
tagging_chain = prompt_final | llm | parser

# 6. Ejecución del Triaje
triaje_resultados = []

print(f"--- Iniciando Triaje Automático de {len(candidatos_cv)} Candidatos ---\n")

for i, cv in enumerate(candidatos_cv, 1):
    try:
        resultado = tagging_chain.invoke({"input": cv})
        triaje_resultados.append(resultado)
        
        print(f"👤 CANDIDATO #{i}")
        print(f"   - Nivel: {resultado.get('nivel_experiencia')}")
        print(f"   - Tech:  {resultado.get('tecnologia_principal')}")
        print(f"   - Disponibilidad: {resultado.get('disponibilidad')}")
        print(f"   - Salario: {resultado.get('salario_esperado')}")
        print(f"   - Perfil: {resultado.get('resumen_perfil')}")
        print("-" * 50)
        
    except Exception as e:
        print(f"❌ Error analizando candidato #{i}: {e}")

print(f"\nTriaje completado. Base de datos actualizada con {len(triaje_resultados)} perfiles.")
