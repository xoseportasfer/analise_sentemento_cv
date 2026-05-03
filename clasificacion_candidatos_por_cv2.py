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
    # Candidato 1: Senior Python / Arquitecto
    "Ingeniero de Software con más de 10 años de experiencia en el sector tecnológico. "
    "Mi especialidad principal es el desarrollo backend utilizando Python y el framework Django. "
    "He liderado equipos de hasta 15 personas en entornos internacionales y dinámicos. "
    "Poseo conocimientos profundos en arquitectura de microservicios y patrones de diseño. "
    "En mi último rol, optimicé la infraestructura en AWS reduciendo costes en un 30%. "
    "Tengo experiencia avanzada trabajando con bases de datos PostgreSQL y Redis. "
    "Implementé pipelines de CI/CD utilizando Jenkins y GitLab CI para despliegues diarios. "
    "Domino el uso de Docker y Kubernetes para la orquestación de contenedores en producción. "
    "Me apasiona el desarrollo guiado por pruebas o TDD para asegurar la calidad del código. "
    "Cuento con un nivel de inglés C2, lo que me permite comunicarme con fluidez total. "
    "He participado como ponente en diversas conferencias de tecnología a nivel nacional. "
    "Busco una posición de responsabilidad como Tech Lead o Arquitecto de Software. "
    "Mi disponibilidad para comenzar en un nuevo proyecto es de 1 mes por preaviso legal. "
    "Mis pretensiones salariales actuales se sitúan en torno a los 70.000 euros anuales. "
    "Me considero un experto en la resolución de problemas complejos bajo alta presión. "
    "He trabajado bajo metodologías ágiles como Scrum y Kanban durante toda mi carrera. "
    "Tengo experiencia en la monitorización de sistemas con herramientas como Prometheus y Grafana. "
    "Valoro mucho el equilibrio entre la vida laboral y personal en las empresas. "
    "Estoy abierto a modelos de trabajo 100% remotos o híbridos en la zona de Madrid. "
    "Mi objetivo es seguir creciendo técnicamente mientras aporto valor estratégico al negocio.",

    # Candidato 2: Junior Java / Recién graduado
    "Recién graduado en el Grado de Ingeniería Informática por la Universidad Politécnica. "
    "Durante mis años de estudio, me enfoqué principalmente en el ecosistema de Java. "
    "Realicé mis prácticas curriculares de 6 meses en una consultora tecnológica de renombre. "
    "Allí aprendí las bases de Spring Boot y la creación de APIs REST funcionales. "
    "Conozco el manejo básico de herramientas de control de versiones como Git y GitHub. "
    "He desarrollado varios proyectos personales que se pueden consultar en mi perfil público. "
    "Tengo nociones básicas de maquetación front-end utilizando HTML5, CSS3 y algo de JavaScript. "
    "Me considero una persona extremadamente proactiva y con una gran curva de aprendizaje. "
    "Estoy muy motivado por conseguir mi primer empleo estable dentro del desarrollo backend. "
    "Tengo un nivel de inglés B2, capaz de leer documentación técnica sin dificultad. "
    "He trabajado con bases de datos relacionales como MySQL en mis proyectos académicos. "
    "Busco un entorno donde pueda tener un mentor senior que guíe mi evolución técnica. "
    "Mi disponibilidad para incorporarme al equipo de trabajo es de carácter inmediato. "
    "Como salario inicial, aceptaría un rango entre los 22.000 y 25.000 euros brutos. "
    "Me gusta participar en hackatones y eventos de programación para poner a prueba mis límites. "
    "Tengo facilidad para trabajar en equipo y me adapto rápido a nuevas culturas. "
    "Estoy familiarizado con el entorno Linux y el uso de la terminal de comandos. "
    "He tomado cursos adicionales sobre seguridad informática y protección de datos. "
    "Mi residencia actual es en Barcelona y no tendría problemas en trabajar de forma presencial. "
    "Deseo aplicar mis conocimientos teóricos en retos reales que impacten a los usuarios.",

    # Candidato 3: Middle .NET / Sector Bancario
    "Desarrollador de software especializado en tecnologías Microsoft .NET con 4 años de experiencia. "
    "He trabajado extensamente con C# y .NET Core para aplicaciones de alta disponibilidad. "
    "Mi trayectoria se ha centrado sobre todo en el sector de la banca y las finanzas. "
    "He desarrollado módulos críticos para el procesamiento de pagos y transferencias SEPA. "
    "Tengo experiencia en la integración de servicios externos mediante protocolos SOAP y REST. "
    "Domino el uso de Entity Framework para el acceso y manipulación de datos en SQL Server. "
    "He implementado soluciones de mensajería asíncrona utilizando Azure Service Bus. "
    "Poseo conocimientos sólidos en la nube de Microsoft Azure, incluyendo App Services y Functions. "
    "Trabajo diariamente bajo la metodología Scrum con sprints de dos semanas de duración. "
    "Me enfoco mucho en la calidad del código siguiendo los principios SOLID y Clean Code. "
    "Tengo un nivel de inglés B2 certificado por la Escuela Oficial de Idiomas. "
    "Actualmente estoy liderando la migración de un sistema legacy a una arquitectura moderna. "
    "Mi disponibilidad para un cambio laboral es de 15 días naturales de preaviso. "
    "Busco una mejora profesional con un salario objetivo de unos 48.000 euros al año. "
    "Tengo experiencia en la creación de pruebas unitarias y de integración con xUnit. "
    "Me gusta colaborar estrechamente con el equipo de QA para reducir la deuda técnica. "
    "He configurado entornos de desarrollo local utilizando Docker para estandarizar procesos. "
    "Soy una persona metódica, puntual y muy comprometida con los plazos de entrega. "
    "Prefiero trabajar en un modelo híbrido que me permita ir a la oficina ocasionalmente. "
    "Mi meta a corto plazo es obtener la certificación de Azure Solutions Architect.",

    # Candidato 4: Senior JavaScript (Fullstack)
    "Desarrollador Fullstack senior con 9 años de experiencia en el desarrollo de aplicaciones web. "
    "Mi stack principal de trabajo incluye Node.js en el backend y React en el frontend. "
    "He diseñado e implementado interfaces de usuario complejas y altamente interactivas. "
    "Tengo experiencia trabajando con bases de datos documentales, especialmente con MongoDB. "
    "He gestionado despliegues automáticos en la plataforma Google Cloud (GCP). "
    "Domino el uso de TypeScript para aportar robustez y tipado a mis proyectos de JS. "
    "Implementé sistemas de autenticación seguros utilizando JWT y OAuth2. "
    "Tengo experiencia en el desarrollo de aplicaciones móviles híbridas con React Native. "
    "He optimizado el rendimiento de carga de aplicaciones web reduciendo el bundle size significativamente. "
    "Cuento con un nivel de inglés C1, lo que me permite trabajar en equipos globales. "
    "He mentorizado a desarrolladores junior ayudándoles a mejorar sus habilidades técnicas. "
    "Soy un fiel defensor del uso de componentes reutilizables y sistemas de diseño. "
    "Mi disponibilidad para incorporarme es de 1 mes tras la firma del contrato. "
    "El rango salarial que estoy considerando está a partir de los 55.000 euros. "
    "Poseo conocimientos en GraphQL para la consulta eficiente de datos desde el cliente. "
    "He trabajado en proyectos de E-commerce con miles de transacciones diarias simultáneas. "
    "Utilizo herramientas de testing como Jest y Cypress para asegurar la estabilidad visual. "
    "Me gusta mantenerme actualizado sobre las últimas tendencias de la comunidad JavaScript. "
    "Busco una empresa que apueste por la innovación y el uso de tecnologías punteras. "
    "Mi enfoque siempre está orientado a mejorar la experiencia final del usuario.",

    # Candidato 5: Lead Go / C++ (Sistemas)
    "Ingeniero de sistemas con más de 12 años de experiencia en programación de bajo nivel. "
    "Experto en el desarrollo de servicios de alta concurrencia utilizando el lenguaje Go. "
    "He pasado gran parte de mi carrera optimizando motores de búsqueda escritos en C++. "
    "Tengo conocimientos profundos en redes, protocolos TCP/IP y sockets de baja latencia. "
    "He trabajado para grandes corporaciones en el desarrollo de sistemas operativos embebidos. "
    "Domino la gestión de memoria manual y la depuración de fugas con herramientas como Valgrind. "
    "Implementé algoritmos de compresión de datos que mejoraron la eficiencia en un 40%. "
    "Tengo experiencia en el liderazgo técnico de proyectos de infraestructura crítica nacional. "
    "Cuento con un nivel de inglés técnico excelente para la redacción de especificaciones. "
    "He contribuido activamente en proyectos de código abierto de la Cloud Native Foundation. "
    "Poseo un doctorado en Ciencias de la Computación enfocado en computación distribuida. "
    "Mi disponibilidad es inmediata dado que actualmente me encuentro realizando consultoría freelance. "
    "Mis expectativas salariales son acordes a mi experiencia, sobre los 85.000 euros. "
    "Tengo experiencia en el uso de gRPC para la comunicación eficiente entre servicios. "
    "He diseñado sistemas tolerantes a fallos con replicación de datos en tiempo real. "
    "Valoro la transparencia, el rigor técnico y la honestidad intelectual en el trabajo. "
    "Soy capaz de leer y entender código ensamblador cuando la optimización lo requiere. "
    "He gestionado equipos de ingenieros de software en diferentes husos horarios. "
    "Busco retos que supongan un desafío intelectual y técnico de primer nivel. "
    "Me interesa la investigación aplicada y la implementación de nuevas arquitecturas.",

    # Candidato 6: Middle Java / Microservicios
    "Desarrollador Java con 6 años de experiencia trabajando en consultoría tecnológica avanzada. "
    "Especialista en el desarrollo de microservicios con Spring Cloud y Netflix OSS. "
    "Tengo experiencia demostrable en la migración de aplicaciones monolíticas a servicios pequeños. "
    "Domino el uso de Apache Kafka para la implementación de arquitecturas orientadas a eventos. "
    "He trabajado con bases de datos NoSQL como Cassandra y bases relacionales como Oracle. "
    "Implementé soluciones de seguridad con Spring Security para la protección de endpoints. "
    "Tengo experiencia en el despliegue de aplicaciones sobre entornos OpenShift y Azure. "
    "Utilizo Maven y Gradle como herramientas de construcción y gestión de dependencias. "
    "Poseo un nivel de inglés B2 capaz de mantener reuniones técnicas de trabajo. "
    "He desarrollado integraciones complejas con sistemas de terceros mediante APIs externas. "
    "Tengo conocimientos en herramientas de análisis de código estático como SonarQube. "
    "Busco una posición estable de programador senior con proyección a corto plazo. "
    "Mi disponibilidad para un cambio es inmediata por situación de finalización de obra. "
    "Mi salario deseado se encuentra en el rango de los 42.000 a 45.000 euros. "
    "Soy una persona proactiva que siempre intenta proponer mejoras tecnológicas al equipo. "
    "He trabajado en proyectos bajo normativa ISO 27001 de seguridad de la información. "
    "Me gusta el trabajo por objetivos y tengo una gran capacidad de autoorganización. "
    "He participado en la definición de esquemas de datos y modelado de entidades. "
    "Estoy disponible para viajar ocasionalmente si el proyecto lo requiere puntualmente. "
    "Mi principal motivación es trabajar en proyectos que utilicen stacks modernos.",

    # Candidato 7: Junior Python / Data Engineer
    "Ingeniero de datos junior con 2 años de experiencia profesional en el sector retail. "
    "Mi lenguaje principal es Python, el cual utilizo para la creación de scripts de ETL. "
    "He trabajado con Spark para el procesamiento de grandes volúmenes de información. "
    "Tengo experiencia en la automatización de flujos de datos utilizando Apache Airflow. "
    "Conozco el manejo de bases de datos analíticas como Snowflake y Amazon Redshift. "
    "He realizado visualizaciones de datos complejas utilizando herramientas como Tableau. "
    "Tengo conocimientos básicos de aprendizaje automático con librerías como Scikit-learn. "
    "Domino el uso de SQL para la realización de consultas complejas y optimización de tablas. "
    "Mi nivel de inglés es B1, estoy trabajando activamente para mejorarlo este año. "
    "Me considero una persona curiosa, analítica y muy orientada al detalle técnico. "
    "He colaborado con equipos de Data Science para preparar los datasets de entrenamiento. "
    "Mi disponibilidad para comenzar un nuevo reto profesional es inmediata en este momento. "
    "Aspiro a un salario bruto anual de 30.000 euros para seguir formándome. "
    "Tengo experiencia en el control de versiones de modelos de datos con DVC. "
    "He trabajado en entornos cloud, específicamente con servicios de datos en AWS. "
    "Me gusta documentar cada proceso que desarrollo para facilitar el mantenimiento futuro. "
    "Soy capaz de trabajar de forma autónoma siguiendo las directrices de los analistas. "
    "Tengo un máster especializado en Big Data y Análisis de Datos Masivos. "
    "Resido en Madrid y tengo total disponibilidad para trabajar en formato híbrido. "
    "Busco integrarme en un equipo de datos sólido donde pueda aprender mejores prácticas.",

    # Candidato 8: Lead .NET / Azure
    "Arquitecto de soluciones y Lead Developer con 15 años de experiencia técnica. "
    "Especializado en tecnologías .NET desde sus primeras versiones hasta .NET 8 actual. "
    "He liderado la transformación digital de grandes empresas del sector energético. "
    "Experto en el diseño de arquitecturas en la nube de Azure (PaaS y IaaS). "
    "Domino el despliegue de infraestructuras como código utilizando Terraform y ARM. "
    "He gestionado equipos multidisciplinares de hasta 20 ingenieros y diseñadores. "
    "Poseo certificaciones oficiales de Microsoft: Azure Solutions Architect Expert. "
    "Tengo experiencia en la implementación de estrategias de seguridad multicapa en el cloud. "
    "Mi nivel de inglés es C1, con experiencia trabajando directamente para clientes en USA. "
    "He desarrollado frameworks internos para acelerar el desarrollo dentro de la empresa. "
    "Soy experto en optimización de rendimiento en bases de datos SQL Server críticas. "
    "Busco un puesto de alta dirección técnica o Principal Engineer en una tecnológica. "
    "Mi disponibilidad es de 1 mes, aunque negociable según la urgencia del proyecto. "
    "Mi pretensión económica se sitúa a partir de los 80.000 euros anuales fijos. "
    "He participado en la selección de talento técnico y definición de planes de carrera. "
    "Valoro enormemente la cultura del feedback y la mejora continua de procesos. "
    "He implementado arquitecturas de tipo Event-Sourcing y CQRS en sistemas complejos. "
    "Me gusta estar cerca del código aunque mis funciones sean de gestión y diseño. "
    "Tengo amplia experiencia en la negociación con proveedores de servicios tecnológicos. "
    "Mi enfoque principal es alinear la tecnología con los objetivos financieros de la empresa.",

    # Candidato 9: Trainee JavaScript / Frontend
    "Estudiante de último año de FP Superior en Desarrollo de Aplicaciones Web (DAW). "
    "Busco mi primera oportunidad profesional en formato de prácticas o contrato trainee. "
    "Tengo una base sólida de JavaScript nativo (ES6+) obtenida durante mis estudios. "
    "He realizado un bootcamp intensivo de 3 meses enfocado exclusivamente en React. "
    "Conozco el funcionamiento de Redux y otras librerías de gestión de estado global. "
    "Tengo experiencia básica maquetando interfaces con Flexbox, CSS Grid y SASS. "
    "He utilizado Figma para la interpretación de diseños y su posterior implementación. "
    "Domino el uso básico de la consola y comandos básicos de Git para versiones. "
    "Me considero una persona con una actitud positiva y muy abierta a las críticas. "
    "Mi nivel de inglés es A2, pero estoy apuntado a una academia para subir nivel. "
    "He subido un par de landing pages sencillas a servicios de hosting como Vercel. "
    "Busco una empresa que me permita terminar mi formación mientras aporto trabajo. "
    "Disponibilidad horaria total para incorporarme de forma inmediata a la plantilla. "
    "En cuanto al salario, busco lo estipulado por convenio para perfiles de formación. "
    "Tengo muchas ganas de ver cómo funciona un equipo de desarrollo real por dentro. "
    "Soy una persona puntual, responsable y que sabe seguir instrucciones precisas. "
    "Me interesa mucho el mundo del diseño de interfaces y la usabilidad (UI/UX). "
    "He realizado pequeños scripts con Node.js para automatizar tareas de clase. "
    "Resido en Valencia y busco opciones tanto presenciales como de teletrabajo. "
    "Mi objetivo este año es convertirme en un desarrollador frontend junior solvente.",

    # Candidato 10: Senior Go / Cloud Native
    "Ingeniero de Software Senior con 7 años de experiencia centrados en Cloud Native. "
    "Mi lenguaje de programación de cabecera es Go (Golang) para el desarrollo backend. "
    "He trabajado extensamente en el desarrollo de operadores y controladores de Kubernetes. "
    "Tengo experiencia avanzada en la gestión de servicios sobre infraestructura de AWS. "
    "Domino el uso de herramientas de observabilidad como Linkerd, Istio y Jaeger. "
    "He implementado arquitecturas de Serverless utilizando AWS Lambda y Step Functions. "
    "Tengo experiencia en el diseño de APIs altamente escalables y seguras. "
    "Domino el uso de bases de datos distribuidas como CockroachDB y DynamoDB. "
    "Poseo un nivel de inglés C1 certificado, con experiencia en empresas británicas. "
    "He liderado la migración de infraestructuras on-premise a entornos de nube pública. "
    "Me considero un experto en la automatización de procesos mediante scripts y Terraform. "
    "Mi disponibilidad para un nuevo proyecto es de 1 mes por compromiso actual. "
    "Busco un rango salarial de entre 60.000 y 65.000 euros anuales brutos. "
    "Tengo experiencia en la gestión de costes cloud (FinOps) para evitar desviaciones. "
    "He participado en guardias 24x7 y entiendo la importancia de la resiliencia del sistema. "
    "Me gusta contribuir a la comunidad técnica escribiendo artículos en mi blog personal. "
    "Soy un apasionado de la cultura DevOps y la automatización total del ciclo de vida. "
    "He trabajado en entornos con regulaciones estrictas como GDPR y PCI-DSS. "
    "Estoy interesado en posiciones de Senior Software Engineer o SRE (Site Reliability). "
    "Mi meta es seguir construyendo sistemas globales que soporten millones de usuarios."
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
