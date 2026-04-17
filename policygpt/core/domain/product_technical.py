"""Domain profile — product and technical documentation."""

from policygpt.core.domain.base import DomainProfile, register

PROFILE = DomainProfile(
    domain_context=(
        "Documents are technical and product documentation for a software/technology team. "
        "This includes architecture decision records, system design documents, deployment "
        "runbooks, AWS infrastructure guides, API references, data-flow diagrams, "
        "capacity planning sheets, and operational playbooks. Users are engineers, "
        "architects, DevOps/SRE staff, and technical leads asking about system design, "
        "component interactions, deployment steps, AWS service configurations, "
        "infrastructure sizing, and troubleshooting procedures."
    ),

    # ── Bot ───────────────────────────────────────────────────────────────────
    persona_description="a software engineering and DevOps team",
    greeting_reply="Hello! Ready to help with architecture, deployment, or technical documentation questions.",
    identity_reply=(
        "I'm a technical documentation assistant — ask me about system design, "
        "architecture, AWS infrastructure, deployment procedures, and more."
    ),
    intent_user_description="engineers, architects, and DevOps/SRE staff",
    intent_policy_description=(
        "architecture decisions, system design, deployment steps, AWS configurations, "
        "infrastructure specs, API references, operational runbooks, or any technical topic"
    ),
    doc_type_label="technical documentation",

    # ── Corpus: document summaries ────────────────────────────────────────────
    doc_summary_focus=(
        "purpose, system/component scope, key design decisions, technology stack, "
        "AWS services used, deployment topology, data flows, dependencies, "
        "constraints, and known limitations"
    ),
    chunk_summary_capture=(
        "component names, AWS services, design decisions, configuration values, "
        "deployment steps, data-flow descriptions, dependencies, and constraints"
    ),
    combine_summary_retain=(
        "component names, AWS services, architecture decisions, configuration values, "
        "deployment topology, data flows, dependencies, and limitations"
    ),
    finalize_summary_focus=(
        "system scope, key design decisions, AWS services, configuration values, "
        "deployment topology, dependencies, and constraints"
    ),

    # ── Corpus: section summaries ─────────────────────────────────────────────
    section_summary_capture=(
        "component names, AWS services with EXACT names/versions, design decisions with rationale, "
        "configuration values with EXACT settings, deployment steps in order, "
        "data-flow descriptions, dependencies with versions, and known limitations"
    ),
    section_combine_preserve=(
        "component names, AWS services, configuration values, deployment steps, "
        "data-flow descriptions, dependencies, and limitations"
    ),
    user_label="engineer",

    # ── Corpus: FAQ ───────────────────────────────────────────────────────────
    faq_cover=(
        "architecture decisions and rationale, AWS service configurations, deployment steps, "
        "component dependencies, data flows, infrastructure sizing, operational procedures, "
        "troubleshooting steps, and known limitations"
    ),

    # ── Bot: aggregate / listing queries ──────────────────────────────────────
    aggregate_response_hint=(
        "Prefer standalone component or service names from document titles or sections that explicitly describe the component. "
        "Include the technology stack, AWS service names, and version numbers where stated. "
        "Do not output bare names without context."
    ),
    aggregate_positive_markers=(
        "component name",
        "service name",
        "system overview",
        "architecture overview",
        "design summary",
        "deployment overview",
        "service is named",
        "component is",
        "the system is",
        "infrastructure overview",
    ),

    # ── Entity extraction ─────────────────────────────────────────────────────
    entity_categories=frozenset({
        "aws_service",    # EC2, RDS, S3, Lambda, EKS, CloudFront, SQS, SNS, etc.
        "component",      # API Gateway, Auth Service, Ingestion Pipeline, etc.
        "technology",     # Python, FastAPI, PostgreSQL, Redis, Kafka, Docker, etc.
        "config_value",   # instance types, timeout values, replica counts, CIDR blocks
        "deployment",     # region, AZ, cluster, environment (prod/staging/dev)
        "data_flow",      # event, queue, topic, stream, webhook, API call
        "action",         # deploy, scale, rollback, failover, restart, migrate
        "team",           # platform team, SRE team, product team, data team
        "abbreviation",   # ADR, SLA, RTO, RPO, CI/CD, IAM, VPC, ALB, ASG
        "other",
    }),
    entity_global_categories=frozenset({"aws_service", "component", "abbreviation", "technology"}),
    entity_extraction_rules=(
        "- Extract ALL meaningful entities: AWS services, components, technologies, "
        "configuration values, deployment details, data flows, teams, and abbreviations.\n"
        "- For AWS services: include the full service name, any version or tier, "
        "and how it is used in this system.\n"
        "- For components: describe the component's responsibility, its interfaces, "
        "and which other components or services it depends on.\n"
        "- For technologies: include the language/framework/database name, version if stated, "
        "and its role in the system.\n"
        "- For config values: describe what the value controls (timeout, replica count, "
        "instance type) and which component it applies to.\n"
        "- For deployment details: describe the environment (prod/staging/dev), region, "
        "AZ count, and deployment method (ECS, EKS, Lambda, EC2 ASG).\n"
        "- For data flows: describe the trigger/source, the transport (SQS, Kafka, HTTP), "
        "and the consumer/destination.\n"
        "- For abbreviations: spell them out fully in the technical context.\n"
        "- Synonyms must reflect how an engineer would search (technical shorthand, "
        "AWS console names, CLI resource types — the way someone types in Slack or a ticket).\n"
        "- Be exhaustive: include every entity an engineer or architect might ask about."
    ),
    entity_examples=(
        '[\n'
        '  {"name":"EKS","category":"aws_service","context":"Amazon Elastic Kubernetes Service — '
        'used to run the containerised ingestion pipeline and API microservices in a managed '
        'Kubernetes cluster","synonyms":["elastic kubernetes service","kubernetes cluster",'
        '"k8s","managed kubernetes"]},\n'
        '  {"name":"Ingestion Pipeline","category":"component","context":"Event-driven pipeline '
        'that reads documents from S3, converts them to HTML, extracts sections, and writes '
        'embeddings to OpenSearch","synonyms":["ingest service","document pipeline","doc ingestion",'
        '"ingestion service","pipeline"]}\n'
        ']'
    ),

    # ── Web UI ────────────────────────────────────────────────────────────────
    ui_assistant_label="Technical Docs Assistant",
    ui_eyebrow="Ask technical questions",
    ui_description=(
        "Get instant answers on architecture, AWS configurations, deployment steps, and system design — "
        "grounded in your indexed technical documentation."
    ),
    ui_sidebar_title="Chat with technical docs",
    ui_sidebar_subtitle="Ask questions across architecture, deployment, and AWS documentation.",
    ui_search_placeholder="Search technical documentation...",
    ui_input_placeholder="Ask about architecture, AWS services, deployments, configurations...",
    ui_prompt_chips=(
        ("Architecture overview", "Give me an overview of the system architecture — key components and how they interact."),
        ("AWS services used", "What AWS services are used and what is each one responsible for?"),
        ("Deployment steps", "What are the step-by-step deployment instructions for the production environment?"),
        ("Data flow", "How does data flow through the system — from ingestion to storage to serving?"),
    ),
)

register("product_technical", PROFILE)
