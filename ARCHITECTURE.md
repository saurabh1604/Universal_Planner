# Universal Migration Planner - System Architecture

This document outlines the **Logical Architecture** of the Universal Migration Planner (UAMP). It illustrates how human intent is transformed into mathematical execution through an intelligent, schema-aware middleware layer.

## The Vision: Production Architecture (DevOps Integration)

While the PoC utilizes a local Streamlit interface, the target architecture integrates directly into the **DevOps Portal**, leveraging **Object Storage** for secure data handling.

```mermaid
graph LR
    %% STYLING
    classDef portal fill:#e3f2fd,stroke:#1565c0,stroke-width:2px;
    classDef storage fill:#fff3e0,stroke:#ef6c00,stroke-width:2px;
    classDef brain fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px;
    classDef solver fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px;

    %% -----------------------------
    %% ZONE 1: USER & DATA LAYER
    %% -----------------------------
    subgraph "Zone 1: The Environment"
        direction TB
        DevOps([DevOps User]):::portal
        Portal[DevOps UI / Console]:::portal
        ObjStore[(Object Storage)]:এরstorage

        DevOps -->|NL Prompt| Portal
        Portal -.->|Fetch Data| ObjStore
    end

    %% -----------------------------
    %% ZONE 2: INTELLIGENT MIDDLEWARE
    %% -----------------------------
    subgraph "Zone 2: Agentic Middleware (The Brain)"
        direction TB
        Profiler[Schema Profiler]:::brain
        Auditor[Auditor Agent]:::brain
        Architect[Architect Agent]:::brain

        %% Data Flow
        ObjStore -->|Raw CSV/Parquet| Profiler
        Profiler -->|Data Passport| Auditor
        Profiler -->|Data Passport| Architect

        %% Chat Loop
        Portal <-->|Validation Loop| Auditor
        Auditor -->|Approved Request| Architect
        Architect -->|Generates CDL| SolverInput(CDL JSON):::brain
    end

    %% -----------------------------
    %% ZONE 3: EXECUTION CORE
    %% -----------------------------
    subgraph "Zone 3: Execution Engine"
        direction TB
        Solver[Universal Solver]:::solver
        Result[Optimized Schedule]:::solver

        SolverInput --> Solver
        ObjStore -->|Source Data| Solver
        Solver -->|Compute| Result
    end

    %% FEEDBACK
    Result -->|Visualization| Portal
```

## Component Roles

### 1. The Environment (DevOps & Storage)
*   **DevOps UI**: The interaction point. Users type natural language constraints directly into the portal they use for deployment.
*   **Object Storage**: The single source of truth for pod data (CSV/Parquet), ensuring security and scalability.

### 2. Intelligent Middleware (The Brain)
*   **Schema Profiler**: Dynamically scans data from Object Storage to create a **Data Passport**. This ensures the agents understand the *current* state of the infrastructure (e.g., specific Exadata names, current utilization).
*   **Auditor Agent**: The "Safety Valve". It validates prompts against the Data Passport and existing DevOps rules (e.g., "Blackout Periods").
    *   *Action*: Rejects hallucinations or ambiguous requests before they trigger expensive computations.
*   **Architect Agent**: The "Translator". Converts the validated prompt into the mathematical **Constraint Definition Language (CDL)**.

### 3. Execution Engine (The Hands)
*   **Universal Solver**: A "blind" CP-SAT engine. It accepts:
    1.  **Data** (from Object Storage)
    2.  **Logic** (CDL from the Architect)
*   It computes the optimal schedule without needing to understand the business intent, only the mathematical constraints.
