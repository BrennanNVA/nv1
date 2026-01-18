# Quantitative Trading Firm Project Structure Research

**Research Date**: January 2025
**Sources**: HRT Engineering Blog, Two Sigma Technical Articles, Industry Best Practices
**Purpose**: Inform Nova Aetus project structure based on institutional quant firm patterns

---

## Executive Summary

Research into how top quantitative trading firms (HRT, RenTec, Two Sigma, D.E. Shaw, Citadel) structure their codebases reveals consistent patterns:

1. **Monorepo Architecture** - Single unified repository for all components
2. **Layered Architecture** - Clear separation between latency-critical core and research/tooling
3. **Modular Design** - Well-defined module boundaries with shared libraries
4. **Data as Code** - Version-controlled data pipelines and transformations
5. **Strong Tooling** - Comprehensive build/test/CI infrastructure
6. **Clear Ownership** - CODEOWNERS files and team boundaries

---

## Key Findings by Firm

### Hudson River Trading (HRT)

**Architecture Pattern**: Layered with language separation

```
[Trading Core / Low Latency C++]
  ├ Exchange adapters
  ├ Order server (Trip, OpenOrderServer)
  ├ Market data handlers
  ├ Risk & exposure
  └ Performance-critical utilities

[Middleware / Infrastructure]
  ├ Distributed storage (Blobby filesystem)
  ├ Messaging, logging, event streams
  ├ Monitoring, backtrace frameworks
  └ Build, test, deployment tools

[Tools & Research / Orchestration]
  ├ Data pipelines / workflows
  ├ Exploratory research / backtesting
  ├ Internal dashboards, UI
  └ Non-critical services (Python/Go)
```

**Key Practices**:
- **C++** for latency-critical paths (exchange connectivity, order entry, market data)
- **Python** for non-latency-critical tooling (orchestration, monitoring, research)
- **Go** for infrastructure/SRE work
- Modular code generation pipelines (RawReflect, TripRaw)
- Contract-first design (protobufs, gRPC)
- Team organization by domain (algo/trading, infrastructure, tools/data)

**Inferred Project Structure**:
```
/trading_core/          # C++ modules
  ├ exch_adapters/
  ├ order_entry/        # Trip, OpenOrderServer
  ├ marketdata/
  ├ risk/
  └ serialization/      # RawReflect, TripRaw

/infra/                 # Infrastructure
  ├ distributed_storage/  # Blobby
  ├ messaging/
  ├ logging/
  ├ monitoring/
  └ system_utils/

/tools/                 # Build/test/deployment
  ├ build/
  ├ test/
  ├ deployment/
  └ scripts/

/research/              # Research & data pipelines
  ├ data_pipelines/
  ├ backtesting/
  └ visualization/

/web_ui/                # Frontend/backend
  ├ frontend/           # React/TypeScript
  └ backend/            # Python services

/shared_libs/           # Common utilities
  ├ metrics/
  ├ encoding/
  ├ concurrency/
  └ common_utils/
```

---

### Two Sigma

**Architecture Pattern**: Monorepo with "Head-to-Head Development"

**Key Practices**:
- **Monorepo**: 6,000+ projects in single repository
- **Trunk-based development**: Frequent merges to main branch
- **Source integration**: Build all components from source
- **Gated sharing**: Changes shared after validation (build/tests)
- **Implicit versioning**: Components versioned by source control revision
- **VATS system**: Internal build/test automation (20,000 builds/day, hundreds of thousands of pre-push tests/hour)

**Data Engineering**:
- **Data as Code**: Version-controlled datasets, transformations, pipelines
- **Automated quality checks**: Data lineage, observability, data contracts
- **Reproducibility**: Backtesting with exact same data & code versions

**Project Structure Principles**:
- Clear module boundaries with clean APIs
- Dependency granularity (file-level and module-level tracking)
- Code ownership (CODEOWNERS files)
- Separate prototyping from production code

---

### Renaissance Technologies (RenTec)

**Architecture Pattern**: Highly secretive, but inferred patterns

**Known Characteristics**:
- Petabyte-scale data warehouses
- Mix of researchers, data programmers, system engineers, trading programmers
- Collaborative model-based research group structure
- Extreme confidentiality and strict NDAs

**Inferred Structure**:
- **Data Layer**: Ingestion pipelines, feature engineering, historical data stores
- **Research & Modeling**: Prototyping environment, ML training, backtesting
- **Trading/Execution**: Real-time signal generation, order placement, risk controls
- **Infrastructure**: High-performance compute clusters, low-latency networking
- **Support**: Version control, CI/CD, security, compliance, audit tools

**Likely Organization**:
- Research projects (small-to-medium, by asset class/signal type)
- Productionization projects (research → simulation → live pipeline)
- Infrastructure/core platform projects (shared components)
- Cross-cutting concerns (logging, monitoring, security)

---

## Common Patterns Across All Firms

### 1. Monorepo Structure

**Benefits**:
- Atomic cross-language changes (change C++ core and Python wrapper together)
- Shared tooling reduces duplication
- Prevents dependency hell and version mismatches
- Promotes code reuse and consistency

**Structure**:
```
/codebase (monorepo)
  /common         # Shared utilities: math, stats, logging, networking
  /data           # Data models, schemas, pipelines (ETL, data as code)
  /research       # Experimental models, notebooks, backtests
  /strategies     # Specific trading strategies/modules
  /execution      # Execution engine: order routing, low-latency paths
  /risk           # Risk engine, limit checks, PnL, compliance
  /infra          # Build scripts, deployment tooling, configuration
  /ops            # Monitoring tools, dashboards, incident response
```

### 2. Language Separation by Latency Requirements

- **C++ (C++17/20+)**: Latency-critical paths
  - Exchange connectivity
  - Order entry/execution
  - Market data processing
  - Risk calculations
  - Optimizations: minimize allocations, lock-free structures, kernel bypass

- **Python**: Non-latency-critical
  - Research and prototyping
  - Data pipelines
  - Backtesting frameworks
  - Internal tools and dashboards
  - Service APIs

- **Go**: Infrastructure/SRE work
  - DevOps tooling
  - System administration
  - Infrastructure automation

### 3. Modularity & Low Coupling

**Principles**:
- Clear module boundaries with defined interfaces
- Avoid tight coupling between modules
- Dependency granularity (track file-level and module-level)
- Semantic versioning for shared libraries (even if implicit)

**Module Organization**:
- Domain-driven design: separate by business domain (`risk/`, `execution/`, `market_data/`)
- Public vs internal headers/interfaces
- Shared libraries for common functionality

### 4. Build Systems & CI/CD

**Requirements**:
- Fast incremental builds (only rebuild changed components)
- Smart caching and parallelism
- Thousands of builds/tests per hour capability
- Selective CI triggers (only test what changed)
- Pre-commit hooks for linting, static analysis, style checks

**Tools**:
- CMake or Bazel for C++
- Python packaging (pyproject.toml, Poetry)
- Cross-language integration (Pybind11, CPython extensions)

### 5. Data as Code

**Practices**:
- Version-control datasets and transformations
- Data lineage tracking
- Automated quality checks
- Data contracts (producer-consumer agreements)
- Reproducibility for backtesting

**Tools**:
- dbt for SQL transformations
- Terraform for infra-as-code
- Experiment tracking built into pipelines

### 6. Testing & Observability

**Testing Strategy**:
- Unit tests (pytest for Python, GoogleTest/Catch2 for C++)
- Integration tests across components
- Performance/regression tests (track latency, throughput, memory)
- Stress testing under live trading conditions

**Observability**:
- Latency metrics (P95, P99)
- Error rates and throughput
- Tracing pipelines of data and code
- Real-time risk controls (circuit breakers, kill switches)

### 7. Code Ownership & Governance

**Practices**:
- CODEOWNERS files for team responsibility
- Code review workflows
- Architectural decision documentation
- Onboarding documentation
- Governance framework for shared library changes

---

## Example Monorepo Structure (Quantitative Trading System)

Based on research findings, here's a comprehensive structure:

```
/QuantSystem/
├── CMakeLists.txt              # Top-level C++ build
├── .clang-format               # C++ style guide
├── CODEOWNERS                  # Ownership per directory
├── README.md
├── LICENSE
├── SECURITY.md
│
├── scripts/                    # Build, lint, util scripts
│   ├── build.sh
│   ├── lint.sh
│   └── format.sh
│
├── data/                       # Market data, provenance
│   ├── raw/                    # Raw market data
│   ├── processed/              # Processed/cleansed data
│   └── schemas/                # Data schemas
│
├── quant/                      # Python layer (research/service logic)
│   ├── pricing/
│   ├── features/
│   ├── models/
│   ├── utils/
│   └── __init__.py
│
├── benchmarks/                 # Performance benchmarks
│   ├── python/                 # ASV benchmarks
│   └── cpp/                    # C++ benchmarks
│
├── QuantCore/                  # High-performance engine (C++)
│   ├── include/QuantCore/      # Public header interfaces
│   │   ├── execution/
│   │   ├── market_data/
│   │   ├── risk/
│   │   └── utils/
│   ├── src/                    # Implementation
│   │   ├── execution/
│   │   ├── market_data/
│   │   ├── risk/
│   │   └── utils/
│   ├── models/                 # Shared proto/schema files
│   ├── configs/                # Config files (YAML/TOML)
│   └── Dockerfile              # Engine deployment
│
├── services/                   # Python services
│   ├── api/                    # REST/gRPC APIs
│   ├── gateway/                # Exchange gateways
│   └── dashboard/              # Dashboard backend
│
├── clients/                    # Frontends
│   ├── web/                    # Web dashboard
│   └── mobile/                 # Mobile app (if applicable)
│
├── research/                   # Research notebooks/experiments
│   ├── notebooks/
│   ├── backtests/
│   └── experiments/
│
├── infra/                      # Infrastructure as code
│   ├── terraform/              # Infrastructure definitions
│   ├── kubernetes/             # K8s configs
│   └── monitoring/             # Prometheus/Grafana configs
│
└── ci/                         # CI/CD workflows
    ├── .github/workflows/      # GitHub Actions
    ├── jenkins/                # Jenkins configs (if used)
    └── scripts/                # CI helper scripts
```

---

## Recommendations for Nova Aetus

### Current State Analysis

Nova Aetus currently follows a **"src layout"** pattern:
```
nova_aetus/
├── src/nova/
│   ├── core/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── strategy/
│   └── dashboard/
├── docs/
├── scripts/
└── mcp_server/
```

### Recommended Evolution

Based on quant firm patterns, consider evolving toward:

1. **Monorepo Structure** (if adding more components)
   - Keep current structure but add clear module boundaries
   - Consider separating research code from production code

2. **Language Separation** (if performance becomes critical)
   - Keep Python for current use case (swing trading, not HFT)
   - Consider C++ modules only if latency becomes bottleneck
   - Current Python-only approach is appropriate for swing trading

3. **Data as Code**
   - ✅ Already using version-controlled configs (`config.toml`)
   - ✅ Data pipelines in `src/nova/data/`
   - Consider: Add data lineage tracking, data contracts

4. **Modularity Improvements**
   - ✅ Already well-modularized (`core/`, `data/`, `features/`, `models/`, `strategy/`)
   - Consider: Add `CODEOWNERS` file for team ownership
   - Consider: Document module interfaces more explicitly

5. **Testing Infrastructure**
   - ✅ Already has `tests/` directory
   - Consider: Add performance benchmarks
   - Consider: Add integration tests across components

6. **Build/CI Improvements**
   - ✅ Already has `requirements.txt`
   - Consider: Add `pyproject.toml` for modern Python packaging
   - Consider: Add pre-commit hooks for linting/formatting
   - Consider: Add selective CI (only test changed modules)

7. **Documentation Structure**
   - ✅ Already well-organized (`docs/guides/`, `docs/research/`, `docs/reference/`)
   - ✅ Matches best practices for documentation organization

### Specific Recommendations

1. **Add CODEOWNERS file**:
   ```
   # Core trading logic
   /src/nova/strategy/ @trading-team
   /src/nova/models/ @ml-team

   # Infrastructure
   /src/nova/core/ @platform-team
   /src/nova/data/ @data-team

   # Documentation
   /docs/ @docs-team
   ```

2. **Add pyproject.toml** for modern Python packaging:
   ```toml
   [project]
   name = "nova-aetus"
   version = "1.0.0"
   dependencies = [...]

   [project.optional-dependencies]
   dev = ["pytest", "black", "ruff", "mypy"]
   ```

3. **Add pre-commit hooks**:
   - Black for formatting
   - Ruff for linting
   - mypy for type checking
   - pytest for running tests

4. **Consider research/production separation**:
   ```
   /src/nova/
     /research/        # Experimental code
     /production/      # Production-ready code
   ```

5. **Add performance benchmarks**:
   ```
   /benchmarks/
     /data_loading/
     /feature_calculation/
     /model_training/
   ```

---

## Key Takeaways

1. **Monorepo is standard** for quant firms, but current structure is fine for single-project
2. **Language separation** by latency requirements (C++ for HFT, Python for swing trading)
3. **Modularity** with clear boundaries is critical
4. **Data as code** ensures reproducibility
5. **Strong tooling** (build/test/CI) enables scale
6. **Clear ownership** prevents chaos in large codebases

**For Nova Aetus**: Current structure is appropriate for swing trading system. Focus on:
- Adding CODEOWNERS for ownership
- Improving testing infrastructure
- Adding performance benchmarks
- Documenting module interfaces
- Consider research/production separation as codebase grows

---

## References

- HRT Engineering Blog: https://www.hudsonrivertrading.com/hrtbeat/
- Two Sigma Technical Articles: https://www.twosigma.com/articles/
- LinkedIn: "Monorepo Map: Trading System" by Colman Marcus Quinn
- Industry best practices from quant trading forums and technical blogs

---

**Note**: This research is based on publicly available information and inferred patterns. Actual internal structures may differ, but these patterns represent industry best practices for quantitative trading systems.
