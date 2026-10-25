# InspectAI — Architecture

This diagram matches the implemented code, not an aspirational design. Every
box maps to a module in `src/`.

## Component flow

```mermaid
flowchart TD
    U[User] -->|upload image / pick sample| UI[Streamlit UI<br/>app.py]

    UI -->|in-process default| PIPE[Pipeline<br/>src/pipeline.py]
    UI -. "if INSPECTAI_API_URL set" .-> API[FastAPI inspection API<br/>src/api/routes.py<br/>runs on AWS App Runner]
    API --> PIPE

    PIPE --> DEC[decode + validate<br/>src/vision/preprocessing.py]
    DEC --> NORM[normalize / resize / grayscale<br/>OpenCV 5]
    NORM --> AGENT[Inspection Agent<br/>src/agent/controller.py]

    subgraph LOOP[Agentic decision loop  perceive to terminate]
        AGENT --> Q[assess_image_quality<br/>src/vision/image_quality.py]
        Q -->|quality fails| HR1[request_human_review]
        Q -->|usable| DET[analyze_image -> detect<br/>src/vision/detector.py<br/>OpenCV 5 anomaly map]
        DET --> CH{choose on<br/>real score + regions}
        CH -->|score >= HIGH| PA[potential_anomaly]
        CH -->|score < LOW| NA[no_anomaly_detected]
        CH -->|borderline| RE[run_targeted_reinspection<br/>lowered threshold]
        RE --> CMP[compare_analysis_results]
        CMP -->|corroborated| PA
        CMP -->|not corroborated| HR2[request_human_review]
    end

    AGENT --> LOG[Decision log<br/>src/agent/decision_log.py]
    DET --> ANN[annotate + heatmap<br/>OpenCV 5 drawing]

    PA --> OUT[Structured result + annotated images]
    NA --> OUT
    HR1 --> OUT
    HR2 --> OUT
    LOG --> OUT
    ANN --> OUT
    OUT --> UI

    EVAL[Evaluation<br/>src/evaluation/evaluate.py] -.reuses.-> DET
    EVAL -.reuses.-> AGENT
    SYN[Synthetic labeled dataset<br/>src/vision/synthetic.py] -.feeds.-> EVAL
```

## Where things run

| Concern | Location |
|---|---|
| Image processing (decode, resize, color, residual, morphology, draw) | OpenCV 5 in `src/vision/*` |
| Agent decisions + bounded loop | `src/agent/controller.py` (pure, deterministic) |
| Decision trace / logging | `src/agent/decision_log.py` |
| HTTP service (the AWS component) | `src/api/routes.py` → container → App Runner |
| UI | `app.py` (Streamlit), in-process or calling the API |
| Metrics / charts | `src/evaluation/*`, `scripts/make_charts.py` |

## AWS services actually used

Only what the component needs:

- **Amazon ECR** — stores the container image.
- **AWS App Runner** — runs the container, provides HTTPS + health checks + autoscaling.

No database, queue, or GPU is used; the classical pipeline is CPU-light
(~5–10 ms/image measured locally). See `deployment/aws-deployment.md` for the
exact (currently **unverified**) deployment commands.
```
