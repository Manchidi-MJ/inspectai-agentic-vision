# Submission Checklist & Competition Materials

Competition: **OpenCV AI Competition 2026, powered by AWS** —
<https://opencv26.devpost.com/>
Verified deadline: **October 26, 2026, 11:59 p.m. Pacific Time.**

---

## Project description (for the Devpost page)

**InspectAI — Agentic Visual Quality Inspector.** InspectAI localizes visual
anomalies on manufactured-object surfaces using OpenCV 5, then runs an explicit
agentic decision loop that evaluates its own evidence and decides what happens
next — re-inspect with adjusted parameters, pass, flag a potential anomaly, or
escalate to human review — with a full, auditable decision trace. A degraded
image is never declared defective, and uncertain evidence is surfaced for a
human rather than silently passed. The OpenCV 5 engine runs as a FastAPI service
containerized for AWS App Runner.

## Innovation statement (short)

Most inspection systems output a score and stop. InspectAI adds the **decision
layer**: an agent whose next action provably depends on the first visual result,
so borderline cases are investigated and escalated instead of guessed. It keeps
three outcomes distinct — potential anomaly, no anomaly, uncertain/human review —
and records why each verdict was reached.

## Definition-of-done status

- [x] OpenCV 5 installed and verified (5.0.0), used substantively across the pipeline.
- [x] Working image-inspection pipeline.
- [x] Functioning agentic decision loop.
- [x] Agent's next action demonstrably depends on visual results (seeds 102/111).
- [x] Usable application interface (Streamlit).
- [ ] **Meaningful AWS component deployed and tested** — config ready, **UNVERIFIED**
      (no credentials/CLI; local Docker daemon down). API verified locally.
- [x] Actual evaluation results and failure cases documented.
- [x] Automated tests run and reported (27 passed).
- [x] Dependencies pinned and reproduction instructions complete.
- [x] Architecture diagram matches implementation.
- [x] Technical report and demo materials prepared.
- [x] Competition rules/deadline checked (page via search; see manual actions).
- [x] Remaining manual actions listed (below).

## What YOU must do manually before submitting

1. **Re-read the official rules** at <https://opencv26.devpost.com/> and the
   Devpost rules tab (the automated fetch got HTTP 403). Confirm deadline,
   eligibility, team rules, and the exact required deliverables.
2. **Register / confirm eligibility** on Devpost; add any required team info.
   *(Do not invent team members or bios.)*
3. **Deploy the AWS component** (optional but recommended for the AWS theme):
   run the steps in `deployment/aws-deployment.md`, capture the live
   `/health` URL, and update the report's §7 status from "unverified" to the
   real result. Delete the service after judging to avoid cost. Requires your
   own AWS account/credentials and authorization for any paid resources.
4. **Record the demo video** using `docs/demo_script.md` (≤ 5 min). Do not
   claim a video exists until recorded.
5. **Capture screenshots** of the UI (normal, spot, the 102/111 agentic trace)
   and, if deployed, the App Runner console + `/health` response.
6. **Push the repo** to the platform the rules require and paste the final URL
   into the submission.
7. **Fill in the Devpost submission fields** (description above, built-with
   tags: OpenCV 5, Python, FastAPI, Streamlit, AWS App Runner, Docker).

## Facts / assets still needed from you (not fabricated here)

- Live AWS deployment URL (if you deploy).
- Demo video URL.
- UI / console screenshots.
- Team name and member details.
- Public repository URL.
