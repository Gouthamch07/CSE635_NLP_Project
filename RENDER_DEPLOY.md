# Deploying to Render

This repo ships a [`render.yaml`](render.yaml) Blueprint, so deployment is mostly:
**New +** → **Blueprint** → pick this repo → **Apply** → fill in the 5 secrets in the dashboard.

The free tier (`plan: free` in the blueprint) is sufficient for a demo. Upgrade if you need more memory or no spin-down.

---

## 1. One-time GCP setup (Vertex Gemini service account)

Local dev uses `gcloud auth application-default login` (ADC). Render needs an explicit service-account key. Run this once on your machine:

```bash
PROJECT_ID="uplifted-valor-474623-c9"     # your GOOGLE_CLOUD_PROJECT
SA_NAME="ub-cse-chatbot-render"

gcloud iam service-accounts create "$SA_NAME" \
  --project "$PROJECT_ID" \
  --display-name "UB CSE Chatbot (Render)"

gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/aiplatform.user"

gcloud iam service-accounts keys create ./gcp-sa.json \
  --iam-account="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
```

You now have `gcp-sa.json` locally. **Do not commit it** (`.gitignore` already blocks `*-sa.json`). You'll paste its contents into Render in the next step.

---

## 2. Deploy the Blueprint

1. Push your code to `main` (already done).
2. Go to <https://dashboard.render.com/> → **New +** → **Blueprint**.
3. Connect this GitHub repo (`Gouthamch07/CSE635_NLP_Project`).
4. Render reads `render.yaml` and proposes the `ub-cse-chatbot` web service. **Apply**.
5. The first build will pause asking for the secret env vars below.

---

## 3. Secrets to paste into the Render dashboard

Under your service → **Environment** → **Add Environment Variable** (or paste during Blueprint apply):

| Key | Value |
|---|---|
| `GOOGLE_CLOUD_PROJECT` | your GCP project ID, e.g. `uplifted-valor-474623-c9` |
| `GOOGLE_APPLICATION_CREDENTIALS_JSON` | full contents of `gcp-sa.json` from step 1 (paste the whole JSON) |
| `PINECONE_API_KEY` | your Pinecone key (`pcsk_...`) |
| `NEO4J_URI` | `neo4j+s://<your-id>.databases.neo4j.io` (from your Aura console) |
| `NEO4J_PASSWORD` | your Aura password |

The non-secret values (region, model, defaults) are already in `render.yaml` — leave them alone unless you want to change them.

---

## 4. First boot

Watch the deploy logs. Successful startup looks like:

```
warm-starting UB CSE agent
entity_index loaded faculty=N labs=N programs=N courses=N
kg.warmup ok
warm-start complete
INFO: Application startup complete.
```

Cold start on free tier is ~60–90s (downloading the cross-encoder model + warming Vertex + Neo4j). Subsequent requests are fast until the service spins down (15min idle on free tier).

The deployed URL is shown at the top of the dashboard, e.g.
`https://ub-cse-chatbot.onrender.com`.

---

## 5. Common gotchas

- **`404 Publisher Model gemini-3-flash-preview was not found`** — `GOOGLE_CLOUD_LOCATION` must be `global` (already pinned in `render.yaml`).
- **`kg.warmup failed`** — `NEO4J_URI` / `NEO4J_PASSWORD` wrong, or Aura free instance is paused (Aura free tier auto-pauses after 3 days idle; resume from the Aura console).
- **Out of memory** during build/start — bump to `Starter` ($7/mo) or `Standard` ($25/mo) by editing `plan:` in `render.yaml` and pushing.
- **Build is slow** — first build downloads torch/transformers (~700MB). Subsequent builds are cached.
- **Blueprint UI doesn't see updated `render.yaml`** — go to **Settings → Manual Sync**, or just push another commit to `main`.

---

## 6. Local vs. deployed

| | Local | Render |
|---|---|---|
| GCP auth | `gcloud auth application-default login` (ADC) | `GOOGLE_APPLICATION_CREDENTIALS_JSON` env → bootstrap writes file |
| Requirements | `requirements.txt` (full) | `requirements.runtime.txt` (slim) |
| BM25 index | `data/processed/bm25.pkl` (now committed) | same (shipped in repo) |
| Reranker | falls back to `cross-encoder/ms-marco-MiniLM-L-6-v2` if `FlagEmbedding` not installed | same — `FlagEmbedding` excluded from runtime requirements |
