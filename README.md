# feedBack — Stem Splitter

Splits `.feedpak` songs that do not yet have per-instrument stems, and transcribes
lyrics when they're absent. Single-song or batch, with automatic "what's missing"
detection driven by the library index.

## Engines

Three ways to run, picked in the plugin's **Settings**:

- **Managed local server (easiest)** — the plugin can install and run
  [`got-feedBack/feedBack-demucs-server`](https://github.com/got-feedBack/feedBack-demucs-server)
  for you. **Prepare server installation** checks available versions, then **Update
  server, libraries and models** downloads and validates a compatible installation,
  including its own verified FFmpeg/FFprobe pair; no system FFmpeg installation is
  required. Use **Start** when ready.
  While it's running the plugin uses it
  automatically. See [Local demucs server](#local-demucs-server) below.
- **Docker container** — if you run feedBack in Docker, the plugin can't install a server
  on your host (a container can't start a process outside itself). Instead it gives you a
  **compose snippet** to paste, or — if you've already mounted the Docker socket — starts
  the server as a **sibling container** in one click. See [Docker](#docker) below.
- **Remote** — posts audio to a demucs/whisperx server you already run
  (`demucs_server_url` in the app settings). No local dependencies. Split model
  defaults to **`bs_roformer_sw`**.
- **Local engine (in-process, opt-in)** — runs the models inside the app instead of
  a server. The heavy libraries (`torch`, `demucs` / `audio-separator`, `whisperx`)
  are **not** installed until you click **Download local engine + models**.

Precedence in "Auto", most-specific first: the **managed local server** if it's running,
else the **managed sibling container** if it's running, else a **configured remote server**
(`demucs_server_url`), else the **in-process local engine** if installed. If none is
available, the action tells you so rather than failing silently.

(The first two are mutually exclusive in practice — a containerized app can't run a local
server process, and someone with a working local server has no reason to add a container —
but the order is defined rather than accidental: a local process is closer and cheaper.)

> **Nothing heavy is ever downloaded implicitly.** No dependency and no model weight
> is fetched on plugin install, on app launch, or as a side effect of anything else —
> only when you explicitly click one of the download buttons. If you ask for a split
> before the models exist, the plugin **asks first** instead of stalling on a hidden
> multi-GB fetch.

> **Engines all yield 6 stems.** `bs_roformer_sw` is a **6-stem** BS-Roformer-SW
> model (`vocals`/`drums`/`bass`/`guitar`/`piano`/`other`), used by both the remote
> server and the local `audio-separator` engine (which loads the same
> `BS-Roformer-SW.ckpt` as the server, for parity). The local **demucs** engine
> (`htdemucs_6s`) is an equivalent 6-stem alternative. Output labels are normalized
> to canonical stem ids regardless of engine.

Lyrics can use a dedicated WhisperX host: set `whisperx.server_url` in the app config
and lyric transcription posts there (falling back to `demucs_server_url`); splitting
always uses `demucs_server_url`.

## What it does

- **Split stems** — extracts the full mix, runs source separation, writes
  `stems/<id>.ogg` back into the pak, rewrites the manifest (`stems:` +
  `stem_separation:`), and reindexes the song. The full mix is **kept** as a
  `default: false` fallback (its original file — e.g. `full.wav` — is preserved
  verbatim), so there's always a guaranteed-playable baseline.
- **Transcribe lyrics** — isolates a vocal stem (splitting first if needed), runs
  WhisperX, writes `lyrics.json` + manifest `lyrics` / `lyrics_source`, reindexes.

## Local demucs server

Settings → **Local demucs server** manages a real
[feedBack-demucs-server](https://github.com/got-feedBack/feedBack-demucs-server) on this
machine, so you don't have to stand one up yourself.

| Control | What it does |
|---|---|
| **Prepare server installation** / **Check for updates** | Checks the selected source revision, supported library versions, portable media tools and verified stem-model catalog. Shows installed and available versions without installing anything. |
| **Update server, libraries and models** | Downloads a candidate server, installs its contained FFmpeg/FFprobe pair, resolves compatible dependencies, verifies the models and runs media, import, health and separation checks before activation. |
| **Cancel update** | Cancels preparation while keeping the working installation. Activation itself must finish or roll back. |
| **Apply prepared update** | Activates a validated candidate after an older server has stopped. Starting the server also activates a pending candidate. |
| **Restore previous version** | Returns to the retained working generation. |
| **Discard prepared update** | Removes an inactive failed or canceled candidate so preparation can be retried. |
| **Start** / **Stop** | Runs it / kills it (and its worker processes). |
| **Test status** | Probes `/health` — device, GPU, per-model warmup state. |
| **Uninstall server** | Removes the source, its dependencies **and its downloaded weights**. |

**Run on** selects CPU, NVIDIA GPU or Auto for the next server start. If it
differs from the running server's selection, Settings asks you to restart the
server. **Using CPU** / **Using NVIDIA GPU** reports its actual execution device.
**GPU support installed** describes the libraries and can appear alongside
**Using CPU**.

The selected separator has its own status. **Installed and verified** means the
managed server verified the model files; **Starts when needed** means startup
preparation was intentionally skipped. **Ready to use** does not mean the model
stays loaded in memory between jobs. Older servers that do not supply verification
evidence show **Not checked at startup**. Optional transcription and pitch features
appear under details, so their state does not imply the selected separator failed.

Status includes component versions, download/install progress, errors, pending activation
and interrupted-operation recovery. A stopped server stays stopped after an update.
For a running server that supports coordinated draining, new work pauses while current
jobs and downloads finish, then the replacement starts. A legacy server stays running
until you explicitly stop it; its prepared replacement waits safely.

Updates use the **latest stable versions within the selected server's published
compatibility profile**. A newer package outside that profile is reported but is not
installed blindly. The complete dependency graph is resolved during preparation;
the lightweight version check alone cannot certify every transitive dependency.
The first model catalog covers `bs_roformer_sw` and `htdemucs_6s`. New model revisions
must be published in the catalog with verified hashes. An upstream upload by itself
does not automatically become a supported update.

Each candidate has separate source, libraries and model copies. The previous generation
is retained for rollback, so sufficient free disk space is required. Existing legacy
weights can be reused after full hash verification. Arbitrary remote models and
servers without a compatible catalog remain usable through their existing paths.

**Updating does not modify existing song stems.** Explicitly re-split a song to use
the updated model and libraries. Replace all relevant generated stems together when
comparing separation quality; replacing only the guitar file cannot remove guitar
that leaked into the other instrument files. This update does not guarantee that
every song will separate cleanly.

**Start with the app** (on by default, no-op until the server is installed) starts it
in the background on launch. This never slows startup and never downloads:

- verified managed generation → start without downloading auxiliary transcription
  models; the selected stem model was already executed during installation
- legacy weights already on disk → start **with warmup** to check model preparation;
  separation workers load the selected model again when a job starts
- weights absent → start with `--skip-warmup`, so launching can't trigger the ~5 GB fetch

**Use for the whole app** additionally writes the local URL into the app's
`demucs_server_url`, so other parts of the app use it too. Without it, only this plugin
does (and your own `demucs_server_url` is left untouched).

### GPU (CUDA)

Select **Install NVIDIA GPU support** for a supported NVIDIA installation. This
controls the libraries installed by the next installation or update; **Run on**
controls execution separately. With the updated server, GPU libraries can run
CPU separation without reinstalling dependencies or models. The updater uses
the same compatibility and execution checks across
GPU models; there is no RTX 4080-specific path. It:

- **detects an NVIDIA GPU** (via `nvidia-smi`) and ticks **Install NVIDIA GPU support** by default
  when one is present;
- installs the **CUDA torch build** (`torch==2.8.0+cu128` from PyTorch's index) — pinned
  inside the same pip resolve as the profile's dependencies, so it cannot introduce a
  conflicting dependency tree;
- **verifies after installing** that `torch.version.cuda` is actually set and a GPU is
  visible, rather than trusting the pin.

The wheels bundle the CUDA runtime; a compatible NVIDIA driver and enough device
memory are still required. GPU installations are larger than CPU installations.

To change an existing installation, select the GPU option and supported CUDA build
in Settings, check again, then apply the new candidate. The first profile supports
`cu126`, `cu128` and `cu129`; future profiles can change the supported builds. The
candidate must pass a CUDA execution probe before activation. CPU remains available
on machines without a supported CUDA configuration; this change adds no new AMD or
Intel acceleration backend.

### Requirements

Needs `pip` and a writable config dir — no `venv` (the packaged Windows app bundles the
embeddable Python, which has none). If the server can't be managed on your setup, the
section disables itself and explains why.

**Running feedBack in Docker?** This in-process install still works (the plugin and server
share the container), but it's usually CPU-only and it fattens your config volume. Use the
Docker section instead — see below.

## Docker

A containerized feedBack **cannot install the server on its host.** That's a namespace
boundary, not a missing feature: a process can only fork children into its *own*
namespaces, so there is no way for code inside a container to start something outside it.

So Settings offers Docker users the two things that *are* possible. The card only appears
when you're in a container or a Docker daemon is reachable.

### 1. Compose service — recommended

Copy the generated service into the `docker-compose.yml` you already have, then
`docker compose up -d`. That's it — the plugin finds the server automatically.

This needs no daemon access and grants nobody root on your host, which is why it's the one
we recommend.

### 2. One-click — only if you've already mounted the Docker socket

If `/var/run/docker.sock` is mounted into the feedBack container (Portainer users often do
this), the plugin can pull the image and start the server as a **sibling container** for
you: a real container on the real host, with real GPU access, and no Python on the host.

> **The Docker socket is root-equivalent on the host.** The plugin never mounts it, never
> asks you to, and never enables it — it only *uses* one you have already chosen to expose.
> If that's not a trade you want, use the compose snippet; the result is identical.

### GPU in Docker

Needs an NVIDIA driver **and** `nvidia-container-toolkit` on the **host**, and Linux or
Windows/WSL2. **macOS cannot pass a GPU into a container at all** — Docker there is a Linux
VM with no GPU passthrough, so it's always CPU.

The plugin only offers the GPU option when the *daemon* reports an `nvidia` runtime;
otherwise it says why rather than ticking a box that silently runs on CPU.

### Notes

- The container is published on **port 7866**, not 7865 — 7865 is the managed *local*
  server's port, and you may legitimately run both. (On Windows that collision does not
  even fail: both bind, and requests silently go to the wrong server. Hence the separation.)
- Model weights live in a **named volume** and survive Stop, restarts, and container
  removal — stopping the server does not cost you a 1.5 GB re-download.
- Memory: a `bs_roformer_sw` split peaks around **6 GB**. Docker Desktop's default VM may be
  smaller than that; if the container dies mid-split and restarts, raise its memory limit.

## Surfaces

- A **Stem Splitter** nav screen: job queue/dashboard, batch actions, missing lists.
- Per-song **Split stems** / **Transcribe lyrics** actions on the v3 song cards
  (registered via `libraryCardActions`).
- A **Settings** panel: engine selection, the managed demucs server, the local engine
  installer, and — in Docker, or wherever a Docker daemon is reachable — the container
  card (compose snippet + one-click).

Target Host: feedBack desktop with the v3 UI (`window.feedBack.uiVersion === 'v3'`).

## License

**AGPL-3.0-only** — the same license as the feedBack app. See [LICENSE](LICENSE).

## Personal GitHub integration test overlay

This section applies only to `test/integration-stem-github`. The overlay is
separate from the clean managed-updater and status pull-request branches.

A dedicated launcher may set these variables **only in the game child process**:

```text
FEEDBACK_STEM_TEST_REPO=vo90/feedBack-demucs-server
FEEDBACK_STEM_TEST_CONFIG_DIR=<absolute config directory of the isolated test profile>
STEM_SPLITTER_SERVER_REF=review/managed-runtime
```

The matching profile checks and downloads the personal GitHub fork through the
normal GitHub API, raw manifest, and commit archive endpoints. No local source
bundle is used. Settings identifies the configured fork and ref before checking;
installed and checked source revisions remain separately visible. The revision
field is locked while this launcher selects the test ref.

Each checked plan binds the repository, ref, profile scope, and full commit.
Changing the source requires checking again, including after restart or before
activating a prepared update. Downloaded source must match the checked manifest;
receipts record its origin and archive SHA-256. Existing dependency, model hash,
device, inference, activation, and rollback checks still apply. A failed or
unresolvable test source never falls back to an official download.

The override permits only the named personal fork. Other profiles ignore the
test repository and its forced default ref. Without the override, normal official
source selection remains in effect; previously installed origin is still recorded.
The variables are not forwarded to downloaded server subprocesses and do not
change user or machine environment settings. Model-only or library-only updates
require the installed server to come from the selected repository; include Server
when switching repositories.
