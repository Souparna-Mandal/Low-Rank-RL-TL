# flame-rl.com

Static holding page for the FLAME RL project. No build step, no dependencies —
`public/index.html` + `public/styles.css`, deployed as a Cloudflare **Worker
with static assets** (`wrangler.jsonc`).

## Local preview

```bash
python -m http.server 8000 --directory website/public
# open http://localhost:8000
```

Or with Wrangler, which serves it exactly as Cloudflare will:

```bash
cd website && npx wrangler dev
```

## Deploying

The Cloudflare dashboard now routes new projects through **Create a Worker**
rather than Pages, which is why there is no "build output directory" or
"production branch" field in the wizard — a Worker reads that from
`wrangler.jsonc` instead.

### Option A — deploy from your machine (fastest)

```bash
cd website
npx wrangler login     # opens a browser, once
npx wrangler deploy
```

Live on `flame-rl.<your-subdomain>.workers.dev` in about 30 seconds. Re-run
`npx wrangler deploy` to publish changes.

### Option B — connect the Git repo (auto-deploy on push)

In the **Create a Worker** wizard:

| Field                     | Value               |
| ------------------------- | ------------------- |
| Project name              | `flame-rl`          |
| Build command             | *(leave empty)*     |
| Deploy command            | `npx wrangler deploy` |
| Path (Advanced settings)  | `/website`          |

The project name must match `name` in `wrangler.jsonc`, and `Path` must point
at `/website` so Wrangler finds the config.

Builds listen to the repo's **default branch** (`main`). Until this branch is
merged there is nothing on `main` to deploy, so either merge first, or after
the project exists change the branch under **Settings → Build**.

## Custom domain

Worker → **Settings** → **Domains & Routes** → **Add** → **Custom domain** →
`flame-rl.com`. Cloudflare creates the DNS record and issues the certificate
itself, since the domain is already on your account. Add `www.flame-rl.com`
the same way, then optionally add a **Redirect Rule** sending
`www.flame-rl.com/*` to `https://flame-rl.com/$1` (301) so there is one
canonical host.
