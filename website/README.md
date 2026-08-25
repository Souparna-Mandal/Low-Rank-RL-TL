# flame-rl.com

Static holding page for the FLAME RL project. No build step, no dependencies —
`index.html` + `styles.css`, deployed as-is.

## Local preview

```bash
python -m http.server 8000 --directory website
# open http://localhost:8000
```

## Hosting on Cloudflare Pages

The domain is registered with Cloudflare, so Pages is the least-friction option:
DNS is created for you and the apex domain works without any A-record juggling.

### Option A — connect the Git repo (auto-deploy on push)

1. Push this branch:
   ```bash
   git push -u origin website/flame-rl-coming-soon
   ```
2. Cloudflare dashboard → **Workers & Pages** → **Create** → **Pages** →
   **Connect to Git** → authorise GitHub → pick `Low-Rank-RL-TL`.
3. Build settings:
   - Framework preset: **None**
   - Build command: *(leave empty)*
   - Build output directory: **`website`**
   - Production branch: **`website/flame-rl-coming-soon`** (switch to `main`
     once this is merged)
4. **Save and Deploy** → you get `flame-rl.pages.dev` in ~30 s.
5. Project → **Custom domains** → **Set up a custom domain** → `flame-rl.com`.
   Cloudflare adds the DNS record itself (CNAME flattening handles the apex).
   Repeat for `www.flame-rl.com`.
6. Optional: **Rules → Redirect Rules** → redirect `www.flame-rl.com/*` to
   `https://flame-rl.com/$1` (301) so there is one canonical host.

### Option B — direct upload from the CLI

```bash
npx wrangler login
npx wrangler pages deploy website --project-name flame-rl
```

Then attach the custom domain as in step 5 above. Re-run the deploy command to
publish updates.

### Notes

- HTTPS certificates are issued automatically; allow a few minutes after the
  domain is attached.
- Every push to a non-production branch gets its own preview URL, so content
  changes can be eyeballed before they go live.
