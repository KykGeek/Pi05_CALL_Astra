# Project website

Public URL: https://kykgeek.github.io/Pi05_CALL_Astra/

## Deployment

GitHub Pages publishes the prebuilt static website in `docs/` on the `main` branch. No local server, Python environment, model credentials, or SSH connection is required to visit it. The original robotics runtime remains separate from the website.

In repository **Settings → Pages**, select **Deploy from a branch → main → /docs**. After a website release is pushed, GitHub rebuilds this directory. Deployment status is shown in **Actions** and **Settings → Pages**.

## Included content

- Latest supplied paper and revised experimental figures.
- English and Chinese content, with a light/dark theme switch.
- Three episode demos: Astra correction, immediate handback, and policy-only execution.
- Control ownership timeline and expandable policy/Responses logs.

PDFs, videos, images, scripts, and styles are separate files with relative paths. Upload **all** files in a build, not just `index.html`; otherwise demos or the paper will fail to load. The `.nojekyll` file preserves static assets.

## Updating the website

The frontend authoring project is maintained separately from this robotics source package. In that project, run the runtime integrity check and paper-data tests, then build with `PUBLIC_WEBSITE_BUILD=1 npm run build`. Preserve the original log files: the public-build plugin redacts server paths and private hosts in published log assets without changing recorded steps, actions, scores, or token values.

Preview the complete `dist/` directory before copying its contents into `docs/`. Review the diff, commit the website files, and push to `main`. Preserve robotics source files and any existing documentation. Stale hashed assets can be removed in a reviewed website-only cleanup after confirming nothing references them.

## Release verification

1. Open the public URL in a fresh browser session.
2. Check that the paper opens as a PDF, not a local filesystem link.
3. Play all three videos and seek through the progress bar.
4. Expand policy ranges and Astra Responses; check environment-step alignment.
5. Test both languages and themes.
6. Check the browser console and failed resource requests.

Historical demo snapshots and paper-wide aggregates have different provenance. Preserve the displayed provenance notes; do not treat an incomplete historical log as a newly measured result.
