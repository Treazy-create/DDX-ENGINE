# Publishing this update yourself

These are instructions only. The assistant does not authenticate, commit or push
to your GitHub account. Existing remote: https://github.com/Treazy-create/DDX-ENGINE.git.

From your local project, inspect the changes first:

```powershell
Set-Location "C:\Users\HomePC\Desktop\DDX engine"
git status --short
```

Remove previously tracked generated files from Git's index. `--cached` keeps the
actual local files on disk:

```powershell
git rm -r --cached --ignore-unmatch frontend/node_modules frontend/dist models data/raw
git rm --cached --ignore-unmatch data/processed/train.jsonl data/processed/validate.jsonl data/processed/test.jsonl
git rm --cached --ignore-unmatch backend/.env backend/usage_counts.json
```

Stage the source and documentation. Inspect the file list before committing:

```powershell
git add .gitignore README.md requirements.txt backend frontend/src frontend/index.html frontend/package.json frontend/package-lock.json scripts tests docs
git diff --cached --name-only
git diff --cached --check
```

Do not commit credentials, recordings, personal patient data, local backups, or
model weights. Gitignore does not remove already-tracked files; use `git rm --cached`
for any remaining local-only file. If a real key was previously committed, revoke
it and address repository history before publishing more changes.

When the staged list is correct:

```powershell
git commit -m "Improve conversations, add voice mode and demo appointments"
git push origin main
```

If the push says the remote has newer commits, stop and inspect before merging.
Do not use force push to get around a rejected push. No history rewrite is part of
this update; older commits may still include dependencies or datasets.
