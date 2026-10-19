# Publish the prepared repository

This is the small **code-only** project, not the earlier110MiB workspace archive.
The top folder containing README.md,rtl/,tests/ and .github/ is your repo root.
Do not put it inside another folder containing the entire old workspace.

## GitHub Desktop

1. Extract `distilbert-fpga-github-ready.zip`.
2. GitHub Desktop -> File -> Add local repository -> select extracted folder.
3. If it is not yet a repository, use the offered create-repository-here action.
4. Review files/commit, then Publish repository. Private is recommended until
   license/provenance is reviewed; public/private is your decision.

## Command line — new empty remote repository

Create an EMPTY repository on GitHub (no initial README/license/.gitignore).
Then replace the example URL below with your actual URL:

```sh
cd /path/to/extracted/distilbert-fpga-github
git init -b main
git add .
git commit -m "Add simulation-verified DistilBERT FPGA RTL and PYNQ integration"
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
git push -u origin main
```

Configure your own git user.name/user.email if requested. Sign in via Desktop or
your credential manager; never paste authentication tokens/private keys into chat.
For an existing repository, clone it and copy these files without `.git`, rather
than force-pushing or overwriting its history.

Only commit the extracted CONTENTS, not the download ZIP. `.gitignore` excludes
weights, generated payloads, caches, FPGA builds and credentials. The included
GitHub Actions workflow runs Python and synthetic RTL tests without model weights.
It has not run on GitHub until you publish and enable/run Actions.

No remote repository has been created/uploaded by preparing this folder. No
license has been chosen for you. No bitstream/board success is implied.
