# Third-party code

External repositories are added here as git submodules (WP1 step 8), e.g.

```bash
git submodule add <SG2Loc repo URL> third_party/sg2loc
git submodule add https://github.com/y9miao/SceneGraphLoc third_party/scenegraphloc
```

Hydra itself is built inside the Docker image (`docker/hydra/`), not vendored here.
