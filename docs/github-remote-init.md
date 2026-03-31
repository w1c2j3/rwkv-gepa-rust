# GitHub Remote Init Template

下面是把当前目录初始化为独立 GitHub 仓库的一套最小命令模板。

## 1. 初始化本地仓库

```bash
cd ~/GitHub/rwkv-gepa-rust-v1
git init -b main
git add .
git commit -m "Initial standalone release"
```

## 2. 使用 GitHub CLI 直接创建远端

```bash
gh repo create rwkv-gepa-rust-v1 \
  --source=. \
  --private \
  --remote=origin \
  --push
```

如果你想公开仓库，把 `--private` 改成 `--public`。

## 3. 手动添加远端

```bash
git remote add origin git@github.com:<your-account>/rwkv-gepa-rust-v1.git
git push -u origin main
```

## 4. 首次发布前建议检查

```bash
cargo fmt --check
cargo check
```
