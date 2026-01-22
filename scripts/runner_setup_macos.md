# GitHub Self-Hosted Runner Setup Guide (macOS)

This guide walks you through installing and configuring a GitHub Self-Hosted Runner on your Mac for auto-deployment of the HEAN project.

---

## Prerequisites

- macOS (any recent version)
- Docker Desktop installed and running
- Admin/sudo access on your Mac
- GitHub repository access (owner/admin permissions)

---

## Step 1: Get Runner Installation Commands from GitHub

1. Go to your GitHub repository: **https://github.com/nadirzhon/HEAN-META**

2. Navigate to **Settings** → **Actions** → **Runners**

3. Click **New self-hosted runner**

4. Select **macOS** as the operating system

5. GitHub will show you commands like this (example):

```bash
# Download
mkdir actions-runner && cd actions-runner
curl -o actions-runner-osx-x64-2.311.0.tar.gz -L https://github.com/actions/runner/releases/download/v2.311.0/actions-runner-osx-x64-2.311.0.tar.gz
tar xzf ./actions-runner-osx-x64-2.311.0.tar.gz

# Configure
./config.sh --url https://github.com/nadirzhon/HEAN-META --token YOUR_GITHUB_TOKEN
```

**⚠️ Important:** Copy these commands from GitHub (the token is unique and expires).

---

## Step 2: Install the Runner

Open Terminal on your Mac and run the commands from GitHub:

```bash
# Create a directory for the runner
mkdir -p ~/actions-runner
cd ~/actions-runner

# Download the runner (use the exact URL from GitHub)
curl -o actions-runner-osx-x64-2.311.0.tar.gz -L https://github.com/actions/runner/releases/download/v2.311.0/actions-runner-osx-x64-2.311.0.tar.gz

# Extract the installer
tar xzf ./actions-runner-osx-x64-2.311.0.tar.gz
```

---

## Step 3: Configure the Runner with Custom Label

Run the configuration script with the **CRITICAL LABEL** `nadir-mac`:

```bash
cd ~/actions-runner

# Configure (replace TOKEN with your actual token from GitHub)
./config.sh \
  --url https://github.com/nadirzhon/HEAN-META \
  --token YOUR_GITHUB_TOKEN \
  --name "nadir-mac-runner" \
  --labels "nadir-mac"
```

**Configuration prompts:**
- **Runner group:** Press Enter (default)
- **Runner name:** `nadir-mac-runner` (or any name you prefer)
- **Work folder:** Press Enter (default: `_work`)
- **Labels:** Type `nadir-mac` and press Enter

**Expected output:**
```
√ Runner successfully added
√ Runner connection is good

# Runner settings
√ Settings Saved
```

---

## Step 4: Test the Runner (Optional but Recommended)

Before installing as a service, test the runner manually:

```bash
cd ~/actions-runner
./run.sh
```

You should see:
```
√ Connected to GitHub

Listening for Jobs
```

Press **Ctrl+C** to stop the test run.

---

## Step 5: Install as a macOS Service (Launchd)

Install the runner as a service so it starts automatically:

```bash
cd ~/actions-runner

# Install the service
./svc.sh install

# Start the service
./svc.sh start
```

**Expected output:**
```
Creating launch runner in /Users/yourname/Library/LaunchAgents/actions.runner.nadirzhon-HEAN-META.nadir-mac-runner.plist

Run as agent: actions.runner.nadirzhon-HEAN-META.nadir-mac-runner
Success
```

---

## Step 6: Verify the Service is Running

Check the runner status:

```bash
cd ~/actions-runner
./svc.sh status
```

**Expected output:**
```
status actions.runner.nadirzhon-HEAN-META.nadir-mac-runner:

/Users/yourname/Library/LaunchAgents/actions.runner.nadirzhon-HEAN-META.nadir-mac-runner.plist

Started:
25348 0 actions.runner.nadirzhon-HEAN-META.nadir-mac-runner
```

**Alternatively, check in GitHub:**
- Go to **Settings** → **Actions** → **Runners**
- You should see your runner with a green "Idle" status

---

## Step 7: Verify Auto-Deploy Works

1. Make a test commit to the `main` branch:
   ```bash
   git checkout main
   git commit --allow-empty -m "Test auto-deploy"
   git push origin main
   ```

2. Go to **Actions** tab in GitHub

3. You should see a new workflow run: **"Auto-Deploy to Mac (Self-Hosted Runner)"**

4. Click on it to see live logs

5. If successful, your Mac will have the updated containers running

---

## Managing the Runner Service

### Start the service:
```bash
cd ~/actions-runner
./svc.sh start
```

### Stop the service:
```bash
cd ~/actions-runner
./svc.sh stop
```

### Check status:
```bash
cd ~/actions-runner
./svc.sh status
```

### Uninstall the service:
```bash
cd ~/actions-runner
./svc.sh uninstall
```

### Remove the runner from GitHub:
```bash
cd ~/actions-runner
./config.sh remove --token YOUR_GITHUB_TOKEN
```

---

## Troubleshooting

### Runner not appearing in GitHub

**Check if the service is running:**
```bash
cd ~/actions-runner
./svc.sh status
```

**Check logs:**
```bash
cd ~/actions-runner
tail -f _diag/Runner_*.log
```

### Deployment fails with "Docker not found"

**Ensure Docker Desktop is running:**
```bash
docker ps
```

**Add Docker to PATH** (if needed):
```bash
echo 'export PATH="/Applications/Docker.app/Contents/Resources/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

### Permission errors

**Ensure runner user has Docker access:**
```bash
docker ps
```

If you get a permission error, add your user to the Docker group or ensure Docker Desktop is configured for your user.

### Port conflicts (8000, 3000, 6379)

**Check if ports are already in use:**
```bash
lsof -i :8000
lsof -i :3000
lsof -i :6379
```

**Stop conflicting processes or change ports in docker-compose.yml**

### Workflow doesn't trigger

**Common issues:**
- Runner is offline (check in GitHub Settings → Actions → Runners)
- Label mismatch (ensure workflow uses `nadir-mac` label)
- Branch protection rules blocking pushes
- Workflow file has syntax errors

**Check workflow syntax:**
```bash
# Use GitHub's action-validator or yamllint
yamllint .github/workflows/deploy_on_macos_runner.yml
```

---

## Security Best Practices

1. **Never run self-hosted runners on public repositories** (risk of code execution from forks)
2. **This repo is private** - we only deploy on push to `main`, never on `pull_request`
3. **The runner runs under your Mac user account** - ensure you trust all code in the repo
4. **Keep runner software updated:**
   ```bash
   cd ~/actions-runner
   ./config.sh --check
   ```

---

## Advanced: Running Multiple Runners

If you need multiple runners (e.g., for different projects):

```bash
mkdir ~/actions-runner-hean
cd ~/actions-runner-hean
# Download and configure with different labels
```

Each runner needs a unique name and can have different labels.

---

## Next Steps

After setup:
1. ✅ Runner installed and running as a service
2. ✅ Test deployment with empty commit
3. ✅ Monitor GitHub Actions tab for successful runs
4. ✅ Check containers on your Mac: `docker ps`
5. ✅ Access the app: http://localhost:8000 (API), http://localhost:3000 (UI)

---

## Support

If you encounter issues:
- Check runner logs: `~/actions-runner/_diag/Runner_*.log`
- Check GitHub Actions run logs
- Check Docker logs: `docker compose logs --tail=200`
- Run local deploy script for debugging: `./scripts/deploy_local.sh`

---

**Happy deploying! 🚀**
