# Branch Management Instructions

## Current Status
- **Current default branch**: main2
- **Desired default branch**: main
- **Branches to delete**: main2

## Steps to Complete This Task

### Step 1: Change the Default Branch to `main`

This must be done through the GitHub repository settings:

1. Go to your repository on GitHub: https://github.com/kuldeep0291-hub/MLproject101
2. Click on **Settings** tab
3. In the left sidebar, click on **Branches**
4. Under "Default branch", you'll see the current default branch (main2)
5. Click the switch/pencil icon next to it
6. Select `main` from the dropdown
7. Click **Update** or **I understand, update the default branch**
8. Confirm the change

### Step 2: Delete the `main2` Branch

After changing the default branch, you can safely delete `main2`:

**Option A: Via GitHub Web UI**
1. Go to your repository on GitHub
2. Click on the **branches** link (shows "X branches")
3. Find `main2` in the list
4. Click the trash/delete icon next to `main2`
5. Confirm the deletion

**Option B: Via Git Command Line**
```bash
git push origin --delete main2
```

## Why This Matters

- Having `main` as the default branch follows GitHub's current standard naming convention
- It ensures new pull requests and clones default to the `main` branch
- Removing unused branches (main2) keeps the repository clean and reduces confusion

## Verification

After completing these steps:
- Visit your repository on GitHub
- Verify that `main` is shown as the default branch
- Verify that `main2` no longer appears in the branches list
