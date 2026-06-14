## Description
Please describe what this PR does and how it improves the repository. List any dependencies that are required for this change.

## Related Issues
Closes #[issue-number]

## Type of Change
Please check the options that apply:
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature / project (non-breaking change which adds functionality)
- [ ] Documentation update (formatting, typos, README updates)
- [ ] Refactoring (clean-up, performance optimization, no functional changes)

## Verification Checklist
Please ensure the following checks are complete before requesting review:
- [ ] **Functional Test**: The code runs locally and the web app functions correctly.
- [ ] **Formatting**: Code is formatted and linted (run `ruff check .` and `ruff format .`).
- [ ] **No Binaries**: Verified that no model weights, large datasets, or GIF files have been added to the codebase (check `.gitignore`).
- [ ] **Documentation**: The relevant project README has been updated if changes affect hyperparameters or setup.
- [ ] **Hugging Face Sync**: (For maintainers) Verify that `HF_TOKEN` is set up to deploy the changes to Spaces.
