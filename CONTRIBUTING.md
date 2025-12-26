# Contributing to Agentune

## Commit Message Format

We follow [Conventional Commits](https://www.conventionalcommits.org/) for clear, consistent commit history.

### Format

```
<type>: <description>

[optional body]

[optional footer]
```

### Types

- `feat:` \- New feature  
- `fix:` \- Bug fix  
- `docs:` \- Documentation  
- `ci:` \- CI/CD changes

### Examples

**Simple:**

```
fix: handle empty dataset in sampler
docs: add API documentation for Dataset class
```

**Detailed:**

```
feat: add default parameters to feature generator

Added defaults for num_samples_for_generation (30),
num_features_to_generate (30), and num_samples_for_enrichment (200).
Modified sampler validation to handle sample_size > dataset.height.

Fixes #248
```

### Rules

- **Subject line**: ≤50 characters, imperative mood ("add" not "added"), no period
- **Body**: Explain *why*, not *what*. Write naturally with line breaks for readability.
- **Footer**: Reference issues with `Fixes #123` or `Closes #456`

## Branch Naming

Use this format: `<type>/<issue-number>-<description>`

### Examples

```
feat/248-default-generator-params
fix/256-sampler-validation
docs/update-api-docs
```

### Types

Use the same types as commits: `feat`, `fix`, `docs`, `ci`

### Rules

- Use lowercase  
- Use hyphens (not underscores)  
- Keep it short but descriptive

## Pull Requests

### Title

Use the same format as commits: `<type>: <description>`

### Description Template

```
## Summary
Brief description of what this PR does

## Details
Additional context for features or complex changes (optional for simple fixes)

## Related Issues
- Fixes #123
- Relates to #456
```

### Requirements

- All tests must pass  
- Summary and Related Issues are required  
- Details section is optional for simple changes

